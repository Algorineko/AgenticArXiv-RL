"""Bounded single-GPU Qwen native-tool inference over a private stdio channel."""
import argparse
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def normalize_messages(messages):
    converted = copy.deepcopy(messages)
    for message in converted:
        for tool in message.get("tool_calls", []):
            function = tool.get("function") or {}
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    function["arguments"] = json.loads(arguments)
                except ValueError:
                    pass
    return converted


def parse_generation(raw, request_number):
    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", raw, re.DOTALL)
    if raw.count("<tool_call>") != len(blocks) or raw.count("</tool_call>") != len(blocks):
        return None
    result = []
    for index, block in enumerate(blocks):
        try:
            call = json.loads(block)
            args = call.get("arguments")
            if isinstance(args, str):
                args = json.loads(args)
            name = call.get("name")
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name) or not isinstance(args, dict):
                return None
            result.append({"id": f"qwen_{request_number}_{index}", "type": "function",
                           "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
        except (ValueError, TypeError, AttributeError):
            return None
    return result


def send(value):
    sys.stdout.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--physical-gpu", type=int, choices=range(4), required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--max-context-tokens", type=int, default=8192)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    args = p.parse_args()
    if not (1 <= args.max_context_tokens <= 8192 and 1 <= args.max_new_tokens <= 1024):
        p.error("Maximum context is 8192 and maximum generation is 1024 tokens")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise SystemExit("Exactly the selected physical GPU must be visible")
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("model_id") != "Qwen/Qwen3-8B" or not manifest.get("files"):
        raise SystemExit("Unexpected or missing model manifest")
    for name, metadata in manifest["files"].items():
        if Path(name).name != name:
            raise SystemExit("Invalid model manifest filename")
        path = args.model_dir / name
        if path.stat().st_size != metadata["bytes"]:
            raise SystemExit("Model file length differs from manifest")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(16 * 1024 * 1024):
                digest.update(block)
        if digest.hexdigest() != metadata["sha256"]:
            raise SystemExit("Model file hash differs from manifest")
    free = int(subprocess.check_output(["nvidia-smi", "-i", str(args.physical_gpu),
                                        "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
    if free < 40000:
        raise SystemExit("Insufficient free memory to start the single-GPU server")
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise SystemExit("Expected one visible CUDA device")
        torch.cuda.set_per_process_memory_fraction(0.60, 0)
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16,
                                                     device_map={"": 0}, low_cpu_mem_usage=True,
                                                     local_files_only=True, trust_remote_code=False)
    send({"status": "ready", "model_id": "Qwen/Qwen3-8B", "physical_gpu": args.physical_gpu,
          "gpu_name": torch.cuda.get_device_name(0), "memory_before_mib": free,
          "gpu_total_memory_mib": round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 1),
          "max_memory_fraction": 0.60, "max_context_tokens": args.max_context_tokens,
          "max_new_tokens": args.max_new_tokens,
          "peak_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 1024**2, 1),
          "peak_reserved_mib": round(torch.cuda.max_memory_reserved(0) / 1024**2, 1),
          "model_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
          "config_sha256": hashlib.sha256((args.model_dir / "config.json").read_bytes()).hexdigest(),
          "server_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    for sequence, line in enumerate(sys.stdin, 1):
        try:
            request = json.loads(line)
            if request == {"shutdown": True}:
                send({"status": "shutdown"})
                break
            if (not isinstance(request, dict) or set(request) != {"messages", "tools", "max_tokens"}
                    or not isinstance(request["messages"], list) or not isinstance(request["tools"], list)
                    or type(request["max_tokens"]) is not int or not 1 <= request["max_tokens"] <= args.max_new_tokens
                    or len(line.encode()) > 600000):
                send({"status": "invalid_request"})
                continue
            with contextlib.redirect_stdout(sys.stderr):
                prompt = tokenizer.apply_chat_template(normalize_messages(request["messages"]),
                    tools=request["tools"] or None, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                encoded = tokenizer([prompt], return_tensors="pt")
            input_tokens = int(encoded["input_ids"].shape[-1])
            if input_tokens > args.max_context_tokens:
                send({"status": "context_limit", "usage": {"prompt_tokens": input_tokens, "completion_tokens": 0}})
                continue
            free_bytes, _ = torch.cuda.mem_get_info(0)
            if free_bytes < 8 * 1024**3:
                send({"status": "resource_unavailable", "usage": {"prompt_tokens": input_tokens, "completion_tokens": 0}})
                break
            encoded = encoded.to("cuda:0")
            with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
                generated = model.generate(**encoded, max_new_tokens=request["max_tokens"], do_sample=False, use_cache=True)
            suffix = generated[0, input_tokens:]
            raw = tokenizer.decode(suffix, skip_special_tokens=False)
            calls = parse_generation(raw, sequence)
            eos = tokenizer.eos_token_id
            eos_ids = set(eos if isinstance(eos, list) else [eos])
            truncated = len(suffix) >= request["max_tokens"] and int(suffix[-1]) not in eos_ids
            reason = "length" if truncated else ("model_parse_error" if calls is None else "tool_calls" if calls else "stop")
            content = re.sub(r"<tool_call>\s*.*?\s*</tool_call>", "", raw, flags=re.DOTALL)
            content = content.replace("<|im_end|>", "").strip()
            message = {"role": "assistant", "content": content}
            if calls:
                message["tool_calls"] = calls
            send({"status": "ok", "response": {"model": "Qwen/Qwen3-8B",
                  "choices": [{"finish_reason": reason, "message": message}],
                  "usage": {"prompt_tokens": input_tokens, "completion_tokens": int(len(suffix))}},
                  "peak_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 1024**2, 1),
                  "peak_reserved_mib": round(torch.cuda.max_memory_reserved(0) / 1024**2, 1)})
        except torch.cuda.OutOfMemoryError:
            send({"status": "resource_unavailable"})
            break
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            send({"status": "invalid_request"})


if __name__ == "__main__":
    main()
