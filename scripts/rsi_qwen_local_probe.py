"""One bounded, single-GPU open-weight native-tool connectivity check."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, choices=range(4), default=1)
    args = parser.parse_args()
    if args.output.exists() or args.raw_output.exists():
        raise SystemExit("Refusing to overwrite an existing probe")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise SystemExit("Exactly the selected physical GPU must be visible")
    memory = subprocess.check_output(["nvidia-smi", "-i", str(args.physical_gpu), "--query-gpu=memory.free",
                                      "--format=csv,noheader,nounits"], text=True)
    available_mib = int(memory.strip())
    if available_mib < 40000:
        raise SystemExit(f"Selected GPU has only {available_mib} MiB free; skip")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("Expected one visible CUDA device")
    torch.cuda.set_per_process_memory_fraction(0.60, 0)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    tools = [{"type": "function", "function": {"name": "probe_echo", "description": "Echo an integer.",
              "parameters": {"type": "object", "properties": {"value": {"type": "integer"}},
                             "required": ["value"], "additionalProperties": False}}}]
    messages = [{"role": "user", "content": "Call probe_echo exactly once with value 7."}]
    prompt = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
    encoded = tokenizer([prompt], return_tensors="pt")
    if encoded["input_ids"].shape[-1] > 4096:
        raise SystemExit("Unexpectedly long neutral prompt")
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16,
                                                 device_map={"": 0}, low_cpu_mem_usage=True,
                                                 local_files_only=True, trust_remote_code=False)
    free_after_load, _ = torch.cuda.mem_get_info(0)
    if free_after_load < 4 * 1024**3:
        raise SystemExit("Less than 4 GiB remains after model load; skip generation")
    encoded = encoded.to("cuda:0")
    with torch.inference_mode():
        generated = model.generate(**encoded, max_new_tokens=256, do_sample=False, use_cache=True)
    new_tokens = generated[0, encoded["input_ids"].shape[-1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=False)
    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", raw, re.DOTALL)
    valid = False
    if len(blocks) == 1:
        try:
            call = json.loads(blocks[0])
            arguments = call.get("arguments")
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            valid = call.get("name") == "probe_echo" and arguments == {"value": 7}
        except (TypeError, ValueError, AttributeError):
            pass
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.write_text(raw)
    record = {"stage": "single neutral tool-call connectivity, not task performance",
              "native_tool_call_valid": valid, "model_id": "Qwen/Qwen3-8B",
              "gpu_name": torch.cuda.get_device_name(0), "physical_gpu": args.physical_gpu,
              "memory_before_mib": available_mib, "max_memory_fraction": 0.60,
              "gpu_total_memory_mib": round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 1),
              "peak_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 1024**2, 1),
              "peak_reserved_mib": round(torch.cuda.max_memory_reserved(0) / 1024**2, 1),
              "prompt_tokens": int(encoded["input_ids"].shape[-1]), "generated_tokens": int(len(new_tokens)),
              "tool_call_blocks": len(blocks), "do_sample": False, "thinking_enabled": False,
              "torch_version": torch.__version__,
              "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "config_sha256": hashlib.sha256((args.model_dir / "config.json").read_bytes()).hexdigest(),
              "limitation": "One neutral tool probe only, no RSI or task effectiveness."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
