"""Budgeted API-compatible client for a private single-GPU Qwen stdio server."""
import json
from pathlib import Path
import selectors
import subprocess
import time

from rsi_api_pilot import BudgetExceeded, canonical, digest, utc


class LocalQwenClient:
    def __init__(self, command, output, limits, expected, max_completion_tokens=1024,
                 startup_timeout_s=300, response_timeout_s=180):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.limits = dict(limits)
        self.max_completion_tokens = max_completion_tokens
        self.response_timeout_s = response_timeout_s
        self.calls = self.input_tokens = self.output_tokens = self.unknown_usage = 0
        self.stderr = (self.output / "local_model_stderr.log").open("w")
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.stderr, text=True, bufsize=1)
        try:
            self.ready = self._read(startup_timeout_s)
            if (self.ready.get("status") != "ready" or
                    any(self.ready.get(key) != value for key, value in expected.items())):
                raise RuntimeError("Local model ready metadata did not match the frozen protocol")
        except Exception:
            self.close()
            raise

    def _read(self, timeout_s):
        end = time.clock_gettime(time.CLOCK_MONOTONIC) + timeout_s
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while True:
                remaining = end - time.clock_gettime(time.CLOCK_MONOTONIC)
                if remaining <= 0:
                    raise RuntimeError("Local model response timeout")
                if selector.select(min(remaining, 5)):
                    line = self.process.stdout.readline()
                    if not line:
                        raise RuntimeError("Local model closed its stdio channel")
                    try:
                        value = json.loads(line)
                    except ValueError as exc:
                        raise RuntimeError("Local model returned invalid protocol JSON") from exc
                    if not isinstance(value, dict):
                        raise RuntimeError("Local model returned a non-object response")
                    return value
                if self.process.poll() is not None:
                    raise RuntimeError("Local model process exited")

    def append(self, filename, value):
        with (self.output / filename).open("a") as stream:
            stream.write(canonical(value) + "\n")

    def chat(self, messages, tools, episode_id):
        payload = {"messages": messages, "tools": tools, "max_tokens": self.max_completion_tokens}
        reservation = len(canonical(payload).encode()) + 4096
        if (self.calls >= self.limits["calls"] or self.input_tokens + reservation > self.limits["input_tokens"]
                or self.output_tokens + self.max_completion_tokens > self.limits["output_tokens"]):
            raise BudgetExceeded("Local pre-request budget reservation rejected")
        self.calls += 1
        started = time.clock_gettime(time.CLOCK_MONOTONIC)
        record = {"call": self.calls, "episode_id": episode_id, "started_utc": utc(),
                  "request_sha256": digest(payload), "request": payload,
                  "reserved_input_upper_estimate": reservation}
        try:
            self.process.stdin.write(canonical(payload) + "\n")
            self.process.stdin.flush()
            received = self._read(self.response_timeout_s)
        except (RuntimeError, OSError, ValueError, BrokenPipeError) as exc:
            self.input_tokens += reservation
            self.output_tokens += self.max_completion_tokens
            self.unknown_usage += 1
            record.update(status="transport_error", error_type=type(exc).__name__,
                          usage_accounting="unknown_conservatively_reserved",
                          elapsed_s=round(time.clock_gettime(time.CLOCK_MONOTONIC) - started, 4))
            self.append("api_calls.jsonl", record)
            raise RuntimeError("Local model transport failed; details remain private") from exc
        status = received.get("status")
        response = received.get("response") or {}
        usage = response.get("usage") or received.get("usage") or {}
        prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if type(prompt) is int and prompt >= 0 and type(completion) is int and completion >= 0:
            self.input_tokens += prompt
            self.output_tokens += completion
            record["usage_accounting"] = "local_tokenizer"
        else:
            self.input_tokens += reservation
            self.output_tokens += self.max_completion_tokens
            self.unknown_usage += 1
            record["usage_accounting"] = "unknown_conservatively_reserved"
        record.update(status=status, response=response, usage=usage,
                      peak_allocated_mib=received.get("peak_allocated_mib"),
                      peak_reserved_mib=received.get("peak_reserved_mib"),
                      elapsed_s=round(time.clock_gettime(time.CLOCK_MONOTONIC) - started, 4))
        self.append("api_calls.jsonl", record)
        if self.input_tokens > self.limits["input_tokens"] or self.output_tokens > self.limits["output_tokens"]:
            raise BudgetExceeded("Local tokenizer usage exceeded conservative reservation")
        if status == "ok":
            return response
        reasons = {"context_limit": "model_context_limit", "resource_unavailable": "model_resource_unavailable",
                   "invalid_request": "model_protocol_error"}
        if status in reasons:
            return {"choices": [{"finish_reason": reasons[status], "message": {"role": "assistant", "content": ""}}]}
        raise RuntimeError("Unexpected local model server status")

    def totals(self):
        return {"calls": self.calls, "accounted_input_tokens": self.input_tokens,
                "accounted_output_tokens": self.output_tokens, "calls_without_provider_usage": self.unknown_usage,
                "source": "local tokenizer; not provider billing"}

    def close(self):
        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            try:
                process.stdin.write('{"shutdown":true}\n')
                process.stdin.flush()
                self._read(5)
                process.wait(timeout=5)
            except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        if process is not None:
            for stream in (process.stdin, process.stdout):
                if stream:
                    stream.close()
        if getattr(self, "stderr", None):
            self.stderr.close()
