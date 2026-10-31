"""Bounded API baseline preflight for RSI research; stdlib only, no GPU."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from rsi_cpu_tasks import Environment, catalog, validate


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def utc():
    # AppWorld/freezegun freezes datetime.now and time.monotonic in the host.
    # Explicit native clocks keep API accounting independent of simulated time.
    return dt.datetime.fromtimestamp(time.clock_gettime(time.CLOCK_REALTIME), dt.timezone.utc).isoformat()


def monotonic():
    return time.clock_gettime(time.CLOCK_MONOTONIC)


class BudgetExceeded(RuntimeError):
    pass


class API:
    def __init__(self, endpoint, model, output, limits, min_interval, max_completion_tokens=512):
        self.key = os.environ.get("RSI_API_KEY", "")
        if not self.key:
            raise ValueError("RSI_API_KEY is required; credentials are never written to artifacts")
        self.endpoint, self.model, self.output = endpoint.rstrip("/"), model, output
        if not self.endpoint.startswith("https://"):
            raise ValueError("HTTPS endpoint required")
        self.limits = limits
        self.calls = self.input_tokens = self.output_tokens = self.unknown_usage = 0
        self.min_interval, self.last_start = min_interval, 0.0
        self.max_completion_tokens = max_completion_tokens

    def clean(self, value):
        return json.loads(json.dumps(value, ensure_ascii=False).replace(self.key, "[REDACTED]"))

    def append(self, filename, value):
        with (self.output / filename).open("a") as stream:
            stream.write(canonical(self.clean(value)) + "\n")

    def chat(self, messages, tools, episode_id):
        payload = {"model": self.model, "messages": messages,
                   "temperature": 0, "max_tokens": self.max_completion_tokens, "stream": False}
        if tools:
            payload.update(tools=tools, tool_choice="auto")
        # UTF-8 byte count plus overhead is a conservative reservation, not a tokenizer claim.
        reservation = len(canonical(payload).encode()) + 4096
        for retry in range(2):
            time.sleep(max(0, self.last_start + self.min_interval - monotonic()))
            if (self.calls >= self.limits["calls"] or
                self.input_tokens + reservation > self.limits["input_tokens"] or
                self.output_tokens + payload["max_tokens"] > self.limits["output_tokens"]):
                raise BudgetExceeded("pre-request budget reservation rejected")
            self.calls += 1
            self.last_start = monotonic()
            record = {"call": self.calls, "episode_id": episode_id, "started_utc": utc(),
                      "request_sha256": digest(payload), "request": payload, "retry": retry,
                      "reserved_input_upper_estimate": reservation}
            started = monotonic()
            response, status = None, None
            try:
                req = urllib.request.Request(self.endpoint + "/chat/completions",
                    data=json.dumps(payload).encode(), headers={"Authorization": "Bearer " + self.key,
                                                              "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=45) as stream:
                    response = json.load(stream)
                record.update(status="ok", response=response)
            except urllib.error.HTTPError as exc:
                status = exc.code
                record.update(status="http_error", http_status=status,
                              message=exc.read(2000).decode(errors="replace"))
            except (TimeoutError, OSError, ValueError) as exc:
                record.update(status="transport_or_response_error", error_type=type(exc).__name__)
            record["elapsed_s"] = round(monotonic() - started, 4)
            usage = (response or {}).get("usage") or {}
            prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
            if type(prompt) is int and prompt >= 0 and type(completion) is int and completion >= 0:
                self.input_tokens += prompt
                self.output_tokens += completion
                record["usage_accounting"] = "provider_reported"
            else:
                # Failures/timeouts can still be billed: charge the reservation to our cap.
                self.input_tokens += reservation
                self.output_tokens += payload["max_tokens"]
                self.unknown_usage += 1
                record["usage_accounting"] = "unknown_conservatively_reserved"
            self.append("api_calls.jsonl", record)
            if response is not None:
                if self.input_tokens > self.limits["input_tokens"] or self.output_tokens > self.limits["output_tokens"]:
                    raise BudgetExceeded("provider-reported usage exceeded conservative estimate")
                return response
            if retry == 0 and (status in (429, 500, 502, 503, 504) or status is None):
                time.sleep(30 if status == 429 else 1)
                continue
            raise RuntimeError("API request failed; sanitized details recorded in api_calls.jsonl")

    def totals(self):
        return {"calls": self.calls, "accounted_input_tokens": self.input_tokens,
                "accounted_output_tokens": self.output_tokens,
                "calls_without_provider_usage": self.unknown_usage}


def episode(api, task, repeat, max_steps, environment=None, harness_instruction=""):
    env = environment if environment is not None else Environment(task)
    identifier = f"{task['id']}:repeat{repeat}"
    messages = [{"role": "system", "content": "You operate a local tool environment. Complete the user request "
                 "through the available tools. Tool observations describe real state. Do not assert completion "
                 "without performing the requested operations. When finished, give a short final answer."},
                {"role": "user", "content": task["prompt"]}]
    if harness_instruction:
        messages[0]["content"] += "\n" + harness_instruction
    trace = {"episode_id": identifier, "task_id": task["id"], "domain": task["domain"],
             "repeat": repeat, "started_utc": utc(), "steps": [], "stop": "step_budget",
             "tool_errors": 0, "tool_calls": 0}
    try:
        for step in range(max_steps):
            response = api.chat(messages, env.tools(), identifier)
            choice = response.get("choices", [{}])[0]
            message = choice.get("message") or {}
            calls = message.get("tool_calls") or []
            action = {"role": "assistant", "content": message.get("content") or ""}
            if calls:
                # Some compatible gateways add index: don't send this nonstandard response field back.
                action["tool_calls"] = [{k: v for k, v in call.items() if k in ("id", "type", "function")}
                                        for call in calls]
            messages.append(action)
            step_record = {"step": step, "finish_reason": choice.get("finish_reason"),
                           "assistant": action, "observations": []}
            trace["steps"].append(step_record)
            if choice.get("finish_reason") == "length":
                trace["stop"] = "generation_truncated"
                break
            if choice.get("finish_reason") in {"model_parse_error", "model_context_limit",
                                                "model_resource_unavailable", "model_protocol_error"}:
                trace["stop"] = choice["finish_reason"]
                break
            if not calls:
                trace["stop"] = "assistant_final"
                break
            if len(calls) > 4:
                trace["stop"] = "parallel_tool_limit"
                break
            for call in calls:
                function = call.get("function") or {}
                try:
                    args = json.loads(function.get("arguments", "{}"))
                    observation = env.execute(function.get("name"), args)
                except (TypeError, ValueError):
                    observation = {"error": "malformed tool arguments"}
                trace["tool_calls"] += 1
                trace["tool_errors"] += int("error" in observation)
                step_record["observations"].append({"name": function.get("name"), "result": observation})
                messages.append({"role": "tool", "tool_call_id": call.get("id", ""),
                                 "content": canonical(observation)})
        trace["verdict"] = env.check()
    except BudgetExceeded:
        trace["stop"] = "global_budget"
        trace["verdict"] = env.check()
    except RuntimeError:
        trace["stop"] = "api_error"
        trace["verdict"] = env.check()
    finally:
        env.close()
    trace["completed_utc"] = utc()
    api.append("traces.jsonl", trace)
    return trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="doubao-seed-2.0-mini")
    parser.add_argument("--endpoint", default="https://api.qnaigc.com/v1")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--max-calls", type=int, default=100)
    parser.add_argument("--max-input-tokens", type=int, default=300000)
    parser.add_argument("--max-output-tokens", type=int, default=50000)
    parser.add_argument("--min-interval", type=float, default=7.0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 2 or not 1 <= args.steps <= 6 or not 1 <= args.max_calls <= 100:
        parser.error("P0 limits: repeats<=2, steps<=6, calls<=100")
    if not 0 < args.max_input_tokens <= 300000 or not 0 < args.max_output_tokens <= 50000 or args.min_interval < 7:
        parser.error("P0 caps: input<=300000, output<=50000, min interval>=7 seconds")
    args.output.mkdir(parents=True, exist_ok=False)
    controls = validate()
    (args.output / "preflight.json").write_text(json.dumps(controls, indent=2) + "\n")
    tasks = catalog()
    (args.output / "catalog.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n")
    sources = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
               for name in ("rsi_api_pilot.py", "rsi_cpu_tasks.py")}
    frozen = args.output / "frozen_sources"
    frozen.mkdir()
    for name in sources:
        (frozen / name).write_bytes((Path(__file__).parent / name).read_bytes())
    protocol = {"created_utc": utc(), "stage": "P0 development calibration, not RSI effectiveness",
                "model": args.model, "endpoint": args.endpoint, "max_steps": args.steps,
                "repeats": args.repeats, "max_parallel_tools": 4, "catalog_sha256": digest(tasks),
                "source_sha256": sources, "limits": {"calls": args.max_calls,
                "input_tokens": args.max_input_tokens, "output_tokens": args.max_output_tokens},
                "min_request_interval_s": args.min_interval,
                "scoring": "independent goal state; explicit final and truncation reported separately",
                "api_seed_claim": "No reproducibility guarantee from temperature=0 or provider model aliases"}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    if args.validate_only:
        print(canonical(controls))
        return
    api = API(args.endpoint, args.model, args.output, protocol["limits"], args.min_interval)
    results = []
    # Round-major order makes both passes observe all families; no selection from these repeats.
    for repeat in range(args.repeats):
        for task in tasks:
            trace = episode(api, task, repeat, args.steps)
            results.append(trace)
            summary = {"stage": protocol["stage"], "episodes": len(results),
                       "planned_episodes": len(tasks) * args.repeats,
                       "successes": sum(x["verdict"]["success"] for x in results),
                       "normal_completion_successes": sum(x["verdict"]["success"] and x["stop"] == "assistant_final"
                                                          for x in results),
                       "by_domain": {d: {"episodes": sum(x["domain"] == d for x in results),
                                         "successes": sum(x["domain"] == d and x["verdict"]["success"] for x in results)}
                                     for d in sorted({x["domain"] for x in results})},
                       "tool_errors": sum(x["tool_errors"] for x in results),
                       "stop_reasons": {s: sum(x["stop"] == s for x in results)
                                        for s in sorted({x["stop"] for x in results})},
                       "usage": api.totals(), "complete": len(results) == len(tasks) * args.repeats}
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(canonical({"episode_id": trace["episode_id"], "success": trace["verdict"]["success"],
                             "stop": trace["stop"], "usage": api.totals()}), flush=True)
            if trace["stop"] in ("global_budget", "api_error"):
                return


if __name__ == "__main__":
    main()
