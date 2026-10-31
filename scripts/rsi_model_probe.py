"""One bounded native-tool connectivity probe; no task or reference contents."""
import argparse
import hashlib
import json
from pathlib import Path

from rsi_api_pilot import API, utc
from rsi_cpu_tasks import function


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["doubao-seed-2.0-mini", "qwen/qwen3.6-27b"], required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    limits = {"calls": 2, "input_tokens": 30000, "output_tokens": 2048}
    api = API("https://api.qnaigc.com/v1", args.model, args.output, limits,
              min_interval=15, max_completion_tokens=1024)
    protocol = {"requested_model": args.model, "limits": limits, "created_utc": utc(),
                "max_completion_tokens": 1024, "purpose": "Neutral connectivity only, not capability or weight identity",
                "sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                    Path(__file__), Path(__file__).with_name("rsi_api_pilot.py"),
                    Path(__file__).with_name("rsi_cpu_tasks.py"))}}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    status, response = "not_started", {}
    try:
        response = api.chat([{"role": "user", "content": "Call probe_echo exactly once with value 7."}],
                            [function("probe_echo", "Echo a neutral integer.", {"value": {"type": "integer"}}, ("value",))],
                            "neutral-model-connectivity")
        status = "response_received"
    except RuntimeError:
        status = "api_failure"
    choice = (response.get("choices") or [{}])[0]
    calls = (choice.get("message") or {}).get("tool_calls") or []
    valid = False
    if len(calls) == 1 and (calls[0].get("function") or {}).get("name") == "probe_echo":
        try:
            valid = json.loads(calls[0]["function"].get("arguments", "{}")) == {"value": 7}
        except (ValueError, TypeError):
            pass
    summary = {"status": status, "requested_model": args.model, "returned_model": response.get("model"),
               "native_tool_call_valid": valid and choice.get("finish_reason") != "length",
               "finish_reason": choice.get("finish_reason"), "usage": api.totals(),
               "limitation": "Catalog and response aliases do not attest actual backend weights; no task capability claim."}
    (args.output / "summary.json").write_text(json.dumps(api.clean(summary), indent=2) + "\n")
    print(json.dumps(api.clean(summary)))


if __name__ == "__main__":
    main()
