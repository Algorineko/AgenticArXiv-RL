"""Instruction-level candidate generation for the three declared RSI controls.

This module proposes text only. Evaluation, acceptance, generation scheduling
and data access remain external and must be frozen in the experiment protocol.
Valid syntax is never interpreted as a successful or accepted improvement.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json


ARMS = {"fixed_experience", "fixed_outer", "recursive"}
TARGETS = {"solver", "improver"}
MAX_TEXT_CHARS = 12000


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def text_digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class ProposalRequest:
    messages: list
    tools: list
    metadata: dict


def make_request(arm, target, solver, active_improver, seed_improver, experience):
    if arm not in ARMS or target not in TARGETS:
        raise ValueError("Unknown arm or target")
    if arm == "fixed_experience" and target == "improver":
        raise ValueError("The fixed-experience arm cannot modify its improver")
    for text in (solver, active_improver, seed_improver):
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS:
            raise ValueError("Instruction text must be nonempty and within the text cap")
    parent = solver if target == "solver" else active_improver
    uses_active = arm == "recursive" or (arm == "fixed_outer" and target == "solver")
    policy = active_improver if uses_active else seed_improver
    evidence = deepcopy(experience)
    user = {"target": target, "current_instruction": parent, "experience": evidence}
    messages = [{"role": "system", "content": policy +
                 "\nYou may revise only the target instruction string. "
                 "Return one submit_revision tool call containing the complete replacement. "
                 "The executor, tools, API budgets and independent evaluator are immutable. "
                 "An improver instruction must remain usable to propose later solver or improver revisions."},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False, sort_keys=True)}]
    tools = [{"type": "function", "function": {
        "name": "submit_revision", "description": "Submit a candidate instruction; it is not accepted until evaluated.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string", "maxLength": MAX_TEXT_CHARS}},
                       "required": ["text"], "additionalProperties": False}}}]
    metadata = {"arm": arm, "target": target, "parent_text_sha256": text_digest(parent),
                "proposer_policy_sha256": text_digest(policy),
                "seed_improver_sha256": text_digest(seed_improver),
                "active_improver_sha256": text_digest(active_improver),
                "solver_sha256": text_digest(solver), "experience_sha256": digest(evidence),
                "request_content_sha256": digest({"messages": messages, "tools": tools}),
                "policy_source": "active_improver" if uses_active else "seed_improver"}
    return ProposalRequest(messages, tools, metadata)


def propose(api, request, identifier):
    """Use the shared attempt/token-accounting client, including its retries."""
    response = api.chat(request.messages, request.tools, identifier)
    record = deepcopy(request.metadata)
    record.update(valid_syntax=False, response_sha256=digest(response))
    choices = response.get("choices") or []
    if len(choices) != 1 or choices[0].get("finish_reason") == "length":
        record["invalid_reason"] = "missing choice or truncated generation"
        return record, None
    calls = (choices[0].get("message") or {}).get("tool_calls") or []
    if len(calls) != 1 or (calls[0].get("function") or {}).get("name") != "submit_revision":
        record["invalid_reason"] = "expected exactly one submit_revision call"
        return record, None
    try:
        arguments = json.loads(calls[0]["function"].get("arguments", "{}"))
    except (ValueError, TypeError):
        arguments = None
    if (not isinstance(arguments, dict) or set(arguments) != {"text"}
            or not isinstance(arguments["text"], str) or not arguments["text"].strip()
            or len(arguments["text"]) > MAX_TEXT_CHARS):
        record["invalid_reason"] = "invalid replacement text"
        return record, None
    text = arguments["text"]
    record.update(valid_syntax=True, candidate_text_sha256=text_digest(text),
                  unchanged=text == json.loads(request.messages[1]["content"])["current_instruction"])
    return record, text
