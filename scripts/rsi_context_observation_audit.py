"""Aggregate final-request tool-output sizes without exporting task content."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--private-run', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
calls = [json.loads(line) for line in (a.private_run / 'api_calls.jsonl').read_text().splitlines()]
result = []
for episode in sorted({call['episode_id'] for call in calls}):
    last = [call for call in calls if call['episode_id'] == episode][-1]
    messages = last['request']['messages']
    tool_text = [m.get('content') or '' for m in messages if m['role'] == 'tool']
    invocations = [c['function'] for m in messages if m['role'] == 'assistant' for c in m.get('tool_calls', [])]
    invocation_groups = collections.Counter(hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest() for x in invocations)
    tool_names = collections.Counter(x['name'] for x in invocations)
    groups = collections.Counter(hashlib.sha256(x.encode()).hexdigest() for x in tool_text)
    lengths = {hashlib.sha256(x.encode()).hexdigest(): len(x) for x in tool_text}
    result.append({'episode_sha256_prefix': episode[:12], 'last_prompt_tokens': last['usage']['prompt_tokens'],
        'invocations_by_tool': dict(tool_names),
        'identical_invocation_group_sizes': sorted((n for n in invocation_groups.values() if n > 1), reverse=True),
        'tool_messages': len(tool_text), 'tool_content_characters': sum(map(len, tool_text)),
        'unique_exact_tool_contents': len(groups), 'duplicate_group_sizes': sorted((n for n in groups.values() if n > 1), reverse=True),
        'duplicate_tool_content_characters': sum((n-1)*lengths[d] for d,n in groups.items()),
        'system_content_characters': sum(len(m.get('content') or '') for m in messages if m['role'] == 'system'),
        'user_content_characters': sum(len(m.get('content') or '') for m in messages if m['role'] == 'user')})
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps({'episodes': result, 'note': 'Characters are not model tokens. Exact duplicate content is not evidence that removing messages preserves tool-call semantics.'}, indent=2) + '\n')
