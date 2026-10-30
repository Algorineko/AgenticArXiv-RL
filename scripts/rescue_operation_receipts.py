"""Narrow request compiler and offline operation-receipt checker.

Uses only user text and executed observations. Does not read benchmark gold.
Receipts prove an offline call contract, never real PDF/translation quality.
"""
from __future__ import annotations
import ast
import json
import re


def compile_request(text):
    """Fail closed outside a single explicit numbered translate/force request."""
    refs = re.findall(r'第\s*(\d+)\s*篇', text)
    if len(refs) != 1 or int(refs[0]) < 1:
        return None
    translate = '翻译' in text
    download = '下载' in text
    if translate == download or any(x in text for x in ('不要', '不必', '无需', '不要再', '先', '然后', '再翻译', '两篇')):
        return None
    if translate:
        return {'name': 'translate_arxiv_pdf', 'ref': int(refs[0])}
    if not re.search(r'force\s*=\s*true', text, re.I):
        return None
    return {'name': 'download_arxiv_pdf', 'ref': int(refs[0]), 'force': True}


def decode_action(value):
    if isinstance(value, dict):
        return value
    try:
        action = json.loads(str(value))
    except (ValueError, TypeError):
        return None
    return action if isinstance(action, dict) else None


def receipt(step, request, target_paper_id):
    action = decode_action(step.get('action'))
    if not action or action.get('name') != request['name']:
        return False
    args = action.get('args') or {}
    if not isinstance(args, dict):
        return False
    if request.get('force') and args.get('force') is not True:
        return False
    observation = str(step.get('observation', ''))
    if request['name'] == 'translate_arxiv_pdf':
        m = re.fullmatch(r'已创建翻译任务 task_id=([A-Za-z0-9_-]+), paper_id=([A-Za-z0-9._/-]+)，状态=PENDING。.*', observation, re.S)
        return bool(m and m[2] == target_paper_id)
    try:
        obj = ast.literal_eval(observation)
    except (ValueError, SyntaxError):
        return False
    return (isinstance(obj, dict) and obj.get('paper_id') == target_paper_id
            and obj.get('status') == 'READY' and isinstance(obj.get('size_bytes'), int)
            and obj['size_bytes'] > 0 and bool(obj.get('local_path')))


def assess(history, request, target_paper_id):
    seen = any(receipt(step, request, target_paper_id) for step in history)
    finished = bool(history and str(history[-1].get('action', '')).strip() == 'FINISH')
    return {'operation_seen': seen, 'operation_and_finish': seen and finished}


def history_steps(history_text):
    """Parse only current ReAct records, ignoring the setup-state prose."""
    return [{'action': m[1].strip(), 'observation': m[2].strip()} for m in
            re.finditer(r'(?:^|\n\n)Thought:.*?\nAction: (.*?)\nObservation: (.*?)(?=\n\nThought:|$)',
                        history_text, re.S)]
