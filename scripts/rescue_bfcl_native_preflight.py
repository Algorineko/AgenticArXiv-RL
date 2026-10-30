"""Replay filesystem-only official BFCL v4 gold trajectories on pinned native code.

This is backend preflight, not a model leaderboard score or full official checker.
No model text is evaluated as Python; dispatch literal calls to public methods.
"""
import argparse
import ast
from copy import deepcopy
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys


def parse_literal_call(text,allowed):
    node=ast.parse(text,mode='eval').body
    if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name) or node.func.id not in allowed or node.func.id.startswith('_'):
        raise ValueError('Only whitelisted direct tool calls are supported')
    positional=[ast.literal_eval(v) for v in node.args]
    keywords={}
    for k in node.keywords:
        if k.arg is None or k.arg in keywords:raise ValueError('Unpacking/duplicate keywords forbidden')
        keywords[k.arg]=ast.literal_eval(k.value)
    return node.func.id,positional,keywords


def execute(instance,text,allowed):
    name,args,kw=parse_literal_call(text,allowed)
    function=getattr(instance,name)
    inspect.signature(function).bind(*args,**kw)
    return function(*args,**kw)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()


def serialize_node(node):
    if hasattr(node,'contents'):
        return {'name':node.name,'contents':{k:serialize_node(v) for k,v in sorted(node.contents.items())}}
    return {'name':node.name,'content':node.content}


def preflight(source_root):
    manifest=json.loads((source_root/'source_manifest.json').read_text())
    for name,h in manifest['files'].items():
        if hashlib.sha256((source_root/name).read_bytes()).hexdigest()!=h:raise ValueError('Changed official source: '+name)
    package_root=source_root/'berkeley-function-call-leaderboard';sys.path.insert(0,str(package_root))
    module=importlib.import_module('bfcl_eval.eval_checker.multi_turn_eval.func_source_code.gorilla_file_system')
    backend=module.GorillaFileSystem
    allowed={n for n,f in inspect.getmembers(backend,inspect.isfunction) if not n.startswith('_')}
    data=package_root/'bfcl_eval/data'
    questions={r['id']:r for r in map(json.loads,(data/'BFCL_v4_multi_turn_base.json').read_text().splitlines())}
    answers={r['id']:r for r in map(json.loads,(data/'possible_answer/BFCL_v4_multi_turn_base.json').read_text().splitlines())}
    records=[];excluded=[]
    for key,q in questions.items():
        if 'GorillaFileSystem' not in q['initial_config']:continue
        turns=answers[key]['ground_truth'];calls=[x for turn in turns for x in turn]
        try:
            for call in calls:parse_literal_call(call,allowed)
        except (ValueError,SyntaxError):excluded.append(key);continue
        instance=backend();instance._load_scenario(deepcopy(q['initial_config']['GorillaFileSystem']),long_context=False)
        initial=digest(serialize_node(instance.root));outputs=[];errors=[];observed_errors=0
        for turn_index,turn in enumerate(turns):
            for call_index,call in enumerate(turn):
                try:
                    result=execute(instance,call,allowed);observed_errors+=isinstance(result,dict) and 'error' in result
                    outputs.append({'turn':turn_index,'call':call_index,'observation_sha256':digest(result)})
                except Exception as exc:
                    errors.append({'turn':turn_index,'call':call_index,'error_type':type(exc).__name__,'error':str(exc)})
        records.append({'id':key,'turns':len(turns),'calls':len(calls),'returned_error_observations':observed_errors,'exceptions':errors,'initial_root_sha256':initial,'final_root_sha256':digest(serialize_node(instance.root)),'final_pwd_sha256':digest(instance.pwd()),'observation_hashes':outputs})
    return {'kind':'official_native_filesystem_reference_preflight','official_repo':manifest['repo'],'official_commit':manifest['commit'],'source_manifest_sha256':hashlib.sha256((source_root/'source_manifest.json').read_bytes()).hexdigest(),'native_backend':'GorillaFileSystem','selected_cases':len(records),'selected_calls':sum(r['calls'] for r in records),'cases_without_exceptions':sum(not r['exceptions'] for r in records),'root_state_changed_cases':sum(r['initial_root_sha256']!=r['final_root_sha256'] for r in records),'excluded_mixed_or_unsupported':excluded,'records':records,'limitations':['Gold replay only; no model evaluated','Filesystem-only trajectories selected by supported native functions, not model scores','This is not the full official BFCL state/response/path checker or a BFCL leaderboard result','Source initial configurations and all gold turns retained; non-filesystem distractor tools not instantiated in this preflight']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=preflight(a.source_root);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('selected_cases','selected_calls','cases_without_exceptions','root_state_changed_cases')}))
