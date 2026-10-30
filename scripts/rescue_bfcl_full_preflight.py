"""Pinned eight-API gold replay and deterministic native-checker preflight."""
from copy import deepcopy
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
from rescue_bfcl_native_preflight import parse_literal_call,execute,digest,serialize_node
from rescue_bfcl_native_eval import serialized_result


def world(question,mapping,stateless):
    instances={};dispatch={}
    for name in question['involved_classes']:
        cls=getattr(importlib.import_module(mapping[name]),name);instance=cls()
        if name not in stateless:instance._load_scenario(deepcopy(question['initial_config'].get(name,{})),long_context=False)
        instances[name]=instance
        for method,_ in inspect.getmembers(instance,predicate=inspect.ismethod):
            if not method.startswith('_'):dispatch[method]=name
    return instances,dispatch


def snapshot(instances):
    result={}
    for name,instance in instances.items():
        attrs={k:v for k,v in vars(instance).items() if not k.startswith('_')}
        if name=='GorillaFileSystem' and 'root' in attrs:attrs['root']=serialize_node(attrs['root'])
        result[name]=attrs
    def canonical(value):
        if isinstance(value,dict):return {type(k).__name__+":"+str(k):canonical(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [canonical(v) for v in value]
        if isinstance(value,set):return sorted((canonical(v) for v in value),key=repr)
        return value
    return digest(canonical(result))


def preflight(source):
    for filename in ('source_manifest.json','integration_manifest.json','full_backend_manifest.json'):
        manifest=json.loads((source/filename).read_text())
        for name,h in manifest['files'].items():
            if hashlib.sha256((source/name).read_bytes()).hexdigest()!=h:raise ValueError('Changed source '+name)
    package=source/'berkeley-function-call-leaderboard';sys.path.insert(0,str(package))
    config=importlib.import_module('bfcl_eval.constants.executable_backend_config')
    checkers=importlib.import_module('bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker')
    data=package/'bfcl_eval/data'
    questions=list(map(json.loads,(data/'BFCL_v4_multi_turn_base.json').read_text().splitlines()))
    answers={x['id']:x for x in map(json.loads,(data/'possible_answer/BFCL_v4_multi_turn_base.json').read_text().splitlines())}
    records=[]
    for q in questions:
        a,dispatch=world(q,config.CLASS_FILE_PATH_MAPPING,config.STATELESS_CLASSES)
        b,dispatch_b=world(q,config.CLASS_FILE_PATH_MAPPING,config.STATELESS_CLASSES)
        assert dispatch==dispatch_b
        initial=snapshot(a);history=[];exceptions=[];returned_errors=0;turns=[];calls=0
        for turn_index,gold in enumerate(answers[q['id']]['ground_truth']):
            observations=[];expected=[]
            for call in gold:
                calls+=1
                try:
                    name,_,_=parse_literal_call(call,dispatch)
                    values=[]
                    for instances in (a,b):
                        try:value=execute(instances[dispatch[name]],call,dispatch)
                        except Exception as e:
                            exceptions.append({'turn':turn_index,'call_sha256':hashlib.sha256(call.encode()).hexdigest(),'type':type(e).__name__,'message':str(e)})
                            value='Error during execution: '+str(e)
                        returned_errors+=isinstance(value,dict) and 'error' in value
                        values.append(serialized_result(value))
                    observations.append(values[0]);expected.append(values[1])
                except (ValueError,SyntaxError) as e:
                    exceptions.append({'turn':turn_index,'type':'unsupported_literal_call','message':str(e)})
            history.extend(observations)
            state=checkers.state_checker(a,b)['valid'];response=checkers.response_checker(history,expected,turn_index)['valid']
            turns.append({'turn':turn_index,'state_reproducible':state,'responses_reproducible':response})
        records.append({'case_id':q['id'],'classes':q['involved_classes'],'gold_calls':calls,'gold_turns':len(turns),'exceptions':exceptions,'error_observations_both_replays':returned_errors,'root_or_public_state_changed':initial!=snapshot(a),'checks_reproducible':all(t['state_reproducible'] and t['responses_reproducible'] for t in turns),'turn_checks':turns})
    return {'kind':'full_original_base_gold_backend_preflight','official_commit':manifest['commit'],'cases':len(records),'gold_calls':sum(r['gold_calls'] for r in records),'gold_turns':sum(r['gold_turns'] for r in records),'cases_without_exceptions':sum(not r['exceptions'] for r in records),'native_checks_reproducible_cases':sum(r['checks_reproducible'] for r in records),'records':records,'limitations':['Gold replay only, not model accuracy','Exact same gold path in independent instances tests executable reproducibility, not ground-truth semantic validity','Source simulation APIs have no network/message delivery; not real external actions']}

if __name__=='__main__':
    result=preflight(Path(sys.argv[1]));Path(sys.argv[2]).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print({k:result[k] for k in ('cases','gold_calls','gold_turns','cases_without_exceptions','native_checks_reproducible_cases')})
