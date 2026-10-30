"""Re-execute recorded calls and native checks without additional model inference."""
from copy import deepcopy
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'AgenticArxiv'),str(ROOT/'scripts')]
from benchmark.rescue_catalog import sha256
from rescue_bfcl_native_eval import load,serialized_result
from rescue_bfcl_native_preflight import execute,serialize_node,digest


def replay(record,q,answer,schemas,backend,checkers,interface):
    actual=backend();gold=backend()
    actual._load_scenario(deepcopy(q['initial_config']['GorillaFileSystem']))
    gold._load_scenario(deepcopy(q['initial_config']['GorillaFileSystem']))
    allowed={s['name'] for s in schemas if s['name'] not in q.get('excluded_function',[])}
    observations=[]
    if [t['turn'] for t in record['turns']]!=list(range(len(q['question']))):raise ValueError('Missing native user turns')
    for turn in record['turns']:
        current=[]
        if not 1<=len(turn['steps'])<=6 or [s['iteration'] for s in turn['steps']]!=list(range(len(turn['steps']))):raise ValueError('Changed native step budget/order')
        for step in turn['steps']:
            if step.get('parse_error'):continue
            if interface=='qwen_native':
                actions=step.get('actions',[]);saved=[v['text'] for v in step.get('observations',[])]
            else:
                action=step.get('action');actions=[action] if isinstance(action,dict) else [];saved=[step['observation']] if actions else []
            if len(actions)!=len(saved):raise ValueError('Action/observation count mismatch')
            for action,stored in zip(actions,saved):
                name=action['name'];args=action.get('args',{})
                try:
                    if name not in allowed or any(not re.fullmatch(r'[A-Za-z_]\w*',k) for k in args):raise ValueError('Unavailable function/argument')
                    call=name+'('+', '.join(k+'='+repr(v) for k,v in args.items())+')'
                    value=serialized_result(execute(actual,call,allowed))
                except Exception as exc:value='Error during execution: '+str(exc)
                if value!=stored:raise ValueError('Recorded observation does not match native re-execution')
                current.append(value)
        observations.extend(current)
        expected=[serialized_result(execute(gold,c,allowed)) for c in answer['ground_truth'][turn['turn']]]
        state=checkers.state_checker({'GorillaFileSystem':actual},{'GorillaFileSystem':gold})['valid']
        response=checkers.response_checker(observations,expected,turn['turn'])['valid']
        native=bool(current or not expected) and state and response
        if (state!=turn['state_valid'] or response!=turn['response_valid'] or native!=turn['native_checks_valid'] or bool(native and turn['finished'])!=turn['success'] or digest(serialize_node(actual.root))!=turn['actual_root_sha256'] or digest(serialize_node(gold.root))!=turn['gold_root_sha256']):raise ValueError('Recorded native checks/state differ from replay')
    if all(t['success'] for t in record['turns'])!=record['success']:raise ValueError('Recorded scenario aggregate differs')


def audit(source,artifact,roster):
    protocol=json.loads((artifact/'protocol.json').read_text());interface=protocol.get('interface','react')
    backend,checkers,cases,answers,schemas=load(source,protocol);byid={q['id']:q for q in cases};out={}
    for name in roster:
        root=artifact/'model_runs'/name;m=json.loads((root/'run_manifest.json').read_text())
        if m['status']!='completed' or m['policy']!='model' or m['protocol_sha256']!=sha256(artifact/'protocol.json') or m['source_sha256']!=protocol['evaluator_sha256'] or m.get('interface','react')!=interface:raise ValueError('Invalid native run manifest')
        if m['case_ids']!=protocol['case_ids'] or m['completed_case_ids']!=protocol['case_ids']:raise ValueError('Native coverage differs')
        if sha256(root/'traces.jsonl')!=m['traces_sha256']:raise ValueError('Changed native trace')
        model=protocol['models'][m['model_id']]
        if m['weights_sha256']!=model['weights_sha256'] or m['config_sha256']!=model['config_sha256']:raise ValueError('Native checkpoint differs')
        if any(m[k]!=v for k,v in {'max_calls_per_turn':6,'max_new_tokens':384,'seed':45,'decode':'greedy'}.items()):raise ValueError('Native budget differs')
        records=[json.loads(x) for x in (root/'traces.jsonl').read_text().splitlines()]
        if [r['case_id'] for r in records]!=protocol['case_ids']:raise ValueError('Native trace IDs differ')
        for r in records:replay(r,byid[r['case_id']],answers[r['case_id']],schemas,backend,checkers,interface)
        turns=[t for r in records for t in r['turns']];steps=[s for t in turns for s in t['steps']]
        case_successes=sum(r['success'] for r in records);turn_successes=sum(t['success'] for t in turns)
        if case_successes!=m['case_successes'] or turn_successes!=m['turn_successes'] or len(turns)!=m['n_turns'] or len(records)!=m['n_cases']:raise ValueError('Native aggregate differs')
        if interface=='qwen_native':
            calls=sum(len(s.get('actions',[])) for s in steps);tool_errors=sum(bool(o.get('tool_error')) for s in steps for o in s.get('observations',[]))
        else:
            calls=sum(isinstance(s.get('action'),dict) for s in steps);tool_errors=sum(bool(s.get('tool_error')) for s in steps)
        out[name]={'model_id':m['model_id'],'scenario_successes':case_successes,'n_scenarios':len(records),'turn_successes':turn_successes,'n_turns':len(turns),'native_check_turns':sum(t['native_checks_valid'] for t in turns),'native_check_scenarios':sum(all(t['native_checks_valid'] for t in r['turns']) for r in records),'model_steps':len(steps),'parse_error_steps':sum(bool(s.get('parse_error')) for s in steps),'function_attempts':calls,'tool_error_attempts':tool_errors,'forced_turn_stops':sum(not t['finished'] for t in turns),'tokens':sum(s.get('usage',{}).get('total_tokens',0) for s in steps),'all_observations_and_native_checks_replayed':True,'run_manifest_sha256':sha256(root/'run_manifest.json')}
    return {'kind':'native_backend_reexecution_audit','interface':interface,'arms':out,'official_leaderboard_score':False,'limitations':['Four previously used filesystem calibration scenarios; no broad benchmark claim','Native and ReAct interfaces have different formatting and parallel call affordances','Checks/observations independently replayed; no additional model inference used']}

if __name__=='__main__':
    source=Path(sys.argv[1]);artifact=Path(sys.argv[2]);roster=sys.argv[3:] or ['raw','hidden','visible']
    result=audit(source,artifact,roster);(artifact/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:{a:d[a] for a in ['scenario_successes','turn_successes','n_turns','parse_error_steps','model_steps','tool_error_attempts']} for k,d in result['arms'].items()}))
