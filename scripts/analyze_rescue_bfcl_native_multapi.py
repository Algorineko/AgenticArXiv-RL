"""Independently re-execute recorded multi-API native tool traces."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'AgenticArxiv'),str(ROOT/'scripts')]
from benchmark.rescue_catalog import sha256
from rescue_bfcl_native_multapi import load_multi,instantiate,safe_call,digest,snapshot


def audit(source,artifact,roster):
 protocol=json.loads((artifact/'protocol.json').read_text())
 if sha256(ROOT/'scripts/rescue_bfcl_native_eval.py')!=protocol['evaluator_sha256']:raise ValueError('Changed shared evaluator')
 for name,h in protocol['support_source_sha256'].items():
  if sha256(ROOT/name)!=h:raise ValueError('Changed pinned support source '+name)
 backend,checkers,questions,answers,_=load_multi(source,protocol);byid={q['id']:q for q in questions};out={}
 for name in roster:
  root=artifact/'model_runs'/name;m=json.loads((root/'run_manifest.json').read_text())
  if m['status']!='completed' or m['policy']!='model' or m['protocol_sha256']!=sha256(artifact/'protocol.json') or m['source_sha256']!=protocol['evaluator_sha256'] or m.get('interface')!='qwen_native':raise ValueError('Invalid native run manifest')
  if m['case_ids']!=protocol['case_ids'] or m['completed_case_ids']!=protocol['case_ids']:raise ValueError('Native coverage differs')
  if sha256(root/'traces.jsonl')!=m['traces_sha256']:raise ValueError('Changed native trace')
  model=protocol['models'][m['model_id']]
  if m['weights_sha256']!=model['weights_sha256'] or m['config_sha256']!=model['config_sha256']:raise ValueError('Native checkpoint differs')
  if any(m[k]!=v for k,v in {'max_calls_per_turn':6,'max_new_tokens':384,'seed':45,'decode':'greedy'}.items()):raise ValueError('Native budget differs')
  records=[json.loads(x) for x in (root/'traces.jsonl').read_text().splitlines()]
  if [r['case_id'] for r in records]!=protocol['case_ids']:raise ValueError('Native trace IDs differ')
  for r in records:
   q=byid[r['case_id']];ans=answers[q['id']];actual=instantiate(q,backend);gold=instantiate(q,backend);schemas=[s for s in q['_task_schemas'] if s['name'] not in q.get('excluded_function',[])];owners={s['name']:backend['schema_owner'][s['name']] for s in schemas};obs=[]
   if [t['turn'] for t in r['turns']]!=list(range(len(q['question']))):raise ValueError('Missing user turns')
   for ti,t in enumerate(r['turns']):
    if len(t['steps'])>6 or [s['iteration'] for s in t['steps']]!=list(range(len(t['steps']))):raise ValueError('Native step budget/order changed')
    current=[]
    for step in t['steps']:
     if step.get('parse_error'):continue
     actions=step.get('actions',[]);saved=[o['text'] for o in step.get('observations',[])]
     if len(actions)!=len(saved):raise ValueError('Action/observation count mismatch')
     for action,stored in zip(actions,saved):
      name=action['name'];args=action.get('args',{})
      if name not in owners or any(not isinstance(k,str) for k in args):raise ValueError('Unavailable tool/arguments')
      call=name+'('+', '.join(k+'='+repr(v) for k,v in args.items())+')';value,_=safe_call(actual,owners,call)
      if value!=stored:raise ValueError('Recorded observation differs on native replay')
      current.append(value)
    expected=[]
    for call in ans['ground_truth'][ti]:value,_=safe_call(gold,owners,call);expected.append(value)
    obs.extend(current);sv=checkers.state_checker(actual,gold)['valid'];rv=checkers.response_checker(obs,expected,ti)['valid'];valid=bool(current or not expected) and sv and rv
    if (sv!=t['state_valid'] or rv!=t['response_valid'] or valid!=t['native_checks_valid'] or bool(valid and t['finished'])!=t['success'] or digest(snapshot(actual))!=t['actual_state_sha256'] or digest(snapshot(gold))!=t['gold_state_sha256']):raise ValueError('Native checks/state differ on independent replay')
   if all(t['success'] for t in r['turns'])!=r['success']:raise ValueError('Scenario aggregate mismatch')
  turns=[t for r in records for t in r['turns']];steps=[s for t in turns for s in t['steps']];calls=sum(len(s.get('actions',[])) for s in steps);errs=sum(bool(o.get('tool_error')) for s in steps for o in s.get('observations',[]));sc=sum(r['success'] for r in records);ts=sum(t['success'] for t in turns)
  if (sc,ts,len(records),len(turns))!=(m['case_successes'],m['turn_successes'],m['n_cases'],m['n_turns']):raise ValueError('Aggregate differs from run manifest')
  out[name]={'model_id':m['model_id'],'scenario_successes':sc,'n_scenarios':len(records),'turn_successes':ts,'n_turns':len(turns),'native_check_turns':sum(t['native_checks_valid'] for t in turns),'model_steps':len(steps),'parse_error_steps':sum(bool(s.get('parse_error')) for s in steps),'function_attempts':calls,'tool_error_attempts':errs,'forced_turn_stops':sum(not t['finished'] for t in turns),'tokens':sum(s.get('usage',{}).get('total_tokens',0) for s in steps),'all_observations_and_native_checks_replayed':True,'run_manifest_sha256':sha256(root/'run_manifest.json')}
 return {'kind':'native_multi_api_backend_reexecution_audit','interface':'qwen_native','arms':out,'official_leaderboard_score':False,'limitations':['32 two-API scenarios selected from the public BFCL source; pretrained contamination cannot be ruled out','Two checkpoint sizes are not a controlled causal scale intervention','Only the eight pinned simulated API classes; no live service effects','Every recorded action observation and native state/response check independently replayed']}

if __name__=='__main__':
 source=Path(sys.argv[1]);artifact=Path(sys.argv[2]);roster=sys.argv[3:]
 result=audit(source,artifact,roster);(artifact/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:{f:d[f] for f in ('scenario_successes','turn_successes','n_turns','parse_error_steps','function_attempts','tool_error_attempts','forced_turn_stops')} for k,d in result['arms'].items()}))
