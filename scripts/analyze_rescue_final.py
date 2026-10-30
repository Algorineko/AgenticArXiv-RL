"""Fail-closed audit of the separately frozen, one-shot final composition runs."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'AgenticArxiv'),str(ROOT/'scripts')]
from benchmark.rescue_catalog import sha256,verify_catalog
from benchmark.semantic_oracle import attach_expected_paper_ids
from tools.bootstrap import require_all_tools
from rl.self_evolve import strict_success
from rescue_plan_receipts import assess_plan


def analyze(catalog,runs,protocol_path):
 protocol=json.loads(protocol_path.read_text());_,tasks,splits=verify_catalog(catalog)
 if sha256(catalog/'manifest.json')!=protocol['catalog_manifest_sha256']:raise ValueError('Final catalog changed')
 tasks=[t for t in tasks if t['rescue_split']=='final_test']
 if [t['id'] for t in tasks]!=protocol['final_task_ids']:raise ValueError('Final task order changed')
 require_all_tools('frozen final receipt audit')
 byid={t['id']:t for t in attach_expected_paper_ids(tasks,catalog/'snapshot.json')};expected=set(byid)
 roster={'raw_baseline':'Qwen2.5-1.5B-Instruct','hidden_baseline':'new-SFT-hidden-16steps','visible_baseline':'new-SFT-visible-16steps'}
 output={};manifests=[]
 for arm,model_id in roster.items():
  directory=runs/arm;m=json.loads((directory/'run_manifest.json').read_text());s=json.loads((directory/'benchmark/summary.json').read_text());rows=s['details'];traces=[json.loads(x) for x in (directory/'traces.jsonl').read_text().splitlines()]
  if (m['status']!='completed' or m['policy']!='model' or m['split']!='final_test' or not m['final_test_model_evaluated'] or m['frozen_final_protocol_sha256']!=sha256(protocol_path) or m['catalog_manifest_sha256']!=protocol['catalog_manifest_sha256'] or m['model_id']!=model_id or m['variant']!='baseline' or s['error_count'] or s.get('errors')):raise ValueError('Invalid final arm '+arm)
  if m['weights_sha256']!=protocol['models'][model_id]['weights_sha256'] or m['model_config_sha256']!=protocol['models'][model_id]['config_sha256']:raise ValueError('Wrong checkpoint '+arm)
  for source,h in protocol['source_sha256'].items():
   if m['source_sha256'].get(source)!=h:raise ValueError('Changed frozen evaluator '+source)
  for records in (rows,traces):
   if len(records)!=len(expected) or {t['task_id'] for t in records}!=expected or any(t['trial']!=0 or t['agent_type']!='regex' for t in records):raise ValueError('Invalid final coverage')
  if m['task_ids']!=protocol['final_task_ids'] or m['completed_task_ids']!=protocol['final_task_ids'] or any(t.get('error') for t in traces):raise ValueError('Incomplete/error final traces')
  if sha256(directory/'traces.jsonl')!=m['traces_sha256']:raise ValueError('Changed final trace')
  if any(m[k]!=v for k,v in {'seed':45,'repeat':1,'decode':'greedy','max_new_tokens':384,'max_iterations':6,'timeout_seconds':900}.items()):raise ValueError('Changed final budget')
  strict={r['task_id']:strict_success(r) for r in rows}
  if sum(strict.values())!=m['strict_successes']:raise ValueError('Non-regenerating strict score')
  detail={t['task_id']:assess_plan(t['raw_result']['history'],byid[t['task_id']]) for t in traces}
  families=defaultdict(lambda:{'n':0,'operations_and_finish':0,'strict':0})
  for k,t in byid.items():
   family=families[t['rescue_family']];family['n']+=1;family['operations_and_finish']+=int(detail[k]['operations_and_finish']);family['strict']+=int(strict[k])
  output[arm]={'model_id':model_id,'n':len(expected),'operations_and_finish':sum(x['operations_and_finish'] for x in detail.values()),'ordered_operations_seen':sum(x['ordered_operations_seen'] for x in detail.values()),'strict_successes':sum(strict.values()),'false_finishes':sum(bool(r['false_finish']) for r in rows),'first_action_finish':sum(str(t['raw_result']['history'][0]['action']).strip()=='FINISH' for t in traces),'force_stops':sum(r['termination']=='FORCE_STOP' for r in rows),'parse_failures':sum(r['parse_fail'] for r in rows),'tool_failures':sum(r['tool_fail'] for r in rows),'tokens':sum(r['tokens'] for r in rows),'by_family':dict(families),'details':detail,'run_manifest_sha256':sha256(directory/'run_manifest.json'),'trace_sha256':m['traces_sha256']}
  manifests.append(m)
 for m in manifests[1:]:
  for k in ('source_sha256','versions','physical_gpu'):
   if m[k]!=manifests[0][k]:raise ValueError('Uncontrolled final deployment: '+k)
 return {'kind':'one_shot_frozen_final_descriptive_result','protocol_sha256':sha256(protocol_path),'arms':output,'limitations':['One training seed and short 16-step recipe; not a universal SFT/RL claim','Six correlated composition families, four parameter instances each; no significance claim','Ordered offline-operation witnesses are broader than exact chains, not independent full goal-state/PDF-quality verification','Final set now used; no further tuning or seed selection on these final scores'],'final_test_model_evaluated':True}

if __name__=='__main__':
 root=ROOT/'artifacts/rescue_sft_pair_v1';result=analyze(ROOT/'artifacts/rescue_receipt_catalog_v1',root/'final/model_runs',root/'frozen_final_protocol.json')
 (root/'final/analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:{a:v[a] for a in ['operations_and_finish','strict_successes','n','false_finishes','force_stops']} for k,v in result['arms'].items()}))
