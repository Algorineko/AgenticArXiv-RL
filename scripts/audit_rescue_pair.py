"""Fail-closed same-checkpoint paired audit, including promotion split receipts."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'AgenticArxiv'))
from benchmark.rescue_catalog import sha256,verify_catalog
from rl.self_evolve import strict_success
from analyze_rescue_receipts import analyze as receipts

def audit(catalog,runs,model,variant,split):
 _,tasks,splits=verify_catalog(catalog)
 expected=set(splits['split'][split]);manifests={};metrics={}
 for arm in (model+'_baseline',model+'_'+variant):
  root=runs/arm;m=json.loads((root/'run_manifest.json').read_text());summary=json.loads((root/'benchmark/summary.json').read_text())
  rows=summary['details'];trace=[json.loads(x) for x in (root/'traces.jsonl').read_text().splitlines()]
  if (m['status']!='completed' or m['policy']!='model' or m['split']!=split or m['final_test_model_evaluated'] or m['catalog_manifest_sha256']!=sha256(catalog/'manifest.json') or m['split_sha256']!=sha256(catalog/'splits.json') or summary['error_count'] or summary.get('errors')):raise ValueError('Invalid manifest or execution errors: '+arm)
  for items in (rows,trace):
   if len(items)!=len(expected) or {t['task_id'] for t in items}!=expected or any(t['trial']!=0 or t['agent_type']!='regex' for t in items):raise ValueError('Invalid coverage: '+arm)
  if len(m['completed_task_ids'])!=len(expected) or set(m['completed_task_ids'])!=expected or any(t.get('error') for t in trace):raise ValueError('Incomplete/error traces: '+arm)
  if m['variant']!=arm.split('_',1)[1] or sha256(root/'traces.jsonl')!=m['traces_sha256']:raise ValueError('Variant/trace hash changed: '+arm)
  if any(m[k]!=v for k,v in {'seed':45,'repeat':1,'decode':'greedy','max_new_tokens':384,'max_iterations':6}.items()):raise ValueError('Budget changed: '+arm)
  score=sum(strict_success(r) for r in rows)
  if score!=m['strict_successes']:raise ValueError('Non-regenerating score')
  manifests[arm]=m;metrics[arm]={'strict':score,'n':len(rows),'first_finish':sum(str(t['raw_result']['history'][0]['action']).strip()=='FINISH' for t in trace),'tool_failures':sum(t['tool_fail'] for t in rows),'force_stop':sum(t['termination']=='FORCE_STOP' for t in rows),'tokens':sum(t['tokens'] for t in rows)}
 left,right=manifests.values()
 for key in ('weights_sha256','model_config_sha256','model_id','source_sha256','versions','physical_gpu','timeout_seconds'):
  if left[key]!=right[key]:raise ValueError('Paired arm mismatch: '+key)
 r=receipts(catalog,runs,split=split)['arms'];a=r[model+'_baseline']['details'];b=r[model+'_'+variant]['details']
 fixed=[k for k in a if not a[k]['operation_and_finish'] and b[k]['operation_and_finish']]
 regressed=[k for k in a if a[k]['operation_and_finish'] and not b[k]['operation_and_finish']]
 return {'model':model,'variant':variant,'split':split,'strict_metrics':metrics,'receipts':r,'fixed':fixed,'regressed':regressed,'passes_positive_gain_no_regression':bool(fixed) and not regressed,'final_test_model_evaluated':False}
if __name__=='__main__':
 catalog=ROOT/'artifacts/rescue_receipt_catalog_v1'
 result=audit(catalog,Path(sys.argv[1]),sys.argv[2],sys.argv[3],sys.argv[4])
 Path(sys.argv[5]).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v['operation_and_finish'] for k,v in result['receipts'].items()}));print('passes',result['passes_positive_gain_no_regression'])
