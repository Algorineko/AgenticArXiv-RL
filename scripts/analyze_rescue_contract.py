"""Validate development arms and apply the predeclared receipt promotion rule."""
import json
from pathlib import Path
import sys
from analyze_rescue_screen import analyze as strict_analysis
from analyze_rescue_receipts import analyze as receipt_analysis
ROOT=Path(__file__).resolve().parents[1]

def analyze(catalog,runs):
 result=strict_analysis(catalog,runs,variant='receipt_contract')
 receipts=receipt_analysis(catalog,runs)
 result['kind']='static_request_contract_development_screen'
 result.update(independent_task_families=2,parameter_instances_per_family=3,request_pairs=6)
 result['operation_receipts']=receipts['arms']
 decision={}
 for model in ('base','grpo'):
  a=receipts['arms'][model+'_baseline']['details'];b=receipts['arms'][model+'_receipt_contract']['details']
  fixed=[k for k in a if not a[k]['operation_and_finish'] and b[k]['operation_and_finish']]
  regressed=[k for k in a if a[k]['operation_and_finish'] and not b[k]['operation_and_finish']]
  decision[model]={'fixed':fixed,'regressed':regressed,'evaluate_promotion_once':bool(fixed) and not regressed}
 result['promotion_decision']=decision
 result['limitations'] += ['Operation receipts were declared before this screen; earlier state-screen receipt rescoring was post-hoc', 'Static handwritten compiler and receipt prompt, not autonomous improvement', 'Rule-only executor baseline must be reported; narrow grammar does not establish broad language understanding']
 return result
if __name__=='__main__':
 catalog=ROOT/'artifacts/rescue_receipt_catalog_v1';runs=Path(sys.argv[1]);output=Path(sys.argv[2])
 result=analyze(catalog,runs)
 output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 for arm,d in result['operation_receipts'].items():print(arm,d['operation_and_finish'],'/',d['n'])
 print(json.dumps(result['promotion_decision']))
