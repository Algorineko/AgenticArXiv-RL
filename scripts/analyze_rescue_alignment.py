"""Audit two fixed setup-presentation ablations and select by frozen rule."""
import json
from pathlib import Path
import sys
from analyze_rescue_screen import analyze as strict_analysis
from analyze_rescue_receipts import analyze as receipt_analysis
ROOT=Path(__file__).resolve().parents[1]
def analyze(catalog,runs):
 variants=('current_actions_only','state_facts_only')
 strict={v:strict_analysis(catalog,runs,variant=v) for v in variants}
 receipts=receipt_analysis(catalog,runs)
 decision={}
 for model in ('base','grpo'):
  base=receipts['arms'][model+'_baseline']['details']
  candidates={}
  for variant in variants:
   changed=receipts['arms'][model+'_'+variant]['details']
   fixed=[k for k in base if not base[k]['operation_and_finish'] and changed[k]['operation_and_finish']]
   regressed=[k for k in base if base[k]['operation_and_finish'] and not changed[k]['operation_and_finish']]
   candidates[variant]={'fixed':fixed,'regressed':regressed,'eligible':bool(fixed) and not regressed,'net_gain':len(fixed)-len(regressed)}
  eligible=[v for v in variants if candidates[v]['eligible']]
  selected=max(eligible,key=lambda v:(candidates[v]['net_gain'],v=='state_facts_only')) if eligible else None
  decision[model]={'candidates':candidates,'promotion_candidate':selected}
 return {'kind':'setup_presentation_ablation_development_screen','strict_audits':strict,
         'receipts':receipts['arms'],'promotion_decision':decision,
         'final_test_model_evaluated':False,'promotion_val_model_evaluated':False,
         'limitations':['Two request families/six pairs, not independent replications','Both interventions are static interface diagnostics, not RSI','current_actions_only hides state needed for more general requests; not a production default','Rule-only executor already covers these narrow tasks perfectly']}
if __name__=='__main__':
 result=analyze(ROOT/'artifacts/rescue_receipt_catalog_v1',Path(sys.argv[1]))
 Path(sys.argv[2]).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v['operation_and_finish'] for k,v in result['receipts'].items()}))
 print(json.dumps(result['promotion_decision']))
