"""Audit actual paired optimizer steps and all four development deployments."""
import json
from pathlib import Path
import sys
from audit_rescue_pair import audit
from analyze_rescue_receipts import analyze as receipts
ROOT=Path(__file__).resolve().parents[1]
def analyze(artifact):
 catalog=ROOT/'artifacts/rescue_receipt_catalog_v1';runs=artifact/'model_runs'
 manifests={m:json.loads((artifact/'training'/m/'rescue_training_manifest.json').read_text()) for m in ('hidden','visible')}
 a,b=manifests.values()
 for mode,m in manifests.items():
  if m['status']!='completed' or m['actual_steps']!=16 or not m['parameters_changed'] or m['probe_before_sha256']==m['probe_after_sha256']:raise ValueError('Invalid optimizer evidence: '+mode)
  steps=[json.loads(x) for x in (artifact/'training'/mode/'steps.jsonl').read_text().splitlines()]
  if [s['step'] for s in steps]!=list(range(1,17)):raise ValueError('Missing optimizer logs')
  for variant in ('baseline','current_actions_only'):
   evaluation=json.loads((runs/(mode+'_'+variant)/'run_manifest.json').read_text())
   if evaluation['weights_sha256']!=m['export_weights_sha256']:raise ValueError('Evaluated wrong training export')
 for k in ('base_weights_sha256','base_config_sha256','row_indices','row_task_ids','supervised_tokens','actual_steps','learning_rate','seed','source_sha256','versions'):
  if a[k]!=b[k]:raise ValueError('Unpaired training: '+k)
 pairs={m:audit(catalog,runs,m,'current_actions_only','search_dev') for m in manifests}
 r=receipts(catalog,runs)['arms'];left=r['hidden_baseline']['details'];right=r['visible_baseline']['details']
 return {'kind':'one_seed_16_step_setup_training_feasibility','training_manifests':manifests,'deployment_audits':pairs,'operation_receipts':r,
 'visible_vs_hidden_under_visible_prompt':{'fixed':[k for k in left if not left[k]['operation_and_finish'] and right[k]['operation_and_finish']],'regressed':[k for k in left if left[k]['operation_and_finish'] and not right[k]['operation_and_finish']]},
 'limitations':['One seed, 16 optimizer steps, narrow shared primitive families; no recipe-general conclusion','Equal decisions/steps/labels, unequal input tokens due to visibility intervention','Real metadata but offline operation receipts, not PDF quality','No autonomous improvement or RL training']}
if __name__=='__main__':
 root=ROOT/'artifacts/rescue_sft_pair_v1';result=analyze(root);(root/'development_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print({k:v['operation_and_finish'] for k,v in result['operation_receipts'].items()})
