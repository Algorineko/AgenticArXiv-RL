"""Static prompt/task instruction conflicts; frozen scores are not overwritten."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def audit(tasks):
 return [t['id'] for t in tasks if 'translate_arxiv_pdf' in t.get('expected_tools',[]) and any(n!='translate_arxiv_pdf' for n in t['expected_tools'][t['expected_tools'].index('translate_arxiv_pdf')+1:])]
if __name__=='__main__':
 tasks=json.loads((ROOT/'artifacts/rescue_catalog_v1/tasks.json').read_text());flagged=audit(tasks)
 source=ROOT/'AgenticArxiv/agents/prompt_templates.py'
 assert '调用 translate_arxiv_pdf 之后，直接 FINISH 即可' in source.read_text()
 result={'kind':'posthoc_prompt_contract_audit','conflict':'Runtime prompt requires FINISH immediately after translate, some task plans request later operations','flagged_catalog_task_ids':flagged,'frozen_primary_scores_unchanged':True,'clean_final_sensitivity':{}}
 final=json.loads((ROOT/'artifacts/rescue_sft_pair_v1/final/analysis.json').read_text())
 for arm,d in final['arms'].items():
  eligible={k:v for k,v in d['details'].items() if k not in flagged}
  result['clean_final_sensitivity'][arm]={'n':len(eligible),'operations_and_finish':sum(v['operations_and_finish'] for v in eligible.values())}
 import sys
 sys.path.insert(0,str(ROOT/'AgenticArxiv'))
 from benchmark.tasks_expanded import EXPANDED_TASKS
 result['flagged_current_expanded_ids']=audit(EXPANDED_TASKS)
 result['limitations']=['Flags only a directly explicit translate-stop conflict, not a complete semantic task audit','Historical paper prompt versions are not reconstructed by this current-source check','Clean subset is a post-hoc sensitivity analysis, not a replacement preregistered score']
 (ROOT/'artifacts/rescue_sft_pair_v1/final/prompt_contract_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(result))
