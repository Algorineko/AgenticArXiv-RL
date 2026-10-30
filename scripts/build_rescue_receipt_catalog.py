"""Freeze unchanged dev plus a new CL promotion probe; original final untouched."""
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'AgenticArxiv'),str(ROOT/'scripts')]
from benchmark.rescue_catalog import sha256,verify_catalog
from benchmark.task_spec import Step,TaskSpec
from rescue_operation_receipts import compile_request
source=ROOT/'artifacts/rescue_state_catalog_v1'
output=ROOT/'artifacts/rescue_receipt_catalog_v1'
manifest,tasks,splits=verify_catalog(source)
output.mkdir(exist_ok=False)
shutil.copytree(source/'feeds',output/'feeds');shutil.copy2(source/'snapshot.json',output/'snapshot.json')
tasks=[t for t in tasks if t['rescue_split']!='promotion_val']
promotion=[]
for family in ('translate','force_download'):
 for ref in (4,5,6):
  texts = [f'请把当前论文列表的第{ref}篇翻译成中文。',f'当前列表里第{ref}篇论文，请翻译。',f'需要翻译当前列表中第{ref}篇论文。'] if family=='translate' else [f'请将当前论文列表第{ref}篇的PDF强制重新下载，force=true。',f'当前列表第{ref}篇论文的PDF需要重新下载（force=true）。',f'对当前列表第{ref}篇论文执行强制下载（force=true）。']
  text=texts[ref-4];assert compile_request(text)
  search=Step('get_recently_submitted_cs_papers',{'aspect':'CL','days':7,'max_results':8})
  op=Step('translate_arxiv_pdf',{'ref':ref}) if family=='translate' else Step('download_arxiv_pdf',{'ref':ref,'force':True})
  for downloaded in (False,True):
   state='downloaded' if downloaded else 'listed'
   setup=(search,Step('download_arxiv_pdf',{'ref':ref})) if downloaded else (search,)
   t=TaskSpec(id=f'receipt_v1_promotion_{family}_{ref}_{state}',task=text,steps=(op,),setup=setup,category='rescue_'+family,requires_offline=True,max_iterations=6).to_task()
   t.update(rescue_split='promotion_val',rescue_family=family,catalog_aspect='CL',pair_id=f'{family}_{ref}',initial_state=state)
   promotion.append(t)
tasks.extend(promotion);splits['split']['promotion_val']=[t['id'] for t in promotion];splits['version']='rescue_receipts_v1'
for name,value in [('tasks.json',tasks),('splits.json',splits)]:
 (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
manifest.update(protocol='receipt_contract_v1',parent_manifest_sha256=sha256(source/'manifest.json'))
manifest['files']={str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}
(output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
verify_catalog(output);print(sha256(output/'manifest.json'))
