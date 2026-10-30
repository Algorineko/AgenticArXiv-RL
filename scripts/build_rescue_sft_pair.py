"""Generate paired new train-only expert data differing only in setup rendering."""
from pathlib import Path
import dataclasses
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'AgenticArxiv'),str(ROOT/'scripts')]
from rescue_replay_eval import set_runtime
from benchmark.rescue_catalog import sha256
from benchmark.task_spec import TaskSpec,Step

def build(output):
 output.mkdir(parents=True,exist_ok=False);set_runtime(output,1)
 import generate_sft_data as gen
 from agents.prompt_templates import format_tool_description
 from tools.tool_registry import registry
 from rl.env import MockArxivEnv
 source=ROOT/'artifacts/rescue_catalog_v1'
 train=[t for t in json.loads((source/'tasks.json').read_text()) if t['rescue_split']=='train']
 specs=[]
 for t in train:
  specs.append(TaskSpec(id=t['id'],task=t['task'],steps=tuple(Step(n,a) for n,a in zip(t['expected_tools'],t['expected_tool_args'])),setup=tuple(Step(s['name'],s['args']) for s in t.get('setup',[])),category=t['category'],requires_offline=True,max_iterations=6))
 for family in ('translate','force_download'):
  for ref in (1,2,3):
   search=Step('get_recently_submitted_cs_papers',{'aspect':'AI','days':7,'max_results':6})
   op=Step('translate_arxiv_pdf',{'ref':ref}) if family=='translate' else Step('download_arxiv_pdf',{'ref':ref,'force':True})
   text=f'翻译当前列表中第{ref}篇论文。' if family=='translate' else f'请强制重新下载当前列表中第{ref}篇论文的PDF（force=true）。'
   for downloaded in (False,True):
    specs.append(TaskSpec(id=f'new_train_{family}_{ref}_{downloaded}',task=text,steps=(op,),setup=(search,Step('download_arxiv_pdf',{'ref':ref})) if downloaded else (search,),category='rescue_'+family,requires_offline=True,max_iterations=6))
 prefixes=('', '任务：', '请按要求完成：', '现在需要你完成以下请求：')
 specs=[dataclasses.replace(s,id=s.id+f'_view{v}',task=prefix+s.task) for s in specs for v,prefix in enumerate(prefixes)]
 (output/'train_tasks.json').write_text(json.dumps([s.to_task() for s in specs],ensure_ascii=False,indent=2)+'\n')
 tools=format_tool_description(registry.list_tools());original=gen.build_visible_setup_context
 data={}
 for mode in ('hidden','visible'):
  gen.build_visible_setup_context=original if mode=='visible' else lambda task:''
  env=MockArxivEnv(snapshot_path=source/'snapshot.json',mode='replay')
  rows=gen.generate_deterministic_trajectories(specs,env,tools,source_split='new_AI_train_only')
  data[mode]=rows
  (output/f'{mode}.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
 gen.build_visible_setup_context=original
 assert len(data['hidden'])==len(data['visible'])
 for a,b in zip(data['hidden'],data['visible']):
  assert a['source_task_id']==b['source_task_id'] and a['trajectory_step']==b['trajectory_step']
  assert a['messages'][1]==b['messages'][1]
 manifest={'kind':'new_train_only_setup_rendering_pair','n_tasks':len(specs),'n_rows_per_arm':len(data['hidden']),
  'snapshot_sha256':sha256(source/'snapshot.json'),'files':{n:sha256(output/n) for n in ('train_tasks.json','hidden.jsonl','visible.jsonl')},
  'source_sha256':{n:sha256(ROOT/n) for n in ('scripts/build_rescue_sft_pair.py','scripts/generate_sft_data.py','AgenticArxiv/agents/prompt_templates.py')},
  'papers':'AI train pool only','prefix_views':list(prefixes),'limitations':['New synthetic expert demonstrations over frozen real metadata, not historical training reconstruction','Parameter families overlap development; no unseen primitive-family claim','Offline download/translation stubs']}
 (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'n_tasks':len(specs),'n_rows':len(data['hidden']),'manifest_sha256':sha256(output/'manifest.json')}))
if __name__=='__main__':build(Path(sys.argv[1]))
