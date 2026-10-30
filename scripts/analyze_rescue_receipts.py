"""Post-hoc operation receipts for the already archived state probe."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'AgenticArxiv'))
from benchmark.semantic_oracle import attach_expected_paper_ids
from rescue_operation_receipts import compile_request,assess

def analyze(catalog,runs,split="search_dev"):
    from tools.bootstrap import require_all_tools
    require_all_tools("operation receipt audit")
    tasks=json.loads((catalog/'tasks.json').read_text())
    tasks={t['id']:t for t in attach_expected_paper_ids(tasks,catalog/'snapshot.json') if t['rescue_split']==split}
    result={'kind':'posthoc_offline_operation_receipts','limitations':['Post-hoc diagnostic, not predeclared primary outcome','Target identity uses frozen semantic oracle; receipt parsing is separate from strict chain scoring','Offline receipts do not prove actual PDF download/translation quality'],'arms':{}}
    for directory in sorted(runs.iterdir()):
        if not directory.is_dir():continue
        records=[json.loads(s) for s in (directory/'traces.jsonl').read_text().splitlines()]
        assert len(records)==len(tasks) and {t['task_id'] for t in records}==set(tasks)
        details={}
        for record in records:
            task=tasks[record['task_id']]; request=compile_request(task['task'])
            assert request is not None
            details[task['id']]=assess(record['raw_result']['history'],request,task['expected_paper_ids'][0])
        result['arms'][directory.name]={'n':len(details),'operation_seen':sum(v['operation_seen'] for v in details.values()),'operation_and_finish':sum(v['operation_and_finish'] for v in details.values()),'details':details}
    return result
if __name__=='__main__':
    result=analyze(ROOT/'artifacts/rescue_state_catalog_v1',ROOT/'artifacts/rescue_state_v1/model_runs')
    (ROOT/'artifacts/rescue_state_v1/operation_receipts.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='details'} for k,v in result['arms'].items()}))
