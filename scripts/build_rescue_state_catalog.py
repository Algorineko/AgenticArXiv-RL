"""Derive a paired development-only catalog from the frozen v1 metadata."""
import json
from pathlib import Path
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'AgenticArxiv'))
from benchmark.rescue_catalog import sha256, verify_catalog
from benchmark.task_spec import Step, TaskSpec

def build(source, output):
    manifest, tasks, splits = verify_catalog(source)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source / 'feeds', output / 'feeds')
    shutil.copy2(source / 'snapshot.json', output / 'snapshot.json')
    tasks = [t for t in tasks if t['rescue_split'] != 'search_dev']
    paired = []
    for family in ('translate', 'force_download'):
        for ref in (1, 2, 3):
            search = Step('get_recently_submitted_cs_papers', {'aspect': 'LG', 'days': 7, 'max_results': 6})
            prior = Step('download_arxiv_pdf', {'ref': ref})
            operation = Step('translate_arxiv_pdf', {'ref': ref}) if family == 'translate' else Step('download_arxiv_pdf', {'ref': ref, 'force': True})
            text = f'翻译当前列表中第{ref}篇论文。' if family == 'translate' else f'请强制重新下载当前列表中第{ref}篇论文的PDF（force=true）。'
            for downloaded in (False, True):
                state = 'downloaded' if downloaded else 'listed'
                task = TaskSpec(id=f'state_v1_{family}_{ref}_{state}', task=text, steps=(operation,),
                    setup=(search, prior) if downloaded else (search,), category='rescue_'+family,
                    requires_offline=True, max_iterations=6).to_task()
                task.update(rescue_split='search_dev', rescue_family=family, catalog_aspect='LG',
                            pair_id=f'{family}_{ref}', initial_state=state)
                paired.append(task)
    tasks.extend(paired)
    splits['split']['search_dev'] = [t['id'] for t in paired]
    splits['version'] = 'rescue_state_counterfactual_v1'
    for name, value in [('tasks.json', tasks), ('splits.json', splits)]:
        (output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    manifest.update(protocol='state_counterfactual_v1', parent_manifest_sha256=sha256(source/'manifest.json'),
                    n_tasks=len(tasks), counts={k:len(v) for k,v in splits['split'].items()})
    manifest['files'] = {str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}
    manifest['limitations'] += ['Force-download measures requested call arguments only: offline stub does not perform network refresh',
        'Six request pairs are development probes, not independent benchmark samples; label variant is manually fixed, not RSI']
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    verify_catalog(output)
    print(sha256(output/'manifest.json'))
if __name__ == '__main__':
    build(Path(sys.argv[1]), Path(sys.argv[2]))
