"""Describe frozen paired initial-state probes; reuse fail-closed arm audits."""
import argparse
import json
from pathlib import Path
from analyze_rescue_screen import analyze

def state_analysis(catalog, runs):
    result = analyze(catalog, runs, variant='state_boundaries')
    tasks = {t['id']: t for t in json.loads((catalog/'tasks.json').read_text()) if t['rescue_split']=='search_dev'}
    detail = {}
    for arm in result['arms']:
        traces = [json.loads(x) for x in (runs/arm/'traces.jsonl').read_text().splitlines()]
        rows = json.loads((runs/arm/'benchmark/summary.json').read_text())['details']
        from rl.self_evolve import strict_success
        scores = {r['task_id']: strict_success(r) for r in rows}
        first = {t['task_id']: str(t['raw_result']['history'][0]['action']).strip() == 'FINISH' for t in traces}
        detail[arm] = {state: {
            'n':sum(t['initial_state']==state for t in tasks.values()),
            'strict_successes':sum(scores[k] for k,t in tasks.items() if t['initial_state']==state),
            'first_action_finish':sum(first[k] for k,t in tasks.items() if t['initial_state']==state)}
            for state in ('listed','downloaded')}
        pairs = {}
        for k,t in tasks.items():
            pairs.setdefault(t['pair_id'],{})[t['initial_state']] = {'strict':scores[k],'first_finish':first[k]}
        detail[arm]['pairs'] = pairs
    result.update(kind='paired_initial_state_development_probe', initial_state=detail,
                  independent_task_families=2, parameter_instances_per_family=3,
                  request_pairs=6, held_out_models_evaluated=False)
    result['limitations'] += ['Local base and A40 GRPO run on different hardware/torch versions; only within-model paired prompt comparisons',
                              'Force argument is checked but offline download does not implement real network refresh',
                              'No independent requested-goal witness; extra calls may fail strict scoring despite performing the operation']
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    data=state_analysis(a.catalog,a.runs)
    a.output.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({arm:{'strict':v['strict_successes'],'n':v['n'],'first_finish':v['single_action_finishes']} for arm,v in data['arms'].items()}))
