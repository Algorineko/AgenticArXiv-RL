import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from rescue_state_boundary import boundary_history

def test_prior_and_current_are_separate():
    rendered = boundary_history('先前已下载\n\nThought: 翻译\nAction: translate')
    assert rendered.index('先前已下载') < rendered.index('[/请求开始前的状态记录]')
    assert rendered.index('[本轮请求的动作记录]') < rendered.index('Thought: 翻译')

def test_no_invented_actions():
    rendered = boundary_history('先前已下载')
    assert '[本轮请求的动作记录]\n无\n' in rendered
    assert 'translate_arxiv_pdf' not in rendered

def test_pair_invariance():
    tasks = json.loads((ROOT/'artifacts/rescue_state_catalog_v1/tasks.json').read_text())
    dev = [t for t in tasks if t['rescue_split']=='search_dev']
    assert len(dev)==12
    for left,right in zip(dev[::2],dev[1::2]):
        assert left['pair_id']==right['pair_id']
        for key in ('task','expected_tools','expected_tool_args','max_iterations'):
            assert left[key]==right[key]
        assert right['setup'][:-1]==left['setup']
        assert right['setup'][-1]['name']=='download_arxiv_pdf'

def test_current_actions_only_preserves_actions_not_setup():
    from rescue_state_boundary import current_actions_only
    actual='Thought: x\nAction: FINISH\nObservation: 任务完成'
    assert current_actions_only('先前已下载')==''
    assert current_actions_only('先前已下载\n\n'+actual)==actual
    assert current_actions_only(actual)==actual

def test_facts_only_removes_just_legacy_imperative():
    from rescue_state_boundary import state_facts_only
    facts='先前已下载ref=1。\n'
    directive='以上是已存在的会话状态，不属于当前任务动作，请勿重复执行。'
    actions='\n\nThought: x\nAction: FINISH\nObservation: 任务完成'
    assert state_facts_only(facts+directive+actions)==facts+actions
