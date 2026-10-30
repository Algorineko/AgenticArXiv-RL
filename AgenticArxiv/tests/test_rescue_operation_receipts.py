import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from rescue_operation_receipts import compile_request, receipt, assess, history_steps

@pytest.mark.parametrize('text', ['不要翻译当前列表中第1篇论文。','先下载第1篇再翻译它。','翻译这篇论文。','翻译第0篇。','下载第1篇。','翻译第1篇和第2篇。'])
def test_unsupported_request_fails_closed(text):
    assert compile_request(text) is None

def test_extra_actions_do_not_destroy_receipt():
    req=compile_request('翻译当前列表中第2篇论文。')
    good={'action':{'name':'translate_arxiv_pdf','args':{'ref':2}},'observation':'已创建翻译任务 task_id=mocktask0001, paper_id=2610.1v1，状态=PENDING。前端可订阅 SSE: /events'}
    assert assess([{'action':'cache'},good,{'action':'FINISH'}],req,'2610.1v1')=={'operation_seen':True,'operation_and_finish':True}
    assert not receipt(good,req,'2610.2v1')
    assert not receipt(dict(good,observation='ERROR paper_id=2610.1v1'),req,'2610.1v1')
    assert assess([good,{'action':'FORCE_STOP'}],req,'2610.1v1')['operation_and_finish'] is False

def test_force_must_be_true_and_observation_successful():
    req=compile_request('强制重新下载第1篇（force=true）。')
    action={'name':'download_arxiv_pdf','args':{'ref':1,'force':True}}
    step={'action':action,'observation':str({'paper_id':'p1','status':'READY','size_bytes':39,'local_path':'/tmp/p1.pdf'})}
    assert receipt(step,req,'p1')
    action['args']['force']=False
    assert not receipt(step,req,'p1')

def test_history_parser_ignores_setup():
    text='[先前状态] 已下载\n\nThought: x\nAction: FINISH\nObservation: 任务完成'
    assert history_steps(text)==[{'action':'FINISH','observation':'任务完成'}]

def test_contract_reflects_actual_receipt_not_prior_state():
    from rescue_receipt_prompt import contract_text
    req=compile_request('翻译当前列表中第2篇论文。')
    assert '尚未收到' in contract_text(req,'此前已下载第2篇','p2')
    history='Thought: x\nAction: {"name":"translate_arxiv_pdf","args":{"ref":2}}\nObservation: 已创建翻译任务 task_id=t1, paper_id=p2，状态=PENDING。已排队'
    assert '已收到' in contract_text(req,history,'p2')
    assert '尚未收到' in contract_text(req,history,'p3')
