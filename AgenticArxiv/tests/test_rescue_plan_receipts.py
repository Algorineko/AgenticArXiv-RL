from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from rescue_plan_receipts import assess_plan,step_receipt

def download(p):
 return {'action':{'name':'download_arxiv_pdf','args':{'ref':p}},'observation':str({'paper_id':p,'status':'READY','size_bytes':39,'local_path':'/tmp/pdf'})}

def test_extra_calls_allowed_but_operations_cannot_be_reused():
 task={'expected_tools':['download_arxiv_pdf','download_arxiv_pdf'],'expected_tool_args':[{'ref':1},{'ref':2}],'expected_paper_ids':['p1','p2']}
 good=[{'action':'unused'},download('p1'),{'action':'cache'},download('p2'),{'action':'FINISH'}]
 assert assess_plan(good,task)['operations_and_finish']
 assert not assess_plan([download('p1'),{'action':'FINISH'}],task)['operations_and_finish']
 assert not assess_plan([download('p2'),download('p1'),{'action':'FINISH'}],task)['operations_and_finish']

def test_cache_receipt_parses_dates_without_execution():
 step={'action':{'name':'get_paper_cache_status','args':{'ref':1}},'observation':"{'paper_id':'p1','pdf_ready':True,'translated_ready':False,'updated_at':datetime.datetime(2000, 1, 1, 0, 0)}"}
 assert step_receipt(step,'get_paper_cache_status',{'ref':1},'p1')
 assert not step_receipt(step,'get_paper_cache_status',{'ref':1},'p2')
 step['observation']="{'paper_id':'p1','pdf_ready': __import__('os').getcwd(), 'translated_ready':False}"
 assert not step_receipt(step,'get_paper_cache_status',{'ref':1},'p1')

def test_search_receipt_requires_request_arguments_and_count():
 step={'action':{'name':'get_recently_submitted_cs_papers','args':{'aspect':'CV','days':7,'max_results':3}},'observation':'成功获取 3 篇论文:\n - title'}
 assert step_receipt(step,'get_recently_submitted_cs_papers',{'aspect':'CV','days':7,'max_results':3},None)
 assert not step_receipt(step,'get_recently_submitted_cs_papers',{'aspect':'AI','days':7,'max_results':3},None)
