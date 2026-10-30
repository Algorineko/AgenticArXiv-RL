"""Static request-contract/receipt representation, supported numbered tasks only."""
import json
from rescue_operation_receipts import compile_request, history_steps, assess


def contract_text(request, history, target_paper_id):
    state = assess(history_steps(history), request, target_paper_id)
    contract = {'tool':request['name'], 'ref':request['ref']}
    if request.get('force'):
        contract['force'] = True
    return ('\n\n[从本轮用户请求提取的操作契约]\n' + json.dumps(contract, ensure_ascii=False) +
            '\n[本轮实际工具观察中匹配此操作与目标论文的凭据]\n' +
            ('已收到' if state['operation_seen'] else '尚未收到') +
            '\n上述凭据仅统计本轮调用，不统计请求前的状态。')


def install():
    from agents.agent_engine import ReActAgent
    from models.store import store
    original_run = ReActAgent.run
    original_messages = ReActAgent.build_messages
    def run(self, task, agent_model=None, session_id='default', initial_history=''):
        self.receipt_request = compile_request(task)
        self.receipt_target = None
        if self.receipt_request:
            paper = store.resolve_paper(session_id,self.receipt_request['ref'])
            if paper is not None:
                self.receipt_target = paper.id
        return original_run(self, task, agent_model, session_id, initial_history)
    def messages(self, task, tools_description, history_text):
        messages,extra = original_messages(self,task,tools_description,history_text)
        if self.receipt_request and self.receipt_target:
            messages[0]['content'] += contract_text(self.receipt_request,history_text,self.receipt_target)
        return messages,extra
    ReActAgent.run = run
    ReActAgent.build_messages = messages
