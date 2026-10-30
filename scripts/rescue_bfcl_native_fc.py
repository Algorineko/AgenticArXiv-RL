"""Canonical Qwen chat-template tools interface; no custom ReAct instruction."""
from copy import deepcopy
import inspect
import json
import re
from rescue_bfcl_native_eval import main,serialized_result
from rescue_bfcl_native_preflight import parse_literal_call,execute,serialize_node,digest


def canonical_schema(value):
    if isinstance(value,list):return [canonical_schema(v) for v in value]
    if not isinstance(value,dict):return value
    return {k:('object' if k=='type' and v=='dict' else canonical_schema(v)) for k,v in value.items()}


def parse_calls(text):
    blocks=re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>',text,re.S)
    if text.count('<tool_call>')!=len(blocks) or len(blocks)>8:raise ValueError('Incomplete or excessive native tool calls')
    calls=[]
    for block in blocks:
        action=json.loads(block)
        if not isinstance(action,dict) or not isinstance(action.get('name'),str) or 'arguments' not in action:raise ValueError('Native name/arguments schema required')
        args=action['arguments']
        if isinstance(args,str):args=json.loads(args)
        if not isinstance(args,dict):raise ValueError('Native arguments must be an object')
        calls.append({'name':action['name'],'args':args})
    return calls


def make_client(*args,**kwargs):
    from utils.llm_client import TransformersLLMClient
    class NativeClient(TransformersLLMClient):
        def _prepare_inputs(self,messages):
            prompt=self.tokenizer.apply_chat_template(messages,tools=self.native_tools,tokenize=False,add_generation_prompt=True)
            return self.tokenizer(prompt,return_tensors='pt'),[]
    return NativeClient(*args,**kwargs)


def run_case(question,answer,schemas,backend,checkers,policy,client,model_id):
    actual=backend();gold=backend()
    actual._load_scenario(deepcopy(question['initial_config']['GorillaFileSystem']))
    gold._load_scenario(deepcopy(question['initial_config']['GorillaFileSystem']))
    schemas=[s for s in schemas if s['name'] not in question.get('excluded_function',[])]
    allowed={s['name'] for s in schemas}
    messages=[{'role':'system','content':'You are a helpful assistant.'}]
    if client:
        client.native_tools=[{'type':'function','function':{'name':s['name'],'description':s['description'],'parameters':canonical_schema(s['parameters'])}} for s in schemas]
        client.start_generation_stream(question['id']+':0')
    all_observations=[];turns=[]
    for turn_index,user_messages in enumerate(question['question']):
        messages.extend(deepcopy(user_messages));gold_calls=answer['ground_truth'][turn_index]
        steps=[];observed=[];finished=False;position=0
        for iteration in range(6):
            usage={}
            if policy=='reference':
                if position<len(gold_calls):
                    name,args,kw=parse_literal_call(gold_calls[position],allowed)
                    bound=inspect.signature(getattr(actual,name)).bind(*args,**kw)
                    raw='<tool_call>\n'+json.dumps({'name':name,'arguments':dict(bound.arguments)})+'\n</tool_call>';position+=1
                else:raw='Done.'
            elif policy=='finish_only':raw='Done.'
            else:
                result=client.chat_completions(model=model_id,messages=messages,temperature=0,max_tokens=384)
                raw=result['choices'][0]['message']['content'];usage=result.get('usage',{})
            messages.append({'role':'assistant','content':raw});record={'iteration':iteration,'raw_response':raw,'usage':usage}
            try:actions=parse_calls(raw)
            except (ValueError,TypeError) as e:
                record['parse_error']=str(e);messages.append({'role':'tool','content':'Tool call parsing failed: '+str(e)});steps.append(record);continue
            record['actions']=actions;record['observations']=[]
            if not actions:
                finished=True;record['action']='FINISH';steps.append(record);break
            for action in actions:
                name=action['name'];kw=action['args']
                try:
                    if name not in allowed or any(not re.fullmatch(r'[A-Za-z_]\w*',k) for k in kw):raise ValueError('Unavailable function/argument')
                    call=name+'('+', '.join(k+'='+repr(v) for k,v in kw.items())+')'
                    result=execute(actual,call,allowed);observation=serialized_result(result)
                    tool_error=isinstance(result,dict) and 'error' in result
                except Exception as e:observation='Error during execution: '+str(e);tool_error=True
                record['observations'].append({'text':observation,'tool_error':tool_error});observed.append(observation)
                messages.append({'role':'tool','name':name,'content':observation})
            steps.append(record)
        expected=[serialized_result(execute(gold,call,allowed)) for call in gold_calls]
        all_observations.extend(observed)
        state=checkers.state_checker({'GorillaFileSystem':actual},{'GorillaFileSystem':gold})
        response=checkers.response_checker(all_observations,expected,turn_index)
        valid=bool(observed or not gold_calls) and state['valid'] and response['valid']
        turns.append({'turn':turn_index,'finished':finished,'native_checks_valid':valid,'success':finished and valid,'state_valid':state['valid'],'response_valid':response['valid'],'steps':steps,'actual_root_sha256':digest(serialize_node(actual.root)),'gold_root_sha256':digest(serialize_node(gold.root))})
    return {'case_id':question['id'],'success':all(t['success'] for t in turns),'turns':turns}

if __name__=='__main__':main(case_runner=run_case,client_factory=make_client,interface='qwen_native')
