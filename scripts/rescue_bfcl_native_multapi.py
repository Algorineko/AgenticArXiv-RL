"""Safe Qwen-native tool evaluation on pinned BFCL multi-API scenarios."""
from copy import deepcopy
import importlib
import inspect
import json
from pathlib import Path
import re
import sys
from rescue_bfcl_native_eval import main,serialized_result
from rescue_bfcl_native_fc import canonical_schema,parse_calls,make_client
from rescue_bfcl_native_preflight import parse_literal_call,execute,digest


def _canonical(value):
    if isinstance(value,dict):
        return {f'{type(k).__name__}:{k}':_canonical(v) for k,v in sorted(value.items(),key=lambda item:(type(item[0]).__name__,str(item[0])))}
    if isinstance(value,(list,tuple)):return [_canonical(v) for v in value]
    if isinstance(value,set):return sorted((_canonical(v) for v in value),key=str)
    return value

def snapshot(instances):
    out={}
    for cls,obj in instances.items():
        out[cls]={k:_canonical(v) for k,v in vars(obj).items() if not k.startswith('_')}
    return out


def instantiate(question,backend):
    initial=question['initial_config'];instances={}
    for cls in question['involved_classes']:
        obj=backend[cls]()
        if cls not in backend['stateless']:
            obj._load_scenario(deepcopy(initial.get(cls,{})),long_context=False)
        instances[cls]=obj
    return instances


def safe_call(instances,owners,call_text):
    name,args,kwargs=parse_literal_call(call_text,set(owners))
    obj=instances[owners[name]];fn=getattr(obj,name)
    inspect.signature(fn).bind(*args,**kwargs)
    value=fn(*args,**kwargs)
    return serialized_result(value),bool(isinstance(value,dict) and 'error' in value)


def run_case(question,answer,schemas,backend,checkers,policy,client,model_id):
    actual=instantiate(question,backend);gold=instantiate(question,backend)
    tools=question['_task_schemas'];tools=[s for s in tools if s['name'] not in question.get('excluded_function',[])]
    owners={s['name']:backend['schema_owner'][s['name']] for s in tools}
    if len(owners)!=len(tools):raise ValueError('Duplicate function schema name')
    messages=[{'role':'system','content':'You are a helpful assistant.'}]
    if client:
        client.native_tools=[{'type':'function','function':{'name':s['name'],'description':s['description'],'parameters':canonical_schema(s['parameters'])}} for s in tools]
        client.start_generation_stream(question['id']+':0')
    observations=[];turns=[]
    for ti,user_msgs in enumerate(question['question']):
        if any(m['role']!='user' for m in user_msgs):raise ValueError('Unexpected source dialogue role')
        messages.extend(deepcopy(user_msgs));gold_calls=answer['ground_truth'][ti]
        steps=[];current=[];position=0;finished=False
        for iteration in range(6):
            usage={}
            if policy=='reference':
                if position<len(gold_calls):
                    name,args,kw=parse_literal_call(gold_calls[position],set(owners));owner=owners[name]
                    bound=inspect.signature(getattr(actual[owner],name)).bind(*args,**kw)
                    raw='<tool_call>\n'+json.dumps({'name':name,'arguments':dict(bound.arguments)})+'\n</tool_call>';position+=1
                else:raw='Done.'
            elif policy=='finish_only':raw='Done.'
            else:
                result=client.chat_completions(model=model_id,messages=messages,temperature=0,max_tokens=384)
                raw=result['choices'][0]['message']['content'];usage=result.get('usage',{})
            messages.append({'role':'assistant','content':raw});rec={'iteration':iteration,'raw_response':raw,'usage':usage}
            try:actions=parse_calls(raw)
            except (ValueError,TypeError) as exc:
                rec['parse_error']=str(exc);messages.append({'role':'tool','content':'Tool call parsing failed: '+str(exc)});steps.append(rec);continue
            rec['actions']=actions;rec['observations']=[]
            if not actions:
                rec['action']='FINISH';finished=True;steps.append(rec);break
            for action in actions:
                name=action['name'];kw=action['args'];tool_error=False
                try:
                    if name not in owners or any(not re.fullmatch(r'[A-Za-z_]\w*',k) for k in kw):raise ValueError('Unavailable function/argument')
                    call=name+'('+', '.join(k+'='+repr(v) for k,v in kw.items())+')'
                    observation,tool_error=safe_call(actual,owners,call)
                except Exception as exc:observation='Error during execution: '+str(exc);tool_error=True
                rec['observations'].append({'text':observation,'tool_error':tool_error});current.append(observation)
                messages.append({'role':'tool','name':name,'content':observation})
            steps.append(rec)
        expected=[]
        for call in gold_calls:
            text,_=safe_call(gold,owners,call);expected.append(text)
        observations.extend(current)
        sv=checkers.state_checker(actual,gold)['valid'];rv=checkers.response_checker(observations,expected,ti)['valid']
        valid=bool(current or not gold_calls) and sv and rv
        turns.append({'turn':ti,'finished':finished,'state_valid':sv,'response_valid':rv,'native_checks_valid':valid,'success':finished and valid,'steps':steps,'actual_state_sha256':digest(snapshot(actual)),'gold_state_sha256':digest(snapshot(gold))})
    return {'case_id':question['id'],'success':all(t['success'] for t in turns),'turns':turns}


def load_multi(source,protocol):
    from benchmark.rescue_catalog import sha256
    for filekey in ('source_manifest.json','integration_manifest.json','full_backend_manifest.json'):
        if sha256(source/filekey)!=protocol[filekey+'_sha256']:raise ValueError('Changed pinned manifest '+filekey)
        manifest=json.loads((source/filekey).read_text())
        for path,h in manifest['files'].items():
            if sha256(source/path)!=h:raise ValueError('Changed pinned official source '+path)
    pkg=source/'berkeley-function-call-leaderboard';sys.path.insert(0,str(pkg))
    from bfcl_eval.constants.executable_backend_config import CLASS_FILE_PATH_MAPPING,STATELESS_CLASSES,MULTI_TURN_FUNC_DOC_FILE_MAPPING
    import bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker as checkers
    data=pkg/'bfcl_eval/data';qmap={x['id']:x for x in map(json.loads,(data/'BFCL_v4_multi_turn_base.json').read_text().splitlines())};amap={x['id']:x for x in map(json.loads,(data/'possible_answer/BFCL_v4_multi_turn_base.json').read_text().splitlines())}
    questions=[];all_schemas=[];schema_owner={};owners={}
    supported={'GorillaFileSystem','MathAPI','MessageAPI','TwitterAPI','TicketAPI','TradingBot','TravelAPI','VehicleControlAPI'}
    for cls in CLASS_FILE_PATH_MAPPING:
        if cls not in supported:continue
        module=importlib.import_module(CLASS_FILE_PATH_MAPPING[cls]);owners[cls]=getattr(module,cls)
        fname=MULTI_TURN_FUNC_DOC_FILE_MAPPING[cls];doc=data/'multi_turn_func_doc'/fname
        if not doc.exists():continue
        for line in doc.read_text().splitlines():
            schema=json.loads(line)
            if schema['name'] in schema_owner:raise ValueError('Official tool name collision '+schema['name'])
            schema_owner[schema['name']]=cls;all_schemas.append(schema)
    for key in protocol['case_ids']:
        q=deepcopy(qmap[key]);classes=q['involved_classes']
        if len(classes)!=2 or any(c not in owners for c in classes):raise ValueError('Task outside frozen two-API scope')
        local=[]
        for s in all_schemas:
            if schema_owner[s['name']] in classes:local.append(s)
        q['_task_schemas']=local;questions.append(q)
    # Keep metadata beside constructors for safe dispatch and initialization.
    owners['stateless']=STATELESS_CLASSES;owners['schema_owner']=schema_owner
    return owners,checkers,questions,amap,{'all':all_schemas}


def main_multapi():
    import rescue_bfcl_native_eval as common
    common.load=load_multi
    main(case_runner=run_case,client_factory=make_client,interface='qwen_native')

if __name__=='__main__':main_multapi()
