"""Four fixed original multi-turn filesystem scenarios, native BFCL checkers.

Custom ReAct calibration subset, not a full/official BFCL leaderboard score.
"""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'AgenticArxiv'),str(ROOT/'scripts')]
from benchmark.rescue_catalog import sha256
from rescue_bfcl_native_preflight import parse_literal_call,execute,serialize_node,digest

PROMPT='''You are an assistant operating a virtual filesystem through the tools below.
Use only the listed tools and their documented arguments. Discover hidden backend state by calling tools when needed. Follow the current user request using the previous dialogue and actual observations. A request may require multiple tool calls.
Return exactly:
Thought: brief reasoning
Action: {"name":"tool_name","args":{...}}
Use Action: FINISH only when this user turn is complete, or explain why it cannot be completed. Do not fabricate observations. Execute one action at a time.
Available tools (complete schemas):
{schemas}
Dialogue so far:
{dialogue}
Next assistant action:
'''

def parse_response(text):
    match=re.search(r'(?:^|\n)Action:\s*(.*?)(?=\nObservation:|$)',text,re.S)
    if not match:raise ValueError('Missing Action')
    value=match[1].strip()
    if value=='FINISH':return 'FINISH'
    if value.startswith('```'):
        fenced=re.fullmatch(r'```(?:json)?\s*(.*?)\s*```',value,re.S)
        if fenced:value=fenced[1]
    action=json.loads(value)
    if not isinstance(action,dict) or not isinstance(action.get('name'),str) or not isinstance(action.get('args',{}),dict):raise ValueError('Invalid action schema')
    return action

def serialized_result(result):
    if isinstance(result,str):return result
    if isinstance(result,dict):
        try:return json.dumps(result)
        except (TypeError,ValueError):return str(result)
    return str(result)

def load(source,protocol):
    for manifest_file in ('source_manifest.json','integration_manifest.json'):
        m=json.loads((source/manifest_file).read_text())
        if sha256(source/manifest_file)!=protocol[manifest_file+'_sha256']:raise ValueError('Changed native manifest')
        for name,h in m['files'].items():
            if sha256(source/name)!=h:raise ValueError('Changed native source '+name)
    package=source/'berkeley-function-call-leaderboard';sys.path.insert(0,str(package))
    backend=importlib.import_module('bfcl_eval.eval_checker.multi_turn_eval.func_source_code.gorilla_file_system').GorillaFileSystem
    checkers=importlib.import_module('bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker')
    data=package/'bfcl_eval/data'
    qs={x['id']:x for x in map(json.loads,(data/'BFCL_v4_multi_turn_base.json').read_text().splitlines())}
    ans={x['id']:x for x in map(json.loads,(data/'possible_answer/BFCL_v4_multi_turn_base.json').read_text().splitlines())}
    schemas=[json.loads(x) for x in (data/'multi_turn_func_doc/gorilla_file_system.json').read_text().splitlines()]
    cases=[qs[k] for k in protocol['case_ids']]
    if any(q['involved_classes']!=['GorillaFileSystem'] for q in cases):raise ValueError('Mixed-class scenario outside calibration scope')
    return backend,checkers,cases,ans,schemas


def run_case(question,answer,schemas,backend,checkers,policy,client,model_id):
    actual=backend();gold=backend()
    actual._load_scenario(deepcopy(question['initial_config']['GorillaFileSystem']))
    gold._load_scenario(deepcopy(question['initial_config']['GorillaFileSystem']))
    schemas=[s for s in schemas if s['name'] not in question.get('excluded_function',[])]
    allowed={s['name'] for s in schemas};dialogue=[];all_observations=[];turns=[]
    if client:client.start_generation_stream(question['id']+':0')
    for turn_index,messages in enumerate(question['question']):
        for m in messages:
            if m['role']!='user':raise ValueError('Unexpected source dialogue role')
            dialogue.append('User: '+m['content'])
        gold_calls=answer['ground_truth'][turn_index];steps=[];observed=[];finished=False;reference_position=0
        for iteration in range(6):
            usage={}
            if policy=='reference':
                if reference_position<len(gold_calls):
                    name,args,kw=parse_literal_call(gold_calls[reference_position],allowed)
                    import inspect
                    bound=inspect.signature(getattr(actual,name)).bind(*args,**kw)
                    raw='Thought: Execute the requested operation.\nAction: '+json.dumps({'name':name,'args':dict(bound.arguments)})
                    reference_position+=1
                else:raw='Thought: This turn is complete.\nAction: FINISH'
            elif policy=='finish_only':raw='Thought: This turn is complete.\nAction: FINISH'
            else:
                prompt=PROMPT.replace('{schemas}',json.dumps(schemas,ensure_ascii=False)).replace('{dialogue}','\n'.join(dialogue))
                raw_result=client.chat_completions(model=model_id,messages=[{'role':'user','content':prompt}],temperature=0,max_tokens=384)
                raw=raw_result['choices'][0]['message']['content'];usage=raw_result.get('usage',{})
            dialogue.append('Assistant: '+raw)
            record={'iteration':iteration,'raw_response':raw,'usage':usage}
            try:
                action=parse_response(raw)
                if action=='FINISH':
                    finished=True;record['action']='FINISH';steps.append(record);break
                name=action['name'];kw=action.get('args',{})
                if name not in allowed or any(not re.fullmatch(r'[A-Za-z_]\w*',k) for k in kw):raise ValueError('Unavailable function/argument name')
                call=name+'('+', '.join(k+'='+repr(v) for k,v in kw.items())+')'
                record['action']=action
                try:
                    result=execute(actual,call,allowed);observation=serialized_result(result)
                    record['tool_error']=isinstance(result,dict) and 'error' in result
                except Exception as e:
                    observation='Error during execution: '+str(e);record['tool_error']=True
                observed.append(observation);record['observation']=observation
            except (ValueError,SyntaxError,TypeError) as e:
                record['parse_error']=str(e);observation='Invalid tool action: '+str(e);record['observation']=observation
            dialogue.append('Observation: '+observation);steps.append(record)
        # Gold and checks are strictly outside the model-visible dialogue.
        expected=[serialized_result(execute(gold,call,allowed)) for call in gold_calls]
        all_observations.extend(observed)
        state=checkers.state_checker({'GorillaFileSystem':actual},{'GorillaFileSystem':gold})
        response=checkers.response_checker(all_observations,expected,turn_index)
        native_valid=bool(observed or not gold_calls) and state['valid'] and response['valid']
        turns.append({'turn':turn_index,'finished':finished,'native_checks_valid':native_valid,'success':finished and native_valid,
                      'state_valid':state['valid'],'response_valid':response['valid'],'steps':steps,
                      'actual_root_sha256':digest(serialize_node(actual.root)),'gold_root_sha256':digest(serialize_node(gold.root))})
    return {'case_id':question['id'],'success':all(t['success'] for t in turns),'turns':turns}


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha256',required=True);p.add_argument('--policy',choices=['model','reference','finish_only'],required=True);p.add_argument('--model',type=Path);p.add_argument('--model-id',default='');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if sha256(a.protocol)!=a.protocol_sha256:p.error('Changed calibration protocol')
    protocol=json.loads(a.protocol.read_text())
    if a.policy=='model' and (not a.model or a.model_id not in protocol['models']):p.error('Undeclared model')
    if sha256(Path(__file__))!=protocol['evaluator_sha256']:p.error('Changed calibration evaluator')
    for name,h in protocol['support_source_sha256'].items():
        if sha256(ROOT/name)!=h:p.error('Changed support source '+name)
    backend,checkers,cases,answers,schemas=load(a.source_root,protocol)
    a.output.mkdir(parents=True,exist_ok=False)
    m={'kind':'native_BFCL_filesystem_ReAct_calibration','policy':a.policy,'model_id':a.model_id,'status':'running','protocol_sha256':a.protocol_sha256,'case_ids':[q['id'] for q in cases],'completed_case_ids':[],'source_sha256':sha256(Path(__file__)),'started_at':datetime.now(timezone.utc).isoformat(),'max_calls_per_turn':6,'max_new_tokens':384,'seed':45,'decode':'greedy','official_leaderboard_score':False}
    def save():(a.output/'run_manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
    save();started=time.monotonic();lock=None;client=None
    def timeout(*_):raise TimeoutError('1800 second calibration arm limit')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(1800)
    try:
        if a.policy=='model':
            os.environ.update(CUDA_VISIBLE_DEVICES='1',AA_EVAL_GREEDY='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
            lock=open(f'/tmp/agenticarxiv-rescue-{os.getuid()}-gpu1.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            import subprocess
            if int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())<16000:raise RuntimeError('Insufficient GPU1 memory')
            m['weights_sha256']={x.name:sha256(x) for x in sorted(a.model.glob('*.safetensors'))};m['config_sha256']=sha256(a.model/'config.json')
            declared=protocol['models'][a.model_id]
            if m['weights_sha256']!=declared['weights_sha256'] or m['config_sha256']!=declared['config_sha256']:raise ValueError('Undeclared checkpoint bytes')
            import torch
            from utils.llm_client import TransformersLLMClient
            torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.35,0)
            client=TransformersLLMClient(str(a.model),device='cuda',dtype='bfloat16',seed=45)
            import importlib.metadata
            m['versions']={v:importlib.metadata.version(v) for v in ('torch','transformers')};save()
        results=[]
        with (a.output/'traces.jsonl').open('x') as log:
            for q in cases:
                result=run_case(q,answers[q['id']],schemas,backend,checkers,a.policy,client,a.model_id)
                results.append(result);log.write(json.dumps(result,ensure_ascii=False)+'\n');log.flush();m['completed_case_ids'].append(q['id']);save();print(q['id'],result['success'],flush=True)
        passed=sum(r['success'] for r in results)
        if a.policy=='reference' and passed!=len(cases):raise RuntimeError('Native reference did not pass all cases')
        if a.policy=='finish_only' and passed:raise RuntimeError('Trivial FINISH passed nonempty native scenarios')
        m.update(status='completed',case_successes=passed,n_cases=len(cases),turn_successes=sum(t['success'] for r in results for t in r['turns']),n_turns=sum(len(r['turns']) for r in results),traces_sha256=sha256(a.output/'traces.jsonl'))
    except Exception as e:m.update(status='failed',error_type=type(e).__name__,error=str(e));raise
    finally:
        signal.alarm(0);m['wall_seconds']=round(time.monotonic()-started,3);save()
        if lock:lock.close()
    print(json.dumps({k:m[k] for k in ('status','policy','model_id','case_successes','n_cases','turn_successes','n_turns')}))
if __name__=='__main__':main()
