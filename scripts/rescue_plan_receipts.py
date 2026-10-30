"""Ordered requested-operation witnesses allowing extra calls, offline only."""
import ast
import re
from rescue_operation_receipts import decode_action,receipt


def literal_observation(text):
    # Offline cache observations contain fixed datetime reprs. Strip only that
    # whitelisted constructor shape; never execute tool observation text.
    text=re.sub(r'datetime\.datetime\(\d+(?:,\s*\d+){2,6}\)', 'None', str(text))
    try:return ast.literal_eval(text)
    except (ValueError,SyntaxError):return None


def step_receipt(step,name,args,target):
    action=decode_action(step.get('action'))
    if not action or action.get('name')!=name or not isinstance(action.get('args',{}),dict):return False
    actual=action.get('args') or {}
    if name=='get_recently_submitted_cs_papers':
        if any(actual.get(k)!=v for k,v in args.items()):return False
        matched=re.match(r'^成功获取\s+(\d+)\s+篇论文(?:[:。]|\s*$)',str(step.get('observation','')))
        return bool(matched and int(matched[1])==args['max_results'])
    if name in ('download_arxiv_pdf','translate_arxiv_pdf'):
        if not target:return False
        if any(actual.get(k)!=v for k,v in args.items() if k not in ('ref',)):return False
        return receipt(step,{'name':name,**({'force':True} if args.get('force') else {})},target)
    if name=='get_paper_cache_status':
        obj=literal_observation(step.get('observation',''))
        return (isinstance(obj,dict) and obj.get('paper_id')==target and
                isinstance(obj.get('pdf_ready'),bool) and isinstance(obj.get('translated_ready'),bool))
    raise ValueError('Unsupported plan receipt tool: '+name)


def assess_plan(history,task):
    names=task['expected_tools'];args=task['expected_tool_args'];targets=task['expected_paper_ids']
    if not (len(names)==len(args)==len(targets)):raise ValueError('Incomplete declared operation plan')
    cursor=0;matched=[]
    for name,kw,target in zip(names,args,targets):
        found=False
        while cursor<len(history):
            index=cursor;cursor+=1
            if step_receipt(history[index],name,kw,target):
                matched.append(index);found=True;break
        if not found:break
    seen=len(matched)==len(names)
    finished=bool(history and str(history[-1].get('action','')).strip()=='FINISH')
    return {'ordered_operations_seen':seen,'operations_and_finish':seen and finished,'matched_step_indices':matched,'requested_operations':len(names)}
