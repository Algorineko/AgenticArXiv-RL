"""Post-hoc alias correction; frozen final-v1 primary scores remain unchanged."""
from copy import deepcopy
from rescue_operation_receipts import decode_action
from rescue_plan_receipts import step_receipt as frozen_step_receipt


def step_receipt(step,name,args,target):
    if name=='get_recently_submitted_cs_papers':
        action=decode_action(step.get('action'))
        if action and isinstance(action.get('args'),dict) and isinstance(action['args'].get('aspect'),str):
            # Match the already existing tool/strict-metric case-insensitive cs prefix contract.
            action=deepcopy(action)
            action['args']['aspect']=action['args']['aspect'].upper().removeprefix('CS.')
            step={**step,'action':action}
    return frozen_step_receipt(step,name,args,target)


def assess_plan(history,task):
    names=task['expected_tools'];args=task['expected_tool_args'];targets=task['expected_paper_ids']
    if not (len(names)==len(args)==len(targets)):raise ValueError('Incomplete plan')
    cursor=0;matched=[]
    for name,kw,target in zip(names,args,targets):
        found=False
        while cursor<len(history):
            index=cursor;cursor+=1
            if step_receipt(history[index],name,kw,target):matched.append(index);found=True;break
        if not found:break
    seen=len(matched)==len(names)
    finished=bool(history and str(history[-1].get('action','')).strip()=='FINISH')
    return {'ordered_operations_seen':seen,'operations_and_finish':seen and finished,'matched_step_indices':matched,'requested_operations':len(names)}
