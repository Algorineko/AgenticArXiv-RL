from agents.prompt_templates import get_react_prompt,get_task_contract_react_prompt

def test_opt_in_rule_preserves_request_and_prior_history():
    task='翻译第1篇，然后查看缓存状态。'
    history='此前已下载第1篇'
    legacy=get_react_prompt(task,'tools',history)
    fixed=get_task_contract_react_prompt(task,'tools',history)
    assert '直接 FINISH 即可，不要再调用 get_paper_cache_status' in legacy
    assert '直接 FINISH 即可，不要再调用 get_paper_cache_status' not in fixed
    assert '用户还明确要求查询缓存或执行其他操作' in fixed
    assert task in fixed and history in fixed
    assert '不要为了等待异步翻译完成而自行轮询' in fixed
