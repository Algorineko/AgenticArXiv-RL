test_modules.py
组件临时测试：
```sh
python tests/test_modules.py arxiv --aspect "*" --max 20 > ./output/results.txt
python ./tests/test_modules.py llm --prompt "请介绍计算机科学领域新颖的研究课题"
python ./tests/test_modules.py tools
```

test_tool_registry.py
```sh
# 列出所有工具
python ./tests/test_tool_registry.py list
# 测试arXiv工具
python ./tests/test_tool_registry.py test --tool arxiv --aspect AI --max-results 5
# 测试格式化工具
python ./tests/test_tool_registry.py test --tool format
# 测试参数验证
python ./tests/test_tool_registry.py validate
```
