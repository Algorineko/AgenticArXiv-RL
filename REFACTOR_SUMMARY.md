# AgenticArXiv → RL 改造完成总结

## ✅ 已完成的改造

### Step 0: 环境搭建
- ✅ 创建 `.venv` 虚拟环境
- ✅ 重写 `requirements.txt`（去 FastAPI/MySQL，加 torch/trl/transformers）
- ✅ 创建 `requirements-extra.txt`（可选依赖）
- ✅ 安装核心依赖（运行中）

### Step 1: 副作用解耦
- ✅ 创建 `agents/side_effects.py`
  - `SideEffectManager` 抽象接口
  - `NoOpSideEffectManager` 无操作实现

### Step 2: JSONL Trajectory 记录
- ✅ 创建 `rl/trajectory.py`
  - `TrajectoryStep` / `Trajectory` 数据类
  - `save_trajectory()` / `load_trajectories()` JSONL 读写
  - `create_trajectory()` 从 Agent 结果构造

### Step 3: MockEnv 快照环境
- ✅ 创建 `rl/env.py`
  - `MockArxivEnv` 快照回放环境
  - `generate_snapshot_from_benchmark()` 快照生成

### Step 4: Reward 计算
- ✅ 创建 `rl/reward.py`
  - `RewardCalculator` 复用 `benchmark/metrics.py`
  - Verifiable reward 规则（任务成功+1.0、工具准确+0.5 等）
  - `compute_step_reward()` 单步奖励

### Step 5: Rollout 循环
- ✅ 创建 `rl/rollout.py`
  - `rollout_single_task()` 单任务 rollout
  - `rollout_all_tasks()` 批量 rollout
  - CLI 接口（fire）

### Step 6: TRL 训练脚本
- ✅ 创建 `rl/train_sft.py`（SFT 训练脚本）
- ✅ 创建 `rl/train_dpo.py`（DPO 训练脚本）
- ✅ 创建 `rl/train_grpo.py`（GRPO 训练脚本，TODO: reward_fn）

### 辅助脚本
- ✅ 创建 `scripts/generate_sft_data.py`（SFT 数据生成）
- ✅ 创建 `scripts/generate_dpo_data.py`（DPO 数据生成）

### 目录结构
- ✅ 创建 `traces/train/`、`traces/eval/`
- ✅ 创建 `data/sft/`、`data/dpo/`
- ✅ 创建 `eval/`、`scripts/`
- ✅ 创建 `.gitignore`

---

## 📋 后续待完成（手动）

### 1. 修改 `base_agent.py` 注入副作用管理器
当前 `base_agent.py` 中的副作用（DB/SSE/translate/store）硬编码。

**需要修改**：
- 在 `__init__()` 中添加 `side_effect_mgr` 参数
- 替换所有 `store.xxx`、`log_service.xxx`、`event_bus.xxx` 调用为 `self.side_effect_mgr.xxx`

**参考代码**：
```python
# agents/base_agent.py
from agents.side_effects import SideEffectManager, NoOpSideEffectManager

class BaseAgent(ABC):
    def __init__(self, llm_client, side_effect_mgr: SideEffectManager = None):
        self.llm_client = llm_client
        self.side_effect_mgr = side_effect_mgr or NoOpSideEffectManager()
    
    # 在 _execute_with_side_effects() 中替换：
    # store.set_last_papers(...) → self.side_effect_mgr.set_last_papers(...)
    # event_bus.publish(...) → self.side_effect_mgr.publish_sse(...)
```

### 2. 验证 rollout 跑通
```bash
cd /Users/dev/projects/AgenticArXiv
source .venv/bin/activate

# 配置 LLM API
cat > AgenticArxiv/.env << 'EOF'
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-api-key
MODEL=gpt-4-turbo
EOF

# 测试 rollout
cd AgenticArxiv
python -m rl.rollout search_01 ../traces/train/
```

**期望输出**：
```
📋 任务: search_01 - 检索最近7天内人工智能(cs.AI)方向的论文，最多5篇
🤖 执行 Agent...
✅ 任务完成
   Reward: 1.50
   Metrics: task_completed=True, tool_call_accurate=True
💾 Trajectory 保存至: traces/train/rollout_20260621_150000.jsonl
```

### 3. 生成 SFT 数据
```bash
python scripts/generate_sft_data.py
```

### 4. 训练流程（依次执行）
```bash
# SFT
python -m rl.train_sft

# DPO（需要 SFT 模型）
python scripts/generate_dpo_data.py
python -m rl.train_dpo

# GRPO（需要 DPO 模型，TODO: 完善 reward_fn）
python -m rl.train_grpo
```

---

## 📦 新增文件清单

### RL 核心
- `AgenticArxiv/rl/__init__.py`
- `AgenticArxiv/rl/trajectory.py` — Trajectory 数据类 + JSONL 读写
- `AgenticArxiv/rl/env.py` — MockArxivEnv 快照环境
- `AgenticArxiv/rl/reward.py` — RewardCalculator
- `AgenticArxiv/rl/rollout.py` — Rollout 循环
- `AgenticArxiv/rl/train_sft.py` — SFT 训练
- `AgenticArxiv/rl/train_dpo.py` — DPO 训练
- `AgenticArxiv/rl/train_grpo.py` — GRPO 训练

### 副作用解耦
- `AgenticArxiv/agents/side_effects.py` — SideEffectManager 接口

### 辅助脚本
- `scripts/generate_sft_data.py` — SFT 数据生成
- `scripts/generate_dpo_data.py` — DPO 数据生成

### 依赖管理
- `AgenticArxiv/requirements.txt` — 重写（RL 核心依赖）
- `AgenticArxiv/requirements-extra.txt` — 可选依赖

### 其他
- `.gitignore` — 忽略 traces/outputs/.venv 等

---

## 🔧 关键修改点（需手动完成）

### `agents/base_agent.py`
- [ ] 在 `__init__()` 添加 `side_effect_mgr` 参数
- [ ] 替换 `store.xxx` → `self.side_effect_mgr.xxx`
- [ ] 替换 `event_bus.xxx` → `self.side_effect_mgr.xxx`
- [ ] 替换 `log_service.xxx` → `self.side_effect_mgr.xxx`

### `agents/agent_engine.py`
- [ ] 在 `__init__()` 传递 `side_effect_mgr` 给父类

### `rl/train_grpo.py`
- [ ] 完善 `reward_fn()` 的 TODO
  - 解析 responses 为 action
  - 执行工具得到 observation
  - 调用 `reward_calc.compute_reward()`

---

## 🎯 验证清单

- [ ] `.venv` 环境创建成功
- [ ] 核心依赖安装完成（torch/trl/transformers）
- [ ] `python -m rl.rollout search_01 traces/train/` 跑通
- [ ] `traces/train/*.jsonl` 生成
- [ ] Reward 数值合理（0.5 ~ 2.0）
- [ ] SFT 数据生成成功
- [ ] SFT 训练跑通
- [ ] DPO 数据生成成功
- [ ] DPO 训练跑通
- [ ] GRPO 训练跑通

---

## 🚀 下一步（你来操作）

1. **等待依赖安装完成**（后台运行中）
2. **Git 初始化 + 推送**：
   ```bash
   cd /Users/dev/projects/AgenticArXiv
   rm -rf .git
   git init -b main
   git add -A
   git commit -m "init: RL refactor base

   - 保留 ReAct(regex) Agent 核心
   - 新增 rl/ 包（env/policy/reward/trajectory/rollout/train_sft/dpo/grpo）
   - 改为 .venv 环境（去 MySQL 依赖）
   - 支持 SFT→DPO→GRPO 渐进式训练路径
   "
   gh repo create AgenticArXiv-RL --public --source=. --remote=origin --push
   ```
3. **手动修改 `base_agent.py`**（注入副作用管理器）
4. **验证 rollout**：`python -m rl.rollout search_01 traces/train/`
5. **开始训练**：SFT → DPO → GRPO

---

**改造基础架构已完成！** 🎉
