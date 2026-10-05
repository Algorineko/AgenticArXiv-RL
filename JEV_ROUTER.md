# Jev guided routing

AgenticArXiv 原本让 Qwen2.5-1.5B-GRPO 在一次生成里同时决定两件事：调用哪个工具，
以及参数怎么写。对小模型来说，这两个问题会互相干扰。它可能理解了用户想找论文，
却在 `search_arxiv_papers` 和 `get_recently_submitted_cs_papers` 之间选错；也可能认出了
下载意图，却把一个可以直接使用的 arXiv ID 当成非法列表序号。

这个实验把封闭集合里的工具选择交给 Jev，把明确可验证的参数交给代码，只让 Qwen
处理剩余歧义。它不是替换 Agent，而是给小模型加一个轻量、可关闭、可审计的决策前层。

## 实验结果

使用同一 Qwen checkpoint、同一离线 snapshot、seed=42，在预先固定的 10 个混合任务上
各重复 3 次。两组各得到 30 条有效轨迹，没有运行异常。

| 指标 | Qwen policy | Jev guided | 变化 |
| --- | ---: | ---: | ---: |
| strict success | 40% | **60%** | **+20 pp** |
| tool accuracy | 60% | **80%** | **+20 pp** |
| argument accuracy | 50% | **70%** | **+20 pp** |
| reference accuracy | 60% | **80%** | **+20 pp** |
| false FINISH | 40% | **20%** | **-20 pp** |
| 平均 Qwen token | 3864.7 | **1004.5** | **-74.01%** |
| 平均总延迟 | **3447.0 ms** | 4720.9 ms | +36.96% |

policy 在 10 个任务中稳定完成 4 个，guided Jev 完成 6 个；
原本成功的任务没有退化。外部路由增加约 1.27 秒平均延迟。

也就是说在不重训 Qwen、不更换环境的前提下，Jev guided routing 能让现有 1.5B Agent 在一部分任务上从失败变成成功。

## Jev 工作范围


任务要求按 `all:agentic reinforcement learning` 检索最近 30 天的 5 篇论文。

原 policy 三次都走成了分类浏览：

```text
get_recently_submitted_cs_papers(aspect="RO", days=30)
```

工具本身执行成功，但它没有执行用户要求的关键词查询，因此 strict success 为 0/3。

guided Jev 三次都先选中 `search_arxiv_papers`，随后参数解析器直接保留用户给出的查询、
时间窗和数量：

```text
search_arxiv_papers(
    query="all:agentic reinforcement learning",
    days=30,
    max_results=5,
)
```

结果为 3/3 strict success。这个案例体现的是 Jev 最适合的工作：在两个语义相近、但
用途不同的工具之间做封闭选择。

### 2. arXiv ID 下载问题：从 0/3 到 3/3

任务直接给出 `2608.14528v1` 并要求下载。原 policy 把它误解成列表序号，三次都以
“ref 超出 1-based 索引范围”为由提前结束。

guided Jev 以 0.99 confidence 选择 `download_arxiv_pdf`，参数解析器识别出这是合法的
arXiv ID，而不是候选列表序号：

```text
download_arxiv_pdf(ref="2608.14528v1")
```

三次下载均成功，strict success 从 0/3 变为 3/3。这里的收益不只来自“选对下载工具”，
也来自把清晰的标识符交给确定性代码，而不是让小模型重新解释一遍。

### 3. `infeasible_zero_index`：

这个任务要求下载“第 0 篇论文”。项目的候选论文索引从 1 开始，所以正确行为是解释
参数非法并结束，不能真的调用下载工具。

原 policy Qwen 本身已经能完成这题，三次都是正确的；因此它不是一个“Jev 超过 Qwen”
的样本。值得记录的是，某次 Jev 以 0.88 confidence 错选了下载工具，但 guided 层在工具
执行前识别出 `ref=0`：

```text
Jev: download_arxiv_pdf
validator: non_positive_reference
final: 无法执行——论文序号必须从 1 开始
tool calls: 0
```

也就是说，高置信度 router 出错并没有直接变成一次错误副作用。这个案例说明 Jev 不是
被盲目信任的单点分类器；它被放在 schema 与业务合法性检查之后使用。

## 在 Jev 缩小工具列表基础上采用 guided

最初的 `legacy` 实现只做一件事：Jev 选出工具，再让 Qwen 在单工具 prompt 中生成参数。
真实 A/B 中，这个版本将 Qwen token 减少了 29.52%，但 strict success 仍是 40% 对 40%，
平均延迟还从 3374.9 ms 增加到 7236.1 ms。

问题并不在 Jev 有没有选对。1.5B checkpoint 面对一个训练时没见过的“固定工具、只写
参数”prompt，仍可能输出另一个工具名或错误参数。继续润色 prompt 没有稳定解决这个
分布偏移，因此最终结构改成：

```text
task + state + available tools
              |
              v
       Jev chooses next tool
              |
       confidence >= threshold? ---- no ----> original Qwen policy
              |
             yes
              v
   resolve explicit observable arguments
        | resolved          | ambiguous
        v                   v
 schema + legality      Qwen sees only the
    validation          selected tool schema
        |                   |
        +---------+---------+
                  v
             environment
```

分工理由：

- Jev 只做它擅长的离散决策，不负责自由生成 ID、查询字符串或整数；
- 明确出现在请求里的参数不会经过一次不必要的语言模型“转述”；
- 低 confidence、网络失败、未知工具、schema 错误和业务非法参数都有明确回退或拦截路径。

## 优势区间

为避免偶然性，还从先前 trace 中预选了 6 个“Jev 高置信度选对、
Qwen 多数失败”的任务，每题重复 3 次：

| 指标 | Qwen policy | Jev guided |
| --- | ---: | ---: |
| 轨迹数 | 18 | 18 |
| strict success | 0% | **100%** |
| tool accuracy | 0% | **100%** |
| argument accuracy | 8.33% | **100%** |
| reference accuracy | 12.50% | **100%** |
| 平均 Qwen token | 3348.7 | **0** |
| 平均总延迟 | **2852.1 ms** | 3137.1 ms |

这组任务本来就是按历史表现挑出的“Jev 优势区间”，所以不能拿 100% 外推整体分布。
它的用途是机制验证：当工具选择和参数都能被明确接管时，收益是否真的能传到完整
rollout。答案是可以。前面的 mixed 10-task 结果则用来检查这种收益是否仍有净增益。

## 实验设计

| 项目 | Baseline | 实验组 |
| --- | --- | --- |
| checkpoint | Qwen2.5-1.5B-GRPO | 同一 checkpoint |
| 工具环境 | MockArxivEnv replay | 同一 snapshot |
| seed / repeat | 42 / 3 | 42 / 3 |
| 工具选择 | Qwen | Jev-latest |
| 明确参数 | Qwen | 确定性解析器 |
| 含糊参数 | Qwen | 同一 Qwen |

参数由 `expected_tool_args` 判定，`strict_success` 同时要求正常结束。

实验设计中路由器看不到 `expected_tools`、reward 或任何答案字段。每次决定、confidence、是否采用、
回退原因、延迟和 token 都写入 trace，故而“Jev 真正控制了动作”和“API 失败后 Qwen
完成任务”可以被区分开。

## 更大范围的路由探测

在 81 条 expanded 任务上对 Jev 做过一次 routing-only probe，只测首步选择，不执行
完整 Agent：

| 指标 | 结果 |
| --- | ---: |
| next-tool accuracy | 70/81 = 86.42% |
| train | 43/51 = 84.31% |
| dev | 7/8 = 87.50% |
| iid_test | 16/18 = 88.89% |
| ood_test | 4/4 = 100% |
| 平均 confidence | 0.902 |
| 平均延迟 | 1.219 s |
| 估算费用 | USD 0.00333 |

11 个错误集中在标题文字指代和 infeasible 请求。正确样本平均 confidence 为 0.933，错误
样本为 0.702；这也是默认把阈值设为 0.80、低于阈值交回 policy 的依据。需要强调：
86.42% 是首步分类准确率，不是完整 Agent 成功率。

## 开关与失败处理

默认不启用外部路由：

```env
TOOL_ROUTER=policy
```

启用 guided Jev：

```env
TOOL_ROUTER=jev
ROUTER_ARGUMENT_MODE=guided
TYPESAFE_API_KEY=your-key
JEV_MIN_CONFIDENCE=0.80
```
配置模板见 `jev_config.example.env`。真实 key 放在被 `.gitignore` 排除的 `.env.local`。
`ROUTER_ARGUMENT_MODE=legacy` 只用于复现实验，不是推荐设置。

以下情况自动退回原 Qwen policy：

- Jev confidence 低于阈值；
- TLS、超时、429/5xx 或响应格式错误；
- Jev 返回未知工具；
- Qwen 生成的 routed arguments 无法通过 schema 检查。

明确非法的参数不会回退后继续尝试，而是给出具体原因并安全结束；`ref=0` 就属于这一类。

## 失败尝试与保留的限制

1. routing-only probe 证明 Jev 有分类信号，但不能证明端到端收益；
2. legacy A/B 只降低 Qwen token，没有提高 strict success；
3. 两版 prompt-only 参数生成尝试仍被 1.5B 模型的分布偏移抵消；
4. guided 版本加入确定性参数解析和合法性保护后，mixed 10-task 才从 40% 提升到 60%。

当前限制同样明确：样本仍小，API 延迟高于本地 1.5B 推理，标题级模糊指代和不支持能力
仍是弱项。要声称整体 benchmark 提升，还需要在预先注册的 81 题或独立 holdout 上完成
同样的端到端 A/B。

