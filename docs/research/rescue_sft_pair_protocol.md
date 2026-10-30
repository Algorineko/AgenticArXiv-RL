# 新数据 SFT 状态呈现配对可行性实验（训练前冻结）

目的：检查修正新训练输入中的状态可见性是否改变模型在相同真实运行提示下的行为。不是历史复现，不把16步可行性训练称作健康完整SFT，更不据此归因RL。

数据固定208个决策样本/臂，来自96条AI训练轨迹：原冻结train12任务＋新编AI两族三编号两状态12任务，每任务4个固定语言前缀。论文仅AI训练池。hidden/visible两份样本逐行共享source_task_id、轨迹位置与assistant标签，只改变prompt是否呈现真实setup；所有数据生成在本次模型训练前完成。数据manifest SHA256：87cb9cd229c139b6483950303dd9be67b2fe060cbeedb60a97c88fcc6c7134b0。原始dev/promotion/final均未作为训练样本。

两臂从同一原始Qwen2.5-1.5B-Instruct权重重新初始化，都在本机GPU1依次训练；相同seed123、相同64个决策行与顺序、batch1×累积4、16个真实optimizer steps、lr2e-5、weight_decay0、clip1。FP32参数/优化器＋BF16 autocast，梯度检查点，最终同样导出BF16。拒绝截断超过4096token的行。最多1800秒/臂，至少38GB空闲，85%显存分配上限。记录每步loss/grad_norm、训练参数前后哈希、真实步数与输出权重哈希。输入token数可因状态文本长度不同而不同，行/步骤/输出标签严格配对；不声称等token训练。

训练完成后固定做2×2开发评测：hidden/visible训练权重各搭配baseline可见状态与current_actions_only隐藏状态，沿用12条LG开发探针，greedy seed45/384新token/6轮。主比较为同样可见状态下visible训练臂对hidden训练臂的操作凭据＋FINISH，补交叉部署与原始模型8/12、发布SFT0/12/隐藏6/12的描述性参照。

不根据中间loss挑checkpoint，不增加步数或重跑挑种子，不训练既有发布SFT/GRPO。如果16步没有改善，作为短训练可行性负结果，不声称“训练对齐无效”。若出现收益，也只是两族窄语法探针的内部证据，规则执行器12/12仍是必要基线。暂不解封原final；是否做更广泛新任务验证须先审计完整训练与开发结果，不能仅凭train loss下降升级主张。
