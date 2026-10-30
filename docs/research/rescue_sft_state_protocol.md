# SFT 阶段对照（模型推理前冻结）

使用ModelScope公开发布的Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-SFT；下载逐文件固定Revision并校验API给出的SHA256。模型自己的training_manifest标注2 epochs全参SFT，而非健康1 epoch模型。本实验不能把它当作缺失的1 epoch checkpoint，也不证明它与所测GRPO之间是唯一直接父子关系。

仅在rescue_receipt_catalog_v1的search_dev 12条状态反事实请求上运行baseline一次。保持greedy/seed45/384新token/6轮/900秒，本机GPU1，模型不访问promotion/final。汇报strict、操作凭据、首步FINISH和每对状态差异；与本机原始权重比较硬件一致，与A40 GRPO比较仅作行为诊断，不比较时延。

主要问题：在已发布SFT产物上是否已经出现“同文翻译请求仅列表会执行、此前下载即直接结束”。若存在，不能将该现象专属归因于后续RL；若不存在，也不能仅凭两个发布checkpoint差异将原因归为RL。预先关心三对翻译的逐题变化，其他6题作为force参数控制。

该对照不更新权重、不搜索提示、不选择候选。2 epoch SFT健康性尚未建立；健康SFT起点仍需独立对照。规则执行器已在同一12题达12/12，所以任何好分数都不构成RSI或广泛能力突破。
