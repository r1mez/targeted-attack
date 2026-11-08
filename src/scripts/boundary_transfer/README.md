# BA3Motif 连续软边界原型工具

`generate_soft.py` 是独立的连续边权生成入口。具体优化、固定自环的加权构图和训练图结构参照由 [`soft_boundary.py`](../../counterfactuals/soft_boundary.py) 实现；冻结模型加载与概率查询使用统一模型接口，审批、文件校验和输出目录使用共享运行工具。

该工具保持已有的类对 logits 目标、结构距离、anchor L1、候选排序及类对概率通过阈值。它保存连续加权邻接与代理 embedding，按需要调用冻结目标模型进行概率验收。它不接入当前反事实模型训练；当前训练的 `L_boundary` 仍为 0，后续用途待确认。

当前[配置](../../configs/ba3motif_soft_boundary.json)为 `approval_status="待确认"`、`plan_path=null`。历史批准仅保存在 `historical_approval` 中；入口会在模型加载、数据读取和输出写入前拒绝运行。开展新实验须先为当前实现提交具体方案并获得确认，包含 smoke、正式规模、查询预算及输出位置。本次清理未运行训练、推理、优化、评估或 smoke。

新入口预留 `smoke` 与 `full` 两个阶段，后者仅复用相同入口、源码指纹、配置和新批准方案下通过的 smoke。配置使用独立目录 `results/ba3motif-soft-boundary-current/{阶段}-{编号}/`，不会复用旧实现的 smoke。

- `run.json` 与 `approved_plan.md` 保存执行配置、来源校验值和批准方案。
- `reference_stats.json` 保存实际使用的训练图结构参照。
- 每个候选的 JSONL 轨迹保存连续优化过程；`graphs/` 保存加权邻接与代理 embedding。
- `queries.jsonl` 在目标前向开始前登记用量，`candidates.jsonl` 与 `summary.json` 保存逐图结果和逐类对汇总。
- smoke 中保留原始硬图与其浮点权重表示的一致性检查，调用统一的概率查询接口。

旧硬边界采样、离散化诊断和对应报告入口已清理。历史产物仍保留在 `results/ba3motif-boundary-transfer/`、`results/ba3motif-boundary-discretization/` 与 `results/ba3motif-soft-boundary/`，只读用于查阅与累计目标查询记账。既有账本累计 599 次目标前向，保留的上限为 600；未来新实验必须在新方案中明确可用预算，现有配置不足以再完成 smoke。
