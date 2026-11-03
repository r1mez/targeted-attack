# 实验入口与结果整理

GNN 训练脚本集中在 `src/scripts/gnn_training/`。[训练命令与输出说明](gnn_training/README.md)

已训练 BA3Motif 目标模型的 edge edit 诊断脚本位于 `src/scripts/diagnostics/`，只通过类别概率检查目标模型。[诊断说明](diagnostics/README.md)

BA3Motif 同构代理的概率蒸馏入口位于 `src/scripts/surrogate_training/`；执行配置和授权状态见[蒸馏方案](../../notes/experiment-ba3motif-surrogate-distillation-plan.md)。

运行实验前，先提交包含假设、数据集、指标、预期结论和执行规模的方案并取得用户确认；smoke test 也适用。所有实验入口统一使用 `/home/ll_yqs2/data/anaconda3/envs/myexplainer/bin/python`，不得自行切换解释器。
