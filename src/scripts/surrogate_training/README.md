# 同构代理蒸馏

本目录保存固定目标 GNN 的概率蒸馏入口。代理与目标沿用同一 LEConv 架构，但代理独立初始化；目标模型仅通过三类 softmax 概率提供蒸馏信号。

BA3Motif 的查询预算、边翻转和连通性约束、smoke 与正式训练步骤记录在[实验方案](../../../notes/experiment_plans/2026-11-03/experiment-ba3motif-surrogate-distillation-plan.md)。已批准运行 `distill-002` 完成蒸馏：

```bash
/usr/bin/timeout 30m /home/ll_yqs2/data/anaconda3/envs/myexplainer/bin/python \
  -m src.scripts.surrogate_training.train_ba3motif_surrogate \
  --run-dir results/ba3motif-surrogate-distillation/distill-002
```

该运行的结果汇总、查询账本与覆盖统计保存在对应运行目录，validation KL 选出的代理 checkpoint 写入 `src/params/gnns/ba3motif/ba3motif_leconv_surrogate.pt`。复现实验时必须使用新的 run ID；入口拒绝覆盖已有运行目录或 checkpoint。
