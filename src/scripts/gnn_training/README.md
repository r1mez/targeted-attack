# GNN 训练

从仓库根目录运行 BA3Motif 目标 GNN 训练，将 `manual-001` 换成本次使用的独立运行标识。设备由 `--device` 指定。[训练入口](train_ba3motif_gnn.py)

```bash
/home/ll_yqs2/data/anaconda3/envs/myexplainer/bin/python \
  -m src.scripts.gnn_training.train_ba3motif_gnn \
  --device cuda:0 \
  --run-dir results/ba3motif-target/manual-001
```

默认沿用源项目的 300 epochs、batch size 128、Adam（学习率 0.001）和 seed 42。使用 700 张训练图拟合模型，按 150 张验证图的交叉熵选择 checkpoint。输出位于 `src/params/gnns/ba3motif/ba3motif_leconv.pt`，可用 `--output-path` 指定其他位置。已有权重和运行目录受覆盖检查保护。运行目录记录 `run.json`、逐轮验证指标 `training.jsonl` 和最优权重 `best.pt`。CUDA 散射加法的浮点累加顺序可能导致不同运行之间的数值差异。
