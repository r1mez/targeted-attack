# 实验代码

所有实际代码统一放入 `src/`。BA3Motif 已有数据加载器、LEConv 分类器和训练入口。实验审批与解释器要求见[项目 README](../README.md)。

- [data/](data/README.md)：数据接口、预处理、划分与本地数据集。
- [models/](models/README.md)：分类模型、检查点加载和预测接口。
- [attacks/](attacks/README.md)：定向攻击方法及基线。
- [counterfactuals/](counterfactuals/README.md)：反事实生成与可行性约束。
- [configs/](configs/README.md)：明确、可记录的实验配置。
- [scripts/](scripts/README.md)：训练、攻击、生成、评估及结果整理入口。

所有实验使用 `/home/ll_yqs2/data/anaconda3/envs/myexplainer/bin/python`。每次实验运行均须关联用户已确认的方案；长任务先运行批准范围内的小规模 smoke test。

本地数据集按数据集名存放在 `src/data/` 下，GNN 检查点存放在 `src/params/gnns/{数据集名}/`。通过配置指定各项路径。Git 忽略本地数据集、权重和 Python 缓存。依赖说明见 [requirements.txt](../requirements.txt)。
