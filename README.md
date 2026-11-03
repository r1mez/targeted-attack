# GNN 图分类定向攻击与反事实生成

给定已训练的图分类模型、输入图和指定目标类别，研究满足约束且扰动尽可能小的图修改方法。

当前研究设想是以边界图的中间层 embedding 提供定向攻击的 latent 方向。BA3Motif 加载器和目标 GNN 训练入口已准备好。[训练命令](src/scripts/gnn_training/README.md)

数据采用 `myexplainer` 的 BA3Motif（C5 / House / K5），本地副本位于 `src/data/ba3motif/`，固定划分为 700/150/150。目标 GNN 沿用源项目的 LEConv 架构，训练输出为 `src/params/gnns/ba3motif/ba3motif_leconv.pt`。后续反事实实验中，目标 GNN 仅开放 softmax 概率，同构代理提供表示与梯度。编辑固定节点和节点特征，仅做边增删。

源码与依赖入口：

- [src/](src/README.md)：数据接口、模型、攻击、反事实、配置及实验入口。
- [README.md](README.md)：项目入口与协作要求。
- [requirements.txt](requirements.txt)：指定 `myexplainer` 环境的现有依赖快照，并非已确认的项目最小依赖集。

本地研究资料保存在 `papers/`（原文及索引）、`notes/paper_notes/YYYY-MM-DD/`（按日期归档的逐篇论文笔记）、`notes/`（科研 TODO、论文研读与讨论、实验进度及方案）和 `results/`（获批实验输出）。论文调研可以先阅读并在对话中汇报，是否撰写阅读笔记及撰写范围由用户明确决定。研究日志不代替未获授权的阅读笔记，也不记录目录调整或 Git 等工程维护事项。`.gitignore` 显式列出这些研究目录、本地 `AGENTS.md`、`.gitignore` 和工具配置等忽略项。在已有本地工作区开展协作前，先阅读 `AGENTS.md`。未经用户明确允许，不得自主创建或修改 Git 提交。

所有实验，包括训练、推理评估、攻击、生成与 smoke test，均须先提交具体方案并经用户明确确认。统一使用 `/home/ll_yqs2/data/anaconda3/envs/myexplainer/bin/python`，依赖查询和管理也使用该解释器的 `-m pip`；长任务先执行批准范围内的小规模 smoke test。攻击或生成期间固定分类器参数并使用评估模式，区分命中指定目标类别与普通类别翻转，未证明最优性时不声称扰动全局最小。
