# 多分类图反事实生成

当前主线是 **Graph AE B＋无配对类别条件 CFM**：从指定原图的节点 latent 出发，执行源类别 lift 与目标类别 land，解码得到对称软邻接。节点数、原始节点特征与自环固定；当前输出为软图，不要求连通。

Flow 随机初始化，从第一次参数更新起使用三项联合目标： `L_CFM + lambda_cf * L_cf + lambda_dist * L_dist`。`L_cf` 是最终软图在冻结代理 GNN 上的目标类别交叉熵；`L_dist` 是相对原图、每图对无向非自环节点对求和再对 batch 求均值的软 L1。只更新共享 flow，梯度完整经过冻结 decoder、land 和 lift。训练不需要为每张原图寻找配对的目标反事实图。边界损失暂为 0，边界引导与硬图输出留作后续研究。

已完成目标 LEConv 训练、同构代理蒸馏、Graph AE B 验收，以及本机 RTX A6000 上的单阶段三项损失全量训练：700 图训练、150 图验证、100 轮。全部 150 测试图的 300 个目标任务中，目标 GNN 预测翻转率为 97.33%（292/300），定向翻转率为 89.33%（268/300）。输出为软图，边界引导尚未接入。此前仅 matching loss 的 CFM 与旧硬图测试保留为历史证据，当前 Flow 不加载其初始化。

## 数据、权重与入口

- 数据采用 `myexplainer` 的 BA3Motif（C5 / House / K5），位于 `src/data/ba3motif/{raw,processed}/`，划分为 700/150/150。
- 目标权重为 `src/params/gnns/ba3motif/ba3motif_leconv.pt`，同构代理为同目录的 `ba3motif_leconv_surrogate.pt`。反事实任务中目标仅返回类别概率，以代理提供输入梯度。
- [src/](src/README.md)：源码目录与职责。
- [反事实训练与评价](src/scripts/counterfactual_training/README.md)：B AE 预训练、单阶段 Flow 联合训练与软图评价。
- [GNN 训练](src/scripts/gnn_training/README.md)及[代理蒸馏](src/scripts/surrogate_training/README.md)：既有分类器入口。
- [软边界工具](src/scripts/boundary_transfer/README.md)：后续原型研究工具，不接入当前任务训练。
- [requirements.txt](requirements.txt)：既有环境依赖快照，不是已确认的最小依赖集。

## 协作与运行

所有实际代码位于 `src/`。本地研究资料位于 `notes/`、`papers/` 与 `results/`；这些资料和本地 `AGENTS.md` 不纳入 Git。在已有工作区协作前先读 `AGENTS.md`。未经明确允许，不创建或修改 Git 提交。

所有模型训练、前向评价、梯度检查与 smoke 均需独立实验方案及用户确认。统一使用解释器 `/home/ll_yqs2/data/anaconda3/envs/myexplainer/bin/python`，模型训练使用本机 CUDA GPU，不回退 CPU。ba3motif_flow.json 保存最小 smoke 配置，ba3motif_flow_full.json 保存 2026-11-08 已批准并完成的全量训练/评价设置；不能将已有许可自动推广到新增实验。完整本地结果位于 results/ba3motif-soft-joint-full/。
