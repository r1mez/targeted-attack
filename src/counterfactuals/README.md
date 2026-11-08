# Graph AE＋无配对 CFM

当前只保留一条软图反事实主线：

- [graph_ae.py](graph_ae.py)：确定性结构 Graph AE B；节点编码为 `25×32`，对称节点对 MLP 解码。兼容既有 B 权重。
- [unpaired_flow.py](unpaired_flow.py)：共享 ClassFlow、matching MSE、RK4、源类 lift 与目标类 land。
- [soft_counterfactual.py](soft_counterfactual.py)：checkpoint 出处加载、固定标准化、可微软图输出与任务损失。
- [training.py](training.py)：B AE 的独立 BCE 预训练，以及随机初始化 Flow 的单阶段三项损失训练。
- [soft_boundary.py](soft_boundary.py)：独立软原型优化与结构参照；当前任务训练不加载原型。

Flow 随机初始化，训练目标为 `L_CFM + lambda_cf * L_cf + lambda_dist * L_dist`。`L_cf` 使用冻结代理 logits 计算最终软图的目标类别交叉熵。`L_dist` 对最初原图与软图的 `i<j` 差值每图求和、batch 求均值。只更新 flow，完整保留 decoder、land、lift 的输入梯度。encoder 固定输入编码可 no_grad。

生成保留节点、原始 X 和原对角线，非自环边权镜像为对称矩阵；不阈值、不采样、不 Top-k 翻边。边界损失为 0，硬图输出为后续待办。

B AE 已有冻结权重；历史仅 matching loss 的 CFM 不作为当前初始化。2026-11-08 已完成 RTX A6000 上 700 图、100 轮的单阶段 Flow 正式训练，验证 150 图；150 测试图/300 任务上目标预测翻转 292/300、定向命中 268/300。只适用于当前软图协议。旧配对编辑 VAE、A/C 对照、硬搜索与随机方向比较已退出活跃源码；历史结果与清理前快照保留于本地 results。入口见[训练说明](../scripts/counterfactual_training/README.md)。
