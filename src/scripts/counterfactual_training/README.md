# AE 预训练与单阶段 Flow 训练

生成器只有 Graph AE 先预训练并冻结。Flow 随机初始化，从第一次参数更新起联合优化：

`L_CFM + lambda_cf * L_cf + lambda_dist * L_dist`

CFM 是联合目标中的一项损失，不是 Flow 的前置训练阶段。当前训练不加载历史 CFM 权重，也不提供 Flow 预训练/微调分支。

模型训练和训练 smoke 使用本机 GPU。配置显式指定 `device`，当前选择 `cuda:1`（RTX A6000）；训练入口在 PyG 导入前设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，核对型号后将模型及训练/验证数据迁移到该 GPU。GPU 不可用或显存不足时停止，不自动回退 CPU。

| 入口 | 职责 |
| --- | --- |
| train_ae.py | 预训练 B 结构 AE；独立邻接 BCE，新重训按验证软 BCE 选模 |
| train_flow.py | 随机初始化 Flow，单阶段三项损失联合训练 |
| evaluate.py | 加载该训练的输出权重，生成软图并分别评价代理与目标概率 |

## 调用方式

从项目根目录用模块方式调用：

```bash
/home/ll_yqs2/data/anaconda3/envs/myexplainer/bin/python -m src.scripts.counterfactual_training.train_flow --help
```

实验方案批准且配置补齐后，训练命令为：

```bash
/home/ll_yqs2/data/anaconda3/envs/myexplainer/bin/python -m src.scripts.counterfactual_training.train_flow smoke
/home/ll_yqs2/data/anaconda3/envs/myexplainer/bin/python -m src.scripts.counterfactual_training.train_flow full
```

`smoke/full` 仅表示小规模检查或正式运行，不是两个 Flow 训练阶段。CLI 没有 `--stage`。默认 ba3motif_flow.json 仍只批准最小 train/smoke，上述不带 --config 的 full 命令会被审批检查拒绝。2026-11-08 正式实验单独使用已批准的 --config src/configs/ba3motif_flow_full.json，训练和评价已完成。full 需要同操作、源码和配置的新 smoke。

AE 和软图评价入口使用相同调用方式，将模块名替换为 train_ae 或 evaluate。

## 前向、损失与参数

训练开始时，先用冻结 AE 编码原始训练图，计算 mean/std 并固定；验证、测试图不参与统计，也无需先训练 Flow。辅助节点输入沿用 AE seed＋graph ID，保持节点对应。

每次 Flow 更新都计算两条分支：

- 真实图 latent＋原预测类别＋独立噪声，计算 L_CFM。
- 源图 latent 经同一 Flow 的源类 lift、目标类 land，冻结 decoder 输出最终软图，以冻结代理的目标类 CE 计算 L_cf，并计算相对原图的 L_dist。

两项任务损失完整经过 decoder、land、lift；三项合并，只更新 Flow。AE 与代理参数冻结，decoder 和代理保留输入梯度；目标 GNN 只用于评价。

L_dist 每图对 i<j 求和、batch 求均值。L_boundary=0。固定原始 X 与自环，不阈值、不采样、不 Top-k。目标策略必须显式配置；现支持 all_other，即每张原图指定其他类别，未寻找配对目标图。

## 权重与记录

训练输入只需要冻结 AE、冻结代理、原始数据及目标原预测缓存。旧 B checkpoint 缺少内嵌 architecture，继续核对其 run / summary / samples 出处。历史 CFM 和旧硬图结果保留为证据，不是训练输入。

Flow 训练输出 `flow.pt`，记录 joint_flow、random 初始化、参数、architecture、由 AE 训练 latent 拟合的 mean/std、AE/代理出处、优化器和 RNG。评价通过 `stages.evaluate.inputs.flow_checkpoint` 显式指定这一训练结果及其 SHA-256；该字段待填不会影响训练输入核验。

训练保留原 700/150/150 划分，训练/验证原预测来自目标缓存。最小配置验证沿用 selection_ids；全量配置明确使用全部 150 验证图。每次独立保存配置、方案、源码、环境、样本和指标。最小 smoke 抽取训练/验证各 6 图，只作技术检查；后续获批的全量实验已经完成。评价保存软 L1、六方向的代理/目标命中，并单列原预测错误样本。

## 全量训练与逐类翻转率

2026-11-08 全量方案已批准并完成，配置为 src/configs/ba3motif_flow_full.json。训练 700 图/1400 目标任务，100 轮，验证使用全部 150 图/300 任务，测试覆盖全部 150 图/300 任务；测试集未加入训练。

已执行顺序为 train_flow smoke/full，随后将第 100 轮 flow.pt 路径和 SHA-256 机械填入 evaluate.inputs.flow_checkpoint，再运行 evaluate smoke/full。训练和生成均在 RTX A6000，目标概率服务在 CPU。产物位于本地 results/ba3motif-soft-joint-full/；测试目标预测翻转为 292/300（97.33%），指定目标命中为 268/300（89.33%）。训练目标查询 0，评价累计 486 次。旧 smoke 对源码和配置的校验不会自动替代新配置 smoke。

评价的 all、by_source、by_direction 均按“原图—指定目标”任务计数：any_flip_rate 为目标预测变成任意其他类的比例，targeted_asr 为命中该任务指定类别的比例。by_source 按目标 GNN 的原预测类分组；by_true_label 补充按真实标签的分组。per_graph 与 per_graph_by_source 则按原图统计两个目标尝试中至少一次翻转/定向命中的比例，分母与任务指标不同。保存全部原预测错误图并单列结果，空分组返回 null。
