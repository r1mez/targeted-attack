# 数据接口与预处理

`load_ba3motif` 使用 PyG `InMemoryDataset`：从 `raw/graphs.npz` 校验并处理固定划分，首次加载时在数据集目录下建立 `processed/` 缓存；之后直接读取缓存。处理保留节点顺序、原始边和自环。[加载器](ba3motif_dataset.py)

用户已确定使用 `myexplainer` 中的 BA3Motif C5 / House / K5 版本。原始数据位于 `src/data/ba3motif/raw/graphs.npz`，处理缓存位于 `src/data/ba3motif/processed/`，沿用固定 700/150/150 划分。来源及哈希保存在 `src/data/ba3motif/provenance/`。

当前反事实输出为对称软邻接，固定节点、原始节点特征和原自环，不要求连通。原始图仍保持二值，不因生成任务改变数据。load_originals 将训练/验证原图与既有目标概率缓存匹配，以目标原预测作为 classes；load_graph_split 可读取测试原图，select_validation 沿用既有 selection_ids。缓存校验和原图读取不执行模型查询。

本地数据集按 `src/data/<数据集名>/` 存放，各目录由 `.gitignore` 排除。通过配置指定加载路径，记录来源、版本、许可、预处理、划分和随机种子。保留原始数据，并遵守根目录的实验审批规范。
