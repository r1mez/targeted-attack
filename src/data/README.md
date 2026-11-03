# 数据接口与预处理

`load_ba3motif` 使用 PyG `InMemoryDataset`：从 `raw/graphs.npz` 校验并处理固定划分，首次加载时在数据集目录下建立 `processed/` 缓存；之后直接读取缓存。处理保留节点顺序、原始边和自环。[加载器](ba3motif_dataset.py)

用户已确定使用 `myexplainer` 中的 BA3Motif C5 / House / K5 版本。原始数据位于 `src/data/ba3motif/raw/graphs.npz`，处理缓存位于 `src/data/ba3motif/processed/`，沿用固定 700/150/150 划分。来源及哈希保存在 `src/data/ba3motif/provenance/`。

已明确的反事实编辑范围为固定节点和节点特征、仅边增删。自环与连通性规则仍需在实验方案中确定，不应因加载便利而静默改变原图。

本地数据集按 `src/data/<数据集名>/` 存放，各目录由 `.gitignore` 排除。通过配置指定加载路径，记录来源、版本、许可、预处理、划分和随机种子。保留原始数据，并遵守根目录的实验审批规范。
