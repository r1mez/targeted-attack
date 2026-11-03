# 图分类模型

`BA3MotifLEConv` 沿用 `myexplainer/gnns/ba3motif_gnn.py` 的两层 LEConv、均值池化与三类分类头。源类名为 `BA3MotifGCN`，项目以实际卷积类型命名 checkpoint。[模型实现](ba3motif_gnn.py)

checkpoint 位于 `src/params/gnns/{数据集名}/{数据集名}_{GNN类型}.pt`。加载 BA3Motif 权重时，用其中的 `model_kwargs` 实例化 `BA3MotifLEConv`，再将 `model_state_dict` 传给 `load_state_dict`。`class_names` 按输出顺序保存为 C5、House、K5。

攻击或生成期间固定分类器参数并使用评估模式，记录输出类别顺序及预处理。白盒方法需要输入梯度时，不应误用会关闭全部梯度的上下文；不得因此更新分类器参数。训练和推理评估均须按已获批准的方案执行。
