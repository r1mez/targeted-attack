# 分类器与冻结访问

[ba3motif_gnn.py](ba3motif_gnn.py) 保留源项目 LEConv architecture：两层 LEConv、均值池化和三类分类头，输出顺序 C5、House、K5。

[ba3motif_model_access.py](ba3motif_model_access.py) 收拢权重加载、参数指纹、可微带权批处理与概率目标访问：

- 冻结代理返回 logits，对软邻接的输入梯度保留，用于 L_cf。
- 目标访问仅返回类别概率并关闭反传；支持二值原图与对称软邻接。任务训练不读取目标内部表示或梯度。
- DenseProxy 用带权 LEConv 的等价稠密公式计算完整节点对，零权边仍保留输入梯度；避免旧 PyG 的 CUDA scatter 与确定性模式冲突。原模型 architecture 与权重不变，目标概率接口仍调用原模型。

冻结参数不等于关闭输入梯度。代理保持 eval 和参数冻结，但训练 flow 时不能对代理前向使用 no_grad。目标概率查询独立记账。

2026-11-08 的训练 smoke 已用生成软图核对稠密代理与原 PyG 公式的数值一致性；具体误差和梯度检查随运行记录保存。浮点运算顺序不同，不保证逐位相同。

GNN 权重位于 `src/params/gnns/ba3motif/`，保持 model_kwargs、model_state_dict、class_names 与数据的一致性。代理的软图迁移能力仍需评价，代理命中不替代目标命中。
