# 当前源码

当前研究主干是 Graph AE B、无配对类别 CFM 与最终软图反事实损失。所有实际代码放在本目录。

- [data/](data/README.md)：BA3Motif raw/processed、固定划分、原图目标概率缓存对应。
- [models/](models/README.md)：LEConv、冻结模型访问、可微带权代理与概率目标访问。
- [counterfactuals/](counterfactuals/README.md)：B AE、类别 flow、软图生成、三项训练目标及独立软原型工具。
- [experiments/](experiments/README.md)：共享审批、独立运行目录、指纹和资源记录。
- [scripts/](scripts/README.md)：训练与评价入口；入口之间不相互导入算法实现。
- [configs/](configs/README.md)：当前 AE / Flow / 软原型配置。
- [attacks/](attacks/README.md)：尚无当前主线攻击实现。

数据位于 `src/data/ba3motif/`；分类器权重位于 `src/params/gnns/ba3motif/`。既有 B AE / CFM 权重及其出处保留在原 `results/` 目录，由配置显式引用。固定使用项目解释器，模型运行需另行批准。
