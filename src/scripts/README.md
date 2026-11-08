# 当前实验入口

- [gnn_training/](gnn_training/README.md)：既有目标 LEConv 训练。
- [surrogate_training/](surrogate_training/README.md)：既有同构代理概率蒸馏。
- [counterfactual_training/](counterfactual_training/README.md)：B AE 预训练、单阶段 Flow 联合训练与软图评价。
- [boundary_transfer/](boundary_transfer/README.md)：独立软原型工具，当前训练不依赖它。
- [diagnostics/](diagnostics/README.md)：既有模体/背景编辑诊断，作为本地资料保持 Git 忽略。

算法从 src/data、models、counterfactuals 导入，共享运行控制从 src/experiments 导入；入口不再互相作为共享库使用。

模型前向、训练、梯度检查和 smoke 必须关联新获批实验方案，统一使用 `/home/ll_yqs2/data/anaconda3/envs/myexplainer/bin/python`。历史批准和 smoke 不授权新源码运行。
