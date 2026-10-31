# 实验代码

当前仅有目录骨架，数据、模型和具体实现设置尚待确认。源码统一放入 `src/`。修改前说明意图；在已有本地工作区中遵守根目录 `AGENTS.md`，实验审批与解释器要求也见[项目 README](../README.md)。

- [data/](data/README.md)：数据接口、预处理和划分。
- [models/](models/README.md)：分类模型、检查点加载和预测接口。
- [attacks/](attacks/README.md)：定向攻击方法及基线。
- [counterfactuals/](counterfactuals/README.md)：反事实生成与可行性约束。
- [configs/](configs/README.md)：明确、可记录的实验配置。
- [scripts/](scripts/README.md)：训练、攻击、生成、评估及结果整理入口。

所有实验使用 `/home/ll_yqs2/data/anaconda3/envs/myexplainer/bin/python`。每次实验运行均须关联用户已确认的方案；长任务先运行批准范围内的小规模 smoke test。

依赖说明见根目录 [requirements.txt](../requirements.txt)。数据、检查点及实验输出存放在源码目录外，通过配置指定；Python 缓存等生成文件不纳入提交。
