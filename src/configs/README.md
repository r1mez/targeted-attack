# 当前配置

- [ba3motif_ae.json](ba3motif_ae.json)：仅 B architecture 的 AE 训练；保留数据与验证清单来源。
- [ba3motif_flow.json](ba3motif_flow.json)：随机初始化 Flow 的单阶段三项损失训练及软图评价；训练输入只引用冻结 AE，不依赖历史 CFM 权重。
- [ba3motif_flow_full.json](ba3motif_flow_full.json)：2026-11-08 已批准并完成的全量联合训练与测试评价配置，100 轮、batch=32、700/150/150；包含训练和评价的 smoke/full，评价输入已指向本轮第 100 轮 Flow。原最小 smoke 配置保持原样。
- [ba3motif_soft_boundary.json](ba3motif_soft_boundary.json)：独立软原型工具；旧批准与结果仅作为历史来源。

2026-11-07 用户确认了方法分工和仓库整理；2026-11-08 明确 Flow 为单阶段随机初始化训练，先完成最小 GPU smoke，再单独批准并完成全量训练及评价。最小 ba3motif_flow.json 的范围仍仅 train/smoke；全量配置为 ba3motif_flow_full.json。AE 重训配置仍待确认。入口拒绝缺失或未批准设置；旧硬搜索的预算、倍率网格和轮数不自动沿用。

批准时需记录方案路径/指纹、阶段、种子、超参数及资源上限。旧六份历史配置已压缩快照，不保留在活跃目录；历史 results 不改写。

Flow 配置只含 train 与 evaluate 两个运行操作。train 从随机参数开始联合训练三项损失；evaluate.inputs.flow_checkpoint 用于显式指定训练完成后的输出，其空值不会阻塞训练输入检查。CLI 不再有 --stage 参数。

AE 与 Flow 训练配置显式选择本机 CUDA GPU（当前 device=cuda:1，预期型号 RTX A6000），并用 PCI_BUS_ID 固定 CUDA 排序；训练入口在加载 PyG 前设置排序，运行控制再次核对型号。threads 是 CPU 辅助线程数，不表示使用 CPU 训练；训练不得静默回退 CPU。

全量方案使用 data_selection.validation_policy=full_split，明确选取全部 150 张验证图；原配置缺省 ae_selection，沿用既有 90 个 AE selection_ids。train/evaluate 各自的 smoke/full 对象可以覆盖对应操作的 seconds/queries，避免训练继承评价查询预算。评价使用训练结束后机械填入的 flow_checkpoint 路径和指纹，不能用旧模型替代。
