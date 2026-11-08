# 实验运行控制

`runtime.py` 由各实验入口直接导入，不依赖其他 `src/scripts/` 入口。这里只维护真实共享的审批、文件指纹、运行目录、种子、超时和目标查询记录；模型与训练算法留在各自模块。

## 使用约定

在任何实验模型加载前进入 `ExperimentRun(config, config_path, phase, stage=None)`。配置必须同时具备已批准状态、确认日期、方案路径及其 SHA-256；传入的 `phase` 和 `stage` 分别需要位于 `approved_phases` 和 `approved_stages`。`validate_approval` 仅校验审批，不能把代码准备授权或历史运行记录当作新实验授权。

资源取自 `config[phase]` 的 `seconds`（正数）和 `queries`（非负整数）；`config.stages[stage][phase]` 如存在，则覆盖该操作相应字段。这样可分别固定训练零查询与评价预算。随机种子优先取 `config.stages[stage].seed`，未指定 stage 时取顶层 `seed`；`threads` 必须显式配置。空值会报错，没有无限资源或隐式默认种子。

训练操作必须显式配置本机 CUDA `device`（当前 `cuda:1`，RTX A6000），GPU 不可用、编号无效或 CUDA 失败时停止，不回退 CPU。训练入口在 PyG 导入前设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，运行控制核对配置的 `gpu_name`。上下文提供 `run.device`；AE/Flow 入口据此迁移模型和训练/验证张量，训练函数同时拒绝 CPU 训练输入。环境记录 GPU 编号、型号、总显存、CUDA 与 CUBLAS 设置，CPU 线程只用于辅助工作。

上下文提供 `directory`、`paths`、`signature`、`query_limit`、`check_time()` 和 `log(row)`。`paths` 是 `verified_inputs(config, stage=None)` 验证后的全部输入文件路径；通用输入从 config.inputs 读取；当前操作专用输入从 config.stages[stage].inputs 合并。已训练 Flow 的 checkpoint 仅在 evaluate 操作核验，训练不依赖它。AE 关联的 run、summary、samples 等来源也需声明，指纹通过不代替内容关系核对。

运行目录采用 `results_root/<stage>-<phase>-NNN/`（没有 stage 时为 `<phase>-NNN/`），保存原配置、方案、Python 源码快照、依赖与环境、种子、源码/配置指纹、事件和汇总。调用方可以先向 `summary.json` 写入实验指标，退出上下文时合并执行状态。异常和超时写入 `failure.json`；已有结果不会覆盖。`full` 要求相同 stage、代码和配置下由此运行控制生成的已完成 smoke，不沿用旧入口的 smoke 记录。

`query_limit` 已扣除当前 stage、phase 下历史 `forward_started` 数量。需要查询目标时，使用 `TargetLedger(oracle, cache, path, query_limit, check_time)`；`query(x, adjacency, **context)` 返回 `(probabilities, new_forward)`，目标调用开始前即记入预算。bool 图与 float32 软图均按原始 dtype、形状、全部数值构造缓存键，不做阈值化或去除零权边。缓存必须对应已验证的相同目标检查点。

## 自由函数

- `ROOT`、`EXPECTED_PYTHON`：项目根目录与唯一实验解释器。
- `digest(path)`、`verified_inputs(config, stage=None)`：SHA-256 与配置输入核验。
- `write_json(path, data)`、`append_jsonl(path, row)`、`read_jsonl(path)`：JSON/JSONL 文件读写。
- `new_directory(root, prefix)`：原子分配独立目录。
- `validate_approval(config, phase=None, stage=None)`：返回已批准且指纹一致的方案路径。

代码整理本身不授权实验。2026-11-08 用户先单独批准最小 Flow GPU smoke，再另行批准并完成全量训练和评价；两次批准分别关联对应配置与方案，不将最小 smoke 许可推广到正式运行。

Flow 入口内部使用 train / evaluate 作为运行操作标识，仅用于审批、目录和资源记录，不代表 Flow 预训练/微调阶段。

evaluate 同样用配置的本机 CUDA 设备生成软图并运行代理；目标概率接口独立保留原 CPU 实现，不提供训练梯度。其设备与职责须在实验方案中说明。
