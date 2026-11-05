# NeMo-RL（NVIDIA）

<!-- contribution-summary:start -->
**【仓库】** [NeMo-RL](https://github.com/NVIDIA-NeMo/RL)（NVIDIA 开源的 RL 后训练框架：SFT / DPO / GRPO 等算法，vLLM、SGLang、TRT-LLM 生成后端，FSDP / Megatron 训练后端）

**【贡献点】** 扩池第十批的评估与本仓库的取证入口汇总。[扫描](../../tools/pyflag/readme.md) · [证据](evidence/preserving-dataset-version-drift.md)

**【类型】** 项目跟踪

**【价值评估】** 6 / 10 — 活跃度与工程规范都高（单日多次合入、带 pydantic 配置校验、有专门的 unsupported-key 告警机制），与既有 RL 训练侧积累（verl／AReaL／SkyRL／rLLM）可迁移；但配置面主要在 `TypedDict` 与 pydantic 模型里，pyflag 这类"形参即配置"的扫描器覆盖有限。

**【进展】** 2026-11-05 完成 pyflag 第十批扫描（34 个开关 / 1 条候选），逐条追到 `use_preserving_dataset` 的**依赖版本漂移**：锁定版本下开关空转。未提 issue、未开 PR，未触上游。

**【大白话讲明白】**

NeMo-RL 是英伟达的强化学习后训练框架，和 verl 是同一类东西。我们这轮把它纳入扫描池，想看看"配置了但没生效"的问题在它身上是什么形状。

结果和训练侧前几家一致：能扫出来的开关不多，而且这一条的真相比"没接线"复杂——开关的接线是好的，但它要防的那个问题已经被上游依赖库自己解决了，于是它变成了空转。

**【What–Why–How】**

- **What：** 数据加载开关 `use_preserving_dataset`（是否用 `PreservingDataset` 保留异构 tool schema）。
- **Why：** `datasets` 库 4.7.0 起不再对结构不一致的嵌套 dict 补 `None`，而 NeMo-RL 在 2026-10-21 把锁升到 5.0.1 ⇒ 开关的两档输出逐字节相同。
- **How：** 从上游快照抽真实类体执行，两个锁定版本各跑 12 条判据；扫 10 个 `datasets` 版本定位分界。详见[证据页](evidence/preserving-dataset-version-drift.md)。
<!-- contribution-summary:end -->

## 为什么纳进来

RL 训练侧的扫描池里，verl／AReaL／SkyRL／rLLM／OpenRLHF 都扫过了。NeMo-RL 的几个特点让它值得单独看：

- **工程规范**：配置走 pydantic `MasterConfig` + `TypedDict`，还有一套 `warn_on_unsupported_dataset_config_keys`（数据集配置键写错会告警）——正好可以检验 pyflag 在这类仓库上的盲区有多大。
- **活跃度**：2026-11-05 单日就有 `feat(vllm): support per-token-head FP8 KV cache (#5028)` 这类合入。
- **与本专项相邻**：它有独立的 rollout／generation 后端抽象，与 `notebook/rollout-performance-research.md` 的跨框架比较有关。

## 结论索引

| 事项 | 结论 | 价值 | 状态 |
|---|---|---|---|
| pyflag 第十批扫描 | 34 个开关 / 1 条候选 | 方法 | 完成 |
| `use_preserving_dataset` | 可配，但锁定版本下两档输出相同（依赖 `datasets` 4.7.0 起不补 `None`） | 4/10 | 取证完成，**不对外** |

## 边界

- 未安装 nemo_rl 本体，未跑训练，未上 GPU/NPU。
- import 面用替身（清单在读数里），只执行从快照 AST 抽出的真实类/函数体。
- 对外动作（文档澄清 / 依赖约束 / 删开关）**待用户逐项授权**，本轮未触上游。
