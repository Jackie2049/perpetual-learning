# NeMo-RL 的 `use_preserving_dataset`：配置能到达，但在锁定版本下已经不改变任何数据

<!-- contribution-summary:start -->
**【仓库】** [NeMo-RL](https://github.com/NVIDIA-NeMo/RL)（NVIDIA 开源的 RL 后训练框架，SFT/DPO/GRPO 等算法 + vLLM/SGLang 生成后端）

**【贡献点】** `nemo_rl/data/datasets/response_datasets/oai_format_dataset.py:141` 的分支靠 `use_preserving_dataset` 决定是否用 `PreservingDataset` 保留异构 tool schema。键能一路到达形参，但在仓库 `uv.lock` 锁定的 `datasets==5.0.1` 下，**两档输出逐字节相同**——原来要防的"缺失键被补成 `None`"从 `datasets 4.7.0` 起已不再发生。[源码](https://github.com/NVIDIA-NeMo/RL/blob/main/nemo_rl/data/datasets/response_datasets/oai_format_dataset.py) · [实验](../../experiments/nemo-rl-pyflag-triage/README.md)

**【类型】** 问题诊断（B 类主动发掘；扩池第十批；**不对外，用户未授权**）

**【价值评估】** 4 / 10 — 缺陷可机读、可证伪、且关系到训练数据正确性；但当前锁定版本下没有任何用户受影响（用不用这个开关结果一样），可提的只有"文档与实现脱节"和一行依赖约束，采纳与否属维护者的产品决定。

**【进展】** CPU 真跑取证完成：同一份探针在 `datasets 5.0.1`（上游锁定）与 `4.0.0`（引入时的锁）下各 12 条判据全过，自检两条注入均命中；版本矩阵扫了 10 个 `datasets` 版本定位到 4.6→4.7 的分界。**纯 CPU，未装 nemo_rl 本体、未跑训练、未触上游。**

**【大白话讲明白】**

NeMo-RL 用 SFT 训模型时可以直接吃 OpenAI 格式的 JSONL。当数据里带工具（tool calling）定义时，不同样本的工具参数结构可能不一样——样本 A 的工具只有 `query`，样本 B 的工具多一个 `precision`。HuggingFace 的 `datasets` 库早期会把所有样本对齐成同一套字段，缺的补成 `None`，于是样本 A 莫名多出 `"precision": null`，喂给模型就成了坏数据。

为此仓库加了一个开关 `use_preserving_dataset`：打开它就不走 HF 的对齐逻辑，改用自己写的 `PreservingDataset` 原样保留结构。官方文档专门写了一节，告诉用户"异构工具 schema 必须打开它"。

问题在于：`datasets` 库自己升级了。**4.7.0 起**，遇到这种结构不一致的嵌套字段，它不再拆开逐键对齐，而是整体当成 JSON 块存，**不补 `None` 了**。而这个仓库在 2026-10-21 的提交里把锁升到了 5.0.1。于是现在不管开不开这个开关，读出来的数据一模一样——开关变成了空转，文档里那段"不打开就会补 None"的警告也已经不成立。

**【What–Why–How】**

- **What：** 三个并列问题——① 键能不能从 YAML 到达形参（能）；② 到了之后有没有效果（**在锁定版本下没有**）；③ 为什么（依赖库的推断策略变了）。三者混在一起就会被判成"配置没接线"这类错误结论。
- **Why：** `datasets<=4.6.0` 在合并异构嵌套 dict 时会取键的并集并补 `None`；`>=4.7.0` 对同层结构不一致的 dict 退回 `Json(decode=True)` 类型，原样保留、不补键。NeMo-RL 的代码没改一行，失效来自依赖升级。仓库锁版本：2025-09 到 2025-12 是 4.0.0、2026-01 到 2026-10 是 4.4.1（这两个区间里开关是有效的），2026-10-21 起是 5.0.1（开关空转）。
- **How：** 从上游快照用 AST 抽出 `OpenAIFormatDataset`／`PreservingDataset`／`get_formatted_message_log`／`sft_processor` 的真实源码原样执行，`datasets` 与 `transformers` 用真库；两档各跑一遍，比较输出的 `tools` 与训练样本的 `token_ids`。判据挂在**依赖版本**上：同一份脚本在 4.0.0 与 5.0.1 下都全绿，但"两档是否相同"的真值相反。**边界：** 未装 nemo_rl 本体（import 面用替身，清单在读数里）；未跑真实训练；样例是两条构造出来的异构 tool 定义，不是用户数据。
<!-- contribution-summary:end -->

## 一、怎么来的

pyflag 第十批扫 `NVIDIA-NeMo/RL@d82b79c`（2026-11-05，1096 个 `.py`，53 秒）：**34 个开关 / 1 条 `silently_inert`**（`use_preserving_dataset`，`mode=never`）。

34 个开关的密度与 AReaL（42）、DeepSpeed（44）同档，再次印证"配置面在 `TypedDict`／dataclass 里的仓库，pyflag 只能看到函数层面那一小部分"。这批只有一条候选，但**它有官方文档专节、有专门测试、有用户提过的 issue**（#2116 当年就是用它时崩在数据集合并上，由 #2515 修掉）——顺着它走比再扫一个仓库的信息量更高。

## 二、三个问题要分开问

| 问题 | 读数 | 判据 |
|---|---|---|
| 键能到达形参吗 | 能 | `load_response_dataset` 用 `dataset_class(**data_config)`；`DATASET_REGISTRY["openai_format"]` 直接绑类；`ResponseDatasetConfig` 是 `TypedDict`，运行时不做校验 |
| 到了以后有效果吗 | **锁定版本下没有** | 两档 `tools` 逐字节相同、`sft_processor` 产出的 `token_ids` 逐字节相同 |
| 那什么时候有过效果 | `datasets<=4.6.0` | 关闭档给样本 1 补出 `strict: null` 与 `expr: null, precision: null`，开启档保留原结构 |

第三条同时解释了为什么文档那样写：文档描述的是 2025-09 引入时的真实行为。

## 三、版本矩阵

同一份异构 tool 数据，真跑 `OpenAIFormatDataset` 的两档后比较第 1 个样本：

| `datasets` | 两档输出相同？ | 关闭档样本 1 的 `strict` | `parameters.properties` 被补 `None` |
|---|---|---|---|
| 3.6.0 | 否 | `None` | 是 |
| 4.0.0 | 否 | `None` | 是 |
| 4.2.0 | 否 | `None` | 是 |
| 4.4.0 | 否 | `None` | 是 |
| 4.4.1 | 否 | `None` | 是 |
| 4.5.0 | 否 | `None` | 是 |
| 4.6.0 | 否 | `None` | 是 |
| **4.7.0** | **是** | 不存在 | 否 |
| 4.8.4 | 是 | 不存在 | 否 |
| 5.0.1（上游锁定） | 是 | 不存在 | 否 |

分界在 **4.6.0 → 4.7.0**。机制是 `datasets` 的类型推断：结构一致时逐键建 `Sequence`/`Value` 特征，缺的键补 `None`；结构不一致时整块退回 `Json(decode=True)`，原样往返。

## 四、上游锁版本时间线

| 提交 | 日期 | 事件 | `datasets` |
|---|---|---|---|
| `6fe56b01d7` | 2025-09-26 | #1181 引入 `use_preserving_dataset` | 4.0.0 |
| `7e5df0cc8c` | 2025-12-13 | 数据集模块重构 | 4.0.0 |
| `967d2c8565` | 2026-01-22 | train/val 拆分 | 4.4.1 |
| `174fa2011b` | 2026-02-04 | preference 数据集拆分 | 4.4.1 |
| `4fdd49c225` | 2026-10-21 | #3286 xtoken 对齐 | **5.0.1** |
| `d82b79c58f17` | 2026-11-05 | 本次取样 HEAD | 5.0.1 |

⇒ 按仓库锁装环境的使用者：**2025-09 至 2026-10 之间开关有效；2026-10-21 之后开关空转**。代码 diff 里没有任何相关改动，所以 review、单元测试、扫描器都不会报。

## 五、为什么不能说成"开关坏了"

- 写成"配置项无效"会误导维护者去找接线问题——接线是好的（四条可达性判据）。
- 写成"回归"也不对——不是代码回归，是依赖升级让既有代码失去意义。
- 准确的表述是：**该开关当前的唯一效果是把数据加载换成自定义实现；它原本要防的事已经由依赖库自己解决，其文档描述与该库当前行为不符。**

## 六、混合数据下还剩下的差异（无害）

部分样本带 tools、部分不带的常见情形：锁定版本下唯一差异是关闭档给出 `tools: null`、开启档没有这个键。下游 `sft_processor` 用 `datum_dict.get("tools", None)`，`get_formatted_message_log` 再判 `if tools is not None` ⇒ 两者等价，不影响训练。

## 七、可以做什么（**均未执行，待用户授权**）

| 动作 | 内容 | 代价 |
|---|---|---|
| (a) 文档澄清 | `docs/guides/sft.md:527-544` 那段"不打开会补 None"在锁定的 datasets 5.x 下已不成立，注明按版本判断 | 最小、无争议 |
| (b) 依赖约束 | 若仍希望开关有效，需把 `datasets` 上限钉回 `<4.7`（会牵连整仓） | 大，属维护者决定 |
| (c) 删开关 | 承认依赖已解决该问题，移除 `use_preserving_dataset` 与 `PreservingDataset` 的默认路径 | 破坏兼容，需 deprecation 周期 |

## 八、核对产物

| 产物 | 说明 |
|---|---|
| [实验 README](../../experiments/nemo-rl-pyflag-triage/README.md) | 完整叙述与复跑命令 |
| [探针](../../experiments/nemo-rl-pyflag-triage/probe_preserving_dataset.py) | 12 条判据 + 2 条注入自检 |
| [锁定版本读数](../../experiments/nemo-rl-pyflag-triage/raw/probe-datasets-5.0.1.json) | `datasets 5.0.1` / `transformers 5.12.1`，12/0 |
| [历史锁定读数](../../experiments/nemo-rl-pyflag-triage/raw/probe-datasets-4.0.0.json) | `datasets 4.0.0` / `transformers 5.5.4`，12/0 |
| [版本矩阵](../../experiments/nemo-rl-pyflag-triage/raw/version-matrix.json) | 10 个 `datasets` 版本 + 上游锁时间线 |
| [扫描归档](../../experiments/nemo-rl-pyflag-triage/raw/scan-nemo-rl.json) | pyflag 第十批原始读数 |
| [源文件快照](../../experiments/nemo-rl-pyflag-triage/raw/) | 8 个逐字节副本 + sft.md 摘录，无克隆也能复跑 |
