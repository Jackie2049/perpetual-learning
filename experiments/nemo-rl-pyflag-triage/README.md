# NeMo-RL pyflag 第十批：`use_preserving_dataset` 在锁定版本下**已经不改变任何数据**

取样：`NVIDIA-NeMo/RL@d82b79c58f17`（2026-11-05T04:30:55Z，HEAD 当时最新）。
`github.com:443` 本轮不可达，源码走 `api.github.com` 的 tarball 端点（20.7 MB）取回，`raw/snapshot-*` 是逐字节副本，无克隆也能复跑。

## 一、扫描读数

`tools/pyflag/scan_unwired_flags.py`，**1096 个 `.py`、53 秒 → 34 个开关 / 1 条 `silently_inert`**。

| 仓库 | 快照 | 文件 | 开关 | 失效 | 人工结论 |
|---|---|---|---|---|---|
| NeMo-RL | `d82b79c` | 1096 | 34 | 1 | `use_preserving_dataset`（`mode=never`）——**开关可达、但当前锁定版本下两档输出逐字节相同** |

开关密度 34 与 AReaL（42）、DeepSpeed（44）同档，远低于推理侧（SGLang 386、TRT-LLM 352）。NeMo-RL 的配置面同样主要在 `TypedDict` + pydantic `MasterConfig`，**印证了第九批那条判断：pyflag 适合"形参即配置"的仓库**。读数：[scan-nemo-rl.json](raw/scan-nemo-rl.json)。

## 二、为什么值得顺着这一条往下走

`use_preserving_dataset` 与其他"配了没反应"的开关不同：它有**官方文档专节**、有**专门的测试文件**、有**用户提过的 issue**（#2116），而且它防的是一个明确的坏结果——异构 tool schema 被补齐成 `None`。扫描器只报"零生产者"，回答不了"用户配了它到底拿不拿得到想要的东西"。

## 三、结论：可配，但在当前锁定版本下**无害也无用**

三种结局里，这一条落在中间那个最容易被漏掉的：

| 可能 | 是否成立 | 证据 |
|---|---|---|
| **不可配**（键到不了形参） | 否 | `dataset_class(**data_config)` 直接解包；注册表把 `openai_format` 绑到类本身；`TypedDict` 不做运行时校验 ⇒ 键一路到达形参 |
| **可配但没有效果** | **是** | 锁定 `datasets==5.0.1` 下，两档输出的 `tools` 逐字节相同，训练样本 `token_ids` 也相同 |
| 可配且有效 | 曾在旧版本成立 | `datasets<=4.6.0` 下关闭档给样本 1 补出 `strict: null`、给 `parameters.properties` 补出 `expr: null, precision: null`；开启档保留原结构 |

**根因不在 NeMo-RL，在 `datasets` 库的推断策略变了。** 4.7.0 起，同层结构不一致的嵌套 dict 不再被拆成逐键字段，而是整体退回 `Json` 类型，**不补 `None`** ⇒ `PreservingDataset` 与 HF `Dataset` 拿到的数据没有区别。

## 四、版本矩阵（同一份数据，真跑 `OpenAIFormatDataset`）

| `datasets` | 两档输出相同？ | 关闭档样本 1 的 `strict` | `parameters.properties` 被补 `None` |
|---|---|---|---|
| 3.6.0 / 4.0.0 / 4.2.0 / 4.4.0 / 4.4.1 / 4.5.0 / 4.6.0 | 否 | `None` | 是（`expr`、`precision`） |
| 4.7.0 / 4.8.4 / 5.0.1 | **是** | 不存在 | 否 |

分界在 **4.6.0 → 4.7.0**。读数：[version-matrix.json](raw/version-matrix.json)，脚本 [version-matrix.py](raw/version-matrix.py)。

## 五、上游锁版本的时间线（决定"谁受影响"）

| 提交 | 日期 | 事件 | `datasets` 锁 |
|---|---|---|---|
| `6fe56b01d7` | 2025-09-26 | #1181 引入 `use_preserving_dataset` | 4.0.0 |
| `967d2c8565` | 2026-01-22 | train/val 拆分 | 4.4.1 |
| `4fdd49c225` | 2026-10-21 | #3286 xtoken 对齐 | **5.0.1** |
| `d82b79c58f17` | 2026-11-05 | 本次取样 HEAD | 5.0.1 |

⇒ **2025-09 到 2026-10 之间按仓库锁装的环境里，这个开关是真的有用的；从 2026-10-21 起它变成空转。** 这不是代码回归（代码一行没动），是**依赖升级让一个特性静默失效**——文档里那段"会错误地补 `None`"的描述在锁定版本下已不成立，而新装环境拿不到任何提示。

## 六、探针与自检

[probe_preserving_dataset.py](probe_preserving_dataset.py)：从 `raw/snapshot-*.py` 用 AST 抽出 `PreservingDataset`、`OpenAIFormatDataset`、`get_formatted_message_log`、`sft_processor`、`TaskDataSpec` 的**真实源码**原样 `exec`；`datasets.load_dataset`、`transformers` 的 tokenizer 与 chat template 都是真的。

替身清单（读数里打印）：`RawDataset` 基类、`torch.Tensor`（仅注解）、`nemo_rl.data.multimodal_utils` 四个名字（走纯文本路径）、`BatchedDataDict`、tokenizer 的 `return_tensors="pt"` 包装（本机无 torch，改走 numpy 形状）。

| 环境 | 结果 | 读数 |
|---|---|---|
| `datasets 5.0.1` + `transformers 5.12.1`（上游锁定） | **12 checks / 0 failed**，`--selftest` 两条注入都让目标判据变红 | [probe-datasets-5.0.1.json](raw/probe-datasets-5.0.1.json) |
| `datasets 4.0.0` + `transformers 5.5.4`（#1181 时的锁） | **12 checks / 0 failed**（同一份脚本走历史分支） | [probe-datasets-4.0.0.json](raw/probe-datasets-4.0.0.json) |

**同一份探针在两个版本上都全绿，这本身就是读数**：判据是"两档是否相同"，而它随版本取不同的真值——不是"结果对了就算过"。

### 本批两处判据错（都由自检抓出）

1. **把旧版行为写成了普遍事实。** 第一版判据是"关闭档两条样本键集相同"（假设 HF 一定会补齐），在锁定版本上**自检注入后仍绿**——因为新版根本不补。查下去才发现 4.7.0 是分界点。⇒ **"HF Dataset 会给缺失键补 None"这句话必须带版本号**；拿它当判据，会在依赖升级后悄悄变成假绿。
2. **把"同名/同位置"当成判据。** 第一版拿 `tool_props(off)[0] == tool_props(off)[1]` 判"被补齐"，而新版两样本键集本来就不同（各自保留自己的键）。改成比较**两档输出是否逐字节相同**才是这个开关的定义。

## 七、还测了一条：混合数据（部分样本带 tools）在锁定版本下的差异

| `datasets` | 带 tools 的行 | 不带 tools 的行 |
|---|---|---|
| 4.0.0 | 两档不同（关闭档被补 `None`） | 不同（关闭档 `tools: null`，开启档无该键） |
| 5.0.1 | **两档相同** | 不同（同样 `None` vs 缺键） |

锁定版本下唯一剩下的差异是"`tools: null` vs 没有 `tools` 键"，而下游 `sft_processor` 走 `datum_dict.get("tools", None)`、`get_formatted_message_log` 再判 `if tools is not None` ⇒ **`None` 与缺键等价，无害**。读数：[mixed-rows-datasets-5.0.1.json](raw/mixed-rows-datasets-5.0.1.json)。

## 八、价值与判断

**4/10，不对外。** 理由：

- 缺陷确凿、可机读、可证伪，而且**影响的是数据正确性**（对旧版本而言），不是只影响性能；
- 但**当前锁定版本下没有用户受影响**——两档一模一样，用不用这个开关结果相同；
- 可提的只有"文档与实现脱节"（`docs/guides/sft.md:527-544` 描述的行为在锁定版本下已不成立）和一行的 fix（把 `datasets` 上限钉住，或删掉这个开关）。这类改动触及依赖策略，采纳与否是维护者的产品决定，与 TM-020 同构。
- **对外待用户逐项授权**，本轮未触上游。

## 九、方法产出

1. **"开关有没有效果"要和"开关能不能配上"分开问**，而且**效果判据必须有版本坐标**。本批第一版判据栽在把旧版行为当恒定真值上；🔗 TM-019 的教训（"工厂被当成决策者"）是静态侧的同族错误。
2. **依赖升级可以让一个特性静默失效，而代码 diff 里什么都没有。** 扫描器、单测、文档三样在版本升级后都可能仍然"看起来正常"：扫描器只看有没有生产者、单测只测新旧两档的类类型、文档写着旧行为。⇒ **跨版本特性，判据要挂在依赖版本上**。
3. **顺着"有文档专节"的开关走比顺着高密度候选走更划算**：34 个开关里只有 1 条候选，但这条有文档、有测试、有用户 issue，追下去的信息量远超那 33 条。

## 复跑

```bash
# 上游源码（github.com 不通时走 tarball）
curl -sL -o /tmp/nemo.tar.gz https://api.github.com/repos/NVIDIA-NeMo/RL/tarball/d82b79c58f175917182ed7f53283ae590af67924
tar xzf /tmp/nemo.tar.gz -C /tmp

# 锁定环境（uv.lock = datasets 5.0.1 / transformers 5.12.1）
uv venv --python 3.13 /tmp/nemo-ds-venv
uv pip install --python /tmp/nemo-ds-venv/bin/python "datasets==5.0.1" "transformers==5.12.1" jinja2 tokenizers numpy
/tmp/nemo-ds-venv/bin/python probe_preserving_dataset.py --selftest --json raw/probe-datasets-5.0.1.json

# 历史环境（#1181 时的锁）
uv venv --python 3.12 /tmp/ds400
uv pip install --python /tmp/ds400/bin/python "datasets==4.0.0" "transformers==5.5.4" jinja2 tokenizers numpy
/tmp/ds400/bin/python probe_preserving_dataset.py --json raw/probe-datasets-4.0.0.json
```

纯 CPU、无 GPU/NPU、未跑训练、未装 nemo_rl 本体、**未触上游**。
