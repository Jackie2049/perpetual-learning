# SkyRL 的 `use_liger_kernel`：开关从仓库第一个提交起就没接过线，**没有用户受影响**

<!-- contribution-summary:start -->
**【仓库】** [SkyRL](https://github.com/NovaSky-AI/SkyRL)（NovaSky 团队的 RL 训练框架，FSDP / Megatron 两套后端）

**【贡献点】** 在 `f854d6d9` 上重扫 41 个开关 / 7 条候选，逐条查证后**真失效 1 条**（`use_liger_kernel`），并把"从没接过线"从 18 个 ref 扩到**整条历史 1458 个提交**。[实验与读数](../../../experiments/skyrl-pyflag-head-audit/README.md)

**【类型】** 问题诊断 / 方法沉淀

**【价值评估】** 3 / 10 — 判定可机读、可证伪（20 条判据 + 17 条注入），但**没有用户受影响**：配置面拒绝这个键、依赖里没有 `liger-kernel`、两处生产构造点都不传。留下的是一条方法约束与一套可复跑的判据。

**【进展】** 2026-11-05 完成；纯静态 + 本机 import 读数，未跑训练、未触上游。对外（提 issue 或只改 docstring）**待用户逐项授权**。

**【大白话讲明白】**

SkyRL 里有个开关叫 `use_liger_kernel`，名字的意思是"模型改用 Liger Kernel 那套融合算子来加载，省显存、跑得快"。代码里它确实写了：函数签名上有个 `use_liger_kernel=False`，下面有 `if use_liger_kernel:` 的分支，分支里真的去 import Liger 的模型类。

问题在于：**没有任何地方给它传过值**。三处都堵死了——

- 配置文件（SkyRL 用的是强类型 dataclass）里**根本没有这个字段**，用户写 `use_liger_kernel: true` 会在拼配置那一步就被拒；
- 真正构造模型包装器的两处代码（`fsdp_worker.py` 的两个 `HFModelWrapper(...)`）也没传它；
- 依赖清单里没有 `liger-kernel`，所以就算有人想办法传了 `True`，只会得到一个 `ModuleNotFoundError`。

所以结论是：**这个开关是死的，但没人会被它坑**——因为你根本打不开它。

那它还值得说吗？有一点点。函数注释写着 "Defaults to False"，读源码的人会以为这是个可以传的参数；更准确的说法是"这一段是留给未来的、目前没接线"。我们上一轮（11-04）只能追到 2026-02 的 18 次提交，说不清"是不是某次重构把它漏掉了"。这一轮把整个仓库 1458 个提交都扫了：**从 2025-06-24 第一个提交开始，这个形参就在，一次都没被传过**。文件中间改过三次名，每一版都带着它。

**【What–Why–How】**

- **What：** SkyRL `f854d6d9`（2026-11-04）上 pyflag 报的 7 条「有消费者、无生产者」开关逐条查证；重点是唯一真失效的 `use_liger_kernel` 的历史归属与用户可达性。
- **Why：** 这类开关的危害是"看起来能配、配了没用"且不报错。上一批写明的边界是"只能证明 2026-02-16 起没接过线"，这一轮补的就是那句边界。
- **How：** `probe.py` 13 条判据逐条绑行号（11 条注入验证每条判据都真的会被打红）；`history-liger.py` 用每条提交一次 `git grep` 扫全历史（1458 rev / 27 秒），并用 AST 区分"形参声明"与"关键字实参"；`reach-liger.py` 从配置面、构造点、依赖面、本机 import 四侧否掉可达性。**边界**：只扫了 HEAD 一个点；历史只对 `use_liger_kernel` 做了全量；未真跑 Hydra 配置校验。
<!-- contribution-summary:end -->

## 一、读数

| 口径 | 文件 | 数 | 结果 |
|---|---|---|---|
| pyflag 扫描 | 972 个 `.py` | 41 个开关 / 7 条候选 | 真失效 1、工具盲区 2、非用户面 4 |
| `probe.py` | 13 条判据（7 组） | — | **13/13 通过**；11 条注入全部按预期变红 |
| `history-liger.py` | 1458 个提交 | — | H1–H4 全过；该文件 38 条提交全部带形参；历史关键字实参 **0** |
| `reach-liger.py` | 4 面 6 条判据 | — | 全过：配置无字段、两处构造不传、依赖里没有、本机 import 失败（正向对照 `json` 可 import） |

## 二、七条候选

| 开关 | 判定 | 一句话 |
|---|---|---|
| `use_liger_kernel` | **DEAD** | 形参 + 分支都在，全仓零调用；配置无字段、依赖无包 |
| `use_remove_padding` | 非用户面 | 分支是 `use_remove_padding or ulysses_sp_size > 1`，唯一调用点又在 `sp > 1` 守卫内 ⇒ 传不传没有观测差异 |
| `enable_sample_support_replay` | 工具盲区 | 读配置字段后存局部变量、再按关键字传，pyflag 只认同名传参 |
| `enable_pd` | 工具盲区 | `ie_cfg.enable_pd` → `enable_pd=True` 显式传 |
| `use_combine` | 误报 | 子类用位置参数 `super().__init__(config, use_combine=False)` |
| `use_signals` | 设计如此 | docstring 标 `(Deprecated)`，默认走多进程 |
| `disable_print` | 公开 API | `pprint(...)` 是公开 helper 的返回字符串开关 |

## 三、为什么历史归属要单独做一次

上一批的边界写得很清楚：浅克隆 + contents API 只覆盖了 `model_wrapper.py` 的 18 次提交，所以只能说"该文件自 2026-02-16 起没接过线"。
而这一轮的成本是 **27 秒**（1458 次 `git grep`），比上一轮的逐版本取文件还便宜 ⇒ **"查不动历史"这个理由不成立**。

新读数顺带修正了一个说法上的粗糙处：该文件在历史上换过三次名，
`skyrl_train/models.py`（`8805e756`，2025-10-01 前后）→ `skyrl_train/model_wrapper.py`（`0c18fb5b`，2026-01-05）→ `skyrl/backends/skyrl_train/workers/model_wrapper.py`（`b408bafa`，2026-10-30）。
只按当前路径 `git log` 会拿到 18 条，按 rename 链才是 38 条——**路径换了，"这个形参有没有被传过"的答案不能在换了名的文件上断档**。

## 四、方法产出

1. **判据必须能被打红，而且要指定打红哪一条。** 11 条注入里有 2 条起初"变红"了但没红在该红的地方（`L5` 把声明文件自己算成生产者、`S1`/`P1` 数行数不数出现次数）；自检不写"期望哪条红"，这两处会被 PASS 掩盖。
2. **负向对照的产物可能污染负向对照本身。** 自检临时树建在实验目录下，`L5` 的全仓扫描把它一起扫了 ⇒ 四条注入多红一条。**判据扫全仓时，先想清楚"全仓"包不包括测试临时物。**
3. **"从没接过线"要有全历史读数才能下，而不是"我看了几个 ref 都是 0"。** 上一批的措辞是保守的（只写 18 个 ref），这一批才敢写整条历史；差别就是 27 秒的扫描。

## 五、对外判断

**不建议提 issue**：没有用户受影响，改法（删形参 / 接线 / 补依赖）都涉及"要不要支持 Liger"的产品决定，维护者的动作会是"要么删要么接"，不是修 bug。
如果要动作，成本最低且无争议的只有一句 **docstring 澄清**（`Defaults to False` → "当前尚未接线"），这同样**待用户逐项授权**。

## 六、复现

见 [实验 README](../../../experiments/skyrl-pyflag-head-audit/README.md#复现)。判据脚本与 11 个源文件快照随仓库提供，无克隆也能跑 `probe.py`。
