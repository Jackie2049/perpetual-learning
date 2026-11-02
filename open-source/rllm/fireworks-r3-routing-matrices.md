# rLLM × Fireworks：默认开启的路由回放（R3）在训练帧里没有落脚点

<!-- contribution-summary:start -->
**【仓库】** [rLLM](https://github.com/rllm-org/rllm)（Agent 强化学习框架，Fireworks 是它的托管训练后端）

**【贡献点】** Fireworks 后端自 PR [#934](https://github.com/rllm-org/rllm/pull/934) 起默认 `router_replay: R3`，但逐 token 路由矩阵在 Tinker/Fireworks 训练路径上装不进训练帧 `tinker.ModelInput`——rLLM 用 `model_copy(update=...)` 塞的属性不在该类声明字段中，赋值被 pydantic 拒绝。默认配置下开关在记录数据，训练侧结构上拿不到。[取证](fireworks-r3-routing-matrices.md) · 复现 `experiments/rllm-fireworks-r3-replay/`

**【类型】** 问题诊断（B 类：主动发掘，附最小复现）

**【价值评估】** 6 / 10 — 触发条件是"Fireworks 后端 + 默认配置"，即该后端全部用户；但服务端实际如何处理未声明的属性未验证（需真实 Fireworks 账号），因此不主张训练结果已出错，只主张"默认开启的回放开关拿不到载荷"。

**【进展】** 2026-11-03 对 main `4e828c8` 完成本地复现与承载能力实测（tinker 0.22.2／0.23.0／0.36.0 三个版本现象一致），并排除了两条一度怀疑的旁支。**对外动作全部未做，待用户逐项授权。**

**【大白话讲明白】**

MoE 模型每个 token 被哪些专家处理，是采样时随机决定的。训练时希望"把推理那一刻的选路原样再走一遍"，避免同一句话在训练和推理走了不同专家——这叫 router replay。

rLLM 的 Fireworks 后端从 10 月 9 日起**默认开这个功能**。rLLM 确实采到了每个 token 的选路，也确实试着把它塞进要发给训练服务的数据包——但那个数据包对象（tinker 的 `ModelInput`）按定义是封闭的，根本没有放这个数据的格子，多塞一个属性会被类型系统当场拒绝。上游 Fireworks 官方 cookbook 的做法是走一个专门接受这个参数、并顺手做位置对齐的构造函数，rLLM 走的是"塞属性"这条不存在的路。

现在能确定的只有"本地塞不进去"；服务端是会静默忽略还是校验报错，需要真实 Fireworks 环境才能回答。

**【What–Why–How】**

- **What：** `rllm/trainer/tinker/transform.py:85-87`（塞属性）、`rllm/types.py:128`（Step 字段）、`rllm/engine/rollout/fireworks_engine.py:262-277`（解析）、`rllm/trainer/config/rllm/backend/fireworks.yaml:78`（默认 R3）。范围限 Fireworks/Tinker 后端，不涉及 verl 路径。
- **Why：** `transform.py` 用 `input_tokens_T.model_copy(update={"routing_matrices": rm})` 注入；`tinker.ModelInput` 的声明字段只有 `['chunks']`、`from_ints()` 不接受该参数、模型是 `frozen=True` + `extra='forbid'`，所以该属性既设不上也不进序列化。另有两处同路径不一致：Python 侧默认是 `"disabled"` 与 yaml 的 `R3` 不同源；同一开关在 verl 0.8.0 里语义是"记录到 record_file"，rLLM 同时另开 `enable_rollout_routing_replay`，两条通道并存。
- **How：** 用真实 rLLM 类两层复现（数据路径 + 引擎端到端打桩 sampler），并用独立脚本测 tinker 三版本的承载能力；均 exit 0 = 缺陷在。**边界**：未跑真实 Fireworks 训练、未跑全量测试；"服务端如何处置"与"是否已影响线上训练"均未验证。
<!-- contribution-summary:end -->

2026-11-03 M1 本地取证（TM-010）。目标仓库 [rllm-org/rllm](https://github.com/rllm-org/rllm)，快照 `4e828c8`（2026-10-09 提交，2026-11-03 确认为 main 头）。**全部在本地完成，未对外发布任何内容。**

## 结论

rLLM 的 Fireworks 后端自 PR #934（2026-10-09 合入）起**默认开启 `router_replay: R3`**。R3 在这条训练路径上的载荷（逐 token 路由矩阵）装不进训练帧 `tinker.ModelInput`：rLLM 用 `model_copy(update=...)` 塞的属性不在该类的声明字段里，赋值被 pydantic 拒绝。默认配置下的一个开关在记录数据，训练侧结构上拿不到。

本轮实跑用到的具体版本与命令见 `run-info.json`；快照 `4e828c8` 与 TM-009（同仓另一条候选）是同一棵树，两次取证共用同一份本地 checkout。

## 证据等级（先说清楚）

| 结论 | 等级 |
|---|---|
| `tinker.ModelInput` 没有 `routing_matrices` 字段，`from_ints(...)` 不接受该参数，赋值抛 `ValidationError` | **实测**（carrier_check.py，tinker 0.36.0，PyPI 最新） |
| rLLM `trajectory_to_datums` 塞字符串列表；verl 路径塞 `(len, layers, topk)` 张量——两套载荷类型不同 | **实测 + 源码定位** |
| FireworksEngine 能解析出矩阵，Step 也带上了，但在 datum 处拿不到承载字段 | **实测**（repro.py stage B，真实引擎 + 打桩 sampler） |
| Fireworks 训练服务端**抛不抛错、会不会忽略**该属性 | **未验证**（需要真实 Fireworks 账号；本轮无凭据，也没有提交任何东西） |
| 该缺陷是否已在线上吃掉正确的 MoE 训练 | **未验证，不主张** |

## 大白话

MoE 模型每个 token 走哪几个专家是随机丢出去的。训练时想"把推理那一刻的选路原样再走一遍"，避免同一句话训练和推理走了不同专家——这叫 router replay。

Fireworks 提供一个训练服务，rLLM 可以把训练数据打包成"datum"发过去。为了让服务端按原路复现，得把"每个 token 选了哪些专家"塞进 datum 的输入部分。rLLM 确实拿到了这份数据，也确实往输入对象上塞了——但**那个对象（tinker 的 `ModelInput`）根本没有这个字段**，按类型定义它是封闭的（pydantic 冻结模型），多塞一个属性会被直接拒绝。

也就是说：**默认开着、数据也采了、到最后一步被类型挡住**。而上游 Fireworks 官方 cookbook 的做法是走一个专门接受该参数、并且要做对齐处理的构造函数——rLLM 用的是"塞属性"这条不存在的路。

## 三处不一致（都在同一条默认配置路径上）

1. **默认值不同源。** Fireworks 后端 yaml `router_replay: R3`，但 `AlgorithmConfig.router_replay` 的默认仍是 `"disabled"`；两处没同步。
2. **语义与其消费方不同。** verl 0.8.0（rLLM 钉的版本）里 `router_replay.mode='R3'` 的注释是"记录到 `record_file`"（离线记录），**不是"回放"**。rLLM 把用户配置转成 `actor_rollout_ref.actor.router_replay.mode=R3`（照抄 verl 语义＝记到文件），同时又另开 `rollout.enable_rollout_routing_replay=True`（在线记录）——**同一开关被拆成两条独立通道**，训练侧真正消费写的是 `routed_experts`。
3. **两条路径的载荷类型不同。** verl 路径写 `(tokens, layers, topk)` 张量并由 `transform.py` 放进 `routed_experts` 张量字段；Tinker/Fireworks 路径写字符串列表塞进 `ModelInput`。

## 复现

需要一份 rLLM 源码 checkout（本轮用 API 拉到 `4e828c8` 的完整 `rllm/` 与 `tests/` 到本地临时目录）与 Python 3.12 环境。

```bash
uv venv -p 3.12 /tmp/rllm-r3-venv
uv pip install --python /tmp/rllm-r3-venv/bin/python \
  "tinker>=0.22.2" "tinker-cookbook>=0.4.1" "fireworks-ai[training]==1.2.20" \
  pydantic PyYAML omegaconf hydra-core codetiming numpy openai
cd <rllm-checkout> && PYTHONPATH=. /tmp/rllm-r3-venv/bin/python repro.py          # exit 0 = 缺陷在
                               PYTHONPATH=. /tmp/rllm-r3-venv/bin/python carrier_check.py  # exit 0 = 承载缺口在
```

`repro.py` 两阶段：

- **A（数据路径）**：用真实 `ModelOutput / Step.from_model_output / trajectory_to_datums`。R3 下 datum 的 `model_input.routing_matrices` = `['', '', '', 'm0', 'm1', 'm2']`（元素是 `str`），而该对象的声明字段只有 `['chunks']`。
- **B（引擎端到端）**：真实 `FireworksEngine` + 打桩 sampler（返回 Fireworks 的 `logprobs.content[*].routing_matrix` 形状）。引擎解析出 `['m0','m1','m2']` → Step 带上 → datum 处仍无承载字段。

输出（2026-11-03 实测）：

```
[A] router_replay=disabled -> None
[A] R3 datum len=6 routing_matrices=['', '', '', 'm0', 'm1', 'm2']
[A] datum input type = tinker.ModelInput
[A] element types     = ['str']
[A] ModelInput declared fields = ['chunks']; can carry routing_matrices = False
[A] target_tokens=[102, 103, 104, 201, 202, 203] mask=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0] aligned_matches=3/3

[B] engine ModelOutput.routing_matrices = ['m0', 'm1', 'm2']
[B] workflow Step.routing_matrices       = ['m0', 'm1', 'm2']
[B] R3 datum routing_matrices            = ['', '', '', 'm0', 'm1', 'm2']
[B] datum ModelInput can carry routing_matrices = False

stage A defect: True   stage B defect: True
```

`carrier_check.py` 单独测承载能力（tinker 0.36.0，PyPI 最新）：

```
[1] ModelInput declared fields = ['chunks']; carries routing_matrices = False
[2] ModelInput.from_ints(tokens: List[int]) -> 'ModelInput'; accepts routing_matrices = False
[3] attribute assignment rejected: ValidationError
[4] nothing to serialize
carrier gap reproduces = True
```

## 做过的排除（避免误判）

| 排除项 | 结果 |
|---|---|
| 是不是 tinker 版本太旧 | 试了 0.22.2（rLLM 下限）/ 0.23.0 / 0.36.0（PyPI 最新）**全部**没有该字段、`from_ints` 都不接受该参数，赋值都被 pydantic 拒绝 |
| 是不是 Fireworks 官方 cookbook 也这么塞属性 | 不是。cookbook（`bba676c`）走 `tinker.ModelInput.from_ints(tokens[:-1], routing_matrices=rm)`，并且先经 `build_r3_routing_matrices` 做对齐 |
| 是不是我漏看了 rLLM 的其它传递点 | 不是。`grep -rn routing_matrices rllm/` 的全部写点只有 `types.py`（Step 字段）、`transform.py`（塞属性）、两个 engine（赋值 ModelOutput）、verl 侧张量 |
| 是不是"多步累积时布局错位"（一度怀疑） | **不成立**，两个形状（cumulative / 重渲染）实测矩阵都落在自己 token 的预测行上；这条排除掉，不写进结论 |
| 是不是蒸馏路径还有别的断点（一度怀疑 OpenAIEngine 缺 `compute_logprobs`） | **不成立**，`openai_engine.py:259` 有该方法 |

## 影响面（保守写法）

- **触发条件**：Fireworks 后端 + 默认配置（R3 默认开）。即"用 Fireworks 后端的 rLLM 用户"默认就在这条路径上。
- **实际后果**：取决于 Fireworks 训练服务端如何处理未声明的属性。**未验证**：可能是忽略（R3 静默无效果），也可能在请求校验处报错。rLLM 本地路径不会报错，因为塞属性发生在本地对象上。
- **不主张**：不主张训练结果已经出错、不主张性能损失、不主张这是安全或正确性回归。

## 可能的修法（仅本地设想，未实施、未提 PR）

1. 服务端形状未知时：**默认改回 `disabled`**（与 `AlgorithmConfig` 默认一致），先把不一致消掉；等承载确定后再默认开。
2. 走 cookbook 路线：`ModelInput.from_ints(..., routing_matrices=...)` + `build_r3_routing_matrices` 对齐；但**前提是实际 tinker 版本提供该参数**——本轮所有可用版本都不提供，所以这不是 rLLM 单方面能改的。
3. 打补丁式（`object.__setattr__`）绕过冻结只让本地不炸，**不解决问题**——属性仍然不会进入序列化。

哪条对取决于"Fireworks 服务端实际吃什么"，这一条只有能跑 Fireworks 或能读其 SDK 源码的人能定。

## 未做

- 未跑任何 Fireworks 真实训练（无凭据；也不在 M1 的资源范围）。
- 未跑 rLLM 全量测试套件。
- 未提 issue / PR / 评论。

## 材料

| 文件 | 作用 |
|---|---|
| `repro.py` | 两阶段复现；exit 0 = 缺陷在 |
| `carrier_check.py` | 承载能力单测 + 两路径载荷类型对照；exit 0 = 缺口在 |
| `run-info.json` | 环境、版本、断言清单、未验证项 |
