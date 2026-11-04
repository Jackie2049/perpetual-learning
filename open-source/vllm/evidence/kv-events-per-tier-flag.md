# vLLM：secondary tier 的 `enable_kv_events` 在库内没有生产者（2026-11-05）

<!-- contribution-summary:start -->
**【仓库】** [vLLM](https://github.com/vllm-project/vllm)（主流大模型推理引擎）

**【贡献点】** `secondary_tiers` 配置里的 `enable_kv_events` 在 [fs/manager.py](https://github.com/vllm-project/vllm/blob/da0b6ac2/vllm/v1/kv_offload/tiering/fs/manager.py#L117) 与 [obj/manager.py](https://github.com/vllm-project/vllm/blob/da0b6ac2/vllm/v1/kv_offload/tiering/obj/manager.py#L114) 声明并分支，文档也列为可配字段，但**库内没有任何代码给它赋值**——与同一矩阵的 `enable_telemetry`（从 `extra_config` 直接读）不同类。[实验与复现](../../experiments/vllm-pyflag-triage/README.md)

**【类型】** B 类（主动发掘）：源码读出的设计不一致候选，**未提交上游**

**【价值评估】** 4 / 10 — 分支是活的、可证伪、可机读；但**用户可以在 tier 条目里手写这个键绕过去**，所以不能写成"开关坏了"。更接近设计一致性问题而非缺陷。

**【进展】** 2026-11-05 在 `da0b6ac2` 上完成扫描 + 逐条取证 + CPU 行为分叉；上游查重无重复 issue，**对外待用户逐项授权**。

**【大白话讲明白】**

vLLM 可以把显存里放不下的 KV 缓存挪到本地磁盘或对象存储里，这些"下一级存储"通过 `secondary_tiers` 配置。这个功能加了一个开关：下一级存储每成功存下一个块，要不要对外广播一条"我这儿有这一块"的事件，让别的机器能来取。

开关的代码写好了、两个存储后端都真的会按它分支、文档的表里也列着"可选，默认 false"。但是 **vLLM 自己从来不打开它**，也不替用户组装这个字段——想用的人只能自己在那段配置里手写一行。对比同一段配置里的另一个开关 `enable_telemetry`，它由代码从配置里读出来再往下传，用户不用操心。两者不在同一个层级上，这就是本轮记录的不一致。

**【What–Why–How】**

- **What：** `enable_kv_events`（fs / obj 两个 secondary tier 各一份）在库内无生产者；同一文件里的 `take_events()` / `get_finished_jobs()` 事件缓冲逻辑完整可用。
- **Why：** PR #47923 的写法是"tier 配置字典原样 `**config` 展开"，所以**透传在字面上成立**，但没有任何默认注入点；而 `kvcr` 的 `enable_telemetry` 走的是 `extra_config.get(...)` 显式读取。推断：两种写法各有来源，未统一。
- **How：** 静态逐行分类（10 处出现、0 处赋值）+ 把 `__init__` 的真实源码取出在 CPU 上跑 `False`/`True` 两档（`None` vs `[]`），并跑了"全局关时 tier 自开会告警"这条门控。**边界**：未启动 vLLM、未发真实事件、未调查 out-of-tree tier 是否依赖手写；"算不算缺陷"取决于维护者对配置层的定位，本文不下判断。
<!-- contribution-summary:end -->

## 证据链（全部可机读复跑）

详见 [实验 readme](../../experiments/vllm-pyflag-triage/README.md)。关键行号：

| 位置 | 内容 |
|---|---|
| `vllm/v1/kv_offload/tiering/fs/manager.py:117` | `enable_kv_events: bool = False,` |
| `vllm/v1/kv_offload/tiering/fs/manager.py:143` | `if enable_kv_events:` → `self.events = []` 或告警 |
| `vllm/v1/kv_offload/tiering/obj/manager.py:114` / `:142` | 同上 |
| `vllm/v1/kv_offload/tiering/spec.py:433` | 唯一构造点，`create_secondary_tier(tier_config, primary_kv_view, self)` |
| `vllm/v1/kv_offload/tiering/factory.py:96` | `**config` 展开 tier 配置 |
| `docs/features/kv_offloading_usage.md:248` / `:291` | 文档字段表把 `enable_kv_events` 列为可选 |
| `vllm/v1/kv_offload/tiering/kvcr/manager.py:289` | 对照：`extra_config.get("enable_telemetry", False)` 是**显式读取** |

## 与 SGLang 两条的区别（对外口径）

| 项目 | 位置 | 用户能否绕开 | 对外写法 |
|---|---|---|---|
| SGLang `enable_deterministic_inference` | Grok 路由 shim 形参 | **不能**（调用点是写死的） | "开关对 Grok 路由无影响" |
| SGLang `use_logits_buffer` | 唯一调用点按块数传值 | 不能 | "默认值不可达" |
| **vLLM `enable_kv_events`** | tier 配置字典 | **能**（自己写 `enable_kv_events: true`） | "库内无生产者；用户可手写" |

⇒ 本条**不能**写成"事件开关坏了"。

## 上游查重（2026-11-05）

- `"enable_kv_events"` in title → 1 条 #61893（open，**KV connector 级**，方向不同）
- `"secondary_tiers" "enable_kv_events"` → 1 条 #47923（closed，引入该键的 PR）
- `tier.enable_kv_events` → 0 条
- 相关但不同的问题：#63037 / #64613 / #65548（事件元数据与身份），#62580（valkey tier）

## 下一步

对外动作（issue / 文档 PR / 代码 PR）**均待用户逐项授权**。若用户判定"不算缺陷"，本条降级为方法读数，
结论只保留"SGLang 与 vLLM 两批共 46 条候选，真缺陷 3 条"这一类统计。
