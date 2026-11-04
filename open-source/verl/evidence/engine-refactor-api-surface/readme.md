# verl #6067 重构的接口面体检：文档还在教的三个名字，代码里已经没有了

<!-- contribution-summary:start -->
**【仓库】** [verl](https://github.com/verl-project/verl)（RL 训练框架）

**【贡献点】** 上一批把 `tiled_mlp` 查成"一类回归"，查的是**配置消费点**（字段还在、没人读）。这一批换一个信号族，查**接口与文档引用**：#6067 删掉的 10 个生产模块里共 50 个公开方法/函数名，有 13 条在 main 上已无定义、但仍有**非历史类文档**在教用户使用，去重后是 3 个名字（`update_policy`、`update_critic`、`compute_values`）。另查出一条**代码在、前端照常接受、后端已无人触发**的 profiler stage（`critic_update` / `compute_values` 的 `@DistProfiler.annotate(role=...)` 随旧 worker 删除，engine 侧只剩 actor 侧三个 role）。[#6067](https://github.com/verl-project/verl/pull/6067)

**【类型】** 问题诊断（B 类：主动发掘）

**【价值评估】** 4 / 10 — **比上一批弱**：这 3 个名字在代码侧**没有任何残留调用点**（`verl/**` 非测试代码 0 处、`tests/` 与 `examples/` 也是 0 处），所以不存在静默失效、不影响任何训练结果；真正的价值只有两条：① 照文档扩展算法的用户会走空；② profiler stage 这一条是**结构性缺口**（新引擎里没有 critic 域的采集入口），要真机才能定严重程度。**不主张对外提 issue**，仅作为"重构范围"的记录。

**【进展】** 2026-11-04 M1 本地完成，纯静态。**未触上游**、未建 issue、未评论、未开 PR。复现与归档见 [实验目录](../../../../experiments/verl-6067-api-surface-scan/readme.md)。

**【大白话讲明白】**

verl 四月把旧的 worker 代码整份删了，换成了 engine。有些名字也随之消失，比如"更新 actor"这个方法，以前叫 `update_policy`，现在叫 `update_actor`。问题是**文档没跟着改**：性能调优文档里还在告诉用户"调这两个参数会影响 `update_policy` 和 `update_critic` 的耗时"，扩展算法教程里还在说"照着 `update_policy` 的样子实现"。

另一种更隐蔽：verl 有个"精度调试器"，可以指定只采集某几个阶段的数值，其中一项叫"算 critic 的值"（`compute_values`）。旧代码里，critic 的取值和更新各自挂了一个采集标签；重构之后这两个标签**没有人再挂了**，但调试器的白名单里这两个名字还留着——用户配了 `compute_values`，程序不报错、也不提示，安安静静什么也不采集。

这两件事今天都不会让训练算错，所以价值不高；但它们说明**重构的"删干净了没有"只能靠逐层核对**，配置层干净不代表接口层和文档层也干净。

**【What–Why–How】**

- **What：** #6067（`044bbba2`）整文件删除的 10 个生产模块（`workers/actor/*`、`workers/critic/*`、`fsdp_workers.py`、`megatron_workers.py`、`sharding_manager/*`）共导出 50 个公开方法/函数名。当前 main 上：32 个仍有同名定义，13 个"无定义但非历史文档仍在引用"（去重后 3 个名字），5 个无定义且无文档引用。
- **Why：** 配置消费点体检（上一批）只回答"用户设了这东西有没有用"；接口名同样会被删除，但它的消费者是**文档和下游脚本**，信号完全不同——代码里 grep 干净不代表文档层干净，也不代表功能还在（profiler role 就是反例：文档里的 stage 名被前端白名单接受，但没人触发）。
- **How：** ① `scan_api_surface.py` 从 `044bbba2^` 的 10 个被删模块抽公开方法/函数名，在 main 的 `verl/**` 查 `def`、在 `docs/**` 查引用，分四档（有定义／仅历史文档引用／有活动文档引用／都没有）；② `verify_findings.py` 对 7 条判定逐条写死证据、跑 0 不一致；③ `--selftest` 往 main 副本种同名 `def` 与 `annotate(role=...)`，四条相关判定必须翻转。**边界**：纯静态，未跑 verl、未上 NPU/GPU；profiler stage 实际采不采得到需 msprobe + NPU，本轮不测；`docs/blog/` 下的发布说明引用已删接口是对的，已在扫描器里排除。纯静态，未跑 verl、未上 NPU/GPU；profiler stage 的"实际采不采得到"需要 msprobe + NPU，**本轮不测**；`docs/blog/` 下的发布说明引用已删接口是正确的，已在扫描器里单独排除。

<!-- contribution-summary:end -->

## 一、结论

| 信号 | 结果 | 是否算问题 |
|---|---|---|
| 50 个被删接口名中仍有同名定义 | 32 | 否（重构后保留/重建） |
| main 无定义、**非历史文档**仍引用 | 13 条 → 去重 3 个名字 | **文档层问题**（`update_policy`／`update_critic`／`compute_values`） |
| main 无定义、仅 `docs/blog/` 引用 | 0 | 否 |
| main 无定义、无任何文档引用 | 5 | 否（正常删除） |
| `@DistProfiler.annotate(role=...)` 的 critic 侧 role | **0 处**（旧 worker 里 2 处） | **结构性缺口，需真机定级** |

代码侧调用点：`verl/**`（非测试）、`tests/`、`examples/` 对这三个名字的**调用**一律 0 处 ⇒ **没有静默失效**，这一点必须先说清楚，否则会把"文档过时"读成"功能坏了"。

## 二、三个还留在文档里的名字

| 名字 | 旧位置 | 文档在哪教 | main 里现在叫什么 |
|---|---|---|---|
| `update_policy` | `actor/{base,dp_actor,megatron_actor}.py` | [perf_tuning.rst:135](https://github.com/verl-project/verl/blob/main/docs/perf/perf_tuning.rst#L135)、`dpo_extension.rst:18`（扩展算法教程） | `update_actor`（`engine_workers.py:656`） |
| `update_critic` | `critic/*` + `fsdp_workers.py:1698` | `ray_trainer.rst:194-195`、`perf_tuning.rst:135` | **RPC 层没有同名方法**；trainer 改调 `critic_wg.train_mini_batch`（`ray_trainer.py:1359`） |
| `compute_values` | `critic/*` + `fsdp_workers.py:1678` | `ray_trainer.rst:162`、`perf_tuning.rst:141`、`precision_debugger_zh.md:48/85` | **RPC 层没有同名方法**；trainer 改调 `critic_wg.infer_batch`（`ray_trainer.py:1221`） |

注意 `dpo_extension.rst` 与 `ray_trainer.rst` **#6067 根本没碰**——它们不是这次重构造成的回归，而是**存量文档没跟上重构**。区分这两者，决定了对外表述该写"重构漏改"还是"文档欠债"。

## 三、结构性缺口：profiler 的 critic 侧采集标签

旧的 `fsdp_workers.py` 里，critic 的两个 RPC 各挂了一个采集标签：

```python
@DistProfiler.annotate(color="cyan", role="compute_values")     # :1677
def compute_values(self, data: DataProto): ...

@DistProfiler.annotate(color="pink", role="critic_update")      # :1697
def update_critic(self, data: DataProto): ...
```

`precision_debugger` 正是靠这个 `role` 决定"这一阶段要不要开始 dump"。重构后：

- `verl/**` 全仓 `role="compute_values"` 与 `role="critic_update"` 各 **0 处**（engine 侧只有 `actor_update`、`actor_compute_log_prob`、`ref_compute_log_prob` 三个 role）；
- 但前端白名单 `precision_debugger_profile.py:26-33` 的 `_STAGE_TO_ROLE` 里，`compute_values → critic`、`critic_update → critic` **仍然在**；
- `strict: False`（默认）下，未知 stage 只 warn 不报错，而这两个 stage **是已知的** ⇒ 连 warn 都没有。

⇒ 用户按 [precision_debugger_zh.md](https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/dev_guide/precision_analysis/precision_debugger_zh.md) 配 `stages: [..., compute_values, critic_update]`，最坏情况是"配了、没报错、也没采到"。

**必须收窄的地方**：静态只能证明"没有装饰器会以这两个 stage 名触发"，**不能**证明"一定采不到"——`_MODEL_ATTRS_BY_ROLE` 里还留着 `critic.engine.module` 的解析路径，说明写这段代码时是预期 critic 侧会被采集的；实际行为要 NPU + msprobe 才能读，属真机范围（M5）。口径：**"前端接受、后端无触发点"**，不是"功能坏了"。

## 四、两条存量文档漂移（不属 #6067）

| 文档 | 说的 | main 里的实际 |
|---|---|---|
| `perf_tuning.rst:140-141` | values 前向的 token 上限由 `critic.forward_micro_batch_size_per_gpu` 控制 | 该键是**个数**；token 上限另有 `critic.forward_max_token_len_per_gpu`（`critic.yaml:49`），文档全文 0 次提及 |
| `perf_tuning.rst:93,96` | 链接到 `fsdp_workers.py#L120-L122`（#6067 改成 `engine_workers.py`） | 新目标文件里没有 batch-size 归一化逻辑（`ppo_mini_batch_size` 只有 `:511` 一处赋值），链接可达但**点名的内容不在那** |

## 五、本批自己的三处错（都由自检或复核抓出）

1. **判定脚本第一版的正则写错 3 条**：`update_policy` 的文档锚点我按中文写了"和 `update_critic`"，实际文档是英文 "and"；`update_critic` 那条把路径拼成 `docs/advanced/../workers/...`；`compute_values` 那条要求行首是 `-`，而文档里是 `*`。⇒ 三处都是**期望值凭记忆写、没有先跑一遍看实际文本**。教训与第七批"两版对照的期望值必须先自证"同族。
2. **`critic-infer-token-cap` 这条第一版写反了**：我把"文档提 `critic.forward_max_token_len_per_gpu`、main 里没有"当成问题，复核 `critic.yaml:49` 才发现 main 里**有的正是这个名字**，缺的是"文档没提它、只提了 micro_batch_size"。判定方向整条反过来后才成立。
3. **`_STAGE_TO_ROLE` 我先读成"删掉了 critic 域"**：实际是表里还留着、删掉的是装饰器。这两者在对外表述里分量完全不同（前者是"配置项失效"，后者是"前端接受后端不触发"），必须分开写。

## 六、与上一批的分工与合起来的口径

两批查的是同一次重构的两个信号族，**都不能说成"这次重构全扫过"**：

| | 上一批（配置消费点） | 本批（接口 + 文档） |
|---|---|---|
| 问的问题 | 用户设了这个字段，还有人读吗 | 代码/文档提到的这个名字，还存在吗 |
| 命中 | 3 条真失效（有静默失效，影响实际行为） | 3 个文档里的旧名（无静默失效）+ 1 个 profiler 结构性缺口 |
| 看不见 | 函数名、文档、装饰器 | 配置字段、行为契约（MTP 由报错变静默清零） |

#6067 另外改动的 71 个文件的**语义变化**仍未扫；两次体检都只覆盖"被整份删除的文件"。
