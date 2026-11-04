# verl 的 `tiled_mlp` 不是"从没接上"，而是引擎重构时丢了接线

<!-- contribution-summary:start -->
**【仓库】** [verl-project/verl](https://github.com/verl-project/verl)（LLM 后训练 / RL 训练框架）

**【贡献点】** 把已有的 `tiled_mlp` 取证（原结论：配置项从未被消费）补上历史维度：该选项在 2026-01-04 至 2026-04-20 之间**确实有效**，接线随 `verl/workers/fsdp_workers.py` 被 [#6067](https://github.com/verl-project/verl/pull/6067) 整体删除而丢失，新引擎从未接过。[取证更新](../../open-source/verl/evidence/tiled-mlp-unwired/readme.md) · 脚本 [check_wiring.py](check_wiring.py)

**【类型】** 问题诊断（B 类，已有候选的口径订正）

**【价值评估】** 6 / 10 — 结论更准确、对维护者也更有说服力（是回归不是设计），但**问题本体早已取证完成**，且仍卡在"对外动作待用户授权"；本轮新增的信息是因果链与一条被一起丢掉的守卫。

**【进展】** 2026-11-04 对 main `74543265` 复核接线状态，并回溯 `fsdp_workers.py` 的删除提交。**未对外发布**。

**【大白话讲明白】**

verl 有个省显存的开关：把 MLP 拆成几片算。今年一月有人做这个功能时，是真的把配置接到了引擎上的。四月一次大重构（"把 workers 换成 engines"）把旧文件整个删了重写，**接线漏在了旧文件里**——新引擎里从来没有这行代码。

这个区别很重要：写成"配置从来没接上"像是功能没做完，维护者会问"那当初的 PR 怎么合进去的"；写成"重构时丢了一条接线"，就是一个具体、可核对、且可能**不止这一条**的回归——同一次重构如果还漏了别的开关，一起找出来比单修一条更有价值。

顺带还丢了一条守卫：旧代码在 FSDP1（老策略）下开这个功能会直接报错。新引擎是先把模型建好、再加上这个补丁，之后才判断"用哪种策略"，所以如果只补接线、不补守卫，在老策略下会先打补丁再走 FSDP1——而默认配置恰好就是老策略。

**【What–Why–How】**

- **What：** `actor_rollout_ref.model.tiled_mlp` 在 verl main `74543265` 上的可达性；对比提交 `63e8a0d1`（删除前最后一个版本）的 `verl/workers/fsdp_workers.py`。
- **Why：** `apply_tiled_mlp_monkey_patch` 只在 `apply_monkey_patch` 的 `if use_tiled_mlp:` 分支里被调用；main 上全仓 704 个 `.py` 里提到 `tiled_mlp` 的只有 6 个文件，**全部是定义、配置与测试**，生产调用点 `verl/workers/engine/fsdp/transformer_impl.py:322` 的实参是 `model / use_remove_padding / ulysses_sp_size / use_fused_kernels / fused_kernels_backend`，不含 `use_tiled_mlp`。删除前那一版则在 `:503`／`:935` 传了 `use_tiled_mlp=` 与 `tiled_mlp_shards=`，并在 `:374` 有 FSDP1 拒绝。
- **How：** `check_wiring.py` 用 AST 取两处的 `apply_monkey_patch` 调用点与关键字实参（`EXPECT_*` 全是写死的常数），负向对照是"把当前调用改成确实传 `use_tiled_mlp`，检查必须转为 False"。**静态分析，未运行 verl**；"接线丢失"是文本＋AST 层面的事实，未排除动态读取（本仓已有 `pyflag` 的同类边界说明）。**未验证**：同期重构是否还丢了其他配置（只查了 `tiled_mlp` 一个）。
<!-- contribution-summary:end -->

## 时间线（全部可核）

| 日期 | 事件 | 依据 |
|---|---|---|
| 2025-09-09 | `verl/workers/engine/fsdp/transformer_impl.py` 出现 | 该文件最早提交 `5c46f4f4` |
| 2026-01-04 | PR [#4649](https://github.com/verl-project/verl/pull/4649) 合入，`tiled_mlp` **接到** `verl/workers/fsdp_workers.py` | 合并提交 `dcc11f29` |
| 2026-04-20 | 提交 `044bbba2`（PR [#6067](https://github.com/verl-project/verl/pull/6067)，BREAKING）**删除** `verl/workers/fsdp_workers.py`（−1768 行），接线随之消失 | `commits/044bbba2` 的 files 列表 `status=removed` |
| 2026-11-04 | main `74543265`：新引擎仍无接线 | `check_wiring.py` |

`verl/workers/engine/fsdp/transformer_impl.py` 从 2025-09 存在至今**从未**出现过 `tiled_mlp`（查了最早提交与当前头两处），所以不是"新引擎曾经有、后来又被删"。

## 复核读数（main `74543265`）

| 检查项 | 结果 |
|---|---|
| 全仓 `.py` 提到 `tiled_mlp` 的文件 | **6 个**：`models/transformers/{__init__,monkey_patch,tiled_mlp}.py`、`workers/config/{model,critic}.py`、`tests/models/test_tiled_mlp_accuracy.py` |
| 生产调用点 `apply_monkey_patch` 个数（FSDP 引擎） | **1**（`transformer_impl.py:322`） |
| 该调用传的关键字 | `model, use_remove_padding, ulysses_sp_size, use_fused_kernels, fused_kernels_backend`——**无 `use_tiled_mlp`** |
| 其他引擎（megatron／torchtitan／veomni／automodel／mindspeed） | 逐个查 `transformer_impl.py` 内 `tiled_mlp` 计数均为 0 |
| 删除前的 `fsdp_workers.py` | 2 处 `apply_monkey_patch`，`tiled_mlp` 出现 **23 行**，含三处取值与三处转发：**actor**（`:913-915` → `:935-936`）、**ref**（`:972-976` → `:989-990`，且 ref 可覆盖 model 的取值）、**critic**（`:1454-1456` → `:1481-1482`），另有两处 FSDP1 拒绝（`:374` actor、`:1459` critic） |

## 一处不要写错的口径

- **不能说"配置项从来没接上"**。它在 2026-01-04 到 2026-04-20 是有效的；`open-source/verl/evidence/tiled-mlp-unwired/` 的原表述基于 2026-11-02 快照，已在本批加了更新说明。
- **不能推断"用户在这期间开过就一定生效"**：生效窗口与"哪个版本的用户实际跑过"是两件事，本批没有版本使用数据。
- **FSDP1 守卫也丢了**，且新引擎**先建模型再分支**（`transformer_impl.py:322` 在 `:421`／`:448` 的策略分支之前），而 `dp_actor.yaml:26` 的默认 `strategy` 是 `fsdp`（FSDP1）。⇒ 只补接线不补守卫，默认配置下会先打 TiledMLP 后走 FSDP1。这一点是补丁评估此前没有覆盖的，已写进 [补丁说明](../../open-source/verl/patches/tiled-mlp-wiring/readme.md) 的更新块。

## 一条只在本批记录、未做扫描的线索

删除的 13 个生产文件里，配置读取远不止 `tiled_mlp`（`megatron.*` 20 余项、`fsdp_config.*`、`router_replay.mode`、`mtp.*` 等）。**本批只核了 `tiled_mlp` 一个**，没有对同期重构做逐项比对——那需要按"删除前读过的路径 ↔ 现在谁在读"逐条人工核，静态 grep 报出来的多半是改名或改层级的假阳性。⇒ 这一条留在队列（TM-006 第②条），不写成结论。

## 复现

```bash
cd experiments/verl-tiled-mlp-regression
python3 check_wiring.py          # 需要 Python 3.10+（快照里有 match 语句）；本机用 3.12
```

快照：`transformer_impl_main_74543265.py`、`fsdp_workers_pre_refactor_63e8a0d1.py`（都来自上游 raw 内容，只读）。

## 没做的

- 未运行 verl，未在 GPU 上复验；
- 未查同期重构是否还丢了别的配置项（这是更值得查的一步，属下一步候选）；
- 未对外提 issue／PR，也未更新已有关联材料的上游状态。
