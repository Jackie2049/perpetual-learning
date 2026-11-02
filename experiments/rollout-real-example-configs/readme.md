# 真实配置下的微批与气泡：前缀共享线已经死了，能落地的是批数与 PP 气泡

2026-11-02，M1。支撑 [rollout 调研主文档 §13.36](../../notebook/rollout-performance-research.md)，
接 §13.34／§13.35 登记的两条残留——「被跳过的 PP 气泡仍未测」「长度分布是构造参数」。

**不是 GPU 实测，不是端到端**。配置逐条抄自 verl `1a9e711e` 的 `examples/grpo_trainer/`；
切分形状按 `rearrange_micro_batches` 逐行复刻并与真实源码逐位核对；**全程不含任何墙钟**，
所有「倍数」都是批数或占比之比（§19.1）。

## 一句话结论

**§13.34／§13.35 量了整整两批的那条前缀共享线，在 HEAD 上已经不可达**——engine 迁移 `#6067`
把 `use_prefix_grouper` 的唯一接线（`dp_actor` 的 forward 分支 + `fsdp_workers` 的透传）连同整批
worker 文件一起删了，新 engine 从未补上。剩下能落地的读数是两件计数量：**官方 example 里一个
rank 每个 mini-batch 只切出 2–550 个 micro-batch（多数 5–33，预算成为瓶颈时被组数封顶）**，
以及**组完整把批数变少会把 PP 气泡放大 2.11–3.95 倍**。

## 判决一：前缀共享线在 `1a9e711e` 上是死代码

§13.34 量了「micro-batch 层把组切开 81.3%」，§13.35 进一步归因到 `force_group_size` 没接线。
本批发现更靠前的一层：**连让它有意义的开关都走不通**。

| 环节 | 位置 | 状态 |
|---|---|---|
| 配置项 | `workers/config/actor.py:173`、`config/actor/actor.yaml:102` | 存在，默认 `false` |
| 唯一读取它的训练逻辑 | `trainer/ppo/ray_trainer.py:367` | 只用来决定 `_balance_batch` 是否按组均衡 |
| 注意力侧开关 | `models/transformers/monkey_patch.py:297` | `apply_monkey_patch(..., use_prefix_grouper=False)` 默认关 |
| **唯一调用方** | `workers/engine/fsdp/transformer_impl.py:300-306` | **没有传这个参数** |
| **构造 `PrefixGrouper` 的函数** | `trainer/ppo/prefix_grouper_utils.py:46` | **全仓零调用方，该模块全仓无 import** |

对比迁移前的代码（`044bbba2^`），接线本来是完整的：`workers/actor/dp_actor.py:130-141` 有 PG forward
分支，`workers/fsdp_workers.py:934/969/988` 把 `use_prefix_grouper` 一路传到 `apply_monkey_patch`。
**`#6067`（engine 迁移）删掉了这两个文件，重建 engine 时没有把这条路径搬过来。**

顺带一处自相矛盾：`examples/prefix_grouper/run_qwen3_8b_fsdp.sh:56` 设 `use_dynamic_bsz=True`，
而同一目录的 `README.md`「Limitations」第 2 条明写 *Incompatible with `use_dynamic_bsz=True`*。
示例还在、文档还在、实现不在——这就是这条线的当前状态。

> **这条对本专项的影响**：§13.34／§13.35 的**数字和推理仍然成立**（它们量的是「若有组完整性会怎样」），
> 但它们描述的是一个**当前不可达的假设路径**。引用时必须一起写「该路径在本 checkout 无调用方」，
> 否则会读成「verl 有个 bug 待修」，而实际是「功能本身没接上」。

## 判决二：真实配置里能切的 micro-batch 有多少

一个 actor DP rank 在**一个 mini-batch** 里看到的样本数 = `ppo_mini_batch_size × rollout.n ÷ dp`
（`ray_trainer.py:1315-1316`），批数由 `ceil(总 token ÷ ppo_max_token_len_per_gpu)` 决定、被组数封顶。

| example | 后端 / 并行 | batch/rank | 组数 | 预算 | 默认批数 | 组完整批数 | 每批样本 |
|---|---|---|---|---|---|---|---|
| qwen3_8b_fsdp | fsdp / dp=8 | 160 | 32 | 24576 | 16 | 16 | 10.0 |
| qwen3_8b_megatron | megatron / TP2 PP2 | 320 | 64 | 24576 | 33 | 33 | 9.7 |
| qwen2_5_32b_fsdp | fsdp / dp=16 | 40 | 8 | 16384 | 5 | 5 | 8.0 |
| qwen3_30b_a3b_fsdp | fsdp / dp=8 | 128 | 16 | 32768 | 20 | **16** | 6.4 |
| qwen3_30b_a3b_megatron_vpp2 | megatron / VPP2 | 64 | 8 | 30720 | 18 | **8** | 3.6 |
| qwen3_5_35b_fsdp | fsdp / dp=8 | 10 | 2 | 16384 | 2 | 2 | 5.0 |
| hy_v3_megatron_pp16 | megatron / PP16 | 2048 | 128 | 30720 | 550 | **128** | 3.7 |
| dsv3_671b_megatron_pp12 | megatron / PP12 | 256 | 32 | 10240 | **shape 失败** | — | — |

（7 个开 `use_dynamic_bsz` 的 example 全部列出；另有 3 个 `use_dynamic_bsz=False` 的 BSHD 配置
——qwen3.5 的 GDN 不支持 packed sequence——**根本不经过** `rearrange_micro_batches`，不在表内。）

三处读法：

1. **批数的量级是「个位数到几十」**，不是几百。中间列 7 个配置里有 6 个在 2–33 之间，
   只有一个 PP=16 的大配置到 550。这意味着任何「按批数摊销」的开销（PP 气泡、每批固定成本）
   在这些配置上权重很大。
2. **有 3 个配置已经顶到组数上限**（`ceil(总 token/预算) > 组数`）：30B fsdp、30B megatron/VPP2、
   hy_v3。对它们来说「组完整」不是可选优化，而是**批数从 20→16、18→8、550→128** 这种量级的重构。
3. **dsv3_671b 的形状冲突是真的**：它的 `ppo_max_token_len_per_gpu` 取 `prompt+response = 10240`，
   而实际轨迹长度分布在 0.8×10240 附近、有相当尾巴超过 10240 ⇒ 只要有一条超过预算，
   `rearrange_micro_batches` 的 405 行断言就触发。**这个 example 的 dynamic-bsz 配置本身不稳健**。
   （脚本里它还设了 `overlong_buffer_len=4096`，本实验未建模 buffer，故只报「有风险」不报「必然失败」。）

## 判决三：组完整把批数变少，PP 气泡随之变大

单级流水（1F1B）的气泡占比是 `1 − m/(m+pp−1)`，`m` 越小气泡越大。组完整**只减少批数、不增加批数**，
所以它的成本方向与「省前缀计算」的收益方向**相反**：

| example | PP | 默认批数 | 组完整批数 | 默认气泡 | 组完整气泡 | 比值 |
|---|---|---|---|---|---|---|
| qwen3_30b_a3b_megatron_vpp2 | 2 | 18 | 8 | 5.26% | 11.11% | **2.11×** |
| hy_v3_megatron_pp16 | 16 | 550 | 128 | 2.65% | 10.49% | **3.95×** |
| qwen3_8b_megatron | 2 | 33 | 33 | 2.94% | 2.94% | 1.00× |
| 其余四个 | 1 | — | — | 0 | 0 | — |

pp=1 的配置气泡恒为 0（没有流水就没有气泡），所以「PP 气泡」这个代价只在**同时开 PP 且预算顶到组数**
的配置上兑现——恰好是表里 30B/VPP2 与 hy_v3 这两个。

**这不是说「组完整一定亏」**：两个方向没有可比单位（一边是前缀块 FLOP，一边是流水气泡占比），
本实验明确**不做合并**，只把两边分别报出来，让决策者知道代价在哪里。

## 识别性检验与负向对照

| 检验 | 期望 | 实测 |
|---|---|---|
| 组内条数 n=1 | 组退化为单条 → 两条路径批数相等 | 4 = 4 ✅ |
| 预算放大 1024× | 两条路径都收敛到 1 批 | 1 = 1 ✅ |
| pp=1 | 气泡恒 0，比值必须为 None 而不是假正数 | None ✅ |
| pp≥2 | 气泡比值必须 > 1 | 2–16 档全为 3.94–4.26 ✅ |
| 组完整分区 | 从不拆组 | 全 True ✅ |
| 组完整实现成空操作 | 段数必须完全相同 | 相同 ✅ |
| 长度全相同时 | 每批约 1.6 条样本、5 条一组必然跨 4.9 批 | 4.94 段 ✅ |

最后一行值得单说：**默认逐条 LDM 每组被切成 4.94 段，不是「浪费」，而是「5 条样本散在 17 个批里」的
算术必然**（均匀散布期望 `k(1−((k−1)/k)^n) = 4.45`，实测 4.94 略高因为 LDM 还要平衡工作量）。
如果没有前缀共享机制，这 4.94 段本身没有成本；它只有在「组完整性 + 前缀共享」同时成立时才是浪费。
**这也是本批把 §13.34／§13.35 的收益读数降级为「不可达」的原因。**

## 与真实 verl 的交叉验证

`xcheck_verl.py`（CPU torch 2.15.0，直接 import 快照源码，只跑切分算法）：

| 形态 | 结果 |
|---|---|
| 真实预算 + `force_group_size=1`（3 个） | 分区完全一致 |
| `force_group_size=n`（3 个） | 完全一致 |
| `force_group_size=n` + vPP（2 个） | 完全一致 |
| 预算 < 单条长度（2 个） | 两侧同判不可行（shape） |
| `roundup` 顶破组数（1 个） | 两侧同判不可行（cap） |

**9/10 分区一致，1 个两侧同判不可行，不一致 0。**

## 模型修正（本批自查记录，全部是跑通之后才发现）

| # | 第一版写法 | 错在哪 | 现在 |
|---|---|---|---|
| 1 | 把 `ppo_max_token_len_per_gpu` 当长度上限喂给 `max_seq_len` | 它是每个 micro-batch 的 **token 预算**，与 `max_prompt+max_response` 是两个量；据此算出两组假冲突 | 分成 `budget` / `max_seq` 两个字段从脚本各自抄出，形状冲突降为 1 个真实风险 |
| 2 | 报 `ratio = 组完整/逐条` 的每组段数，得 0.06–0.5 | 读起来像「省 50–94%」，但这条线在 HEAD 无调用方，且 4.94 段是「5 条散在 17 批」的必然，不是可回收浪费 | 只报计数；结论降级为「收益在当前 checkout 不可达」 |
| 3 | 组完整分支用 `group_lens` 当 `batch_len` 再取 min | 真实代码的 `num_groups = batch // force_group_size`，分母被换了一次 | 与源码同序同式 |
| 4 | 用 `grouped_is_fewer` 当判据 | 它只在 `ceil > 组数` 时为真，几乎恒 False | 改为直接报 `budget_bound` |
| 5 | `identifiability` 用完整分区算 | hy_v3 单案例 LDM 4.5 s，`build_summary` 涨到 408 s | 改为只算批数，降到 0.01 s；加缓存后全量 59.6 s |
| 6 | 三处单测期望值 | 断言顺序用例自己触发 shape；分区用例 `n_micro > 组数`；「轻批排两端」用两条样本测 | 全部改成与被测语义一致的常数期望 |

## 边界（引用时必须一并写）

- **配置是抄的，负载是造的**：11 个 example 的 12 个字段逐条抄自脚本；但**长度分布是构造参数**
  （总长 0.8×(prompt+response)、组级 sd 0.25、组内 sd 0.15），不是训练日志。批数对
  「总 token ÷ 预算」敏感，真实分布可能给出不同批数。
- **dp 是推定的**：由脚本的 GPU 数与 TP/PP 推出「能让 mini-batch 留下完整组」的最小 dp，脚本未显式声明。
- **PP 气泡是公式不是实测**：`1 − m/(m+p−1)` 是 1F1B 的计数式，不含 interleaved/VPP 调度、
  不含各 stage 实际时长不均衡。只能当占比比值看。
- **前缀线死掉的结论只覆盖本 checkout**：未核上游是否有计划恢复；也未核 `_balance_batch` 的组级分组
  （`ray_trainer.py:1160` 起）独立于前缀共享是否仍有价值——这条登记为残留。
- **不支持**「应该关掉 `use_prefix_grouper`」这类决定（它本来就没生效），也不支持
  「应该给 micro-batch 接 `force_group_size`」——只在有前缀共享的前提下才有意义。

## 复现

```bash
python3 real_example_configs.py             # 约 60 s，覆盖 summary.json
python3 real_example_configs.py --verify    # 与归档逐字段比对（连跑两次通过）
python3 -m unittest test_real_example_configs    # 25 passed

# 与真实 verl 交叉验证（CPU torch，不进本仓库依赖）
uv venv /tmp/verl-seqlb-venv --python 3.12
VIRTUAL_ENV=/tmp/verl-seqlb-venv uv pip install torch --index-url https://download.pytorch.org/whl/cpu
VIRTUAL_ENV=/tmp/verl-seqlb-venv uv pip install numpy packaging ray tensordict omegaconf transformers codetiming
VERL_ROOT=../../verl /tmp/verl-seqlb-venv/bin/python xcheck_verl.py
# 一致 9｜不一致 0｜两侧同判不可行 1（共 10 个形态）
```

执行条件与产物 sha256 见 [run-info.json](run-info.json)。

## 源码锚点（verl 1a9e711e）

- `verl/utils/seqlen_balancing.py:402-414`：`num_groups` → `min(组数, ceil(total/budget))` → `roundup` → `assert`
- `verl/utils/seqlen_balancing.py:416-462`：组级 LDM、`use_dynamic_bsz_balance` 排序与两端摆放
- `verl/workers/engine/utils.py:71-77`：`force_group_size` 读非张量键默认 1；`max_token_len = 预算 × sp_size`
- `verl/workers/engine/fsdp/transformer_impl.py:300-306`：`apply_monkey_patch` 唯一调用点，未传 `use_prefix_grouper`
- `verl/models/transformers/monkey_patch.py:40-72`：PG 包装器与 `apply_prefix_grouper_patch`
- `verl/trainer/ppo/prefix_grouper_utils.py:46`：`build_pg_from_micro_batch`，全仓零调用方
- `verl/trainer/ppo/ray_trainer.py:367,1160,1201`：`use_prefix_grouper` 只影响 `_balance_batch` 的分组
- 迁移前：`044bbba2^:verl/workers/actor/dp_actor.py:130-141`、`044bbba2^:verl/workers/fsdp_workers.py:934,969,988`
