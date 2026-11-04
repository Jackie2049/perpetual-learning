# verl 三个「字段在、消费者没了」的开关：把静态结论跑成运行时对照

2026-11-05，M1。配套 [TM-005／TM-006／`freeze_vision_tower`](../../queue_tech-miner.md)。

**不是 GPU 实测、不是端到端训练。** 在真实 verl 树（`31cfccb4`，2026-11-04）上导入真实模块、构造真实 `LlamaForCausalLM`，驱动**真实的消费函数**，并给每个探针配一条正向对照。

## 一句话结论

三个开关在 verl main 上**仍然全部无效**，而这轮换了一种证据：不再只说"`grep` 到 0 处读取"，而是**把真实的消费者函数用真实模型跑两遍**——

| 开关 | 运行时可观测的差异 | 判定 |
|---|---|---|
| `actor_rollout_ref.model.tiled_mlp.enabled` | **0**（引擎调用不传该形参；传了才会改写 `LlamaMLP.forward`，且改写后 logits 差 5.96e-08） | 引擎侧无效 |
| `actor.use_prefix_grouper` | **0**（唯一能构造共享前缀前向的模块 `prefix_grouper_utils.py` 全仓 0 个导入方） | 运行时无效、训练侧副作用仍在 |
| `actor.freeze_vision_tower` | 未测（本机无 VL 模型；静态核对见 [TM-006 第二段](../../open-source/verl/evidence/engine-refactor-config-consumers/readme.md)） | 仍无读取点 |

## 为什么做

TM-006 前三段（配置消费点／接口面／启动配置键）全是**文本与 AST 静态取证**。静态结论有一个共同弱点：它能证明"这个名字没人读"，但不能证明"就算有人读，行为也不变"；反过来，也不能排除"扫描器漏了某条间接路径"。

这一轮换成运行时：**把开关真的传进去，看输出动没动**。并且每个探针都要有一条**正向对照**——同一套机器，在开关真的被消费时，读数必须明显变化。否则"读数没变"分不清是"开关无效"还是"探针坏了"。

## 方法

`probe.py` 直接 `import` 真实 verl 模块（`verl.models.transformers.monkey_patch`、`verl.workers.engine.fsdp.transformer_impl`），用真实的 `transformers` 构造 2 层 `LlamaForCausalLM`（hidden 32／inter 64／vocab 64，`torch.manual_seed` 固定）。

### 探针 1：`tiled_mlp`

三层读数：

1. **调用点**：`inspect.getsource` 取 FSDP 引擎 `_build_module` 里那次 `apply_monkey_patch(...)` 的实参文本，看有没有 `use_tiled_mlp`。
2. **消费者真的能用**（正向对照）：显式 `use_tiled_mlp=True` 调同一个 `apply_monkey_patch`，看 `LlamaMLP.forward` 是否被换成 `_patch_mlp_class.<locals>.tiled_forward`，并比较同一输入下改写前后的 logits。
3. **按引擎实际实参调用**：照抄引擎那次调用的五个关键字（不含 `use_tiled_mlp`），再看 `forward` 是否仍是原生 `LlamaMLP.forward`。

### 探针 2：`use_prefix_grouper`

1. `apply_monkey_patch` 的签名里**有没有** `role`（旧 worker 曾传 `role="actor"`／`role="ref"`）。
2. `use_prefix_grouper=True` 时模块全局 `_PREFIX_GROUPER_PATCHED` 是否翻转（**证明这个开关在补丁层不是空操作**，这是正向对照）。
3. `verl/workers/` 下有没有任何地方把 `prefix_grouper=` 喂进模型前向；`prefix_grouper_utils` 有没有任何导入方。
4. 历史：`git grep` 在 `044bbba2^`（#6067 之前）找谁用过 `prefix_grouper_utils`，以及 `dp_actor.py` 是否还在 main 上。

### 探针 3：`check_docs.py`

三个开关在 `docs/`、`examples/`、非生成 yaml 里的全部提及，脚本输出而不是手抄。

## 读数

### `tiled_mlp`

| 读数 | 值 |
|---|---|
| FSDP 引擎实际实参 | `model=module, use_remove_padding=…, ulysses_sp_size=…, use_fused_kernels=…, fused_kernels_backend=…` |
| 其中含 `use_tiled_mlp` | **否** |
| `use_tiled_mlp=True` 后 `LlamaMLP.forward` | `LlamaMLP.forward` → `_patch_mlp_class.<locals>.tiled_forward`（**换了**） |
| 改写前后 logits 最大差 | **5.96e-08**（fp32 舍入，符合"TiledMLP 只省显存、不改数值"） |
| 照引擎实参调用后 `forward` | **仍是 `LlamaMLP.forward`** |
| 全仓 `apply_monkey_patch(` 调用点 | **1 处**（`verl/workers/engine/fsdp/transformer_impl.py:322`） |

⇒ **能用的机器是活的，只是这台机器从来没被按下去。** 这一条比纯静态强的地方在于：如果哪天有人把它接上，读数会立刻从"仍是 `LlamaMLP.forward`"变成 `tiled_forward`，探针自己会报。

### `use_prefix_grouper`

| 读数 | 值 |
|---|---|
| `apply_monkey_patch` 形参 | `model, ulysses_sp_size, use_remove_padding, use_fused_kernels, fused_kernels_backend, use_prefix_grouper, use_tiled_mlp, tiled_mlp_shards` —— **没有 `role`** |
| 传 `role="actor"` | `TypeError: got an unexpected keyword argument 'role'` |
| `use_prefix_grouper=True` → `_PREFIX_GROUPER_PATCHED` | `False → True`（**补丁层有效**，正向对照成立） |
| `verl/workers/` 里把 `prefix_grouper=` 喂进前向的地方 | **0** |
| 导入 `prefix_grouper_utils` 的模块 | **0**（该文件自己定义 4 个函数，无人调用） |
| trainer 侧仍在读这个开关 | **4 处**（`ray_trainer.py:367/1197/1238`，另 `experimental/` 三个 trainer） |

**历史链（新证据，把 TM-005 的"从没接过"改成"曾经接过、重构后丢了"）**：

- `d6bc6296`（2026-01-05，[#4368](https://github.com/verl-project/verl/pull/4368)）引入 `prefix_grouper_utils.py`。
- `044bbba2^` 上 `verl/workers/actor/dp_actor.py:138` **真的** `from verl.trainer.ppo.prefix_grouper_utils import forward_micro_batch_with_prefix_grouper`，同文件 8 处读 `self.use_prefix_grouper`。
- `044bbba2`（2026-04-20，[#6067](https://github.com/verl-project/verl/pull/6067)）**删除** `dp_actor.py`；main 上已无此文件。
- 新引擎 FSDP `_build_module` 既不传 `use_prefix_grouper`，也没有任何等价前向入口。

⇒ 与 `tiled_mlp` 同形：**#6067 前有效，#6067 后无效**。这修正了 TM-005 原本"引擎侧从没接过"的表述——准确说法是"旧 actor 实现接过、重构时没迁移"。

## 自检

`probe.py --selftest` 注入三个扰动，每个都必须被对应的判定接住：

| 注入 | 期望 |
|---|---|
| `tiled_pass`：假装 FSDP 引擎实参里有 `use_tiled_mlp` | `tiled_mlp.engine_flag_inert` 翻成 `False` |
| `tiled_noeffect`：假装开关传进去也不再改写 forward | `tiled_mlp.machinery_alive` 翻成 `False` |
| `prefix_role`：假装签名里又有 `role` | `prefix.no_role_param` 翻成 `False` |

**本批自己的一处设计错（已改）**：第一版把 `tiled_noeffect` 也接在 `engine_flag_inert` 上，自检立刻报"未检出"。追下去发现是**判据本身错了**——"开关传进去了但不起作用"是另一种 bug，把它算成"引擎没接线"会让结论在两种完全不同的情况下都成立，等于这条判定不区分对象。已改为 `engine_flag_inert` **只**由调用点决定，`machinery_alive` 单独成项。这类错和 TM-004 的"真阳压倒静默失效"同族：**判据必须只对自己那条事实敏感**。

## 复跑

```bash
# 依赖：torch 2.15.0 / transformers 5.22.1 / orjson / codetiming（本机 /tmp/verl-seqlb-venv）
PYTHONPATH=/tmp/verl-6067-w3 /tmp/verl-seqlb-venv/bin/python probe.py --json raw/probe.json   # exit 0 成立，2 表示结论失效
PYTHONPATH=/tmp/verl-6067-w3 /tmp/verl-seqlb-venv/bin/python probe.py --selftest               # exit 0 = 三个注入全被接住
python3 check_docs.py --verl-root /tmp/verl-6067-w3 --json raw/docs.json                      # exit 2 = 文档面已清理
```

`raw/probe.json`、`raw/docs.json` 是本次归档读数。`--verl-git` 默认指向 `~/workspace/verl`（全历史克隆，用于历史链；`/tmp` 那棵树是浅克隆，`git log` 看不到 2026-01 的提交）。

## 边界

- **纯 CPU，未跑训练、未上 GPU/NPU、未触上游。**
- `tiled_mlp` 的显存收益**未在本轮测**——已有 M5 4090 真机冒烟（FSDP2 下 −46.7% 峰值，见 [TM-006](../../queue_tech-miner.md)）。本轮只回答"开关有没有接进去"。
- `use_prefix_grouper` 的**训练侧代价**不在本轮：组级均衡使 DP 不均衡在批 256 时 +6.61%、批 1024 时 +1.27%（[既有实验](../verl-prefix-grouper-side-effects/readme.md)），本轮新增的是运行时长度的归属性："曾经有效、#6067 后失效"。
- 只覆盖 **FSDP 引擎**（全仓唯一 `apply_monkey_patch` 调用点就在那里）；Megatron／mindspeed／automodel 不改 HF 模型类，`tiled_mlp` 本就不适用于它们。
- `prefix_grouper` 是第三方包，本机未安装 ⇒ 这些读数取自 `setup.py`／`requirements` 与源码关键词，而非一次真正的 import 尝试。

## 与既有材料的差异

- 既有 [TM-005 取证](../../open-source/verl/evidence/prefix-grouper-inert/readme.md) 是 `6879e1ab07`（2026-11-02）上的 AST 静态判定；本轮在 `31cfccb4`（2026-11-04）上**重跑运行时探针**，并**修正了历史归属**。
- 既有 [TM-006 取证](../../open-source/verl/evidence/tiled-mlp-unwired/readme.md) 与本轮结论一致（无效），本轮新增的是**正向对照**：证明"把它接上，机器确实会动"。
