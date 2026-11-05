# AReaL 训练侧 pyflag（扩池第七批）：42 个开关 / 7 条候选，真缺陷 0 条

对 [AReaL](https://github.com/areal-project/AReaL) `3b936798`（2026-10-19）跑一遍
「配置项有没有生产者」。推理引擎六批（SGLang／vLLM／LMDeploy／TensorRT-LLM／llama.cpp／MLC-LLM）
扫完后换到**训练侧**，这是第一个训练框架对象。

- 证据页：[open-source/areal/evidence/pyflag-2026-11.md](../../open-source/areal/evidence/pyflag-2026-11.md)
- 扫描器：[tools/pyflag](../../tools/pyflag/readme.md)
- 上游状态：**不对外**（无缺陷可提）

## 读数

| 项 | 值 |
|---|---|
| 快照 | `3b93679888e9f4929f6f155f13c6a286a62dfd72`（2026-10-19，`git status` 干净） |
| `.py` 文件 | 1141 |
| 扫描耗时 | **32 秒** |
| 发现开关 | **42** |
| `silently_inert` | **7** |
| 真缺陷 | **0** |

## 7 条候选的归因

| 开关 | `mode` | 档位 | 事实 |
|---|---|---|---|
| `use_sapo_loss` | never | **R1 真实接线** | 生产者在 `actor.py:612` 的 `functools.partial(grpo_loss_fn, ...)` 里（`:622` 传值）；配置字段 `cli_args.py:2062` |
| `use_cispo_loss` | never | **R1** | 同一个 `partial`（`:625`）；配置字段 `cli_args.py:2076` |
| `use_decoupled_loss` | never | **R1** | 同一个 `partial`（`:626`），另有一处真分支 `actor.py:170`；配置字段 `cli_args.py:2115` |
| `use_cpu` | never | **R2 真不可达** | `archon/moe/kernels.py:155` 形参 + `:211` 分支；唯一调用点 `moe/utils.py:212` 不传，全仓含 `tests/` 零生产者 ⇒ CPU 实现从未执行；docstring `:172` 仍宣称可用 |
| `use_local_output` | mixed | **R1** | `parallel_styles.py:42` 形参 → `:48` 存 self → `:80` partial 喂给 `_prepare_output_fn`；调用方 `fsdp_utils/parallel.py:318` 不传 ⇒ 吃默认 `True`（默认值就是设计意图） |
| `use_dist_ckpt` | always | **R3 内部默认** | `checkpointer.py:315` 形参 + `:343` 分支；唯一调用点 `:459` 不传 ⇒ 恒 `True`。无配置字段、无 CLI ⇒ 内部默认 |
| `with_rng` | always | **R3 内部默认** | `checkpointer.py` 4 处形参 + 真分支；生产调用点 `megatron_engine.py:1406`／`:1437` 都不传 ⇒ 恒 `True`。无配置字段 |

`classify.py` 覆盖全部 7 条，缺一条 `exit 1`。

## 本批的产出：扫描器又多一类盲区

三条 R1 出自同一个原因——**`functools.partial` 隔了一层**。扫描器收集调用点时只看
`ast.Call.func` 的名字是不是目标函数；而这里是 `partial` 的**第一个位置实参**：

```python
loss_fn = functools.partial(
    grpo_loss_fn,
    ...
    use_sapo_loss=self.config.use_sapo_loss,     # ← 真正的生产者在这里
    ...
)
```

⇒ 「找不到生产者」之前必须排除 `partial` / `lambda` / 装饰器。已写进
[pyflag readme 第 4 条](../../tools/pyflag/readme.md)，**尚未进扫描器**。

## 训练侧与推理引擎的形状差异（下一批要注意）

训练框架的开关大多挂在**配置 dataclass 字段**上（`areal/api/cli_args.py` 里的 `field(...)`），
而不是函数形参。pyflag 由「形参默认值」发现开关，所以**只能看见函数层面那一小部分**——
AReaL 42 个开关相对它数百个配置字段覆盖很低。⇒ 训练侧要系统覆盖得另做
「配置字段 → 消费点」的扫描，不能照搬 pyflag 口径。这一条比"又扫了一个仓库"更有价值。

## 脚本

| 文件 | 作用 | 怎么跑 |
|---|---|---|
| `verify_findings.py` | 33 条判据逐条钉到文件行；`--selftest` 3 条注入 | `python3 verify_findings.py <AReaL 根>` |
| `classify.py` | 7 条候选逐条给档位与理由，漏一条 `exit 1` | `python3 classify.py raw/scan-areal.json` |

## 复现

```bash
git clone https://github.com/areal-project/AReaL /tmp/areal && cd /tmp/areal && git checkout 3b936798
cd <repo>
python3 tools/pyflag/scan_unwired_flags.py /tmp/areal --json tools/pyflag/scan-areal-3b936798.json
python3 experiments/areal-pyflag-triage/verify_findings.py /tmp/areal          # 33 checks
python3 experiments/areal-pyflag-triage/verify_findings.py --selftest /tmp/areal
python3 experiments/areal-pyflag-triage/classify.py experiments/areal-pyflag-triage/raw/scan-areal.json
```

## 本批两处判据错（都由重跑抓出）

1. **`get_rng_state` 的"唯一调用点"假红**：Python 版收集调用点按名字匹配，把
   `torch.get_rng_state()`、`current_platform.get_rng_state()` 也算成 AReaL 的
   `MegatronCheckpointManager.get_rng_state`。第一版改成"只要是属性调用就算"——照样错
   （那两个也是属性调用）。最终加 `self_only`：只收 `self.get_rng_state(...)`。
   ⇒ **"同名不同对象"要按接收者区分，不能按调用形态区分。**
2. **`with_rng: bool = True` 数错**：写判据时凭印象写"3 处"，实际 **4 处**
   （`:410/:499/:548/:641`）。⇒ 判据里的常数必须现数，不能凭读代码的印象。

## 边界

纯静态 AST 分析 + 一次全仓扫描，**未装 areal、未跑训练、未上 GPU/NPU、未触上游**。
`use_cpu` 那条只证明"CPU 分支没有生产者"，没有证明"CPU 实现本身是坏的"。

**2026-11-05 追加（后续批）**：这条的后半句已经查完，见
[use-cpu-fallback](../areal-use-cpu-fallback/README.md)——CPU 实现在**上游自己的缓冲口径下是对的**
（与 GPU 排布逐位相同），且 `kernels.py` 顶层依赖 Triton ⇒ 缺 Triton 的机器连模块都进不来。
⇒ 口径改为「开关不可达 + 实现可用」，不是"坏掉的回退"。
