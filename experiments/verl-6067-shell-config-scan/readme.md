# verl #6067 的第三个体检信号族：被删掉的启动配置键，还有谁在传

**目的**：前两批只覆盖"**被整份删除的文件**"——配置消费点看引擎还读不读、接口面看文档还教不教。
本批换到"**被改动**的启动入口"：`verl/trainer/config/ppo_trainer.yaml` 里
`trainer.use_legacy_worker_impl` 被整条删掉，而命令行还在传它的脚本**一行都没被改**。

**大白话**：这个开关以前用来选"旧 worker 实现还是新引擎实现"。重构把旧实现删了，开关也就没用了——顺手把开关本身也删了。但三份启动脚本还在按老习惯写 `use_legacy_worker_impl=disable`。现在分两种情况：**加了 `+` 的**，Hydra 把它当新键收下、程序照跑（配置里多出一个**没人读**的字段，不报错）；**没加 `+` 的**，Hydra 在合并配置时就拒绝，直接报
`Could not override 'trainer.use_legacy_worker_impl'.`

## 读数

对 15 个后缀文件里的一个键做全仓定位（`scan_shell_refs.py`，可复现）：

| 处置 | 处数 | 位置 |
|---|---|---|
| `INVOKE / COMPOSE_FAIL`（**会直接报错**） | **1** | `tests/special_e2e/run_fully_async_policy_genrm.sh:177`（unprefixed） |
| `INVOKE / SILENT_OK`（不报错，也**不生效**） | 2 | `tests/special_e2e/run_fully_async_policy_opd.sh:240`、`examples/grpo_trainer/run_qwen3_30b_veomni.sh:102`（都带 `+`） |
| `DOC_TABLE_CELL`（参数表仍列着） | 1 | `docs/ascend_tutorial/dev_guide/model_dev/parameter_and_metrics.md:368`（默认值写作 `auto`） |
| `EXEMPT_NEG_MARK`（已标"removed"） | 0 | `docs/workers/engine_workers.rst:221` 写的是 `(removed; only the unified engine is available)`，被排除是对的 |

**代码侧读取点 0 处**（`git grep` 计数在 HEAD 上为 0）⇒ `SILENT_OK` 那两条传进去的值**不会被任何代码消费**；
它们今天只是"多写了一个字段"。**不报错的恰好是最安全的一档**，这一点必须写在结论里，否则容易被读成"两个脚本坏了"。

## 真实 Hydra 复现（不是机制推断）

`hydra_repro.py` 用最小树复刻三种形状，实测（hydra **1.3.8**、Python 3.12.13）：

| 形状 | 树里有这个键 | 传法 | 实测 |
|---|---|---|---|
| A（`044bbba2^` 的形状） | 有 | `trainer.use_legacy_worker_impl=disable` | **OK**，值进入配置 |
| B（`044bbba2` 之后的形状） | 无 | `trainer.use_legacy_worker_impl=disable` | **FAIL** `ConfigCompositionException: Could not override 'trainer.use_legacy_worker_impl'.` |
| C（同 B） | 无 | `+trainer.use_legacy_worker_impl=disable` | **OK**，作为一个新键被接受 |
| D（负向对照，存在的键） | 无 | `trainer.device=npu` | **OK** ⇒ B 的失败不是"所有 override 都失败" |

C 那一行还顺带否掉了一条**可能的反驳**："`fully_async` 走的是 `ppo_trainer`，也许 open dict 允许新增键"——
同一条链上 `cfg.trainer.<新键> = ...` 赋值直接抛 `ConfigAttributeError`，不成立。

## 这条时间线（责任归属）

| 时间 | 事件 | 结果 |
|---|---|---|
| 2026-04-20 #6067 | 删掉配置键 | 未改的 `main_ppo`（`${TRAINER[@]}`）调用点立刻进入形状 B |
| 2026-04-20 #6067 自身 | 只修了 `examples/router_replay/*.sh` 三处（把 `USE_LEGACY_WORKER_IMPL` 分支连同传参一起删） | 其余调用点未动 |
| 2026-04-27 / 05-06 | fully-async 的 genrm / opd 脚本新增（晚于 #6067） | **按只有新实现的树写的**，所以 `+` 前缀是对的 |
| 2026-05-19 | veomni 示例更新（只确认了它最后改动日期晚于 #6067，未追它最初是哪次带上 `+` 的） | 同上 |

⇒ **`genrm` 那一条是 #6067 当时的遗留**，`opd` / `veomni` 两条是"照着带 `+` 的模板抄下来的残余"。
后两条今天无害（`+` 让 Hydra 放行），但源码里已经**没有任何代码读这个字段**，
保留它只会让下一个人以为这个开关还活着。

## 顺带查的第二条：`scripts/megatron_merge_lora.py`

同一次重构把该脚本从 `verl.workers.megatron_workers.ActorRolloutRefWorker` 迁到
`verl.workers.engine_workers.ActorRolloutRefWorker`，新写的两行用了 `engine.bridge.save_weights(...)` /
`engine.bridge.save_hf_weights(...)`。**没收进上面的读数**，原因是静态核不全：

- `MegatronEngine.bridge` 是 `AutoBridge` 实例（或 `None`），其方法由**第三方包**提供
  （`mbridge` / `megatron.bridge`），本机未安装，**无法静态确认这两个方法存在**；
- 现有引擎代码**从不调**它们，只调 `export_weights` / `export_hf_weights` / `export_adapter_weights`
  （`verl/workers/engine/megatron/transformer_impl.py:818-826`）；但
  `verl/utils/checkpoint/megatron_checkpoint_manager.py:993-1015` 仍在 `inspect.signature` 里**探测**
  `bridge.save_weights`，说明"有些 bridge 版本确实有"。——**这是推断，不是实测**。

⇒ 记为"需要真机或装 `mbridge` 才能判定的候选"，不进对外材料。

## 文件

| 文件 | 作用 |
|---|---|
| `scan_shell_refs.py` | 取 #6067 前后 `ppo_trainer.yaml` 的叶子键差集 → 全仓找引用点 → 四档判定（`INVOKE` / `DOC_TABLE_CELL` / `EXEMPT_NEG_MARK` / `OTHER`）＋链条解析（`--config-name`、`_self_`、searchpath 回退） |
| `verify_findings.py` | 4 条判定写死 + 3 条结构性自证；`--selftest` 三种扰动全部要被捕捉 |
| `hydra_repro.py` | 真实 Hydra 四形状对照（A/B/C/D），最小树、不装 verl |
| `scan-result.json` / `findings.json` / `hydra-repro.json` | 三份输出 |

## 复现

```bash
python3 scan_shell_refs.py --repo <verl 仓库> --json scan-result.json     # 期望删 1 键、4 处引用
python3 verify_findings.py --repo <verl 仓库> --json findings.json        # 期望 0 处不一致
python3 verify_findings.py --repo <verl 仓库> --selftest                  # 期望 3 个扰动全部被捕捉
pip install hydra-core && python3 hydra_repro.py --json hydra-repro.json  # 期望 4/4 PASS
```

本轮实测环境：verl main `74543265`（2026-11-04 fetch），父提交 `044bbba2^`，Python 3.12.13，macOS arm64；
`hydra-core` 装在一次性 venv 里（`/tmp/hydra-probe-venv`），**没有装 verl、没跑训练、没上 GPU**。

## 判据与边界（写在这里免得下次重犯）

1. **"不报错"不等于"生效"**。`+key` 让 Hydra 放行，但 verl 侧读取点 0 处，值进配置后**没人用**。
   对外只能说"传了个没人读的字段"，不能说"这个脚本靠它走了新实现"。
2. **只查了一个键**。`ppo_trainer.yaml` 的键差集本轮恰好只剩 1 个（脚本会打印差集，下次有变化会直接显示）；
   其余 yaml（`_generated_ppo_*.yaml`、`fully_async_*.yaml`）**未纳入差集**，只做了"现在树里还有没有"的抽查。
3. **`SILENT_OK` 是保守档**：判据是"链里出现过 `ppo_trainer.yaml` ⇒ struct 模式 ⇒ 去掉 `+` 必失败"。
   真实 Hydra 已对上（B/C 两条），但**别的 entry 如果用了 open dict，这一档会误判**——目前四个命中都在同一条链上。
4. **看不见的**：CI 是否真的因为这个原因失败（`genrm` 脚本**不在** `.github/workflows` 的 paths/命令里，
   `opd` 在，但它带 `+` 不会失败）；`scripts/megatron_merge_lora.py` 的两行桥接调用（见上，第三方包缺失）。

## 脚本自身的坑

1. **`git grep -l -- '*.sh' ...` 不会把 glob 当模式**——把它当成待匹配的**文本**，于是返回空集、
   整轮扫描"0 处引用"。改成 `git grep -l -e <key> -- '*.sh'` 之后才命中。
2. **`entry_context` 对"引号内匹配"必须放过**：`"${VERL_ROOT}/examples/data_preprocess/geo3k.py"` 被正则当成启动行，
   于是三个命中全部解析到错误的入口。加上"引号/括号内不算"之后才对。
3. **`cp -R` 到 `/var` 与 `/private/var` 的路径不一致**：自检第一次跑时 `relative_to` 抛 `ValueError`，
   修法是 `resolve()` 两侧。同族的另一次失误更隐蔽——`write_text(x.replace(...))` 中 `replace` **没命中时也照样写出**，
   "注入了"是假的；现在先把替换前后的字符串比一比。
