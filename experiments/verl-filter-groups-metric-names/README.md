# verl `algorithm.filter_groups.metric`：三份「合法值表」互相不自洽，照着旧注释填会在第一个训练步就崩

2026-11-05，M1。来源：主文档 §13.49 结尾「顺带记一个文档不一致」的扩查。

> **一眼结论。** V1 的 DAPO 组过滤按**名字**去轨迹里取指标，取不到就抛
> `RuntimeError: Finished groups are missing DAPO metric '<名字>'`。而「哪些名字能用」这件事，
> 仓里有**三份互相不同的说明**：
>
> | 名字 | 配置 docstring | yaml 注释 | `docs/algo/dapo.md` | 随仓库脚本 | 代码里真的会出现 |
> |---|---|---|---|---|---|
> | `reward`（canonical） | ✓ | ✓ | — | — | ✓ 走 `rm_scores` |
> | `acc` | ✓ | ✓ | ✓ | ✓ 两个脚本 | ✓ 五个 reward manager 都写 |
> | `score` | ✓ | — | ✓ | — | ✓ 但**只有 `math_dapo` 一种 scorer 给** |
> | `seq_reward` | **✓ 声称可用** | — | ✓ | — | ✗ **零写入点** |
> | `seq_final_reward` | — | — | **✓ 举为例子** | — | ✗ **零写入点** |
> | `response_filter` | — | — | — | **✓ 随仓库脚本在用** | ✗ **零写入点** |
>
> 三个不可达的名字里，**两个是文档自己在教用户填**。用真 `ReplayBuffer` 在 CPU 上跑：
> 填 `seq_reward` / `seq_final_reward` 都在**第一个训练步**抛错，错误信息只说「缺这个指标」，
> 不说「这个指标从来没人写」。**零 GPU、零训练、零联网运行时**——只 exec 真实源码 + 替身数据面。

## 一、问题是什么

`verl/trainer/ppo/v1/replay_buffer.py` 的 `_dapo_filtered_keys`（`31cfccb4` 上的 `:263`）做三件事：

```python
use_canonical_reward = self.filter_groups_metric == FILTER_GROUPS_REWARD_METRIC  # "reward"
...
    reward_extra_info = extra_fields.get("reward_extra_info", {}) if isinstance(extra_fields, dict) else {}
    if self.filter_groups_metric not in reward_extra_info:
        missing_metric_uids.add(uid)
    else:
        metrics_by_uid[uid].append(float(reward_extra_info[self.filter_groups_metric]))
...
if missing_metric_uids:
    raise RuntimeError(f"Finished groups are missing DAPO metric {self.filter_groups_metric!r}: {...}")
```

也就是说：**除了 `"reward"`，其余名字必须正好是某条轨迹 `extra_fields.reward_extra_info` 里的一个键**。
键从哪来？`verl/experimental/reward_loop/reward_manager/*.py` 里 reward manager 写：

```python
if isinstance(result, dict):
    score = result["score"]
    for key, value in result.items():
        reward_extra_info[key] = value      # ← 动态：scorer 返回什么就写什么
else:
    score = result
    reward_extra_info["acc"] = score        # ← 静态：非 dict scorer 一律写 acc
```

所以「真的会出现哪些键」= **reward manager 写死的键 ∪ scorer 返回值里的键**。默认 scorer
`verl/utils/reward_score/__init__.py::default_compute_score` 对除了 `math_dapo` 以外的所有
数据集都走 `return float(res)` 分支 ⇒ 命中静态分支 ⇒ **只有 `acc`**。

## 二、三份说明各自怎么写

- **配置 docstring**（`verl/trainer/config/algorithm.py:48-52`）：
  `metric (Optional[str]): Metric to use for filtering. "reward" selects the canonical pre-KL reward from
  rm_scores; "acc", "score", "seq_reward", etc. select a reward_extra_info field.`
- **yaml 注释**（`verl/trainer/config/ppo_trainer.yaml:90`）：
  `"reward" selects canonical pre-KL rm_scores; other names like "acc" select a reward_extra_info field.`
- **文档例子**（`docs/algo/dapo.md:80`）：`metric: acc # score / seq_reward / seq_final_reward / ...`
- **随仓库脚本**：`dapo_30b_a3b_base_math_fsdp.sh:49` 用 `acc`；
  `grpo_qwen35_35b_megatron_async.sh:64` 用 **`response_filter`**——全仓（该脚本自己那一行以外）零处。

## 三、读数

### 读数一（静态，两张表做差）

`probe_metric_tables.py` 从真实源码抽三张表：写死键 5 个（`acc`／`overlong`／`overlong_reward`／
`timeout`／`error`，7+1+1+1+1 处写入点）、scorer 字面量键 3 个（全在 `math_dapo.py:275`：
`score`／`acc`／`pred`）、以及上面四份「说明」里点名的 6 个名字。做差：

- 说明里点名、代码里**完全没有**：`seq_reward`（docstring + 文档）、`seq_final_reward`（文档）、
  `response_filter`（脚本）。
- `score` 能跑，但**只对 `math_dapo` 系列数据集**；换成 gsm8k 就是同一个异常。
- `pred` 是个真会写入的键，但**没有任何说明提到它**（它其实是 `math_dapo` 返回的预测串，拿来过滤没意义）。

### 读数二（运行时，真 `ReplayBuffer`）

`stub_env.py` 用 exec 加载真实 `replay_buffer.py`（只替身 `transfer_queue` 与 `SkipManager`，
两者都不参与该判定），按上游 CPU 测试同一套**写盘顺序**（先每条轨迹、再把 prompt 标 finished）
造一个「2 条同分轨迹的组」，逐名字跑 `_dapo_filtered_keys`：

| 名字 | 输入＝把能给的都给了 | 输入＝真实场景（只有 `acc`） |
|---|---|---|
| `reward` | ran | **ran**（走 `rm_scores`，不依赖 extra_info） |
| `acc` | ran | ran |
| `score` | ran | **raised** |
| `seq_reward` | **raised** | raised |
| `seq_final_reward` | **raised** | raised |
| `response_filter` | **raised** | raised |

- `acc`／`seq_*`／`response_filter` 无论给多少键都抛 ⇒ **不是「scorer 没给」的问题，是没有任何代码写这些键**。
- 「只缺一条轨迹的键」也整组抛（`⑤` 自检）——错误是整组级别的，不是挑掉那一条。

### 读数三（上游自己的单测守的是什么）

`tests/trainer/ppo/v1/` 里 `filter_groups_metric=` 的字面值只有两个：`acc`（13 处）与
`reward`（1 处）。**`seq_reward` 与 `seq_final_reward` 一次都没出现在测试里**——所以 CI 不会因为
这两个名字不可达而红，也就没人发现过。

## 四、历史归属（为什么是「重构吃掉」而不是「从来就是错的」）

逐层确认（只读上游 git 与 GitHub API）：

1. `FilterGroupsConfig.metric` 的 docstring **自该文件被拆出来那天（`c936ec7`，2025-07-04，
   「add BaseConfig for all dataclass config」）就写着** `"acc", "score", "seq_reward", "seq_final_reward", etc.`，
   一路到 **2026-10-12**（`7e9875249`，score-centering 收尾）才改。
2. **DAPO recipe 的 `dapo_ray_trainer.py:240` 真的有 `if metric_name == "seq_final_reward":`
   分支**（verl-recipe 仓，2026-11-05 读取）——所以这些名字在被搬进上游 core 之前**是有主的**。
3. **V1 的 ReplayBuffer 从引入那天（`c88e875`，2026-07-23）就只认 `reward_extra_info` 的字面键**，
   没有任何 `seq_*` 特判 ⇒ 名字搬进 V1 之后**就断了，文档同步慢了两轮**。
4. #7792（`7cb6501`，2026-09-08）把 canonical `reward` 加进来、把 docstring 改成现在这句，但
   **留了一个 `seq_reward`**，而 `docs/algo/dapo.md` 一个字没改（它最近一次正文改动是
   2026-04-19 的改名 PR）。
5. `response_filter` 那个脚本自 **2026-07-28** 引入（`2cf5706`）就带着这个值，全仓从未有过对应写入点。

⇒ 正确表述是：**「`seq_*` 家族在 V1 上是离职的员工，说明书写到 2026-10 还在推荐他们接活」**。

## 五、影响面（不夸大的判断）

- **谁会被打到**：只有同时满足「`filter_groups.enable=True`」＋「`metric` 填了 `seq_reward`／
  `seq_final_reward`／`response_filter`」＋「跑 V1 trainer」的用户。仓内所有 recipe 脚本都用 `acc`，
  所以**按官方 recipe 跑不会碰到**。
- **为什么仍然值得说**：① 崩在**第一个训练步**（不是配置校验期），大集群上等于白起一次作业；
  ② 报错信息 `missing DAPO metric 'seq_reward'` 会被读成「我的 scorer 少返回一个键」，
  而真相是**没有任何约定让 scorer 返回这个键**；③ 文档与代码已经在 2026-09-08 那轮改注释时
  **改了一半**，留着的那个名字最容易被当成「官方背书过」。
- **不是 bug 的方向**：这些名字不是"实现错了"，而是"说明与实现没对齐"。所以对外最小动作应当
  **只是文档/注释与配置校验**，不该去给 V1 补 `seq_*` 实现（那是新增语义，需要设计者定夺）。

## 六、复现

```bash
# 任意 verl 源码树（需要 verl/ 与 tests/ 目录）
python3 probe_metric_tables.py --tree /path/to/verl --json raw/metric-tables.json
python3 probe_metric_tables.py --tree /path/to/verl --selftest   # 8/8
python3 negative_control.py   --tree /path/to/verl               # 4/4
```

依赖：`omegaconf`（`_new_buffer` 里用 `OmegaConf.create({})` 建空的 `trainer_config`）。
**不需要** ray / torch / numpy / transfer_queue / GPU。

- `--selftest` **8/8**：①表里含 `acc`；②正向对照（`acc` 可读时 ran）；③负向对照（无人写的名字
  raised）；④canonical 不依赖 extra_info；⑤缺一条轨迹的键就整组抛；⑥改写可达集合能被发现；
  ⑦不可达名字在真实输入下全 raised；⑧上游测试用到的名字都能在表①找到。
- `negative_control.py` **4/4**：把夹具丢键（A）、uid 写成含下划线（B）、改坏 canonical 常量（C）、
  让所有名字都可读（D）四种破坏各来一次，每种都**必须让指定检查变红**。这条是防「自检写成
  恒真」的：第一版就有两处 —— 破坏施加在外层包装上导致内层没被改到（B 跑出"假通过"），
  以及 C 的期望检查名写错（实际命中的是 `⑧`，因为常量被改后 upstream 测试里的 `reward` 脱离了表①）。
  两者都已按实测重写。

## 七、边界

- 只覆盖 **V1**（`verl/trainer/ppo/v1/`）。旧 core `ray_trainer.py` 根本不读 `algorithm.filter_groups`
  （全仓 grep 零命中），那些名字在旧路径上从来没有消费者；DAPO recipe 自己的 trainer 另有一套
  `seq_final_reward` / `seq_reward` 分支，**不在本批结论范围内**。
- `stub_env.py` 的替身不模拟 TransferQueue 的并发/租约/时序，只做「写进去、读出来」；
  `_dapo_filtered_keys` 的判定逻辑本身是上游原文。
- 「哪些 scorer 会返回 dict」是按 `verl/utils/reward_score/*.py` 的**字面量 return 扫描**得出的，
  自定义 scorer（`reward.custom_reward_function`）可以返回任意键——那正是当前设计留给用户的自由度，
  也说明「不可达」这句话只对**官方 scorer + 官方 reward manager** 的组合成立。
- 单快照（`31cfccb4`，2026-11-04）。再往上，上游随时可能把注释改齐（届时 `⑦`／`⑧` 自检会变红）。

## 八、材料

- [读数](raw/metric-tables.json) · [自检输出](raw/selftest.txt) · [负向对照输出](raw/negative-control.txt)
- [三表扫描 + 运行时探针](probe_metric_tables.py) · [替身环境](stub_env.py) · [负向对照](negative_control.py)
- 上游查重（2026-11-05，只读 GitHub API）：`filter_groups.metric` 0 条、`seq_reward` 8 条（均无关）、
  `seq_final_reward` 8 条（均无关）、`filter_groups` 16 条（均无关）。
