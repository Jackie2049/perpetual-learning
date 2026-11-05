# verl V1 DAPO 组过滤下的采样轮询：4 卡（2+2）A/B——整步 −4.0%，等待收益 ≈0.6×量子

2026-11-02 执行（M5 算力侧，TM-008 边界实验的 4 卡补充）。2 卡 DAPO 的六跑 A/B 见 [verl-v1-poll-quantum-dapo](../verl-v1-poll-quantum-dapo/README.md)（独立复核见其 [verification.md](../verl-v1-poll-quantum-dapo/verification.md)）；固定批 4 卡的 A/B 见 [verl-v1-poll-quantum](../verl-v1-poll-quantum/README.md)；机制背景与 2 卡结论见主文档 §13.44／§13.48，本批结论见 §13.49。

## 目的

首轮（15:03–15:11 UTC）只拿到 4 卡默认 2.0 基线，第二臂因与并行会话抢卡未完成；三次撞车后 A/B 推迟。本批在**同一个独占窗口**内重跑两臂，回答两个问题：

1. **DAPO 的收益在多卡下是保留、放大还是被压缩？** 首轮登记的两条锚点给出相反预测——若省下的是**每步固定量**（2 卡实测固定等待 −1.87s），4 卡的相对收益应约 −1.9/18.5 ≈ **−10%**；而固定批 4 卡只测到 **−2.0%**。哪一侧成立取决于"派发—等待"轮数是否随卡数变化。
2. **0.5 与 0.1 在 4 卡上是否仍等价？** 2 卡 DAPO 上两者等价（gen 7.21 vs 7.29）；固定批 2 卡上 0.5 拿到 0.1 收益的 96%（−0.65s / −0.68s）。4 卡 DAPO 没有读数。

## 配置

- **机器**：4090-1；4 卡 = GPU 0,1,4,5（**2+2 放置**，按 §13.42／§13.45：同 fabric 两两成对，SHM all-reduce 带宽最优）。窗口内 GPU 2 由 liyaowei embed 常驻（21728 MiB），其余 GPU 空闲；每臂启动前用 `nvidia-smi` 与 `ps` 核过无其他训练进程。
- **栈**：verl `0.10.0.dev`（原树 `/home/zxw/venvs/rollout-g8-src/verl`＝`24f25b03`；patched 副本 `/tmp/m5-poll-ab/verl`）＋ vLLM 0.24.0＋torch 2.11.0+cu129；`NCCL_P2P_DISABLE=1`、`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`。
- **任务配置**：与 2 卡 DAPO 主组逐项相同——`train_batch_size=32` prompts × `rollout.n=4`、`ppo_mini_batch_size=16`、micro 1、max_prompt 512／max_response 256、bf16、sdpa、`optimizer_offload=True`、`gpu_memory_utilization=0.20`、15 步；`algorithm.filter_groups.enable=True`、`metric=acc`、`max_inflight_gen_batches=2`。
- **变量**：`trainer.v1.sampler.poll_interval`，2.0（原树，默认）/ 0.5 / 0.1（patched 树）。**每臂两跑**（第二跑换名，第一跑归档不动）。

## 执行账（全部为 UTC，窗口 15:58–16:45）

| 运行 | 树／poll | 窗口 | 结果 |
|---|---|---|---|
| `g4base`（首轮） | 原树／2.0 | 15:03:56–15:11:23 | 15/15，exit 0；与并行会话重叠，归档保留 |
| `g4baseR` | 原树／2.0 | 15:59:06–16:06:23 | 15/15，exit 0 |
| `g4p010R` | patched／0.1 | 16:07:04–16:14:05 | 15/15，exit 0 |
| `g4p050R` | patched／0.5 | 16:14:56–16:21:55 | 15/15，exit 0 |
| `g4baseR2` | 原树／2.0 | 16:22:44–16:29:57 | 15/15，exit 0（重复） |
| `g4p010R2` | patched／0.1 | 16:30:18–16:37:24 | 15/15，exit 0（重复） |
| `g4p050R2` | patched／0.5 | 16:37:45–16:44:52 | 15/15，exit 0（重复） |

六跑由一个服务器端链式脚本串起（臂间等 GPU 回落到 ~1 MiB、清 `RAY_TMPDIR`），**6/6 全部 exit 0**，无 OOM、无 KV 配额错误。16:45 起 GPU 2/3 出现他人新作业（23724 MiB），在本批全部退出之后，不影响结果。

## 结果一：同窗口 A/B（稳态 steps 3–14，每臂两跑合并 n=24）

| 相位 | poll 2.0 | poll 0.5 | poll 0.1 | 0.5 vs 2.0 | 0.1 vs 2.0 |
|---|---|---|---|---|---|
| **gen** | **6.791s** | **5.822s** | **5.974s** | **−14.3%** | **−12.0%** |
| old_log_prob | 1.756 | 1.798 | 1.769 | +2.4% | +0.8% |
| ref | 1.326 | 1.342 | 1.337 | +1.2% | +0.8% |
| adv | 0.112 | 0.113 | 0.113 | +0.6% | +1.0% |
| update_actor | 6.212 | 6.309 | 6.250 | +1.6% | +0.6% |
| update_weights | 2.307 | 2.324 | 2.324 | +0.7% | +0.7% |
| **整步** | **18.588s** | **17.786s** | **17.846s** | **−4.3%** | **−4.0%** |

臂内 sd：gen 0.97／1.01／0.60s，整步 0.95／1.15／0.58s。相位和与整步残差 0.078–0.083s，与 2 卡一致（计时缺口）。

- **gen 以外的相位都在噪声内**（+0.6%–2.4%，跨跑同量级），改 poll 只动生成侧。
- **复现性带量子特征**：0.1 臂两跑 17.836／17.855（差 0.019s）、0.5 臂 17.304／18.269（差 0.97s）、2.0 臂 18.725／18.451（差 0.27s）。细轮询把跨跑抖动压掉，与 2 卡六跑的结论一致（0.1 臂两跑差 0.080s）。
- **答案 1**：−4.0%，与 2 卡的 −4.9% 同量级——**既不是"固定量迁移"预测的 −10%，也不是固定批的 −2.0%**；绝对收益随卡数缩小，相对收益靠 DAPO 更长的整步保住。
- **答案 2**：0.5 与 0.1 在 4 卡 DAPO 上**墙钟等价**（17.79 vs 17.85，差 0.06s，远小于 0.97s 的臂内跨跑差），复现了 2 卡的等价结论。原因见结果二：两者省下的等待差 0.28s，被这两跑多生成的 13.7 组抵消。

## 结果二：把 gen 差拆成"等"与"干"

模型：`gen_i = a_run + b · 本步生成组数`（各跑独立截距、共同斜率；另有以"组数×response_length"为工作量的 token 口径，两口径差 ≤0.05s）。截距差＝轮询量子贡献的**固定等待差**，斜率×工作量差＝多生成带来的**真实计算差**。区间由臂内按 step 重采样 bootstrap（n=2000）给出。脚本：[decompose-gen.py](decompose-gen.py)。

| 跑 | 截距 | 工作量（组/步） | 固定等待差 | 工作量差 | 残差 |
|---|---|---|---|---|---|
| `g4baseR` | 3.004 | 122.7 | —（控制） | — | — |
| `g4baseR2` | 2.854 | 120.5 | **−0.155 [−0.71, +0.38]** | −0.069 | 0 |
| `g4p050R` | 2.091 | 108.8 | −0.914 [−1.46, −0.40] | −0.439 | 0 |
| `g4p050R2` | 2.205 | 122.5 | −0.807 [−1.33, −0.30] | −0.005 | 0 |
| `g4p010R` | 1.937 | 128.7 | −1.067 [−1.61, −0.55] | +0.191 | 0 |
| `g4p010R2` | 1.789 | 130.2 | −1.215 [−1.74, −0.74] | +0.238 | 0 |

共同斜率 **31.8 ms/组**（token 口径 154.2 μs/token），残差 sd 0.529s。

- **负对照**：同配置两次重跑的截距差 −0.155s，**CI 跨 0**。0.1 与 0.5 两跑的 CI 都不含 0，效应在噪声之外。
- **读数一：等待收益不是"每步固定 2s 量子"，而是随量子近似线性。** 0.1 臂 −1.07／−1.22s（均值 −1.14s），0.5 臂 −0.91／−0.81s（均值 −0.86s）。两点定一条过原点的线：**每步平均等待 ≈ 0.59 × poll_interval**（0.59 = 1.14/1.9 = 0.86/1.5，两个独立点重合到 5%）。这直接否定了"4 卡应得 −10%"的固定量推理——2 卡的 −1.87s 对应斜率 0.98，**斜率本身随配置变**（2 卡 DAPO 每步的生成更长，落在更差的相位上）。
- **读数二：4 卡的多生成代价是 +7.8 组（+6.4%），约 1.4 SE，定不下来。** 逐 step 生成量 sd 15–25、每臂 n=24 ⇒ 均值 SE 3–5 组。2 卡六跑的 **+13.1%** 仍是目前唯一落在 ~2 SE 外的估计，PR 风险段应引用它，而不是本批的 +6.4%。

## 结果三：三套配方横向对照

| 配方 | 卡数 | 整步（2.0 → 短量子） | 相对 | gen Δ | 固定等待 | 多生成代价 |
|---|---|---|---|---|---|---|
| 固定批 | 2 | 21.686 → 21.001（0.1） | −3.2% | −0.80s | — | 0（集合与时机无关） |
| 固定批 | 4 | 14.216 → 13.934（0.1） | −2.0% | −0.40s | — | 0 |
| DAPO（inflight=2） | 2 | 26.534 → 25.246（0.1） | −4.9% | −1.19s | −1.87s | +0.71s（+13.1%） |
| DAPO（inflight=2） | 4 | 18.588 → 17.846（0.1） | −4.0% | −0.82s | −1.14s | +0.24s（+6.4%，1.4 SE） |

- 两套配方的**绝对**收益都随卡数下降（固定批 −0.68→−0.28s，DAPO −1.29→−0.74s），符合"每个派发轮等到的相位不因卡数变好"的直觉。
- DAPO 的**相对**收益两档都接近 −4~5%，因为它每步更长、且 refill 派发循环放大了等待面。
- 0.5 在 4 卡 DAPO 上拿到 0.1 的 **98%**（17.786 vs 17.846），与 2 卡 DAPO 一致；固定批 2 卡是 96%。**"0.5 不够"只对固定批 4 卡未测档成立，不是通例。**

## 首轮 4 卡基线的订正

首轮 `g4base`（18.518s）标为"上界"，理由是尾段与并行会话重叠。本批干净窗口两跑是 **18.725／18.451（臂均值 18.588）**，与 18.518 相差 +0.07s——**重叠成本在均值上测不出来（≤0.15s）**。首轮与重跑的真实差别在生成量：114.8 vs 121.6 组/步，按 31.8 ms/组折算 +0.22s，恰好覆盖 gen 的 +0.27s。因此：

- 首轮的 18.518s **不是被并发抬高的上界**，而是一次"生成量偏低"的正常跑；
- 4 卡 DAPO 默认配置的整步应报 **18.59 ± 0.14s**（两次重跑极差 0.27s）；
- 与 2 卡相比的 1.43× 缩放保持不变（26.534/18.588 = 1.43×）。

## 边界

- **单栈单点**：1.5B／gsm8k 短答案／单 seed／15 步／`n=4`／`inflight=2`；每臂两跑只能给 ±0.4s 级别的分辨力，**−4.0% 的幅度要带这个误差读**，方向由负对照与两个 CI 支撑。
- **0.5 臂跨跑差 0.97s**：它的墙钟与 0.1 打平是"少省 0.28s 等待、少生成 13.7 组"两项相抵的结果，属模型解释，不是直接测到的等价。
- **生成量代价未定**（+6.4%，1.4 SE）。若要判定 4 卡上投机是否更便宜，需要每臂 ≥4 跑。
- 只测 sync V1 trainer 与 `inflight=2`；`inflight=1`（DAPO 默认）的补测在 2 卡上完成（§13.48），4 卡未测。更高 in-flight、`ReplayBufferAsync.wait_for_sampleable`、stale 逐出均未覆盖（本窗口 `stale_uids` 恒空）。
- 相位和与整步始终差 0.078–0.083s（计时缺口），跨臂不变，不进入任何差值。

## 复现

```bash
# 服务器：原树 + patched 副本（TM-008 已建好）
MODE=g4baseR GPUS=0,1,4,5 bash run-poll-dapo.sh
MODE=g4p010R GPUS=0,1,4,5 POLL=0.1 SRC=/tmp/m5-poll-ab/verl bash run-poll-dapo.sh
MODE=g4p050R GPUS=0,1,4,5 POLL=0.5 SRC=/tmp/m5-poll-ab/verl bash run-poll-dapo.sh
# 第二跑把 MODE 加后缀 R2；链式脚本见下
# 本地
python3 make-summary-g4.py raw/ --group=main     # 首轮（写 summary.json）
python3 make-summary-g4.py raw/ --group=rerun    # 本批六跑（写 summary-rerun.json）
python3 decompose-gen.py summary-rerun.json
```

臂间清理与等待的链式写法（本批实际使用，服务器 `/tmp/m5-chain2.sh`）：

```bash
wait_idle() {  # 等 0,1,4,5 回到 ~1 MiB
  for i in $(seq 1 60); do
    busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
      | awk -F, '$1==0||$1==1||$1==4||$1==5 {gsub(/ MiB/,"",$2); if ($2+0>2000) c++} END{print c+0}')
    [ "$busy" = 0 ] && return 0; sleep 5
  done
}
```

## 操作教训（续首轮三条）

4. **臂间用服务器端链式脚本串起来，不要由会话逐臂 SSH 启动。** 本批 6 跑在同一窗口内 6/6 exit 0：每臂结束后等 GPU 回落、清 `RAY_TMPDIR`/checkpoint 目录，再起下一臂。首轮的三次失败全部来自"上一臂残留 + 别的会话恰好启动"。
5. **每臂跑两次，第二次就是负对照。** 同配置两跑的截距差 −0.155s（CI 跨 0）给出"什么都不改"时的分辨率；没有这个对照，−0.74s 的整步差很容易被单次运行的 0.97s 跨跑差吃掉。

## 2026-11-05 无卡轮：4 卡 `inflight=1` 的准备——base recipe 少了一个钉，臂会静默换配置

**为什么做这个。** TM-008 只剩 4 卡 `inflight=1` 一条没测，而它要 4 张空闲卡（本机最稀缺的资源）。这条臂是针对 verl `24f25b03` 准备、并在该树上跑到 6/6 exit 0 的；此后 main 已到 `31cfccb4`。开跑脚本 `run-poll-dapo.sh` **完全靠环境变量**驱动 `examples/grpo_trainer/run_qwen3_4b_fsdp.sh` 去拼命令行——环境变量被改名、recipe 少一条 override，结果都不是报错，而是**训练了另一套配置**。这类失败烧掉的是整个 4 卡窗口。

**一、真的少了一条：`enable_chunked_prefill`。** 逐行 diff 该 recipe（`24f25b03` vs `31cfccb4`）只有一处差异——`actor_rollout_ref.rollout.enable_chunked_prefill=False` 被删掉了；而它的默认值是 **True**（`verl/workers/config/rollout.py:291` 与 `_generated_ppo_trainer.yaml:287`，两个版本都相同）。⇒ 同一份 wrapper 在 main 上会把 `True` 交给引擎，而本目录归档的七跑全部是 `False`。

**二、这不是"vLLM V1 反正总开着"那类空转。** vLLM 0.24 的 `EngineArgs` 里该字段是 `bool | None = None`，`None` 才回落到 `is_chunked_prefill_supported`；显式 `False` 走另一条分支——`_set_default_chunked_prefill_and_prefix_caching_args` 打 `warning_once("This model does not officially support disabling chunked prefill…")`，调度器也不再切分 prefill。verl 侧唯一读该值的分支（`_validate_configs` 把 `max_num_batched_tokens` 抬到 `max_model_len`）在本臂上是**空转**：默认 8192 > `max_model_len` 2048，条件不成立。⇒ **在 verl 的配置面完全看不到变化**（回读 `max_num_batched_tokens` 两档都是 8192），变化只在引擎手里。

**三、钉回去，并把"钉还在不在"变成检查。** `run-poll-dapo.sh` 现在显式传 `False`（附原因），与归档批次逐条对齐。新增 [preflight-g4.py](preflight-g4.py) 在 CPU 上走完整条链：桩 `python3` 让 `run-poll-dapo.sh` → recipe **打印真实 argv**（override 表不在检查脚本里重抄，wrapper 是唯一来源），再用这份 argv 做 Hydra 组合并回读生效值。

| 读数（[raw/preflight-g4.json](raw/preflight-g4.json)） | 值 |
|---|---|
| 捕获到的 override | 58 条（recipe 47 + wrapper 11），尾部实参 10 条全在 |
| 与归档树 `24f25b03` 的 recipe 逐键比对 | **相同**（`drift.same=true`）。`n_old=59` 多的那条是"recipe 与 wrapper 都钉了 `False`"的重复出现，按 Hydra 末次生效合并 |
| 去掉这个钉再看（[raw/preflight-g4-nopin.json](raw/preflight-g4-nopin.json)） | 恰好少一条 `…enable_chunked_prefill=False`，main 上生效值 `true`；同一次检查的 `capture.problems` 直接点名 |
| poll 补丁对 `31cfccb4` | 3 个文件全部 `--fuzz=0` 通过（yaml／`trainer_base.py`／`trainer_separate_async.py`） |
| 两臂组合 | base／poll 均成功，18 个预登记键全中 |
| `trainer.v1.sampler.poll_interval` | 未打补丁树**不存在**（buffer 用构造默认 2.0，正确基线）；打补丁树回读 `0.1` |
| 负向对照 | 编造键 `algorithm.filter_groups.no_such_field` 如期组合失败 |
| 自检 | 7/7 |

**边界。** 本轮**零 GPU 读数**："生效值"是 Hydra 解析出来的配置，不是 vLLM 运行时的行为；vLLM 那段是从 tag `v0.24.0` 的 `vllm/engine/arg_utils.py` 与 `vllm/config/model.py` 源码读的，没有执行。本机只有 bash 3.2，recipe 的 `set -xeuo pipefail` ＋ 空 `EXTRA=()` 在 3.2 下是 unbound-variable 错误，本地用"只去掉 `-u`"的副本跑（服务器 bash ≥4 不需要），替换方式记进读数。另记一条对归档的提醒：按源码，生成式 decoder 模型的 `is_chunked_prefill_supported` 为真（`model.py:1790-1791`），所以本目录用 `False` 的七跑会命中 `arg_utils.py:2472-2480` 那条 `warning_once("This model does not officially support disabling chunked prefill…")`——**七跑各 15 步全部 exit 0**，但若将来决定改用默认 `True`，归档数值就不可直接对比，该臂必须在同一窗口内重取基线。

**本批三处自错（都已改）。** ① 本地适配写成 `set -xe pipefail`：`set` 把 `pipefail` 当成**位置参数**，`"$@"` 的 10 条尾部 override 被吃掉，捕获只剩 48 条——这正是"看起来在检查、其实换了对象"，现已是自检项 `eaten_tail_detected`；② 自检把注入打在 `rollout.n` 上，而受检键是 `actor_rollout_ref.rollout.n`，于是"值被改"这条判据**没被覆盖**却报绿；③ `+` 前缀检查拿 `key` 去比 `key=value` 列表，真实运行与自检双双击出误报。

## 2026-11-05 无卡轮（续）：窗口里读不到计数器，等于没跑——补丁修正 + 受测树守卫 + preflight 接入

**为什么做这个。** 这条臂存在的意义就是读 `training/sampler/blocking_polls`（判断一个训练步里有几次派发轮次）。上一轮做好了补丁，但 `run-poll-dapo.sh` 不检查树、preflight 不检查补丁，**未打补丁的树照样 exit 0**——4 张卡窗口最贵，跑完才发现没有计数器，问题一点没答。

**一、上一版补丁有一条不可达的登记（已删）。** `validation/sampler/blocking_polls` 永远不会出现在任何 metrics 里：全仓 `partition_id="val"` 只有 `trainer_base.py::_validate()` 一处，且写的是 `batch, _ = self.replay_buffer.sample(...)`，buffer 返回的 metrics 被丢弃。补丁从 +9 行降到 **+8 行**（`sha256=4c142752…`），对 `31cfccb4` 的 `replay_buffer.py`／`utils.py` `--fuzz=0` 通过；与 `poll-interval-config.patch` 的文件集不相交，应用顺序无关。

**二、上一版自检为什么没抓到：它数的是字符串。** 旧检查 ⑨ 是 `"training/..." in utils_source and "validation/..." in utils_source`——键写错规则、写进死路径都能过。现在把真实 `MetricsAggregator` 类源码 `exec` 出来喂两次迭代（2 与 3）：打补丁树得 **5.0**（sum），未打补丁树得 **2.5**（默认 weighted_avg）。另加一条不变量：**注册了 validation 键 ⇔ val 调用点保留 metrics**（现在两侧一真一假，成立；上游哪天改成记录 val metrics，这条会翻红）。

**三、async 路径也覆盖了。** `sim-loop.py` 改成 exec 两个类（`ReplayBuffer`、`ReplayBufferAsync`），新增 async 场景：`wait_for_sampleable()` 按次清零、`sample()` 原样转发 metric，3 次轮询读数 3。自检 **12/12**（打补丁树与未打补丁树各一遍），惰性仍是 **120 场景 × 4 步 = 480 行逐字段相同**。

**四、把"树必须是受测树"变成硬门槛。**

| 关卡 | 行为 | 反例 |
|---|---|---|
| [prep-tree.sh](prep-tree.sh) | 从干净树复制 → 两补丁 `--fuzz=0` → 自检 4 个标记；dst 已存在拒绝（`FORCE=1` 覆盖） | 目的树已存在 → exit 3 |
| `run-poll-dapo.sh` | `CHECK_TREE=1`（默认）检查 3 个标记，缺一即 **exit 65** 并打印补救命令；`CHECK_ONLY=1` 只查树 | 去掉 counter 标记 → 65；四标记齐 → 0 |
| `preflight-g4.py` | 新增 `--run-tree`（必填）与 `--sim-python`：树标记 + 在**受测树本体**上跑 sim 12/12 + wrapper 守卫正/负对照；已打补丁的树按 `already_applied` 接受 | 不在受测树上跑 sim 就不给 ok |

读数：[raw/preflight-g4-instrumented.json](raw/preflight-g4-instrumented.json) —— `ok=true`：58 条 override 不变、两臂组合 18 键全中、`poll_patch.usable`／`counter_patch.applies` 通过、受测树 4 标记齐、sim `12/12`、守卫 0/65 双通过；自检 **11/11**。计数器的树级验证读数：[sim-polls.json](../verl-v1-poll-quantum-polls/raw/sim-polls.json)。

**五、口径变化（写进下一次窗口的解读）。** ① 基线臂现在也跑**同一个受测副本**（只是不传 `--poll_interval`，回读默认 2.0），与归档的"原树基線"只差一条惰性补丁 ⇒ 窗口内 A/B 仍自洽，但不要把跨批次绝对值直接混用。② `separate_async` 会对同一键记两次（switch-wait 与 `sample()` 各一次 `wait_for_sampleable`），本臂（sync）不涉及。

**边界。** 零 GPU 读数；计数器仍未在服务器上执行过；`CHECK_TREE` 默认开启会挡住"在未打补丁树上重跑旧配置"，需要时显式 `CHECK_TREE=0`（preflight 的 argv 捕获就是这么做的，因为它只跑 recipe 不看运行树）。

## 2026-11-05 无卡轮（三）：窗口的**读数侧**补上了；顺带证明本页原来那条判据推不出来

**为什么做这个。** 上一轮把"树必须是受测树"做成了硬门槛，但产出侧还是空的：仓里没有任何东西从 console 日志里读 `training/sampler/blocking_polls`。日志是 1240 行、带 ANSI 与 `(TaskRunnerV1 pid=…)` 前缀，「这一步没有计数器」和「计数器等于 0」肉眼分不开。更麻烦的是下面「下一步」里那条判据——**预期 ≥2；若恒 1，斜率差不能归因轮数**——它从来没被推导过。本轮查了一遍：**按量化模型，墙钟给不出轮询次数**，所以这条判据不成立。

**一、解析器 [read-polls.py](read-polls.py)，自检 12/12。** 三件事只有它做：
- step 行必须**行首**匹配（`^(?:\([^)]*\))?step:N - `，且至少 4 个 ` - `）。记忆里的那个坑——`timing_s/step:8.2` 被读成 `step:8`——是 S3 负向对照，去掉锚点即红。
- 计数器**缺席报 `absent`、不报 0**；`--expect-counter` 直接 **exit 2**。窗口在未打补丁的树上跑完会死在这里，而不是交出一个"0 次轮询"的假读数。
- 同一步里取 `timing_s/gen`、`timing_s/step`、`training/filter_groups/{evicted_samples,discarded_surplus_samples}`，`work = evicted + surplus + 32`（32 是 train batch size，trainer 不打印，用 `--selected-groups` 补）。

另两条是"读到之后怎么算"：臂内回归 `gen ~ 1 + work + polls`（系数应 ≈ P）与跨臂恒等式（见第四节）。自检的 S5／S6 是拿植入的已知系数反查这两条。

**二、拿七个归档跑当场对账（不是自证）。** 六跑 + 首轮，`--steady 3:14`：

| poll 臂 | gen（本工具） | 页面归档值 | 整步（本工具） | 归档 | work（本工具） | 归档分解表 |
|---|---|---|---|---|---|---|
| 2.0 | **6.791** | 6.791 | **18.588** | 18.588 | **122.7／120.5** | 122.7／120.5 |
| 0.5 | **5.822** | 5.822 | **17.786** | 17.787 | **108.8／122.5** | 108.8／122.5 |
| 0.1 | **5.974** | 5.974 | **17.846** | 17.846 | **128.7／130.2** | 128.7／130.2 |

八项读数逐位相同 ⇒ 解析路径与独立的 `make-summary-g4.py`／`decompose-gen.py` 一致。**七跑全部 `counter absent`**（它们跑在原树上），这是真实数据上的负向对照：对这七个日志加 `--expect-counter` 会 exit 2。读数：[raw/read-polls-archive.json](raw/read-polls-archive.json)。

**三、本页原来的「预期 ≥2」为什么推不出来。** 量化模型（§13.50）是 `wait = ceil(D/P)·P − D`，`D` 是"这批数据什么时候齐"。这个量**永远落在 `[0, P)`**，与 `ceil` 了几次无关：

- `D = 0.4 s`、`P = 2.0` ⇒ 1 次轮询、等待 1.6 s；
- `D = 10.4 s`、`P = 2.0` ⇒ 6 次轮询、等待同样 **1.6 s**。

两者 A/B 省下的时间一模一样。所以归档测到的 1.07–1.14 s 只是 `E[P − (D mod P)]`（4 卡 0.53–0.57 个格；相位均匀时 0.5，吻合），**它不含任何关于 n 的信息**。⇒「预期 ≥2」没有依据；「若恒 1，斜率差不能归因轮数」同样没有依据——n=1 与 n=6 可以有相同斜率。**这是必须上计数器的理由，本轮把它从"断言"变成"证明"。**

**四、窗口里真正能检验的两条（预登记）。**

| 判据 | 形式 | 失败含义 |
|---|---|---|
| **单位**：一次轮询 = 一次 P 秒 sleep | 臂内回归 `gen ~ 1 + work + polls`，`poll` 系数 = P ± se | 系数不 ≈ P ⇒ 计数器数的不是 `_wait_for_next_poll`，`inflight=1` 的读数不能拿来解释 −4.0% |
| **两路一致**：墙钟 vs 计数器 | `gen(2.0) − gen(0.1) = 2.0·n̄₂ − 0.1·n̄₀.₁ + b·Δwork`（`b = 31.8 ms/组`），残差 ≈ 0 | 残差超噪声 ⇒ 两路测的不是同一段等待 |

**校准臂（新增，值一跑）。** `INFLIGHT=2`、poll 保持默认 2.0：这个配置的 gen 已有归档值 **6.791 s**，是唯一能在"墙钟已知"前提下先验计数器的臂。两条判据都过，`inflight=1` 的读数才算数；不过就先停下，别拿它解释斜率差。

**五、口径。** ① `--expect-counter` 只挡"树没打补丁"，挡不住"计数器数错次数"——后者只能靠第四节的回归。② `work` 含常数额 32，别拿去和别的批次比绝对组数（斜率不受影响）。③ 跨臂恒等式的参考臂可用归档的 0.1 臂，但跨批次截距噪声有 −0.155 s（同配置两跑之差），相对 1.1 s 的效应可接受，窗口内自取更好。

**边界。** 零 GPU 读数；`read-polls.py` 在归档上验的是**解析**，不是计数器本身——计数器依然从未在 GPU 上执行过；第四节的判据是预登记，本轮无法执行。

## 材料

| 文件 | 内容 |
|---|---|
| `raw/rollout-poll-dapo-g4base.txt` | 首轮 4 卡默认 2.0 基线（15 步，exit 0，含与新作业的重叠） |
| `raw/rollout-poll-dapo-g4{baseR,baseR2,p010R,p010R2,p050R,p050R2}.txt` | 本批六跑训练日志（各 15 步，exit 0）；`.txt` 后缀避开 `.gitignore` 的 `*.log` |
| `summary.json` | 首轮解析（`--group=main`，只有 `g4base`）；本次因 parser 新增 `phase_values` 字段重写过，数值逐项未变 |
| `summary-rerun.json` | 六跑 + 臂级合并（`--group=rerun`） |
| `make-summary-g4.py` | 解析与汇总（`--group=main\|rerun`，同臂重复自动合并） |
| `decompose-gen.py` | 等待/工作量拆分（共同斜率 + bootstrap CI） |
| `run-poll-dapo.sh` | 运行脚本（`GPUS`／`MODE`／`POLL`／`SRC`／`INFLIGHT` 可覆盖；`CHECK_TREE=1` 守卫，`CHECK_ONLY=1` 只查树） |
| `prep-tree.sh` | 生成受测副本：复制干净树 → 打两个补丁 → 自检 4 标记（`FORCE=1` 覆盖已存在目录） |
| `preflight-g4.py` | 开窗前检查：捕获真实 argv（桩 `python3`）→ 逐键对齐归档树 → Hydra 组合两臂并回读 → 负向对照 → 受测树标记 + sim + 守卫正/负对照；`--selftest` 11/11 |
| `read-polls.py` | 窗口读数侧：解析 console 日志里的 `training/sampler/blocking_polls`／`timing_s/gen`／生成组数；`--expect-counter` 对未打补丁的树 exit 2；臂内回归与跨臂恒等式；`--selftest` 12/12 |
| `raw/read-polls-archive.json` | 七个归档跑的解析读数（与本页归档值逐位相同；七跑均 `counter absent`） |
| `raw/preflight-g4.json` | 上一轮读数（`31cfccb4`，含与 `24f25b03` recipe 的逐键比对） |
| `raw/preflight-g4-instrumented.json` | 本轮读数：`--run-tree` 上的标记／sim 12/12／守卫 0 与 65 双通过 |
| `raw/preflight-g4-nopin.json` | 对照：去掉 `enable_chunked_prefill` 这个钉时的漂移与 main 上生效值 `true` |
| `sha256sums.txt` | 本目录文件的 sha256；六份原始日志与服务器逐文件一致 |

## 下一步

- 4 卡 DAPO 的 A/B 已闭环。未测只剩 **4 卡 `inflight=1`**（需 ≥4 张空闲卡）。开窗前流程（任一检查失败都会在碰 GPU 前退出）：
  ```bash
  bash prep-tree.sh /home/zxw/venvs/rollout-g8-src/verl /tmp/m5-g4i1/verl
  /tmp/hydra-probe-venv/bin/python3 preflight-g4.py --tree <干净树> --run-tree /tmp/m5-g4i1/verl \
      --sim-python /home/zxw/venvs/rollout-g8/bin/python3 --compare-script <归档树同名脚本> --out raw/preflight-g4.json
  # 校准臂（墙钟已知：gen 应复现 6.791s）——先验计数器
  MODE=g4cal GPUS=0,1,4,5 INFLIGHT=2 SRC=/tmp/m5-g4i1/verl bash run-poll-dapo.sh
  # 正题臂
  MODE=g4i1 GPUS=0,1,4,5 INFLIGHT=1 SRC=/tmp/m5-g4i1/verl bash run-poll-dapo.sh
  # 读数：缺席即 exit 2；两条判据见上一节第四节
  python3 read-polls.py --log /tmp/rollout-poll-dapo-g4cal.log=2.0 \
      --log /tmp/rollout-poll-dapo-g4i1.log=2.0 --expect-counter --json raw/read-polls-window.json
  ```
  要跑跨臂恒等式，再补一个 `MODE=g4i1p010 POLL=0.1 INFLIGHT=1` 的细轮询臂（同一窗口，约 7–8 分钟/跑）。**读数的解释规则已换成「臂内回归系数 ≈ P + 跨臂恒等式残差 ≈ 0」；原来的「预期 ≥2／若恒 1 则不能归因轮数」已作废**（理由见上一节第三节）。若要把生成量代价也定下来，需每臂 ≥4 跑（本批分辨力不够），判据是看 +6.4% 能否越过 2 SE。
- TM-008 的 PR 风险段用 2 卡 +13.1% 的生成量代价与 4 卡 −4.0% 的整步收益，并注明"等待收益 ≈0.59×量子、随配置变"。**对外动作仍未做，待用户逐项授权。**
