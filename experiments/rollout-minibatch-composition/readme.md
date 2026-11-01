# mini-batch 组成代价，与四个框架的归约语义核验

**对应**：[rollout 性能调研主文档 §13.32](../../notebook/rollout-performance-research.md)，接续 [§13.31](../rollout-group-split-training-side/readme.md)。
**日期**：2026-11-02（M1）。**状态**：已完成；20 个单测通过、`--verify` 连跑两次通过。

## 一句话结论

**训练侧拆组的第二类代价（mini-batch 组成）确实存在，但只有个位数百分比；而"把一步拆成 K 次 optimizer step"本身会把梯度噪声放大约 58 倍（K 从 1 到 8），这一项跟拆不拆组无关。** 同时核验了 OpenRLHF／SkyRL／NeMo RL 三个框架：**优势的 mean/std 全部按 uid/prompt 分组，没有任何一个框架"按分到的样本块各自归一化"**——所以 §13.30 第一批的 0.69／0.41 倍仍然只是假设口径，§13.31 的撤回可以推广到这四个实现。

## 两件事、为什么这么做

§13.31 把三个统计候选在 verl 默认路径上量成零，留下两个"未量化就不写成结论"的缺口：

1. **mini-batch 组成**：同组样本被拆进不同 mini-batch 后，每个 mini-batch 的梯度噪声组成会变。
2. **跨框架归约语义**：如果某个框架是"在拿到的样本块上各自归一化"，那么 §13.30 的 0.69／0.41 倍就从假设变成真实，§13.31 的撤回只对 verl 成立。

顺序是**先核语义、再量代价**：若没有框架改口径，mini-batch 组成就是剩下唯一可能的代价位置，值得实测。

## 第一部分：四个框架的归约语义（只读源码）

| 框架 | commit | 优势 mean/std 的分组 | loss 归一化 | 判定 |
|---|---|---|---|---|
| verl | `1a9e711e` | `index=uid`（标量版逐 index 收集，向量化版走 `groupwise.group_mean_std`） | 全局 token / seq 后 DP `all_reduce(AVG)` | 按组 |
| OpenRLHF | `dc2a7ad3` | `rewards.reshape(-1, n_samples)` 后按**行**算 | DeepSpeed `train_batch_size` 乘 ring_attn×tp | 按 prompt |
| SkyRL | `40606751` | `id2score[index[i]]` 分组累加 | 全局 token / seq | 按 prompt |
| NeMo RL | `6e68b179` | `torch.unique(prompts)` 分组 | `global_valid_toks` / `global_valid_seqs` | 按 prompt |

三处"看起来像按块"的地方必须单独核，否则会被读成反例：

- OpenRLHF `experience_maker.py:324-339`：整个 batch 的优势再整体标准化 —— **跨组整体缩放**。
- NeMo RL `advantage_estimator.py:341-344`、`373-376`：全 batch adv 再标准化 —— 同上。
- SkyRL `worker_utils.py:28-36`：minibatch 间 rollout/train logprob 差的矩用可加量（`mean`／`sq_mean`）跨 micro-batch 归约 —— 是**诊断度量**，不是优势口径。这一处的做法容易让人误以为优势也按块算。

每条结论都有 anchor 片段，`source-snapshots/` 存了对应文件的离线副本，`--verify`／单测会重新核验片段仍在（上游改了会明确报 anchor 失效，而不是静默沿用旧结论）。

## 第二部分：mini-batch 组成代价

### 真实条件（只读核对）

- `_balance_batch` 之后**不重排**：actor 迭代用 `shuffle=actor.shuffle`，而 `actor.yaml:122` 默认 `shuffle: false` —— mini-batch 就是重排后数组的**连续切片**。
- 每个 mini-batch 的 loss 除以**全局**归一化因子（全局 token 数 / 全局样本数），DP 上再 `all_reduce(AVG)`。
- 因此只有一条路径能让组成影响更新：**优化器步**。若训练循环是"每个 mini-batch 各做一次 `backward` + `step()`"，一步 rollout 做 K 次参数更新。

### 读数一：噪声随块数 K

ratio = Σ_j Var(v_j) / Var(v̄_full)，K=1 时理想值 1.0（实际 0.0039 是"块方差估计 / 全批方差估计"的有限样本值，两端用同一估计量，比值才是可比的）。

| K | 连续切片 | 随机切 | 相对 K=1 |
|---|---|---|---|
| 1 | 0.0039 | 0.0039 | 1.00 |
| 2 | 0.0154 | 0.0154 | 3.93 |
| 4 | 0.0599 | 0.0599 | 15.31 |
| 8 | 0.2265 | 0.2265 | 57.97 |

连续切片与随机切**逐位相同**：LDM 重排后顺序已近似随机，再打乱不改变块的组成分布（一个应当如此、且确实如此的一致性检查）。

### 读数二：扣掉 K 之后，剩多少是"组成"造成的

同一批样本、同一个 K=4，只换块的组成（dp=4，30 种子配对）：

| 比较 | 比值 | 标准误 |
|---|---|---|
| 真实 LDM 顺序 ÷ 组内连续排列 | **1.0508** | 0.0033 |
| 交错排列 ÷ 真实 LDM 顺序 | **1.0000** | 0.0001 |
| 交错排列 ÷ 组内连续排列 | 1.0509 | — |

三条读数合起来解释了一个反直觉的事实：**真实路径的组成已经接近最坏情形**。

- 逐条 LDM 按序列长度排序后分片，重排结果近似把同组样本**均匀摊到各块**。实测：dp=2/4/8 下同组样本在重排后数组里"近距离相邻"的对数占比只有 36% / 22% / 12%。
- K=4 时每组被摊到全部 4 块（组纯度 0，每组平均跨 4 块），这与"交错排列"的组成结构一致，所以交错基线**没有**额外噪声可加。
- 换成"组内连续排列"（每组落进一个块）后噪声下降 4.8% ——这正是"块是组的随机划分"相对"组被打散"的差异。

### 自查修正（三处，都是跑通之后才发现的）

1. **估计量用错分母**。初版用 `1/n` 的极大似然方差，结果 K=2/4/8 的读数几乎不随 K 变（0.0077/0.0150/0.0283，远小于理论上界翻倍）。原因是"同组完全不相交"的切法让每块组内方差估成 0，K>1 的读数凭空消失。改用无偏的 `1/(n−1)` 后与解析值 `1 + 2ρ/(k_g−1)` 对上；单测 `test_must_use_n_minus_one` 守住。
2. **识别性阈值用错量纲**。初版用绝对阈值 1e-9，而噪声比本身量级只有 1e-2，于是 ±1e-4 的漂移也被判成"敏感"。改为相对极差 > 1%，并加了一个"恒定微扰"的对照测试（变化只有 1e-6 时必须判零敏感）。
3. **错误地把"块均值余弦"当成独立性假设**。块是连续切片时，块期望之间本来就不独立（组的排列有结构），余弦读数不构成对 K 次叠加的质疑；而且这块被负向对照证明是探查不到人为植入的块间共享方向的。改为用**显式的矩分解**替代：块内样本方差 / 块均值方差 = 64，组间方差不决定块均值的下限，残差余弦 0.53 —— 说明"块均值沿块内组均值分布 + 块内样本方差/n"这个分解站得住。

## 第三部分：识别性检验

把"组信号强度 ρ"从 0 拉到 0.9 五档，看指标动不动（相对极差 > 1% 才算有识别性）：

| 指标 | 相对极差 | 敏感 | 说明 |
|---|---|---|---|
| 噪声比（K=4） | 10.0% | 是 | 组成效应与组结构有关 |
| 组成比值（交错÷连续，dp=4） | 10.6% | 是 | 同上 |
| 对照：组均值余弦 vs ρ | 8.3% | 是 | 检验不能太严（真信号要能过） |
| 对照：恒常数指标 | 0 | **否** | 检验不能太松（无效指标要被打死） |

另有一条"指标非退化"冒烟检查：K 从 1 到 8 时噪声比必须变化（实测 0.0039 → 0.2265）。

## 边界（不要读出这些）

- **不是实测**：梯度是随机方向模型（`g = √ρ·组方向 + √(1−ρ)·样本方向`），不含真实 token 梯度结构。结论是**聚合语义的推论**，不是对真实训练的测量。
- **不支持"拆组导致噪声放大"**：放大由块数 K 决定；组完整性只决定块的组成，而组成效应只有个位数百分比。
- **不报耗时、不换算任何论文的墙钟读数**；也不主张"某个框架应该/不应该拆组"。
- 三个非 verl 快照通过 GitHub contents API 取文件，**没有克隆仓库**，无法 `git rev-parse` 复核；commit 与取文件时的 `ref` 一致，局限写在 `source-snapshots/README.md`。
- 组成效应的量级依赖 ρ 与 K 的具体取值；ρ 是模型旋钮不是实测参数，正文报的是"在 ρ=0.5、K=4 下"的读数。

## 复现

```bash
python3 train_side_composition.py               # 约 3 s，覆盖 summary.json
python3 train_side_composition.py --verify      # 重跑并与归档逐字段比对（连跑两次通过）
python3 -m unittest test_train_side_composition -v   # 20 个测试
```

执行环境：Python 3.9.6（macOS arm64），仅标准库。LDM 划分复用
[`../rollout-group-split-tradeoff/`](../rollout-group-split-tradeoff/readme.md) 的实现
（已与真实 verl 在 180 组输入上交叉验证分区一致），不在本目录复制第二份。
产物 sha256 见 [run-info.json](run-info.json)。

## 源码锚点

| 文件:行 | 作用 |
|---|---|
| `verl/trainer/ppo/core_algos.py:308-321` | GRPO 按 `index=uid` 分组算 mean/std |
| `verl/utils/groupwise.py:164` | 向量化版 `group_mean_std` 同样按 uid |
| `verl/workers/engine_workers.py:230` | DP 组 loss `all_reduce(AVG)` |
| `verl/trainer/ppo/ray_trainer.py:1356`、`verl/trainer/config/actor/actor.yaml:122` | actor 迭代的 `shuffle` 默认 `false` ⇒ mini-batch 是连续切片 |
| `openrlhf/trainer/ppo_utils/experience_maker.py:252-289` | reshape 成 (prompt, n) 后按行归一化 |
| `openrlhf/trainer/ppo_utils/experience_maker.py:324-339` | 全 batch 优势整体标准化（跨组，非按块） |
| `skyrl/backends/skyrl_train/utils/ppo_utils.py:1411-1460` | `id2score[index[i]]` 分组 |
| `skyrl/backends/skyrl_train/utils/worker_utils.py:28-36` | 矩按可加量跨 micro-batch 归约（诊断量） |
| `nemo_rl/algorithms/utils.py:118-232` | `torch.unique(prompts)` 分组 |
| `nemo_rl/algorithms/advantage_estimator.py:341-344`、`373-376` | 全 batch adv 整体标准化 |
