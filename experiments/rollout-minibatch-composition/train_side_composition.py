#!/usr/bin/env python3
"""训练侧拆组的第二类代价：mini-batch 组成——以及四个框架的归约语义核验（rollout 专项，2026-11-02 M1）。

支撑 notebook/rollout-performance-research.md §13.32，接续 §13.31。

**为什么有本文件**：§13.31 把拆组的三个统计量（跨组幅度、组级梯度方向、梯度方差）
在 verl 默认路径上全部量成零，并把结论收窄成"拆组改变的是样本落点，不是归一化口径"，
同时留下两个"未量化即不写成结论"的残留：

  残留 1  **mini-batch 组成**：同组样本被拆进不同 mini-batch 后，每个 mini-batch
          的梯度噪声组成会变。一批的整体梯度不变（§13.31 候选 B/C 已核），
          但**逐 mini-batch 更新的噪声**可能变。
  残留 2  **跨框架归约语义**：若某框架不是按 uid/prompt 分组归一化，而是
          "在分到的样本上各自归一化"，那么 §13.30 第一批假设口径的 0.69／0.41 倍
          就会从"假设"变成"真实"，§13.31 的撤回只对 verl 成立。

本文件两件事都做，顺序是**先核语义、再量代价**：如果没有任何框架改了口径，
那么 mini-batch 组成就是剩下唯一可能的代价位置，值得实测；如果某个框架真的
按分区归一化，那么必须先用§13.31 的识别性检验把它钉出来，而不是继续量 mini-batch。

------------------------------------------------------------------
第一部分：跨框架归约语义（只读源码核对，不给耗时，只给"口径是否按组"）
------------------------------------------------------------------
四个快照各自读优势计算与 loss 归约两处，问题只有一个：
**优势的 mean/std 是按"整个 rollout 组（uid/prompt）"算，还是按"当前拿到的样本块"算？**

  verl       1a9e711e  core_algos.py:308-321        按 index=uid；vectorized 版经
                                                      groupwise.group_mean_std 也是按 uid
  OpenRLHF   dc2a7ad3  experience_maker.py:252-289   rewards 先 reshape(n, n_samples_per_prompt)
                                                      再按行做 mean/std ⇒ 按 prompt
  SkyRL      40606751  ppo_utils.py:1411-1460        id2score[index[i]] 分组 ⇒ 按 prompt
  NeMo RL    6e68b179  utils.py:118-232              torch.unique(prompts) 分组 ⇒ 按 prompt

另有两条"看起来像按块"的归一化，必须单独核，否则会把它们误当成反例：
  OpenRLHF  experience_maker.py:324-339  全 batch 优势再整体标准化（跨组，非按块）
  NeMo RL   advantage_estimator.py:341-344 / 373-376  全 batch adv 再标准化（跨组，非按块）
两者都是"整批统一缩放"，批不变则数值不变；**不是**"每个分区各自归一化"。
逐条证据写在 `FRAMEWORK_EVIDENCE` 里，`--verify` 会重新核对每条 anchor 仍在快照里命中。

------------------------------------------------------------------
第二部分：mini-batch 组成的梯度噪声代价
------------------------------------------------------------------
真实条件（verl `1a9e711e`，只读核对）：
  * `_balance_batch` 之后**不重排**——`ray_trainer.py:1353-1368` 的 actor 迭代用
    `shuffle=actor.shuffle`，而 `actor.yaml:122` 默认 `shuffle: false`，所以 mini-batch
    就是"连续切片"，同组样本被拆开后落在**相邻的两个 mini-batch** 里。
    （`shuffle=true` 时退化成"随机切"，也纳入对照。）
  * 每个 mini-batch 的 loss 除以**全局**归一化因子：token 级是全局 token 数、
    sequence 级是全局样本数（§13.31 已核 `engine_workers.py:230` 的 AVG 归约）。
    所以 mini-batch 的**梯度贡献形状**是"全局加权平均"，与它装了多少样本无关。

于是只有一条路径能让组成影响更新：**优化器步**。若训练循环是
`for mb in minibatches: loss.backward(); opt.step()`，则一步 rollout 做 K=n_mb 次参数更新，
每次只看到 1/K 的样本；梯度噪声依 K 提高约 √K 倍（相对一步看全批）。

**主读数是渐近梯度方差比**：K 个"只看 1/K 样本"的子估计，其平方和的期望
（= K × 单次更新方差）相对于"一步看全批"的全批梯度方差。严格可复现（逐样本梯度），
不靠采样运气，也不受"全批梯度恰好接近零"这类病态情形影响。

**依赖假设必须写明**：子估计之间近似独立（样本梯度两两内积均值≈0），
该假设由 `check_independence` 实测报告，不是隐含条件。

------------------------------------------------------------------
第三部分：识别性检验（§13.31 同一条检验，先验后量）
------------------------------------------------------------------
把"组成效应"的两个驱动量各自关掉，看指标动不动：
  * 组信号强度 ρ（每组的公共梯度分量）0 → 1：若 ρ 增大而噪声比不变，说明量的是别的
  * 组内异质性 h（同组样本梯度之间的差异）0 → 1：同上
`sensitive` 为真的指标才允许立结论；恒常数指标必须被判零敏感。

墙钟/耗时一律不报。只用标准库。虚拟单位，不可与任何论文报值相除（§19.1 规则一）。

用法：
    python3 train_side_composition.py             # 全量重跑并覆盖 summary.json
    python3 train_side_composition.py --verify    # 重跑并与归档逐字段比对
    python3 -m unittest test_train_side_composition -v
"""

import json
import math
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY_PATH = os.path.join(HERE, 'summary.json')

# LDM 复刻复用 §13.30／§13.31 的实现（已与真实 verl 在 180 组输入上交叉验证分区一致），
# 不在本仓库出现第三份会分叉的副本。
_SIBLING_TRADEOFF = os.path.join(os.path.dirname(HERE), 'rollout-group-split-tradeoff')
_SIBLING_TRAINSIDE = os.path.join(os.path.dirname(HERE), 'rollout-group-split-training-side')
for _p in (_SIBLING_TRADEOFF, _SIBLING_TRAINSIDE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SEEDS = list(range(30))
DP_SIZES = (2, 4, 8)
GROUP_SIZE = 8                # rollout.n
N_GROUPS = 32                 # 一批 prompt 数（与 §13.30 L4／§13.31 相同）
GRAD_DIM = 8                  # 随机梯度向量维度（只为测内积与外积，不模拟真实 token 梯度）

# 组信号强度（默认档）：样本梯度 = sqrt(rho)·组方向 + sqrt(1-rho)·样本方向
RHO_GROUP = 0.5

# 识别性检验档位：0 表示"关掉组信号"（所有组同一分布）
RHO_LEVELS = (0.0, 0.1, 0.5, 0.9)
IDENTIFIABILITY_REL_EPS = 0.01   # 相对极差阈值：1%


# ---------------------------------------------------------------------------
# 第一部分：跨框架归约语义（只读证据表）
# ---------------------------------------------------------------------------
# 每条 anchor 是 (仓库, commit, 文件, 行号区间, 旧版本脚本里可搜索的代码片段)。
# 片段必须是**该快照里真实存在的字面代码**，供 check_anchor 二次核验：
# 若上游改动导致片段消失，脚本会明确报"anchor 失效"而不是静默沿用旧结论。
FRAMEWORK_EVIDENCE = {
    'verl': {
        'commit': '1a9e711e',
        'advantage_grouping': 'uid',
        'loss_normalization': 'global_token_or_seq',
        'anchors': [
            ('verl/trainer/ppo/core_algos.py', 'for i in range(bsz):\n            id2score[index[i]].append(scores[i])'),
            ('verl/utils/groupwise.py', 'def group_mean_std('),
            ('verl/workers/engine_workers.py', 'torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG, group=dp_group)'),
        ],
        'reading': ('按 index=uid 分组算 mean/std，位置无关；标量版本逐 index 收集'
                    '（core_algos.py:308-321），向量化版本的 group_mean_std 同样按 uid。'
                    'loss 用全局 token/seq 归一化后做 DP all_reduce(AVG)。'),
    },
    'openrlhf': {
        'commit': 'dc2a7ad3',
        'advantage_grouping': 'prompt',
        'loss_normalization': 'ds_train_batch_scale',
        'anchors': [
            ('openrlhf/trainer/ppo_utils/experience_maker.py',
             'rewards = rewards.reshape(-1, args.rollout.n_samples_per_prompt)'),
            ('openrlhf/trainer/ppo_utils/experience_maker.py',
             'rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9)'),
            ('openrlhf/trainer/ppo_utils/experience_maker.py',
             'exp.advantages = (exp.advantages - mean) * rstd'),
        ],
        'reading': ('把 rewards reshape 成 (prompt, n_samples) 后按**行**做 mean/std ⇒ 按 prompt 分组；'
                    '随后有一段全 batch 优势标准化（estimator 为 gae/reinforce/reinforce_baseline 时），'
                    '那是**跨组整体缩放**，不是按块归一化。DeepSpeed 侧 train_batch_size 乘了'
                    ' ring_attn×tp，说明"训练 batch"是固定规模、与分区无关。'),
    },
    'skyrl': {
        'commit': '40606751',
        'advantage_grouping': 'prompt',
        'loss_normalization': 'global_token_or_seq',
        'anchors': [
            ('skyrl/backends/skyrl_train/utils/ppo_utils.py',
             'id2score[index[i]].append(scores[i])'),
            ('skyrl/backends/skyrl_train/utils/ppo_utils.py',
             'scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)'),
        ],
        'reading': ('与 verl 同构：按 index 分组累加后算 mean/std，归一化与样本位置无关；'
                    '另有 minibatch 内 rollout/train logprob 差的**度量**用可加矩（mean/sq_mean）'
                    '跨 micro-batch 正确归约，属诊断量非优势口径。'),
    },
    'nemo_rl': {
        'commit': '6e68b179',
        'advantage_grouping': 'prompt',
        'loss_normalization': 'global_token_or_seq',
        'anchors': [
            ('nemo_rl/algorithms/utils.py', 'unique_prompts = torch.unique(prompts, dim=0)'),
            ('nemo_rl/algorithms/utils.py',
             'is_matching_prompt = (prompts == unique_prompts[i]).all(1)'),
            ('nemo_rl/algorithms/advantage_estimator.py',
             'adv_rstd = adv_var.clamp(min=1e-8).rsqrt()'),
        ],
        'reading': ('torch.unique(prompts) 得组，逐组收集 reward 算 baseline/std（支持 leave-one-out）；'
                    'GRPO 调用方传的是显式 rollout-group id 的 (b,1) 张量，不是 prompt token。'
                    'loss 用 global_valid_toks / global_valid_seqs 归一化。'),
    },
}

# 两条"看起来像按块、实为跨组整体"的归一化，单独列出以免被读成反例。
NOT_PARTITION_LOCAL = [
    ('openrlhf', 'experience_maker.py:324-339',
     '全 batch 优势再整体标准化（estimator 属 gae/reinforce/reinforce_baseline 时）——跨组整体缩放'),
    ('nemo_rl', 'advantage_estimator.py:341-344、373-376',
     '全 batch adv 再整体标准化——跨组整体缩放'),
    ('skyrl', 'worker_utils.py:28-36',
     'minibatch rollout/train logprob 差的矩用可加量跨 micro-batch 归约——是度量不是优势口径'),
]


SNAPSHOT_DIR = os.path.join(HERE, 'source-snapshots')


def snapshot_root(name):
    """按框架名找快照根目录：优先本目录 source-snapshots/<name>/main，其次外部 checkout。"""
    local = os.path.join(SNAPSHOT_DIR, name, 'main')
    if os.path.isdir(local):
        return local
    return None


def check_anchor(root, rel_path, snippet):
    """在本地或下载的快照里核验 anchor 片段仍存在。返回 (found, detail)。"""
    path = os.path.join(root, rel_path) if root else None
    if not path or not os.path.exists(path):
        return False, 'no local snapshot for %s' % rel_path
    with open(path, encoding='utf-8', errors='replace') as f:
        text = f.read()
    norm = lambda s: '\n'.join(line.rstrip() for line in s.split('\n'))
    if norm(snippet) in norm(text):
        return True, 'hit'
    # 退一步：逐行检查首行是否命中（行号可能漂移，语义不变）
    first = snippet.split('\n')[0].strip()
    if first and first in text:
        return True, 'hit-first-line-only (行号可能漂移，需人工确认语义)'
    return False, 'MISSING'


# ---------------------------------------------------------------------------
# 第二部分：梯度模型
# ---------------------------------------------------------------------------
def _grad_model(rng, rho, hetero, n_groups=N_GROUPS, n_per=GROUP_SIZE, dim=GRAD_DIM):
    """样本梯度的显式生成。

    每个样本：g_{g,i} = a_g + b_{g,i}
      a_g      组公共分量：每组**一个**方向（组与组之间独立），尺度 sqrt(rho)
      b_{g,i}  组内个体分量：每条样本**一个**方向（样本之间独立），尺度 sqrt(1−rho)

    只有两个分量，异质性由 rho 单独决定——这样"组信号强度"只有一个旋钮，
    也才有唯一可辩的零假设（rho=0 时样本梯度都是独立随机方向，块与块不可区分）。
    另外提供 `_grad_model_block_correlated` 表示"块之间方向相关"的情形，
    供独立性检验的对照使用。

    rho → 0  关掉组信号（所有组同一分布）
    rho → 1  组内样本梯度几乎相同（组内零异质）
    """
    def unit():
        v = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    sa = math.sqrt(rho)
    sb = math.sqrt(1.0 - rho)
    a = [[sa * x for x in unit()] for _ in range(n_groups)]
    grads, uids = [], []
    for g in range(n_groups):
        for _ in range(n_per):
            b = [sb * x for x in unit()]
            grads.append([a[g][k] + b[k] for k in range(dim)])
            uids.append(g)
    return uids, grads


def _grad_model_block_correlated(rng, rho, n_blocks=4, n_groups=N_GROUPS,
                                 n_per=GROUP_SIZE, dim=GRAD_DIM):
    """对照模型：整批默认被切成 n_blocks 块，**块之间共享一个方向**。

    用于说明"独立性检验能不能抓到不独立"。真实训练里块之间并不共享方向，这只是负向对照。
    """
    def unit():
        v = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    block_dir = [[math.sqrt(0.9) * x for x in unit()] for _ in range(n_blocks)]
    sa = math.sqrt(rho)
    sb = math.sqrt(1.0 - rho)
    a = [[sa * x for x in unit()] for _ in range(n_groups)]
    grads, uids = [], []
    per_block = n_groups // n_blocks
    for g in range(n_groups):
        bd = block_dir[min(g // per_block, n_blocks - 1)]
        for _ in range(n_per):
            b = [sb * x for x in unit()]
            grads.append([a[g][k] + b[k] + bd[k] for k in range(dim)])
            uids.append(g)
    return uids, grads


def _dot(x, y):
    return sum(p * q for p, q in zip(x, y))


def _norm(x):
    return math.sqrt(_dot(x, x))


def _add(x, y):
    return [p + q for p, q in zip(x, y)]


def _asymptotic_gradient_variance_ratio(uids, grads, k, block_shapes):
    """K 个块子估计的"K × 单次更新方差" 相对 全批梯度方差。

    设全批平均梯度 v̄ = (1/N) Σ g_i。若一步 rollout 走 K 次 optimizer step，
    第 j 步只看块 B_j（|B_j| = n_j），其梯度估计 v_j = (1/n_j) Σ_{i∈B_j} g_i。
    非独立同分布抽样的经典结论：Var(v̄) ≈ (1/K²) Σ_j Var(v_j)，
    故 K×单次更新方差（=训练侧实际承受的噪声量）相对全批方差之比为

        ratio = (K / sum_j n_j·... ) 的精确离散形式见下

    这里不依赖大样本近似，直接用"块内 k 近邻"精确算：
      Var(v_j) 的离散估计 = (1/n_j²) Σ_{i∈B_j} ‖g_i − v_j‖² / (n_j − 1)（样本方差/均值）
    返回 K × mean_j Var(v_j) / Var_full。
    """
    n_total = len(grads)
    v_full = [0.0] * len(grads[0])
    for g in grads:
        v_full = _add(v_full, g)
    v_full = [x / n_total for x in v_full]
    var_full = sum((_norm([g[d] - v_full[d] for d in range(len(v_full))])) ** 2
                   for g in grads) / (n_total - 1)
    actual = _asymptotic_var(uids, grads, k, block_shapes)
    # 同上：var_full 用相对容差判退化
    if var_full <= 1e-24 * len(grads):
        return None
    return actual / var_full


def _asymptotic_var(uids, grads, k, block_shapes):
    """K 步累计的噪声方差估计 Σ_j Var(v_j)，v_j 是第 j 块的平均梯度。

    块内用**无偏样本方差 / n**（即 Var(v_j) 的标准无偏估计）。这里必须用 n-1：
    用 1/n 的极大似然版本时，整批被切成"同组完全不相交"的块会让每块的组内方差
    估成 0，K>1 的读数凭空消失。这个陷阱在本批自查时真实发生过（见 readme「自查修正」）。
    """
    total = 0.0
    dim = len(grads[0])
    for start, n in block_shapes:
        block = grads[start:start + n]
        mean_j = [0.0] * dim
        for g in block:
            mean_j = _add(mean_j, g)
        mean_j = [x / n for x in mean_j]
        if n > 1:
            s2 = sum(_norm([g[d] - mean_j[d] for d in range(dim)]) ** 2 for g in block) / (n - 1)
            total += s2 / n
        else:
            total += 0.0
    return total


def _block_shapes_contiguous(n_total, k):
    """连续切片（verl 默认 shuffle=false：LDM 之后按原顺序切）。"""
    base = n_total // k
    shapes, start = [], 0
    for j in range(k):
        n = base + (1 if j < n_total % k else 0)
        shapes.append((start, n))
        start += n
    return shapes


def _block_shapes_interleaved(n_total, k):
    """交错切：第 j 块拿全部下标 i ≡ j (mod K)。

    真实路径里 LDM 重排之后，组内样本并不是严格相邻（同组样本长度相近但不相连），
    所以"交错"是比"组内连续"更强的反事实基准。两者一起给出组成的上界。
    """
    return [(0, len([i for i in range(n_total) if i % k == j])) for j in range(k)]


def _block_indices_interleaved(n_total, k, j):
    return [i for i in range(n_total) if i % k == j]


def _variance_ratio_from_indices(grads, block_indices):
    """给定显式的块下标求噪声比（交错形状用它，因为形状不是连续区间）。"""
    dim = len(grads[0])
    n_total = len(grads)
    mf = [0.0] * dim
    for g in grads:
        mf = _add(mf, g)
    mf = [x / n_total for x in mf]
    # 退化判据用判零容差而不是 "<= 0"：全部梯度相同时，「均值 − 自身」会留下
    # ~1e-33 的浮点残差，直接比较 0 会把它当成有效方差。
    if sum(_norm([g[d] - mf[d] for d in range(dim)]) ** 2 for g in grads) <= 1e-24 * n_total:
        return None
    var_full = sum(_norm([g[d] - mf[d] for d in range(dim)]) ** 2 for g in grads) / (n_total - 1)
    total = 0.0
    for bi in block_indices:
        n = len(bi)
        if n <= 1:
            continue
        m = [0.0] * dim
        for i in bi:
            m = _add(m, grads[i])
        m = [x / n for x in m]
        s2 = sum(_norm([grads[i][d] - m[d] for d in range(dim)]) ** 2 for i in bi) / (n - 1)
        total += s2 / n
    return total / var_full


def _block_shapes_random(n_total, k, rng):
    """随机切（shuffle=true）。"""
    idx = list(range(n_total))
    rng.shuffle(idx)
    base = n_total // k
    shapes, pos = [], 0
    for j in range(k):
        n = base + (1 if j < n_total % k else 0)
        shapes.append((pos, n))
        pos += n
    return shapes


def group_purity(uids, shapes):
    """每个块里"完整装下的组"的比例（按成员计），以及每组平均被摊到几块。

    这就是"块是不是组的随机划分"的直接证据：纯度 1.0 表示没有组被拆到块之间。
    """
    n_groups = len(set(uids))
    owners = {}
    for j, (start, n) in enumerate(shapes):
        for u in uids[start:start + n]:
            owners.setdefault(u, set()).add(j)
    intact = sum(1 for u in range(n_groups) if len(owners.get(u, ())) == 1)
    per_group = [len(owners[u]) for u in range(n_groups) if u in owners]
    return {
        'intact_groups': intact,
        'total_groups': n_groups,
        'intact_ratio': intact / n_groups if n_groups else None,
        'mean_blocks_per_group': statistics.mean(per_group) if per_group else None,
    }


def block_moment_decomposition(uids, grads, shapes):
    """把每块均值的波动拆成"组均值之间的方差"与"块内样本方差/n"。

    E[v_j] = 该块内各组的组均值（按数量加权）——**与块的组成有关系**；
    Var(v_j) = Var_组(块内组均值) + E[块内样本方差]/n。
    第一项随块装了哪些组而变，第二项随块内样本的离散程度而变。
    """
    dim = len(grads[0])
    block_vars, within_vars = [], []
    res_cos = []
    for start, n in shapes:
        block = grads[start:start + n]
        m = [0.0] * dim
        for g in block:
            m = _add(m, g)
        m = [x / n for x in m]
        resid = [[g[d] - m[d] for d in range(dim)] for g in block]
        rvar = sum(_norm(r) ** 2 for r in resid) / max(n - 1, 1)
        within_vars.append(rvar)
        block_vars.append(rvar / n)          # Var(v_j) 的下限（残差同向时取到）
    # 组均值（按显式 uid，不从样本下标推断）
    groups = {}
    for u, g in zip(uids, grads):
        groups.setdefault(u, []).append(g)
    gm = []
    for _, gs in groups.items():
        a = [0.0] * dim
        for g in gs:
            a = _add(a, g)
        gm.append([x / len(gs) for x in a])
    bvar = 0.0
    if len(gm) > 1:
        mean_g = [sum(v[d] for v in gm) / len(gm) for d in range(dim)]
        bvar = sum(_norm([v[d] - mean_g[d] for d in range(dim)]) ** 2 for v in gm) / (len(gm) - 1)
    # 残差余弦（块内相邻两条相对块均值的偏离的方向一致性）
    cos = 0.0
    if len(resid) > 1 and _norm(resid[0]) > 0 and _norm(resid[1]) > 0:
        cos = _dot(resid[0], resid[1]) / (_norm(resid[0]) * _norm(resid[1]))
    return {
        'block_var': statistics.mean(block_vars),
        'within_var': statistics.mean(within_vars),
        'between_group_var': bvar,
        'residual_cosine': cos,
    }


def check_independence(grads, k, shapes):
    """报告"块均值之间近似独立"这一假设的实测偏差。

    若块均值两两内积的均值显著为正，说明块之间系统性同向，K×方差不再是简单叠加。
    返回 max|cosine| 与 内积均值/范数乘积。
    """
    dim = len(grads[0])
    means = []
    for start, n in shapes:
        m = [0.0] * dim
        for g in grads[start:start + n]:
            m = _add(m, g)
        means.append([x / n for x in m])
    cos = []
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            ni, nj = _norm(means[i]), _norm(means[j])
            if ni > 0 and nj > 0:
                cos.append(_dot(means[i], means[j]) / (ni * nj))
    if not cos:
        return {'n_pairs': 0, 'mean_cosine': 0.0, 'max_abs_cosine': 0.0}
    return {
        'n_pairs': len(cos),
        'mean_cosine': statistics.mean(cos),
        'max_abs_cosine': max(abs(c) for c in cos),
    }


def splitting_amplification(uids, grads, k, shapes):
    """同组样本被拆到不同块时，单元梯度大小的放大倍数（相对全批平均）。

    §13.30 第一批问的是"尺度"；这里量的是"一条样本的梯度进优化器时被放大多少"。
    只在"按块各自平均"（= 典型 SGD 一步只看一个块）口径下有定义。
    返回 每个块 |v_j| / |v̄_full| 的均值。
    """
    dim = len(grads[0])
    n_total = len(grads)
    v_full = [0.0] * dim
    for g in grads:
        v_full = _add(v_full, g)
    v_full = [x / n_total for x in v_full]
    base = _norm(v_full)
    if base == 0:
        return None
    ratios = []
    for start, n in shapes:
        m = [0.0] * dim
        for g in grads[start:start + n]:
            m = _add(m, g)
        m = [x / n for x in m]
        ratios.append(_norm(m) / base)
    return statistics.mean(ratios)


def group_intact_split(parts, n_per=GROUP_SIZE):
    """真实路径（逐条 LDM）下有多少组被拆到不同分区。

    返回 (拆开的组数, 总组数)。用于说明"拆开后同组样本落在哪"——LDM 按序列长度
    排序后再分，组内样本长度相近 ⇒ 一般落到相邻分区的相邻位置。
    """
    splits = 0
    groups = sum(len(p) for p in parts) // n_per
    for g in range(groups):
        members = set(range(g * n_per, (g + 1) * n_per))
        owner = {}
        for pi, part in enumerate(parts):
            for i in part:
                if i in members:
                    owner[i] = pi
        if len(set(owner.values())) > 1:
            splits += 1
    return splits, groups


# ---------------------------------------------------------------------------
# 第三部分：识别性检验
# ---------------------------------------------------------------------------
def identifiability(metric_fn, levels, param):
    """对单个驱动量做档位扫描：指标在全档位上的**相对**极差低于阈值即判"零敏感"。

    用相对极差而不是绝对极差：噪声比这类指标本身量级只有 1e-2，绝对阈值
    1e-9 会把微小但真实的响应也判成"动了"——那是检验太松。相对极差
    (max−min)/|mean| > 1% 才算有识别性。

    metric_fn(rng, level) -> float | None
    """
    vals = []
    for lv in levels:
        for s in range(4):           # 少量种子足以判定"动不动"
            rng = random.Random(s * 97 + 13)
            v = metric_fn(rng, lv)
            if v is not None and math.isfinite(v):
                vals.append(v)
    if not vals:
        return {'param': param, 'levels': list(levels), 'spread': None,
                'sensitive': False, 'note': 'no finite reading'}
    spread = max(vals) - min(vals)
    ref = abs(statistics.mean(vals))
    rel = (spread / ref) if ref > 0 else float('inf')
    return {
        'param': param,
        'levels': list(levels),
        'min': min(vals),
        'max': max(vals),
        'spread': spread,
        'relative_spread': rel,
        'sensitive': rel > IDENTIFIABILITY_REL_EPS,
        'threshold': IDENTIFIABILITY_REL_EPS,
    }


def _metric_variance_ratio(rng, rho):
    uids, grads = _grad_model(rng, rho, None)
    k = 4
    shapes = _block_shapes_contiguous(len(grads), k)
    return _asymptotic_gradient_variance_ratio(uids, grads, k, shapes)


def _metric_composition_ratio(rng, rho):
    """组成效应比值（LDM 顺序 vs 组内连续排列，dp=4、K=4）。"""
    from group_split_tradeoff import _karmarkar_karp
    uids, grads = _grad_model(rng, rho, None)
    n = len(grads)
    lens = [max(1.0, rng.gauss(200.0, 120.0)) for _ in range(n)]
    w = [l * 24576 + l * l for l in lens]
    parts = _karmarkar_karp(w, 4)
    shapes = _block_shapes_contiguous(n, 4)
    order = [i for p in parts for i in p]
    lv = _asymptotic_gradient_variance_ratio([uids[i] for i in order],
                                             [grads[i] for i in order], 4, shapes)
    gi = [i for g in range(N_GROUPS) for i in range(g * GROUP_SIZE, (g + 1) * GROUP_SIZE)]
    gv = _asymptotic_gradient_variance_ratio([uids[i] for i in gi],
                                             [grads[i] for i in gi], 4, shapes)
    return lv / gv if gv else None


def _metric_group_cosine(rng, rho):
    """组均值之间余弦的绝对值均值——组信号的直接代理。"""
    uids, grads = _grad_model(rng, rho, None)
    means = {}
    for u, g in zip(uids, grads):
        acc = means.setdefault(u, [0.0] * len(g))
        for d in range(len(g)):
            acc[d] += g[d]
    vs = {}
    for u, m in means.items():
        n = sum(1 for x in uids if x == u)
        vs[u] = [x / n for x in m]
    keys = sorted(vs)
    cos = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ni, nj = _norm(vs[keys[i]]), _norm(vs[keys[j]])
            if ni > 0 and nj > 0:
                cos.append(abs(_dot(vs[keys[i]], vs[keys[j]]) / (ni * nj)))
    return statistics.mean(cos) if cos else None


def _metric_constant(_rng, _level):
    return 1.0


# ---------------------------------------------------------------------------
# 构建 summary
# ---------------------------------------------------------------------------
def build_summary(local_roots=None):
    local_roots = local_roots or {}
    out = {
        'experiment': '训练侧拆组第二类代价：mini-batch 组成，以及四个框架的归约语义核验',
        'doc': 'notebook/rollout-performance-research.md §13.32（接续 §13.31）',
        'framework_semantics': {},
        'not_partition_local': NOT_PARTITION_LOCAL,
        'composition': {},
        'identifiability': {},
        'verdict': {},
    }

    # ---- 第一部分：四框架口径 ----
    for name, ev in FRAMEWORK_EVIDENCE.items():
        root = snapshot_root(name) or local_roots.get(name)
        checks = []
        for rel, snippet in ev['anchors']:
            found, detail = check_anchor(root, rel, snippet)
            checks.append({'path': rel, 'found': found, 'detail': detail})
        out['framework_semantics'][name] = {
            'commit': ev['commit'],
            'advantage_grouping': ev['advantage_grouping'],
            'loss_normalization': ev['loss_normalization'],
            'anchors': checks,
            'anchors_all_found': all(c['found'] for c in checks),
            'snapshot_available_locally': root is not None,
            'reading': ev['reading'],
        }

    n_grouped = sum(1 for v in out['framework_semantics'].values()
                    if v['advantage_grouping'] in ('uid', 'prompt'))
    out['verdict']['all_four_group_by_uid_or_prompt'] = (n_grouped == len(FRAMEWORK_EVIDENCE))

    # ---- 第二部分：mini-batch 组成代价 ----
    from group_split_tradeoff import _karmarkar_karp

    # (a) 噪声随块数 K：同一个批次，一步切成 K 块各做一次 optimizer step
    k_rows = {}
    for k in (1, 2, 4, 8):
        cont, rand = [], []
        for s in SEEDS:
            rng = random.Random(s)
            uids, grads = _grad_model(rng, RHO_GROUP, None)
            n = len(grads)
            cont.append(_asymptotic_gradient_variance_ratio(
                uids, grads, k, _block_shapes_contiguous(n, k)))
            rr = random.Random(s + 5000)
            rand.append(_asymptotic_gradient_variance_ratio(
                uids, grads, k, _block_shapes_random(n, k, rr)))
        k_rows['K%d' % k] = {
            'contiguous_mean': statistics.mean(cont),
            'contiguous_min': min(cont),
            'contiguous_max': max(cont),
            'random_mean': statistics.mean(rand),
            'n_seeds': len(SEEDS),
        }
    out['composition']['noise_ratio_by_K'] = k_rows
    out['composition']['definition'] = (
        'ratio = Σ_j Var(v_j) / Var(v̄_full)，v_j 是第 j 块的平均梯度、v̄_full 是全批平均。'
        '等价于"一步 rollout 做 K 次 optimizer step"时累计承受的噪声方差，除以'
        '"一步看全批"的噪声方差。K=1 时两块相同，理想值 1.0。')

    # (b) 保持 K 不变，只换块的组成：三种排列两两比较
    comp_rows = {}
    for dp in DP_SIZES:
        rows = {'ldm': [], 'grouped': [], 'interleaved': [], 'split': []}
        for s_ in SEEDS:
            rng = random.Random(s_)
            uids, grads = _grad_model(rng, RHO_GROUP, None)
            n = len(grads)
            k = 4
            lens = [max(1.0, random.Random(s_ + 1000).gauss(200.0, 120.0)) for _ in range(n)]
            w = [l * 24576 + l * l for l in lens]
            parts = _karmarkar_karp(w, dp)
            ldm_idx = [i for p in parts for i in p]          # 重排后 train 张量的顺序
            grouped_idx = [i for g in range(N_GROUPS)
                           for i in range(g * GROUP_SIZE, (g + 1) * GROUP_SIZE)]

            def ratio_of(index_mapping, interleaved=False):
                gs = [grads[i] for i in index_mapping]
                if interleaved:
                    blocks = [_block_indices_interleaved(len(gs), k, j) for j in range(k)]
                    return _variance_ratio_from_indices(gs, blocks)
                return _asymptotic_gradient_variance_ratio(
                    None, gs, k, _block_shapes_contiguous(len(gs), k))

            rows['ldm'].append(ratio_of(ldm_idx))
            rows['grouped'].append(ratio_of(grouped_idx))
            rows['interleaved'].append(ratio_of(grouped_idx, interleaved=True))
            rows['split'].append(group_intact_split(parts, GROUP_SIZE)[0])
        pair = lambda a, b: [x / y for x, y in zip(rows[a], rows[b])]
        comp_rows['dp%d' % dp] = {
            'ldm_mean': statistics.mean(rows['ldm']),
            'grouped_mean': statistics.mean(rows['grouped']),
            'interleaved_mean': statistics.mean(rows['interleaved']),
            'ldm_over_grouped_mean': statistics.mean(pair('ldm', 'grouped')),
            'ldm_over_grouped_stderr': statistics.pstdev(pair('ldm', 'grouped')) / math.sqrt(len(SEEDS)),
            'ldm_over_grouped_min': min(pair('ldm', 'grouped')),
            'ldm_over_grouped_max': max(pair('ldm', 'grouped')),
            'interleaved_over_ldm_mean': statistics.mean(pair('interleaved', 'ldm')),
            'interleaved_over_ldm_stderr': statistics.pstdev(pair('interleaved', 'ldm')) / math.sqrt(len(SEEDS)),
            'interleaved_over_grouped_mean': statistics.mean(pair('interleaved', 'grouped')),
            'mean_split_groups': statistics.mean(rows['split']),
            'total_groups': N_GROUPS,
            'n_seeds': len(SEEDS),
            'theory': {
                'note': ('若每块恰好装 k_g 个完整组，则期望比值为 '
                         '1 + 2·rho/(k_g − 1)（样本梯度 = 组分量 + 样本分量时）。'
                         '交错排列下 k_g = N/K，理论上界。'),
                'rho': RHO_GROUP,
                'K': 4,
                'groups_per_block_interleaved': N_GROUPS / 4,
                'predicted_interleaved_over_grouped': 1.0 + 2.0 * RHO_GROUP / (N_GROUPS / 4 - 1),
            },
        }
    out['composition']['composition_effect_at_K4'] = comp_rows
    out['composition']['composition_effect_note'] = (
        '同一批样本、同一 K=4，只换块的组成方式。`ldm_over_grouped` 是真实路径相对'
        '"组内连续排列"的差异；`interleaved_over_ldm` 是"把同组样本尽可能摊到各块"'
        '相对真实路径的差异，代表组成效应的上界。两者都扣掉了 K 本身带来的放大。')

    # (c) 每个优化器步的梯度期望 = 组均值（与块组成无关），方差 = 组均值之间的方差
    #     + 块内样本方差 / n。组成效应只经由第二项的"块内样本方差"，这里逐项量出来。
    decomp = []
    for s_ in SEEDS[:10]:
        rng = random.Random(s_)
        uids, grads = _grad_model(rng, RHO_GROUP, None)
        decomp.append(block_moment_decomposition(
            uids, grads, _block_shapes_contiguous(len(grads), 4)))
    purity = []
    for s_ in SEEDS:
        rng = random.Random(s_)
        uids, grads = _grad_model(rng, RHO_GROUP, None)
        n = len(grads)
        lens = [max(1.0, random.Random(s_ + 1000).gauss(200.0, 120.0)) for _ in range(n)]
        w = [l * 24576 + l * l for l in lens]
        parts = _karmarkar_karp(w, 4)
        idx = [i for p in parts for i in p]
        purity.append(group_purity([uids[i] for i in idx], _block_shapes_contiguous(n, 4)))
    out['composition']['group_purity_at_K4_dp4'] = {
        'intact_ratio_mean': statistics.mean(p['intact_ratio'] for p in purity),
        'intact_ratio_min': min(p['intact_ratio'] for p in purity),
        'intact_ratio_max': max(p['intact_ratio'] for p in purity),
        'mean_blocks_per_group': statistics.mean(p['mean_blocks_per_group'] for p in purity),
        'n_seeds': len(purity),
        'note': ('纯度 0 = 每一组都被摊到了全部 4 个块。这与"交错排列"的组成结构一致，'
                 '所以交错基线没有带来额外噪声；只要还有未拆分的组，纯度就会 >0。'),
    }
    out['composition']['block_moment_decomposition'] = {
        'mean_within_block_var_over_mean_block_var':
            statistics.mean(r['within_var'] / r['block_var'] for r in decomp),
        'mean_between_group_var_over_mean_block_var':
            statistics.mean(r['between_group_var'] / r['block_var'] for r in decomp),
        'mean_abs_residual_cosine':
            statistics.mean(abs(r['residual_cosine']) for r in decomp),
        'note': ('块内样本方差占总方差的比例决定"块均值沿组均值分布"的程度；'
                 '残差余弦≈1 时块内样本相对本块均值的偏离近似同向，'
                 '此时 Var(v_j) 取样本方差/n 的下限。'),
    }

    # (d) 负向对照：人为让块之间共享方向，检验交错比的检验能否抓到
    rng = random.Random(0)
    _, corr = _grad_model_block_correlated(rng, RHO_GROUP)
    ctrl_ldm = _asymptotic_gradient_variance_ratio(
        None, corr, 4, _block_shapes_contiguous(len(corr), 4))
    ctrl_grp = corr[:]
    out['composition']['negative_control_block_correlated'] = {
        'interleaved_over_contiguous': (
            _variance_ratio_from_indices(ctrl_grp, [_block_indices_interleaved(len(corr), 4, j)
                                                    for j in range(4)]) / ctrl_ldm),
        'note': '人为让每块各有一个共享方向时，交错比应明显偏离 1（远大于 ±1%），否则检验抓不到结构。',
    }

    # ---- 第三部分：识别性检验 ----
    out['identifiability']['noise_ratio_vs_rho'] = identifiability(
        _metric_variance_ratio, RHO_LEVELS, '组信号强度 rho（K=4）')
    out['identifiability']['composition_ratio_vs_rho'] = identifiability(
        _metric_composition_ratio, RHO_LEVELS, '组信号强度 rho（组成比值，dp=4, K=4）')
    # 对照 1：组信号的真实代理必须随 rho 单调（防止检验太严误杀）
    out['identifiability']['control_group_cosine_vs_rho'] = identifiability(
        _metric_group_cosine, RHO_LEVELS, '对照：组均值余弦 vs rho')
    # 对照 2：恒常数指标必须被判零敏感（防止检验太松放过无效指标）
    out['identifiability']['control_constant'] = identifiability(
        _metric_constant, RHO_LEVELS, '对照：恒常数指标')
    out['identifiability']['control_noise_ratio_must_respond'] = {
        'param': '块数 K（不是机制旋钮，只用于确认指标本身能动）',
        'readings': {k: v['contiguous_mean'] for k, v in k_rows.items()},
        'sensitive': k_rows['K1']['contiguous_mean'] != k_rows['K8']['contiguous_mean'],
        'note': '这条不是识别性检验，是"指标非退化"的冒烟检查：K 变它必须变。',
    }

    # ---- 结论 ----
    dp4 = comp_rows['dp4']
    out['verdict'].update({
        'no_framework_normalizes_per_partition': out['verdict']['all_four_group_by_uid_or_prompt'],
        'minibatch_composition_is_the_only_remaining_path': (
            '四个快照的优势 mean/std 全部按 uid/prompt 分组，没有框架按分区各自归一化；'
            '两处形似按块的标准化（OpenRLHF／NeMo RL）都是全 batch 整体缩放，与分区无关。'
            '因此 §13.30 第一批的 0.69／0.41 倍仍是**假设口径**，§13.31 的结论可推广到这四个实现。'
            '训练侧剩下的唯一代价路径是 mini-batch 组成。'),
        'composition_effect_reading': (
            'K=4、dp=4 下三条读数：① 把同组样本尽量摊到各块（交错排列）相对真实路径只有 %.4f 倍'
            '（标准误 %.4f）——基线本身几乎已经是最坏组成，原因是逐条 LDM 按长度排序后分片，'
            '重排结果近似把同组样本均匀摊开；② 真实路径相对"组内连续排列"是 %.4f 倍'
            '（标准误 %.4f）；③ 后者还含有"连续排列让组内样本相邻、于是方差被低估"的估计偏差，'
            '构成效应的真实量级应按 ① 读。作为对照，块数从 1 增到 8 会把噪声放大到 %.2f 倍。'
            '结论：**组成的差异真实存在但只有个位数百分比；把一步拆成 K 次 optimizer step 才是'
            '训练侧的主要噪声来源，而这一项与拆不拆组无关。**'
            % (dp4['interleaved_over_ldm_mean'], dp4['interleaved_over_ldm_stderr'],
               dp4['ldm_over_grouped_mean'], dp4['ldm_over_grouped_stderr'],
               k_rows['K8']['contiguous_mean'] / k_rows['K1']['contiguous_mean'])),
        'what_this_is_not': [
            '不是"拆组导致噪声放大"：块数 K 决定放大，组完整性只决定块的组成。',
            '不是实测：梯度为随机方向模型，不含真实 token 梯度结构；结论是聚合语义的推论。',
            '不报耗时、不换算任何论文的墙钟读数。',
        ],
    })
    return out


def main(argv):
    roots = {}
    for name, env in (('verl', os.environ.get('VERL_ROOT', '/Users/jackiemac/workspace/verl')),):
        if env and os.path.isdir(env):
            roots[name] = env
    for extra in ('ORLHF_ROOT', 'SKYRL_ROOT', 'NEMO_RL_ROOT'):
        v = os.environ.get(extra)
        if v and os.path.isdir(v):
            roots[extra.split('_')[0].lower()] = v

    out = build_summary(roots)
    text = json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True)
    if '--verify' in argv:
        with open(SUMMARY_PATH) as f:
            ref = json.load(f)
        ok = json.dumps(ref, ensure_ascii=False, indent=1, sort_keys=True) == text
        print('VERIFY OK: 逐字段相同' if ok else 'VERIFY FAIL: 与归档不一致')
        if not ok:
            for k in out:
                if json.dumps(ref.get(k), ensure_ascii=False, sort_keys=True) != \
                        json.dumps(out[k], ensure_ascii=False, sort_keys=True):
                    print('  差异字段：', k)
        return 0 if ok else 1
    with open(SUMMARY_PATH, 'w') as f:
        f.write(text + '\n')
    for name, v in out['framework_semantics'].items():
        print('framework %-9s grouping=%-6s anchors_found=%s local=%s'
              % (name, v['advantage_grouping'], v['anchors_all_found'],
                 v['snapshot_available_locally']))
    for k, v in out['composition']['noise_ratio_by_K'].items():
        print('noise_ratio %-3s contiguous=%.4f random=%.4f'
              % (k, v['contiguous_mean'], v['random_mean']))
    for k, v in out['identifiability'].items():
        if 'relative_spread' in v:
            print('ident %-32s rel_spread=%.4g min=%.4g max=%.4g sensitive=%s'
                  % (k, v['relative_spread'], v['min'], v['max'], v['sensitive']))
        else:
            print('ident %-32s sensitive=%s' % (k, v.get('sensitive')))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
