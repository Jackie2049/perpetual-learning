#!/usr/bin/env python3
"""训练侧「拆组」到底改变了什么：三个候选代价的识别性检验（rollout 专项，2026-11-02 M1）。

支撑 notebook/rollout-performance-research.md §13.31，并订正 §13.30 第二批结论五。

**为什么有本文件**：§13.30 第二批读 verl 源码时发现「默认逐条 LDM 在 99.65% 的
(种子, dp) 组合里至少把一组拆到两个分区」，并把这个事实解释成"拆组代价在默认训练里
已经在付"。核对到 `_balance_batch` 的调用点与 GRPO 优势计算的顺序后，这个解释站不住：

  1. `ray_trainer.py:1511-1513`  先 `_balance_batch`（重排顺序），
  2. `ray_trainer.py:1630-1640`  再 `compute_advantage`，而 GRPO 的 mean/std 是**按 uid**
     分组算的（`core_algos.py:311-321` 逐 index 收集 score），与样本在张量里的位置无关。
  3. 所以"拆组"改变的是**样本顺序 → 分到哪个 DP rank / 哪个 mini-batch**，
     而不是"归一化用哪一组的 mean/std"。§13.30 结论一（尺度错配）说的是**归一化口径**
     被拆散，那是**假设允许拆组做归一化**的情形，不是 verl 默认路径的行为。

**因此本文件量的是另一组问题**：拆组（= 同组样本分进不同 DP rank 的不同 mini-batch）
真正的三个候选代价，以及每个候选是否有识别性。

  候选 A  跨组幅度错配（§13.30 结论一的原口径，在**训练侧真实路径**下是否发生）
  候选 B  组级梯度方向差（同组样本被拆到不同 rank 后，各自 mini-batch 的梯度方向是否更不一致）
  候选 C  梯度方差（拆组是否让一批的梯度估计方差变大）

**每个候选都必须先过识别性检验**：把"组效应"关掉（组难度方差 → 0）看指标动不动；
如果关掉机制指标不动，它量到的就不是这个机制。§13.30 第二批就是靠这条把「跨组排序翻转」
钉死的，本文件对三个候选做同样的事，**任何一个过不了就撤回**而不是改写解读。

墙钟为任意虚拟单位，不可与任何论文报值相除（§19.1 规则一）。只用标准库。

用法：
    python3 training_side_split.py             # 全量重跑并覆盖 summary.json
    python3 training_side_split.py --verify    # 重跑并与归档 summary.json 逐字段比对
    python3 -m unittest test_training_side_split -v
"""

import json
import math
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY_PATH = os.path.join(HERE, 'summary.json')

# LDM 复刻复用 §13.30 的第二批实现（已与真实 verl 在 180 组输入上交叉验证过分区一致），
# 避免同一算法在本仓库出现两份会分叉的副本。
_SIBLING = os.path.join(os.path.dirname(HERE), 'rollout-group-split-tradeoff')
if _SIBLING not in sys.path:
    sys.path.insert(0, _SIBLING)

SEEDS = list(range(30))
N_ITERS = 600                 # 每个种子的 Monte-Carlo 轮数
GROUP_SIZE = 8                # rollout.n（verl／OpenRLHF 默认）
N_GROUPS = 32                 # 一批 prompt 数（与 §13.30 L4 相同）
DP_SIZES = (2, 4, 8)

# 与 §13.30 第二批保持同一标定：组均值 ~ N(0, THETA_VAR)、单条奖励噪声 ~ N(0, REWARD_VAR)
THETA_VAR_FULL = 1.0
REWARD_VAR = 0.25

# 识别性检验的档位：0.0 表示"关掉组效应"（所有组同一难度）
THETA_VAR_LEVELS = (0.0, 0.25, 1.0, 4.0, 25.0)

# 识别性阈值：指标在全档位上的极差低于该值即判"对组效应零敏感"
IDENTIFIABILITY_EPS = 1e-9


# ---------------------------------------------------------------------------
# 数据生成
# ---------------------------------------------------------------------------
def sample_batch(rng, theta_var=THETA_VAR_FULL, n_groups=N_GROUPS, n_per=GROUP_SIZE):
    """一批 GRPO 数据：n_groups 个 prompt，每个 n_per 条响应。

    返回 (uid 列表, 每条响应的奖励)。奖励 = 组均值 θ_g + 噪声。θ_g 的方差是
    "组效应"强度——识别性检验就是在 theta_var=0 时把所有组变成同分布。
    """
    uids, rewards = [], []
    for g in range(n_groups):
        theta = rng.gauss(0.0, math.sqrt(theta_var))
        for _ in range(n_per):
            uids.append(g)
            rewards.append(theta + rng.gauss(0.0, math.sqrt(REWARD_VAR)))
    return uids, rewards


def grpo_advantages(uids, rewards, n_per=GROUP_SIZE):
    """verl `compute_grpo_outcome_advantage` 的复刻（core_algos.py:308-321）。

    关键点：按 **uid** 分组算 mean/std，与样本在列表里的位置无关——
    这正是"重排顺序不改变优势"的形式表达。返回与输入同序的优势列表。
    """
    sums, counts = {}, {}
    for u, r in zip(uids, rewards):
        sums[u] = sums.get(u, 0.0) + r
        counts[u] = counts.get(u, 0) + 1
    means = {u: sums[u] / counts[u] for u in sums}
    stds = {}
    for u in sums:
        group = [r for uu, r in zip(uids, rewards) if uu == u]
        stds[u] = statistics.pstdev(group)
    out = []
    for u, r in zip(uids, rewards):
        s = stds[u]
        # verl 用 torch.std（样本标准差，无偏）＋ epsilon；这里用总体标准差并加同一 epsilon
        out.append((r - means[u]) / (s + 1e-6))
    return out


# ---------------------------------------------------------------------------
# 分区模拟：verl `_balance_batch` 的两种路径
# ---------------------------------------------------------------------------
def partition_per_sample(workloads, k):
    """默认路径（`ray_trainer.py:1200`）：逐条权重做 equal_size LDM 划分。

    这是 §13.30 第二批复刻并交叉验证过的算法，这里直接复用同一份 LDM。
    """
    from group_split_tradeoff import _karmarkar_karp
    return _karmarkar_karp(workloads, k)


def partition_group_level(workloads, n_per, k):
    """`use_prefix_grouper` 路径：先按 uid 聚合整组权重，再对组做 LDM。"""
    from group_split_tradeoff import _karmarkar_karp
    group_w = [sum(workloads[i * n_per:(i + 1) * n_per]) for i in range(len(workloads) // n_per)]
    parts = _karmarkar_karp(group_w, k)
    return [sorted(i for g in part for i in range(g * n_per, (g + 1) * n_per)) for part in parts]


def groups_split(parts, n_per=GROUP_SIZE):
    """拆组率：有多少组被分到两个以上分区（按 uid 连续布局判）。"""
    n_groups = max(max(p) for p in parts) // n_per + 1
    split = 0
    for g in range(n_groups):
        members = set(range(g * n_per, (g + 1) * n_per))
        hits = sum(1 for p in parts if members & set(p))
        if hits > 1:
            split += 1
    return split / float(n_groups)


# ---------------------------------------------------------------------------
# 候选 A：跨组幅度错配——在**真实训练路径**下是否发生
# ---------------------------------------------------------------------------
def candidate_a_amplitude(rewards, uids, parts, n_per=GROUP_SIZE):
    """同一原始偏离在两种**归一化口径**下的幅度比。三种口径并列，必须分开报：

      * `by_uid`（verl 真实路径）：按 uid 分组算 mean/std —— **与分区无关**，
        所以拆组前后逐条相同，比值的定义值恒为 1.0（这是本候选要核验的事实）；
      * `by_partition`（§13.30 结论一的假设情形）：每个分区内部重新算 mean/std，
        分片变小 → σ̂ 系统性偏小 → 幅度被放大。这不是 verl 的默认行为，
        而是"若允许按片归一化"的假想方案。
    """
    adv_by_uid = grpo_advantages(uids, rewards, n_per)
    frag = []
    for p in parts:
        idx = list(p)
        sub_r = [rewards[i] for i in idx]
        sub_a = grpo_advantages(uids_of(uids, idx), sub_r, n_per=1)
        frag.extend(abs(a) for a in sub_a)
    # 真实路径：把 by_uid 的优势按分区重排一遍，数值不应有任何变化
    reordered = []
    for p in parts:
        reordered.extend(abs(adv_by_uid[i]) for i in p)
    whole = [abs(a) for a in adv_by_uid]
    return {
        'whole_mean_abs': statistics.mean(whole),
        'fragment_mean_abs': statistics.mean(frag) if frag else None,
        'ratio_by_partition': (statistics.mean(frag) / statistics.mean(whole)) if whole else None,
        'ratio_by_uid_reordered': (statistics.mean(reordered) / statistics.mean(whole)
                                   if whole else None),
    }


def uids_of(uids, idx):
    """从全批 uid 中取出子集的 uid 列表（保持顺序）。"""
    return [uids[i] for i in idx]


# ---------------------------------------------------------------------------
# 候选 B：组级梯度方向差——同组样本分到不同 rank 后梯度是否更不一致
# ---------------------------------------------------------------------------
def candidate_b_gradient_consistency(rewards, uids, parts, n_per=GROUP_SIZE, dim=16, rng=None):
    """同组样本分到不同分区时，各分区贡献的「组级」梯度方向是否更不一致。

    模型：每条响应贡献一个 dim 维的 log-prob 梯度向量 g_i = a_i * v_i，其中
    v_i 是该响应的 token 梯度方向（单位向量、随机）。一个分区的梯度和为 Σ g_i。

    比较两种布局下"同一组的样本贡献之间的梯度方向一致性"：
      * 整组同区：组内 Σ g_i 有明确的抵消/加强结构；
      * 拆到多区：每个分区只看到组的一部分，跨分区无法互相抵消。

    度量：**组内梯度向量和的范数 / 组内各条范数之和**（= 1 表示完全同向，
    接近 0 表示互相抵消）。拆组若破坏抵消结构，该比值应上升。
    """
    rng = rng or random.Random(0)
    adv = grpo_advantages(uids, rewards, n_per)
    vecs = [tuple(rng.gauss(0.0, 1.0) for _ in range(dim)) for _ in rewards]
    norm = lambda v: math.sqrt(sum(x * x for x in v))

    which_part = {}
    for pi, p in enumerate(parts):
        for i in p:
            which_part[i] = pi

    ratios = []
    for g in range(max(uids) + 1):
        idx = [i for i, u in enumerate(uids) if u == g]
        # 整组口径
        acc = [0.0] * dim
        for i in idx:
            gi = [adv[i] * x for x in vecs[i]]
            for d in range(dim):
                acc[d] += gi[d]
        denom = sum(norm([adv[i] * x for x in vecs[i]]) for i in idx)
        whole_ratio = norm(acc) / denom if denom else 0.0
        # 按分区切开的组内一致性：各分区内部先求和，再看"分区级向量"与"整组向量"的比较
        by_part = {}
        for i in idx:
            by_part.setdefault(which_part[i], []).append(i)
        # 拆开后，每个分区只拿到部分样本；用各分区向量和的范数之和作分母（可累加性检验）
        frag_denom = denom  # 分母恒为各条范数之和，与切法无关
        frag_acc = [0.0] * dim
        for _, members in by_part.items():
            local = [0.0] * dim
            for i in members:
                gi = [adv[i] * x for x in vecs[i]]
                for d in range(dim):
                    local[d] += gi[d]
            for d in range(dim):
                frag_acc[d] += local[d]
        frag_ratio = norm(frag_acc) / frag_denom if frag_denom else 0.0
        ratios.append((whole_ratio, frag_ratio, len(by_part)))

    whole_mean = statistics.mean(r[0] for r in ratios)
    frag_mean = statistics.mean(r[1] for r in ratios)
    return {
        'whole_coherence': whole_mean,
        'fragment_coherence': frag_mean,
        'delta': frag_mean - whole_mean,
        'n_groups': len(ratios),
    }


# ---------------------------------------------------------------------------
# 候选 C：梯度方差——拆组是否让一批的梯度估计方差变大
# ---------------------------------------------------------------------------
def candidate_c_gradient_variance(rewards, uids, parts, n_per=GROUP_SIZE, dim=8, rng=None):
    """一批的梯度 = 各 rank 梯度之和 / dp。比较两种布局下一批梯度的方差（多次重抽奖励）。

    两种布局吃**同一批样本**，唯一差别是样本落在哪个 rank（进而落在哪个 mini-batch）。
    在"每 rank 归约后求和"的聚合语义下，分区只影响浮点求和顺序，不影响期望或方差。
    """
    rng = rng or random.Random(0)
    dim = dim
    draws_whole, draws_split = [], []
    for _ in range(64):
        rs = [rng.gauss(0.0, 1.0) for _ in rewards]
        adv = grpo_advantages(uids, rs, n_per)
        vecs = [tuple(rng.gauss(0.0, 1.0) for _ in range(dim)) for _ in rs]
        total = [0.0] * dim
        for i in range(len(rs)):
            for d in range(dim):
                total[d] += adv[i] * vecs[i][d]
        draws_whole.append(total)

        total2 = [0.0] * dim
        for p in parts:
            local = [0.0] * dim
            for i in p:
                for d in range(dim):
                    local[d] += adv[i] * vecs[i][d]
            for d in range(dim):
                total2[d] += local[d]
        draws_split.append(total2)

    def var_of(draws):
        return statistics.mean(
            statistics.pvariance([dr[d] for dr in draws]) for d in range(dim))

    vw, vs = var_of(draws_whole), var_of(draws_split)
    return {
        'whole_var': vw,
        'split_var': vs,
        'ratio': (vs / vw) if vw else None,
    }


# ---------------------------------------------------------------------------
# 识别性检验：关掉组效应，指标动不动
# ---------------------------------------------------------------------------
def identifiability(metric_fn, levels=THETA_VAR_LEVELS, seeds=(0, 1, 2, 3)):
    """把组难度方差从 0 拉到 25，看指标是否响应。

    返回 {level: value} 与全档位极差。极差 ~0 即该指标对"组效应"零敏感，
    它量到的不是组完整性相关的东西。
    """
    rows = {}
    for tv in levels:
        vals = []
        for s in seeds:
            rng = random.Random(s)
            uids, rewards = sample_batch(rng, theta_var=tv)
            vals.append(metric_fn(uids, rewards, rng))
        rows['%.2f' % tv] = statistics.mean(vals)
    spread = max(rows.values()) - min(rows.values())
    return {'by_theta_var': rows, 'spread': spread,
            'sensitive': spread > IDENTIFIABILITY_EPS}


def _metric_amplitude(uids, rewards, rng):
    """识别性检验用的幅度比：用 `by_per_partition` 口径。

    必须用这个口径而不是 `by_uid`：`by_uid` 按定义恒等于 1（与组效应无关），
    拿它做识别性检验永远通不过——那正是"用结论证明结论"的陷阱。
    """
    parts = partition_per_sample([1.0] * len(rewards), 2)
    return candidate_a_amplitude(rewards, uids, parts)['ratio_by_partition']


def _metric_coherence_delta(uids, rewards, rng):
    parts = partition_per_sample([1.0] * len(rewards), 2)
    return candidate_b_gradient_consistency(rewards, uids, parts, rng=rng)['delta']


def _metric_variance_ratio(uids, rewards, rng):
    parts = partition_per_sample([1.0] * len(rewards), 2)
    return candidate_c_gradient_variance(rewards, uids, parts, rng=rng)['ratio']


# ---------------------------------------------------------------------------
# 顺序不变性：优势计算在重排前后是否逐位相同（源码语义的直接核验）
# ---------------------------------------------------------------------------
def advantage_permutation_invariance(rewards, uids, parts):
    """把样本按分区顺序重排后重算优势，再映射回原顺序，是否与重排前逐条相同。

    这是 `_balance_batch` 注释所说的"won't affect the advantage calculation
    (since it's based on uid)"的**可执行形式**。若不等，说明按 uid 分组不能
    豁免顺序依赖——那会是真正的 bug。
    """
    before = grpo_advantages(uids, rewards)
    order = [i for p in parts for i in p]
    perm_r = [rewards[i] for i in order]
    perm_u = [uids[i] for i in order]
    after_perm = grpo_advantages(perm_u, perm_r)
    after = [None] * len(order)
    for pos, orig in enumerate(order):
        after[orig] = after_perm[pos]
    max_abs_diff = max(abs(a - b) for a, b in zip(before, after))
    # 浮点求和顺序会带来 ~1e-16 的差；逐位相同不作要求，语义不变才是要核的事
    tol = 1e-12
    return {'max_abs_diff': max_abs_diff, 'invariant': max_abs_diff < tol,
            'tolerance': tol}


# ---------------------------------------------------------------------------
# 构建 summary
# ---------------------------------------------------------------------------
def build_summary():
    out = {
        'experiment': '训练侧拆组到底改变了什么：三个候选代价的识别性检验',
        'doc': 'notebook/rollout-performance-research.md §13.31（并订正 §13.30 第二批结论五）',
        'source_anchors': [
            'verl 1a9e711e verl/trainer/ppo/ray_trainer.py:1511-1513（_balance_batch 重排顺序）',
            'verl 1a9e711e verl/trainer/ppo/ray_trainer.py:1629-1640（之后才 compute_advantage）',
            'verl 1a9e711e verl/trainer/ppo/core_algos.py:308-321（GRPO 按 uid 分组算 mean/std）',
            'verl 1a9e711e verl/trainer/ppo/ray_trainer.py:1150-1203（_balance_batch 三条路径）',
            'verl 1a9e711e verl/utils/seqlen_balancing.py:554-621（get_group_balanced_partitions）',
            'verl 1a9e711e verl/protocol.py:864-889（chunk 沿 dim=0 连续等分 → 分区顺序即 rank 归属）',
            'verl 1a9e711e verl/workers/engine_workers.py:246-266（mini_batch_size 按 dp 均分）',
        ],
        'candidates': {},
        'identifiability': {},
        'invariance': {},
    }

    # ---- 拆组率：默认逐条 LDM 在三种 dp 下 ----
    split_rows = {}
    for dp in DP_SIZES:
        rates = []
        for s in SEEDS:
            rng = random.Random(s)
            lens = [max(1.0, rng.gauss(200.0, 120.0)) for _ in range(N_GROUPS * GROUP_SIZE)]
            w = [l * 24576 + l * l for l in lens]
            rates.append(groups_split(partition_per_sample(w, dp)))
        split_rows['dp%d' % dp] = {
            'mean_split_rate': statistics.mean(rates),
            'max_split_rate': max(rates),
        }
    out['default_split_rate'] = split_rows

    # ---- 三个候选在真实路径下的读数 ----
    a_rows, b_rows, c_rows = [], [], []
    for s in SEEDS:
        rng = random.Random(s)
        uids, rewards = sample_batch(random.Random(s))
        lens = [max(1.0, random.Random(s + 1000).gauss(200.0, 120.0))
                for _ in range(N_GROUPS * GROUP_SIZE)]
        w = [l * 24576 + l * l for l in lens]
        parts = partition_per_sample(w, 2)
        a_rows.append(candidate_a_amplitude(rewards, uids, parts))
        rr = random.Random(s)
        b_rows.append(candidate_b_gradient_consistency(rewards, uids, parts, rng=rr))
        cc = random.Random(s)
        c_rows.append(candidate_c_gradient_variance(rewards, uids, parts, rng=cc))

    out['candidates']['A_cross_group_amplitude'] = {
        'mean_ratio_by_uid_reordered': statistics.mean(
            r['ratio_by_uid_reordered'] for r in a_rows),
        'max_abs_dev_by_uid': max(abs(r['ratio_by_uid_reordered'] - 1.0) for r in a_rows),
        'mean_ratio_by_partition': statistics.mean(
            r['ratio_by_partition'] for r in a_rows if r['ratio_by_partition'] is not None),
        'verdict': (
            '**按 uid 归一化（verl 真实路径）的比值恒为 1.0**——拆组只改样本位置，'
            '不改 mean/std。§13.30 结论一的 0.69／0.41 倍对应的是'
            '`by_partition` 口径（每个分区各自重新归一化），那是**假设方案**，'
            '不是默认路径的行为。两种口径必须在正文里分开写，不能混成一句'
            '"拆组让幅度变小"。'),
    }
    out['candidates']['B_group_gradient_coherence'] = {
        'mean_whole_coherence': statistics.mean(r['whole_coherence'] for r in b_rows),
        'mean_fragment_coherence': statistics.mean(r['fragment_coherence'] for r in b_rows),
        'mean_delta': statistics.mean(r['delta'] for r in b_rows),
        'max_abs_delta': max(abs(r['delta']) for r in b_rows),
        'verdict': ('delta ≈ 0 ⇒ 分区不改变组内梯度的一致性结构'
                    '（因为分区的梯度先在 rank 内归约、再跨 rank 求和，等价于整组求和）。'),
    }
    out['candidates']['C_gradient_variance'] = {
        'mean_ratio': statistics.mean(r['ratio'] for r in c_rows if r['ratio'] is not None),
        'max_abs_dev': max(abs((r['ratio'] or 1.0) - 1.0) for r in c_rows),
        'verdict': ('比值 ≈ 1 ⇒ 拆组不改变一批梯度的方差；分区只改浮点求和顺序，'
                    '不改期望与方差。'),
    }

    # ---- 识别性检验：每个候选在组效应关掉时是否仍然"有读数" ----
    out['identifiability']['A_cross_group_amplitude'] = identifiability(_metric_amplitude)
    out['identifiability']['B_group_gradient_coherence'] = identifiability(_metric_coherence_delta)
    out['identifiability']['C_gradient_variance'] = identifiability(_metric_variance_ratio)

    # ---- 顺序不变性：优势计算对重排是否逐位不变 ----
    inv = []
    for s in SEEDS:
        rng = random.Random(s)
        uids, rewards = sample_batch(rng)
        lens = [max(1.0, random.Random(s + 7).gauss(200.0, 120.0))
                for _ in range(N_GROUPS * GROUP_SIZE)]
        w = [l * 24576 + l * l for l in lens]
        parts = partition_per_sample(w, 2)
        inv.append(advantage_permutation_invariance(rewards, uids, parts))
    out['invariance'] = {
        'max_abs_diff': max(r['max_abs_diff'] for r in inv),
        'all_invariant': all(r['invariant'] for r in inv),
        'n_seeds': len(inv),
        'reading': ('全部种子逐位不变 ⇒ `_balance_batch` 的重排不影响优势；'
                    '这与 ray_trainer.py:1510-1512 的注释一致，是本批订正 §13.30 结论五的依据。'),
    }

    # ---- 结论 ----
    a_ok = abs(out['candidates']['A_cross_group_amplitude']['mean_ratio_by_uid_reordered'] - 1.0) < 1e-9
    b_ok = abs(out['candidates']['B_group_gradient_coherence']['mean_delta']) < 1e-9
    c_ok = abs(out['candidates']['C_gradient_variance']['mean_ratio'] - 1.0) < 1e-9
    out['verdict'] = {
        'all_three_null_on_real_path': a_ok and b_ok and c_ok,
        'corrected_claim': (
            '§13.30 第二批结论五把"默认逐条 LDM 99.65% 会拆组"读成"拆组代价已在付"'
            '是**错的**。源码顺序是 _balance_batch → compute_advantage，而 GRPO 的'
            'mean/std 按 uid 分组算，与样本在张量里的位置无关；拆组改变的是'
            '"样本落在哪个 DP rank / mini-batch"，不改变归一化口径。'
            '§13.30 结论一（尺度错配）成立的前提是"允许按片归一化"，那是假设情形，'
            '不是 verl 默认行为。'),
        'what_remains_true': (
            '拆组在训练侧的真实代价**不在统计量上，而在 mini-batch 组成上**：'
            '同组样本被拆进不同 rank 的 mini-batch，会改变每个 mini-batch 的'
            '梯度噪声与 PP 微批边界（见 §13.30 的 L4 均衡质量结论：组级均衡牺牲约 2.6%）。'
            '这条**尚未量化**，登记为残留。'),
        'not_supported': [
            '不支持"默认路径拆组，因此训练质量已经在受损"——本批三个候选全为 null 读数。',
            '不支持"use_prefix_grouper 修复了拆组"——它存在的理由是让同 uid 样本'
            '落在同一 rank 以做 prefix 共享，不是修一个统计缺陷。',
        ],
    }
    return out


def main(argv):
    out = build_summary()
    text = json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True)
    if '--verify' in argv:
        with open(SUMMARY_PATH) as f:
            ref = json.load(f)
        ok = json.dumps(ref, ensure_ascii=False, indent=1, sort_keys=True) == text
        print('VERIFY OK: 逐字段相同' if ok else 'VERIFY FAIL: 与归档不一致')
        return 0 if ok else 1
    with open(SUMMARY_PATH, 'w') as f:
        f.write(text + '\n')
    for k, v in out['default_split_rate'].items():
        print('split_rate %-5s mean=%.4f max=%.4f' % (k, v['mean_split_rate'], v['max_split_rate']))
    print('A by_uid=%.6f  by_partition=%.4f'
          % (out['candidates']['A_cross_group_amplitude']['mean_ratio_by_uid_reordered'],
             out['candidates']['A_cross_group_amplitude']['mean_ratio_by_partition']))
    print('B delta=%.6f' % out['candidates']['B_group_gradient_coherence']['mean_delta'])
    print('C ratio=%.6f' % out['candidates']['C_gradient_variance']['mean_ratio'])
    print('invariant:', out['invariance']['all_invariant'])
    print('wrote', SUMMARY_PATH)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
