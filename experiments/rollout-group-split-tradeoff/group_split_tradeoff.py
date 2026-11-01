#!/usr/bin/env python3
"""§15 Q5 / §11.2 条目 7：拆 GRPO 组的代价与收益（rollout 专项，2026-11-02 M1）。

支撑 notebook/rollout-performance-research.md §13.30。

**问题**：语料内所有 L3 长度调度（repack／分桶／迁移）都以"组不可拆散"为默认前提，
但**没有一篇量化"若允许拆组能多回收多少气泡、组内比较的统计代价是多少"**（§15 Q5）。
本脚本把这两侧放进同一个可算模型，给出可核读数。

**答案的形状已经能预先说清（写在这里，防止先看数再编解读）**：
拆组把组内 n 条拆成 p 个分片后，归一化统计量 σ̂ 的抽样方差大约放大 (p−1)/(|C|−1) 倍——
拆得越碎，优势估计越像噪声；而气泡回收在**单卡引擎级**本来就极小
（§14 候选 D：AR 下 MIX/H2048=1.000、RX8/H2048=0.994）。所以预期是**两侧都很小、且代价侧更快变大**。
本脚本的任务不是证明这个预期，而是把"多小"钉成数字，并给出**转正条件**（什么负载下收益真能压过代价）。

三层模型，逐层可核：
  Layer 1  统计代价（解析＋仿真）：σ̂_p 的抽样方差、伪零方差率、方差排序翻转率
  Layer 2  气泡收益（解析）：按组调度 vs 按轨迹调度在单批内的下界收益
  Layer 3  端到端（进程内排程仿真）：回收的气泡折算成墙钟，与更新侧的估计噪声代价对比
  Layer 4  训练侧同栈均衡（2026-11-02 第二批补）：verl `use_prefix_grouper` 的组级
           LDM 划分与默认逐条 LDM 划分的残余不均衡；组完整性；int64 饱和边界

墙钟是任意虚拟秒，不可与任何论文报值相除（§19.1 规则一）。
只用标准库；随机性全部来自 random.Random(seed)。

**Layer 4 的 LDM 复刻已与真实 verl 逐位核对**：在 180 组（60 组输入 × dp∈{2,4,8}）
随机输入上，`_karmarkar_karp` 与 `get_seqlen_balanced_partitions(..., equal_size=True)`
给出完全相同的分区（见 readme「复现」节的交叉验证记录）。对照组不依赖 torch。

用法：
    python3 group_split_tradeoff.py             # 全量重跑并覆盖 summary.json
    python3 group_split_tradeoff.py --verify    # 重跑并与归档 summary.json 逐字段比对
    python3 -m unittest test_group_split_tradeoff -v
"""

import json
import math
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY_PATH = os.path.join(HERE, 'summary.json')

SEEDS = list(range(30))
N_ITERS = 2000          # 每个种子的 bootstrap 轮数
THETA_VAR = 1.0         # 组内真实均值方差（题目难度）
GLOBAL_MEAN = 0.5       # 全局平均正确率
REWARD_VAR = 0.25       # 单条响应的奖励方差（0/1 奖励的上界）

# 语料里 GRPO 族实际使用的组大小（verl／OpenRLHF 默认 8；语料内 16 是常见上限）
GROUP_SIZES = [4, 8, 16]

# ---------------------------------------------------------------------------
# 与 verl 源码对齐的常量（训练侧均衡，2026-11-02 第二批补）
# ---------------------------------------------------------------------------
# verl/utils/seqlen_balancing.py:46  calculate_workload(l) = 24576*l + l**2
#   （按 hidden_size=4096 的 dense 7B 标定，与本仓库 §8 固定的 verl 快照逐字一致）
WORKLOAD_LINEAR, WORKLOAD_QUAD = 24576, 1
# torch 默认整数 dtype 为 int64，`.item()` 跨越 2^63 时饱和
INT64_MAX = 2 ** 63 - 1
DP_SIZES = (2, 4, 8)      # 训练侧均衡的分区数（= DP rank 数）
N_GROUPS_TRAIN = 32       # 一批 prompt 数（rollout batch，÷ dp_size 后仍为整数）


def calculate_workload(seqlen):
    """verl `calculate_workload` 的标量版（seqlen_balancing.py:46）。

    注意返回值在 verl 里是 **torch int64**，本函数用 Python int 复刻同一溢出语义，
    以便在无 torch 的环境下也能核「纳秒截断」这一条。
    """
    return WORKLOAD_LINEAR * seqlen + WORKLOAD_QUAD * seqlen * seqlen


def workload_int64(seqlen):
    """verl `int(calculate_workload(torch.tensor([l])).item())` 的复刻（含 int64 饱和）。

    常规长度下与 `calculate_workload` 相同；只有 exact 值越过 2^63 时二者才分道，
    这正是本层要量出来的边界（见 `workload_saturation_len`）。
    """
    return min(calculate_workload(seqlen), INT64_MAX)


def workload_saturation_len():
    """exact workload 越过 int64 上界的最短序列长度（饱和从此开始）。

    `calculate_workload` 自带 docstring 说常数按 7B/hidden 4096 标定，
    这里的返回值是"该近似式在 int64 下开始失效的位置"，不是任何模型的真实限制。
    """
    lo, hi = 1, 2
    while calculate_workload(hi) <= INT64_MAX:
        lo, hi = hi, hi * 2
    while lo < hi:            # 不变式：workload(lo) <= MAX < workload(hi)
        mid = (lo + hi) // 2
        if calculate_workload(mid) <= INT64_MAX:
            lo = mid + 1
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------------------
# Layer 1：统计代价
# ---------------------------------------------------------------------------
def sigma_hat_draws(n, p, rng, iters=N_ITERS):
    """拆成 p 个分片后各分片的样本标准差——注意是**各分片自己的**，不是全组的。

    模型：组内 n 条响应的奖励 ~ 组均值 θ_g + 噪声，θ_g ~ N(0, THETA_VAR)。
    拆组后每个分片只有 ~n/p 条，用它自己的 mean/std 做归一化。
    返回全部分片的 σ̂ 序列。
    """
    theta = rng.gauss(0.0, math.sqrt(THETA_VAR))
    rewards = [theta + rng.gauss(0.0, math.sqrt(REWARD_VAR)) for _ in range(n)]
    rng.shuffle(rewards)
    bounds = [round(i * n / p) for i in range(p + 1)]
    out = []
    for s in range(p):
        piece = rewards[bounds[s]:bounds[s + 1]]
        if len(piece) < 2:
            out.append(0.0)
            continue
        out.append(statistics.pstdev(piece))
    return out


def stats_distribution(n, p, rng, iters=N_ITERS):
    vals = []
    for _ in range(iters):
        s = sigma_hat_draws(n, p, rng, iters=1)
        vals.extend(s)
    mean = statistics.mean(vals)
    var = statistics.pvariance(vals)
    if p <= 1:
        return {'mean': mean, 'var': var}
    return {'mean': mean, 'var': var}


def analytic_var_ratio(n, p):
    """χ² 近似下的方差比：σ̂²·(k−1)/σ² ~ χ²_{k−1}，Var(σ̂) ∝ 1/(2(k−1))。

    全组 k=n，分片 k≈n/p。故 Var(σ̂_分片)/Var(σ̂_全组) ≈ (n−1)/(n/p−1)。
    这是本实验唯一的解析式，其余全用仿真与源码读出的机制。
    """
    k_whole = n - 1
    k_frag = n / float(p) - 1
    if k_frag <= 0:
        return float('inf')
    return k_whole / k_frag


def pseudo_zero_variance_rate(n, p, rng, iters=N_ITERS):
    """伪零方差率：某个分片看起来零方差（全部奖励相同）。

    **只统计分片样本量 ≥ 2 的情形**：分片只有 1 条时样本标准差未定义，
    那不是"伪零方差"，是定义域外（初版把它算进去，给出恒为 1.0 的伪读数）。
    """
    hits = 0
    total = 0
    k_frag = int(n // p)
    if k_frag < 2:
        return None
    for _ in range(iters):
        s = sigma_hat_draws(n, p, rng, iters=1)
        total += 1
        if any(v == 0.0 for v in s):
            hits += 1
    return hits / float(total)


def variance_order_flip_rate(n, p, rng, iters=N_ITERS, n_groups=2):
    """**已撤回的口径**：基于"组内最大值"的跨组排序翻转率。

    保留函数与归档数值是为了留证：`_iid_counterfactual` 的识别性检验显示，
    这个指标对组难度方差**零敏感**（0 到 25 的档位给出同一个翻转率），
    因此在 iid 假设下也翻 ~1/3——它量的是 σ̂ 与组内最大值之间的统计耦合。
    **正文与判据均不再使用它**，改用 `advantage_inflation_fixed_offset`。
    """
    flips = 0
    pairs = 0
    for _ in range(iters):
        # 每组一个题目难度、同一难度下抽 n 条
        groups = []
        for _ in range(n_groups):
            theta = rng.gauss(0.0, math.sqrt(THETA_VAR))
            groups.append([theta + rng.gauss(0.0, math.sqrt(REWARD_VAR)) for _ in range(n)])
        # 组级归一化：每组自己的 mean/std（两臂共用同一批奖励）
        whole = [statistics.mean(g) for g in groups]
        whole_sd = [statistics.pstdev(g) + 1e-6 for g in groups]
        frag = []
        for g in groups:
            bounds = [round(i * n / p) for i in range(p + 1)]
            pieces = [g[bounds[k]:bounds[k + 1]] for k in range(p)]
            if any(len(pc) < 2 for pc in pieces):
                break
            frag.append([statistics.mean(pc) for pc in pieces])
        if len(frag) < n_groups:
            continue
        # 跨组比较：组 A 的"最大真值样本"在拆组后落在哪个分片，其归一化幅度是否仍大于组 B
        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                a_whole = (max(groups[i]) - whole[i]) / whole_sd[i]
                b_whole = (max(groups[j]) - whole[j]) / whole_sd[j]
                a_frag, b_frag = [], []
                for k in range(p):
                    pc = groups[i][round(k * n / p):round((k + 1) * n / p)]
                    a_frag.append((max(pc) - statistics.mean(pc)) / (statistics.pstdev(pc) + 1e-6))
                    pc2 = groups[j][round(k * n / p):round((k + 1) * n / p)]
                    b_frag.append((max(pc2) - statistics.mean(pc2)) / (statistics.pstdev(pc2) + 1e-6))
                pairs += 1
                if (a_whole > b_whole) != (max(a_frag) > max(b_frag)):
                    flips += 1
    return flips / float(pairs) if pairs else 0.0


# ---------------------------------------------------------------------------
# Layer 2：气泡收益（解析上下界）
# ---------------------------------------------------------------------------
def group_vs_traj_rescue(group_lens, cap_mult=1):
    """按组调度 vs 按轨迹调度，同一批工作量下的 makespan。

    容量取 **组数 × 组大小 × cap_mult**。cap_mult=1 时整批恰好一次性放得下，
    两臂 makespan 必然相同（没有空位可回收）——这正是"单批内气泡很小"的形式来源；
    因此本函数在 cap_mult>1 时才有意义：多出来的容量代表连续批处理里
    "已完成的请求被新请求替换"的那部分槽位。

    在 cap_mult>1 下两臂差别只有一处：组调度要等整组结束才归还它那 n 个槽位，
    轨迹调度每完成一条即刻归还。
    """
    n_per = len(group_lens[0])
    cap = int(round(len(group_lens) * n_per * cap_mult))
    t_group = _makespan([n_per] * len(group_lens),
                        [max(g) for g in group_lens], True, cap=cap)
    t_traj = _makespan([1] * (len(group_lens) * n_per),
                       [x for g in group_lens for x in g], False, cap=cap)
    return {
        'group_makespan': t_group,
        'traj_makespan': t_traj,
        'rescue_ratio': (t_group - t_traj) / t_group if t_group else 0.0,
    }


def _makespan(slot_counts, durations, group_level, cap=None, slo=None):
    """固定工作量的 makespan：cap 个槽位，任务按到达顺序排队，能放下就放。

    - group_level=True：一组内的 n 个成员必须**全部完成**才一起归还槽位。
    - group_level=False：每完成一条立刻归还它自己的那一格。

    两者吃到完全相同的任务列表，唯一差别是归还时机。工作量若一次放得下
    （总量 ≤ cap），两种调度必然给出同一个 makespan——差别只出现在**排队**时。
    """
    n = len(slot_counts)
    cap = cap if cap is not None else sum(slot_counts)
    n_per = slot_counts[0] if group_level else 1
    free = cap
    running = []
    pending = list(range(n))
    group_left, group_slots = {}, {}
    if group_level:
        for i in range(n):
            g = i // n_per
            group_left[g] = group_left.get(g, 0) + 1
            group_slots[g] = group_slots.get(g, 0) + slot_counts[i]
    now = 0.0
    end = 0.0
    while pending or running:
        for i in list(pending):
            if slot_counts[i] <= free:
                g = (i // n_per) if group_level else i
                running.append((now + durations[i], slot_counts[i], g))
                free -= slot_counts[i]
                pending.remove(i)
        if not running:
            break
        running.sort()
        t, s_i, g = running.pop(0)
        now = max(now, t)
        end = max(end, t)
        if group_level:
            group_left[g] -= 1
            if group_left[g] == 0:
                free += group_slots[g]
        else:
            free += s_i
    return max(end, 1e-9)


def grouping_tax(fragment_lens, backlog_groups=4):
    """**分组税**：同一 workload 在"整组原子"与"逐条连续"两种粒度下的 makespan 比。

    定义（三条口径同时成立，否则不可比）：
      * slot = 引擎并发容量单位，**一条轨迹跑起来占 1 个 slot**，时长 = 它自己的长度；
      * **组臂**：整组 n 条一起进引擎占 n 个 slot，**最慢那条跑完才一起归还**；
        但这 n 条的时刻表与轨迹臂相同——组臂不等于是"只跑最长那条"，它是
        "n 条同时跑，归还被绑定在最长那条"；
      * **轨迹臂**：n 条同时进，跑完哪条归还哪条；
      * 两臂吃**同一批轨迹**（条数、长度完全相同），容量相同 = 一批的 slot 数。

    两者唯一差别就是归还粒度。容量 ≤ 一批 slot 数时，组臂因归还被绑定而反复空等，
    轨迹臂能把腾出的 slot 立刻给下一条。差值即分组税。

    **这是上界**：假设后续轨迹随时可接、无跨批版本一致性与 KV 搬运成本。
    """
    n_per = len(fragment_lens[0])
    n_groups = len(fragment_lens)
    cap = n_groups * n_per
    flat = []
    for b in range(backlog_groups):
        for g in fragment_lens:
            flat.extend(g)

    # ---- 组臂：以组为单位排程，一组占 n_per slot，时长 = 组最大 ----
    group_dur = []
    for b in range(backlog_groups):
        for g in fragment_lens:
            group_dur.append(max(g))
    t_group = _makespan([n_per] * len(group_dur), group_dur, True, cap=cap)

    # ---- 轨迹臂：同样的轨迹逐条排程 ----
    t_traj = _makespan([1] * len(flat), flat, False, cap=cap)

    return {
        'group_makespan': t_group,
        'traj_makespan': t_traj,
        'grouping_tax': (t_group - t_traj) / t_group if t_group else 0.0,
        'backlog_groups': backlog_groups,
        'work_total': sum(flat),
        'group_arm_busy': sum(group_dur) * n_per,
    }


# ---------------------------------------------------------------------------
# Layer 4：训练侧同栈均衡（verl 源码机制，2026-11-02 第二批补）
# ---------------------------------------------------------------------------
def cross_group_pair_flip(n, p, rng, iters=N_ITERS, theta_var=THETA_VAR):
    """**跨组排序翻转（配对口径）**：两条样本各自来自一个组，比较整组归一化与
    拆片后片内归一化两种口径下的跨组大小关系是否被读反。

    `theta_var` 为 0 时两组同分布（无组难度差异）——这是识别性检验的关键参数。
    """
    flips = 0
    total = 0
    for _ in range(iters):
        tx = rng.gauss(0.0, math.sqrt(theta_var))
        ty = rng.gauss(0.0, math.sqrt(theta_var))
        g = [tx + rng.gauss(0.0, math.sqrt(REWARD_VAR)) for _ in range(n)]
        h = [ty + rng.gauss(0.0, math.sqrt(REWARD_VAR)) for _ in range(n)]
        aw = (g[0] - statistics.mean(g)) / (statistics.pstdev(g) + 1e-6)
        bw = (h[0] - statistics.mean(h)) / (statistics.pstdev(h) + 1e-6)
        bounds = [round(i * n / p) for i in range(p + 1)]
        pa, pb = g[bounds[0]:bounds[1]], h[bounds[0]:bounds[1]]
        if len(pa) < 2 or len(pb) < 2:
            continue
        a2 = (g[0] - statistics.mean(pa)) / (statistics.pstdev(pa) + 1e-6)
        b2 = (h[0] - statistics.mean(pb)) / (statistics.pstdev(pb) + 1e-6)
        total += 1
        if (aw > bw) != (a2 > b2):
            flips += 1
    return flips / float(total) if total else 0.0


def fragment_scale_mismatch(n, p, rng, iters=N_ITERS):
    """**尺度错配的直接度量**：每个分片的 σ̂ 与整组 σ 的比值分布。

    比值 = 1 表示分片与整组同一刻度，拆组不改变任何样本进梯度时的幅度；
    比值 ≠ 1 表示同一原始偏离在片内被放大／缩小。这是"拆组破坏跨组可比"的**唯一有识别性的量**
    （排序类指标对本参数零敏感，见 `_iid_counterfactual`）。
    """
    ratios = []
    for _ in range(iters):
        g = [rng.gauss(0.0, math.sqrt(REWARD_VAR)) for _ in range(n)]
        sd_w = statistics.pstdev(g)
        bounds = [round(i * n / p) for i in range(p + 1)]
        for k in range(p):
            pc = g[bounds[k]:bounds[k + 1]]
            if len(pc) > 1 and statistics.pstdev(pc) > 0:
                ratios.append(sd_w / statistics.pstdev(pc))
    ratios.sort()
    L = len(ratios)
    return {
        'mean': statistics.mean(ratios),
        'median': statistics.median(ratios),
        'p10': ratios[int(0.10 * L)],
        'p90': ratios[int(0.90 * L)],
        'n': L,
    }


def advantage_inflation_fixed_offset(n, p, rng, iters=N_ITERS, offset=-2.0):
    """固定原始偏离量下 |adv| 的绝对值：整组归一化 vs 拆片后片内归一化。

    **这才是"尺度被打散"的可测量化**：同一个原始偏离，进梯度时被放大或缩小多少。
    （`cross_group_pair_flip` 量的是排序，排序在带组效应时也不随组难度变化，见 readme 修正四。）
    """
    whole, frag = [], []
    for _ in range(iters):
        q = [offset] + [rng.gauss(1.0, math.sqrt(REWARD_VAR)) for _ in range(n - 1)]
        whole.append(abs(q[0] - statistics.mean(q)) / statistics.pstdev(q))
        bounds = [round(i * n / p) for i in range(p + 1)]
        pc = q[bounds[0]:bounds[1]]
        if len(pc) > 1:
            frag.append(abs(pc[0] - statistics.mean(pc)) / statistics.pstdev(pc))
    return {
        'whole_abs_adv': statistics.mean(whole),
        'fragment_abs_adv': statistics.mean(frag) if frag else None,
        'inflation': (statistics.mean(frag) / statistics.mean(whole)) if frag else None,
    }


def _iid_counterfactual(n=8, iters=300, seeds=8):
    """识别性检验：翻转率是否随组难度方差变化。

    **如果拆组破坏的是"跨组尺度"，那么组难度方差越大、翻转率必须越高。**
    实测（见 summary）：翻转率在组难度方差 0／0.25／1／4／25 下恒为
    p=2 约 14%、p=4 约 23%——**完全不随组效应变化**。结论：这个指标量的是
    拆片后 σ̂ 的估计噪声，不是组尺度的错配；旧口径（基于组内最大值）同样无效，
    它在 iid 下给出 32.9%／49.3%。
    """
    rng_seeds = list(range(seeds))
    rows = {}
    for tv in (0.0, 0.25, 1.0, 4.0, 25.0):
        per_p = {}
        for p in (2, 4):
            per_p['n8_p%d' % p] = statistics.mean(
                cross_group_pair_flip(n, p, random.Random(s), iters=iters, theta_var=tv)
                for s in rng_seeds)
        rows['theta_var_%g' % tv] = per_p
    old = {
        'n8_p%d' % p: statistics.mean(
            variance_order_flip_rate(n, p, random.Random(s), iters=iters) for s in rng_seeds)
        for p in (2, 4)
    }
    spread_p2 = max(v['n8_p2'] for v in rows.values()) - min(v['n8_p2'] for v in rows.values())
    spread_p4 = max(v['n8_p4'] for v in rows.values()) - min(v['n8_p4'] for v in rows.values())
    return {
        'pair_flip_by_group_variance': rows,
        'max_minus_min_across_group_variance': {'n8_p2': spread_p2, 'n8_p4': spread_p4},
        'old_max_based_flip_reference': old,
        'reading': ('翻转率跨五个组难度方差档位的极差只有 p=2 %.4f、p=4 %.4f（噪声级），'
                    '即该指标对"组效应"零敏感。所以 §13.30 结论一不能用翻转率作为'
                    '"跨组尺度被打散"的证据，应改用 `advantage_inflation_fixed_offset`'
                    '（同一原始偏离在片内被缩放到原来的 %.2f／%.2f 倍）。'
                    % (spread_p2, spread_p4,
                       advantage_inflation_fixed_offset(n, 2, random.Random(0), iters=1000)['inflation'],
                       advantage_inflation_fixed_offset(n, 4, random.Random(0), iters=1000)['inflation'])),
    }


def _karmarkar_karp(weights, k, equal_size=True):
    """`get_seqlen_balanced_partitions` 的 LDM 复刻（seqlen_balancing.py:49-190）。

    只复刻 `equal_size=True` 的分支（两个调用点都用它）：
    先把权重升序排序、每 k 个一组塞进堆，再反复合并堆顶两个 spread 最大的集合。
    这不是"顺手写的贪心"——用 LDM 而不是最优划分，正是为什么组级均衡会留下残余不均衡。
    """
    import heapq

    if equal_size:
        assert len(weights) % k == 0, 'equal_size 要求 len % k == 0'

    class Set_:
        __slots__ = ('s', 'items')

        def __init__(self):
            self.s = 0
            self.items = []

        def merge(self, other):
            self.items.extend(other.items)
            self.s += other.s

        def __lt__(self, other):
            if self.s != other.s:
                return self.s < other.s
            if len(self.items) != len(other.items):
                return len(self.items) < len(other.items)
            return self.items < other.items

    class State:
        __slots__ = ('k', 'sets')

        def __init__(self, items, k):
            self.k = k
            self.sets = [Set_() for _ in range(k)]
            for i, (idx, w) in enumerate(items):
                self.sets[i].items.append((idx, w))
                self.sets[i].s += w
            self.sets.sort(reverse=True)

        def merge(self, other):
            for i in range(self.k):
                self.sets[i].merge(other.sets[self.k - 1 - i])
            self.sets.sort(reverse=True)

        def __lt__(self, other):
            a = self.sets[0].s - self.sets[-1].s
            b = other.sets[0].s - other.sets[-1].s
            if a != b:
                return a > b
            return self.sets[0] > other.sets[0]

    ordered = sorted((w, i) for i, w in enumerate(weights))
    heap = []
    for off in range(0, len(ordered), k):
        items = [(ordered[off + i][1], ordered[off + i][0]) for i in range(k)]
        heapq.heappush(heap, State(items, k))
    while len(heap) > 1:
        st0 = heapq.heappop(heap)
        st1 = heapq.heappop(heap)
        st0.merge(st1)
        heapq.heappush(heap, st0)
    return [sorted(idx for idx, _ in st.items) for st in heap[0].sets]


def _imbalance(partitions, weights):
    """最大分区权重 / 平均分区权重 − 1（0 表示完全均衡）。"""
    sums = [sum(weights[i] for i in p) for p in partitions]
    mean = sum(sums) / float(len(sums))
    return (max(sums) - mean) / mean if mean else 0.0


def _cut_by_uid_order(weights, k):
    """不均衡对照组：按 uid 连续切（默认布局且不做任何均衡时的形态）。"""
    n = len(weights)
    bounds = [round(i * n / k) for i in range(k + 1)]
    return [list(range(bounds[i], bounds[i + 1])) for i in range(k)]


def train_side_balance(group_lens, dp_size, grouped, per_sample_weights=None):
    """训练侧负载均衡：某批组在各 DP 副本间的残余不均衡。

    - grouped=True ：verl `use_prefix_grouper` 路径（ray_trainer.py:1159-1181）。
      组先聚合成一个整体权重，再对"组"做 LDM 划分，组内成员一起走。
    - grouped=False：默认路径（ray_trainer.py:1200），按逐条权重做 LDM 划分，
      由于 uid 连续且样本数/组数在默认布局下整除，等分会把每组切成若干份。
    """
    n_per = len(group_lens[0])
    flat = [x for g in group_lens for x in g]
    per_sample = per_sample_weights or [workload_int64(x) for x in flat]
    if grouped:
        group_w = [sum(per_sample[i * n_per:(i + 1) * n_per]) for i in range(len(group_lens))]
        parts = _karmarkar_karp(group_w, dp_size)
        return {
            'model': 'group_level_ldm',
            'imbalance': _imbalance(parts, group_w),
            'groups_intact': True,
            'n_partitions': dp_size,
            'partition_sizes': [len(p) for p in parts],
        }
    parts = _karmarkar_karp(per_sample, dp_size)
    # 组完整性按 **uid 连续布局** 判：默认路径下样本以 (uid 重复 n 次) 的形式排列，
    # 于是"某组的全部成员落在同一分区"等价于"存在一个分区包含该整段连续下标"。
    intact = 0
    for i in range(len(group_lens)):
        members = set(range(i * n_per, (i + 1) * n_per))
        if any(members <= set(p) for p in parts):
            intact += 1
    return {
        'model': 'per_sample_ldm',
        'imbalance': _imbalance(parts, per_sample),
        'groups_intact': intact == len(group_lens),
        'groups_intact_count': intact,
        'n_groups': len(group_lens),
        'n_partitions': dp_size,
        'partition_sizes': [len(p) for p in parts],
    }


def greedy_split(p, k):
    """GRPO 的**朴素等价拆分**：p 条按出现顺序尽量均分到 k 片（无损）。"""
    k = max(1, min(k, p))
    bounds = [round(i * p / k) for i in range(k + 1)]
    return [list(range(bounds[i], bounds[i + 1])) for i in range(k)]


def karmarkar_split(weights, k):
    """GRPO 的**均衡拆分**：允许打乱同组响应的顺序，按 LDM 分到 k 片。

    代价是损失了"抽样顺序"这一语义（某些 trainer 依赖它做同一 prefix 的连排）。
    """
    k = max(1, min(k, len(weights)))
    while len(weights) % k != 0:
        k -= 1
    return _karmarkar_karp(weights, k)


# ---------------------------------------------------------------------------
# Layer 3：把两侧放到同一个判据下
# ---------------------------------------------------------------------------
def sample_group_lens(rng, n_groups=16, n_per=8, mean_len=50.0, cv=0.6):
    """一批组：组内每条长度独立抖动。cv 是组内长度离散度。"""
    out = []
    for _ in range(n_groups):
        out.append([max(1.0, rng.gauss(mean_len, mean_len * cv)) for _ in range(n_per)])
    return out


def _layer3_verdict(out):
    """把统计代价与调度收益放到同一判据下（判据是必要的，不是充分的）。"""
    whole = out['layer1_statistics']['n8_p1']['sigma_var']
    split = out['layer1_statistics']['n8_p2']['sigma_var']
    snr_ratio = math.sqrt(whole / split)
    max_tax = max(v['max'] for k, v in out['layer2_grouping_tax'].items() if k.startswith('cv'))
    return {
        'sigma_var_inflation_p2': split / whole,
        'snr_ratio_p2': snr_ratio,
        'snr_loss_ratio_p2': 1.0 - snr_ratio,
        'best_case_grouping_tax': max_tax,
        'verdict': '净负' if max_tax < (1.0 - snr_ratio) else '需逐负载判断',
        'note': ('判据：拆组能拿回的墙钟（分组税上限）必须超过优势信噪比的相对下降。'
                 '这是必要条件而非充分条件——它还没算上拆组带来的 KV 搬运、'
                 '跨批版本一致性与组内比较语义丢失。'),
    }


def build_summary():
    out = {
        'experiment': '§15 Q5 拆 GRPO 组的代价与收益',
        'doc': 'notebook/rollout-performance-research.md §13.30',
        'layer1_statistics': {},
        'layer2_grouping_tax': {},
        'layer3_verdict': {},
        'sanitization': {},
    }

    # ---- Layer 1：统计代价 ----
    whole_var = {}
    for n in GROUP_SIZES:
        whole_var[n] = statistics.mean(
            stats_distribution(n, 1, random.Random(s))['var'] for s in SEEDS[:8])
    for n in GROUP_SIZES:
        for p in (1, 2, 4):
            rows = [stats_distribution(n, p, random.Random(s)) for s in SEEDS[:8]]
            mean_var = statistics.mean(r['var'] for r in rows)
            out['layer1_statistics']['n%d_p%d' % (n, p)] = {
                'sigma_mean': statistics.mean(r['mean'] for r in rows),
                'sigma_var': mean_var,
                'analytic_var_ratio': (None if n // p < 2 else analytic_var_ratio(n, p)),
                'sim_var_ratio': mean_var / whole_var[n],
                'fragment_size': n // p,
                'note': ('分片样本量 < 2 时样本标准差未定义，本格不可用'
                         if n // p < 2 else ''),
            }
    pz = {}
    for n in GROUP_SIZES:
        for p in (2, 4):
            vals = [pseudo_zero_variance_rate(n, p, random.Random(s)) for s in SEEDS[:8]]
            vals = [v for v in vals if v is not None]
            pz['n%d_p%d' % (n, p)] = statistics.mean(vals) if vals else None
    out['layer1_statistics']['pseudo_zero_rate'] = pz
    out['layer1_statistics']['order_flip_rate'] = {
        'n8_p%d' % p: statistics.mean(
            variance_order_flip_rate(8, p, random.Random(s), iters=300) for s in SEEDS[:8])
        for p in (1, 2, 4)
    }
    # 配对口径（第二批修正）：真正的跨组翻转，必须有组效应才会翻
    out['layer1_statistics']['pair_flip_rate'] = {
        'n8_p%d' % p: statistics.mean(
            cross_group_pair_flip(8, p, random.Random(s), iters=400) for s in SEEDS[:8])
        for p in (2, 4)
    }
    # iid 反事实：拿掉组效应后，旧口径依旧翻、新口径不翻——这是"旧指标无效"的判据
    out['layer1_statistics']['iid_counterfactual'] = _iid_counterfactual()
    out['layer1_statistics']['advantage_inflation_fixed_offset'] = {
        'n8_p%d' % p: advantage_inflation_fixed_offset(8, p, random.Random(0), iters=2000)
        for p in (2, 4)
    }
    out['layer1_statistics']['fragment_scale_mismatch'] = {
        'n8_p%d' % p: fragment_scale_mismatch(8, p, random.Random(0), iters=2000)
        for p in (2, 4)
    }

    # ---- Layer 2：分组税（拆组能拿回的墙钟上限） ----
    for cv in (0.2, 0.6, 1.5):
        rows = [grouping_tax(sample_group_lens(random.Random(s), cv=cv)) for s in SEEDS]
        out['layer2_grouping_tax']['cv%.1f' % cv] = {
            'mean': statistics.mean(r['grouping_tax'] for r in rows),
            'max': max(r['grouping_tax'] for r in rows),
            'min': min(r['grouping_tax'] for r in rows),
        }
    for n_per in (2, 4, 8):
        rows = [grouping_tax(sample_group_lens(random.Random(s), n_per=n_per)) for s in SEEDS]
        out['layer2_grouping_tax']['n%d' % n_per] = {
            'mean': statistics.mean(r['grouping_tax'] for r in rows),
        }
    for backlog in (1, 4, 16):
        rows = [grouping_tax(sample_group_lens(random.Random(s)), backlog_groups=backlog)
                for s in SEEDS]
        out['layer2_grouping_tax']['backlog%d' % backlog] = {
            'mean': statistics.mean(r['grouping_tax'] for r in rows),
        }

    # ---- 与 §14 候选 D 的口径对齐检验 ----
    # 候选 D 测的是"一批之内"长度异质对 makespan 的影响，容量足够、无排队。
    # 同一模型在该口径下必须给出 0，否则说明两套模型量的不是同一件事。
    out['consistency_with_candidate_d'] = {
        'single_batch_one_group': grouping_tax(
            [[80.0, 8.0, 30.0, 61.0, 19.0, 48.0, 55.0, 25.0]], backlog_groups=1)['grouping_tax'],
        'single_batch_all_singletons': grouping_tax(
            [[x] for x in (80.0, 8.0, 30.0, 61.0, 19.0, 48.0, 55.0, 25.0)],
            backlog_groups=1)['grouping_tax'],
        'reading': ('候选 D 的口径（一批、容量足够、无排队）在本模型下同样给出 0——'
                    '两者不矛盾：候选 D 说明"批内长度异质被引擎吸收"，本模型说明'
                    '"组粒度只在有排队时才构成约束"。真正影响结论的是 backlog 假设。'),
    }

    # ---- Layer 3：判据 ----
    out['layer3_verdict'] = _layer3_verdict(out)

    # ---- Layer 4：训练侧同栈均衡（verl use_prefix_grouper 源码机制） ----
    out['layer4_train_side_balance'] = {'by_cv': {}, 'saturation': {}}
    for cv in (0.2, 0.6, 1.2):
        grouped_rows, no_balance_rows = [], []
        for seed in SEEDS:
            lens = sample_group_lens(random.Random(seed), n_groups=N_GROUPS_TRAIN, cv=cv)
            flat = [x for gl in lens for x in gl]
            per_sample = [workload_int64(x) for x in flat]
            for dp in DP_SIZES:
                grouped_rows.append(train_side_balance(lens, dp, grouped=True)['imbalance'])
                naive = _cut_by_uid_order(per_sample, dp)
                no_balance_rows.append(_imbalance(naive, per_sample))
        out['layer4_train_side_balance']['by_cv']['cv%.1f' % cv] = {
            'grouped_ldm_mean': statistics.mean(grouped_rows),
            'grouped_ldm_max': max(grouped_rows),
            'no_balance_cut_by_uid_mean': statistics.mean(no_balance_rows),
            'no_balance_cut_by_uid_max': max(no_balance_rows),
            'grouped_vs_no_balance_ratio': (statistics.mean(grouped_rows)
                                            / statistics.mean(no_balance_rows)
                                            if statistics.mean(no_balance_rows) else None),
            'n_measurements': len(grouped_rows),
            'reading': ('组级 LDM 与"不均衡、按 uid 连续切"同口径分母；'
                        '比值即"分组约束让均衡变差了多少"——这是组完整性在训练侧的直接定价。'),
        }

    # 组完整性：默认路径（逐条 LDM）下有多少组被拆到两个以上分区
    cut_rows = []
    for seed in SEEDS:
        lens = sample_group_lens(random.Random(seed), n_groups=N_GROUPS_TRAIN, cv=0.6)
        for dp in DP_SIZES:
            r = train_side_balance(lens, dp, grouped=False)
            cut_rows.append(1.0 - float(r['groups_intact_count']) / r['n_groups'])
    out['layer4_train_side_balance']['default_group_split_rate'] = statistics.mean(cut_rows)
    out['layer4_train_side_balance']['default_group_split_rate_reading'] = (
        '逐条 LDM 排序跨过组边界，故默认路径几乎必然把每组拆开——'
        '这与 §13.30 结论一（拆组破坏跨组尺度）是同一机制在**训练侧**的复现，'
        '也解释了为什么 `use_prefix_grouper` 必须改用组级均衡。')

    # 纳秒截断：常规长度下损失恒为 0，真正饱和点远在物理范围外
    sat = workload_saturation_len()
    out['layer4_train_side_balance']['saturation'] = {
        'int64_max_workload': INT64_MAX,
        'saturated_at_seqlen': sat,
        'workload_at_first_saturated': calculate_workload(sat),
        'workload_at_last_exact': calculate_workload(sat - 1),
        'truncation_loss_at_512': 0,
        'truncation_loss_at_32768': 0,
        'reading': ('`int(...item())` 的截断损失只在 exact 值越过 2^63 时出现，'
                    '对应序列长度 %d（%d 仍精确、%d 已饱和）。所以"截断改变均衡决策"'
                    '这条在可训练范围内不成立；真实可观察的机制是**离散性**——'
                    '组级均衡一次移动整组 n 条，单组权重是 LDM 的最小不可分单位。'
                    % (sat, sat - 1, sat)),
    }

    # 拆分实现：朴素等分与均衡拆分各自留下的不均衡
    split_rows = {}
    for cv in (0.2, 0.6, 1.2):
        naive, balanced = [], []
        for seed in SEEDS:
            lens = sample_group_lens(random.Random(seed), n_groups=N_GROUPS_TRAIN, cv=cv)
            for g in lens:
                w = [workload_int64(x) for x in g]
                naive.append(_imbalance(greedy_split(len(g), 2), w))
                balanced.append(_imbalance(karmarkar_split(w, 2), w))
        split_rows['cv%.1f' % cv] = {
            'naive_split_imbalance': statistics.mean(naive),
            'balanced_split_imbalance': statistics.mean(balanced),
            'ratio': (statistics.mean(naive) / statistics.mean(balanced)
                      if statistics.mean(balanced) else None),
        }
    out['layer4_train_side_balance']['split_impl'] = split_rows

    # ---- 自查清理（记录在案，防止把过程当结论） ----
    out['sanitization'] = {
        'removed_claims': [
            '初版把"伪零方差率"写成 p=4 时恒为 1.0：那实际是"4 条拆成 4 片每片 1 条"，'
            '样本标准差未定义（statistics.pstdev 单点返回 0），不是统计现象而是定义域外。',
            '初版 _makespan 在 pending 为空时把 now 直接推到全部结束，吃掉了轨迹调度的优势；'
            '该特判已删除。',
            '初版 group_vs_traj_rescue 让两臂任务数不同（组臂 n 条、轨迹臂 n×n_per 条），'
            '相当于给轨迹臂更多工作，量化出来的"收益"不可信；改为两臂吃同一批工作量。',
        ],
        'definition_notes': [
            '修正后：工作量一次放得下（总量 ≤ 容量）时两种调度必然同 makespan，'
            '所以必须有 backlog 才能测出分组税；backlog 是构造参数，不是语料实测值。',
            'capacity 恒设为"一批"的大小；容量低于一组的用例没有物理意义（引擎连一组都放不下），已从结果中剔除。',
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
    for k, v in out['layer2_grouping_tax'].items():
        print('L2 %-10s grouping_tax mean=%.4f' % (k, v['mean']))
    for k in ('order_flip_rate', 'pseudo_zero_rate'):
        print('L1', k, {kk: (round(vv, 4) if vv is not None else 'n/a')
                        for kk, vv in out['layer1_statistics'][k].items()})
    for k in ('n8_p1', 'n8_p2', 'n8_p4'):
        v = out['layer1_statistics'][k]
        print('L1 %-6s sigma_var=%.4f (sim ratio %.2f / analytic %.2f)'
              % (k, v['sigma_var'], v['sim_var_ratio'], v['analytic_var_ratio']))
    l3 = out['layer3_verdict']
    print('L3 sigma_var inflation=%.2fx snr_ratio=%.3f snr_loss=%.1f%% best_tax=%.1f%%'
          % (l3['sigma_var_inflation_p2'], l3['snr_ratio_p2'],
             100 * l3['snr_loss_ratio_p2'], 100 * l3['best_case_grouping_tax']))
    print('verdict:', l3['verdict'])
    print('wrote', SUMMARY_PATH)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
