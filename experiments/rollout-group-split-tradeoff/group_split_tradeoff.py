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

墙钟是任意虚拟秒，不可与任何论文报值相除（§19.1 规则一）。
只用标准库；随机性全部来自 random.Random(seed)。

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
    """**跨组**优势排序翻转率。

    GRPO 的优势是组内归一化（verl `core_algos.py:306-321` 逐 prompt index 分组、
    减去组均值再除以组标准差），所以**组内两两排序在拆组后不变**——
    拆组真正改变的是"不同组的样本如何被放到同一个梯度更新里比较"：
    分片的 std 越小，该分片的优势幅度被放大得越多，于是跨组的大小关系被改写。
    这里量化的是跨组排序反转的比例，口径与"组内比较"分开写。
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
# Layer 3：把两侧放到同一个判据下
# ---------------------------------------------------------------------------
def sample_group_lens(rng, n_groups=16, n_per=8, mean_len=50.0, cv=0.6):
    """一批组：组内每条长度独立抖动。cv 是组内长度离散度。"""
    out = []
    for _ in range(n_groups):
        out.append([max(1.0, rng.gauss(mean_len, mean_len * cv)) for _ in range(n_per)])
    return out


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
    # 代价侧用 n8_p2 的分片方差膨胀（最保守的一次拆分）
    base_sigma_var = whole_var[8]
    split_sigma_var = out['layer1_statistics']['n8_p2']['sigma_var']
    inflation = split_sigma_var / base_sigma_var
    snr_ratio = math.sqrt(base_sigma_var / split_sigma_var)
    max_tax = max(v['max'] for k, v in out['layer2_grouping_tax'].items() if k.startswith('cv'))
    out['layer3_verdict'] = {
        'sigma_var_inflation_p2': inflation,
        'snr_ratio_p2': snr_ratio,
        'snr_loss_ratio_p2': 1.0 - snr_ratio,
        'best_case_grouping_tax': max_tax,
        'verdict': '净负' if max_tax < (1.0 - snr_ratio) else '需逐负载判断',
        'note': ('判据：拆组能拿回的墙钟（分组税上限）必须超过优势信噪比的相对下降。'
                 '这是必要条件而非充分条件——它还没算上拆组带来的 KV 搬运、'
                 '跨批版本一致性与组内比较语义丢失。'),
    }

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
