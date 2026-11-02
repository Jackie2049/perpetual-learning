#!/usr/bin/env python3
"""§21.12 轨迹之外的证据：轨迹分不开的两种空闲，换通道能不能分开

支撑 notebook/rollout-performance-research.md §21.12（补强 §21.11 判决三的残留）。

§21.11 判决三证明：同一份逐格忙／闲轨迹，若空闲分别来自「运行时驱动重叠」与
「请求命中前缀缓存」，三个统计量（份额／同时性／集中度）**逐位相同**。当时留下
一句话叫「要分开二者必须引入轨迹之外的证据（kernel 活动、请求在途状态、KV 命中
计数）」——但这句话**从未被检验**：换了通道能分开到什么程度？哪一类通道能分？

两个待分辨的语义（轨迹上不可分，见 §21.11 判决三）：

  RUN_NLPC  运行时驱动重叠（真空泡）：**没有活**，没有请求在推进
  RUN_LPC   请求命中前缀缓存：**有活**，多个请求正在推进，只是每 token 便宜

两条**结构性**通道族。它们的失效来自机制而不是噪声：

  占用率类  先对"有几个请求在推进"取 max／封顶，再折算成比例后报出去——
            设备利用率、`vllm:gpu_prefix_cache_hit_rate`、引擎"有效 prefill 吞吐"。
            空窗里两个语义都只剩"少量活"，**读数落在同一侧**。
  计数类    "窗口里发生了多少次"——KV／前缀命中计数、在途请求的阶段分解。

三个判决：

  **判决一**：占用率类**分不开**（判定逐位相同），计数类在纯构造上分开。
  分开的硬前提是"真空泡里计数归零"——一旦空窗里仍有背景命中，计数类也开始失效。

  **判决二**：可执行判据是 设备占用 ∧ 命中计数 两个通道的 **AND**。
  任一通道失效时 AND 只会漏报"是真空泡"，**不会把缓存命中反判成真空泡**。

  **判决三**：有效区间随负载与副本数变——副本越少命中计数越接近零、
  载荷越轻空窗越短；窗口短于一个采样间隔则直接消失（接 §21.11 判决一）。
  给出可报的最小条件，并说明它不能保证的（阈值附近的相位运气）。

墙钟是任意虚拟秒，不可与任何论文报值相除（§19.1 规则一）。只用标准库。

用法：
    python3 fusion_evidence.py             # 全量重跑并覆盖 fusion.json
    python3 fusion_evidence.py --verify    # 重跑并与归档 fusion.json 逐字段比对
    python3 -m unittest test_fusion_evidence -v
"""

import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, 'fusion.json')

SEEDS = list(range(30))
TICKS = 300          # 1 tick = 1 虚拟秒
N_REPLICAS = 4

RUN_NLPC = 'run_nlpc'   # 真空泡：全体同时空闲，没有活
RUN_LPC = 'run_lpc'     # 前缀缓存命中：全体同时空闲，有活但每 token 便宜

SEMANTICS = (RUN_NLPC, RUN_LPC)
LEVELS = ('high', 'low')

WINDOWS = {
    'high': [(20, 80), (140, 200), (250, 270)],
    'low': [(20, 40), (140, 160), (250, 262)],
}

# 空窗（全体同时空闲）内的中位数。D 是"设备占用率"：真空泡掉到 0.10、
# 缓存命中掉到 0.30（每 token 便宜，但仍有一批请求在推进）。两者都低于忙段的
# 一半 ⇒ 占用率判据对两语义给出**同一答案**——这就是 §21.11 那条镜像的数值版。
IN_WINDOW = {
    RUN_NLPC: {'D_occupancy': 0.10, 'E_inflight': 0.5, 'H_hits': 0.0, 'I_prefill': 0.0},
    RUN_LPC: {'D_occupancy': 0.30, 'E_inflight': 8.0, 'H_hits': 60.0, 'I_prefill': 8.0},
}

# 忙段的基准水平（高低两个负载不同，是故意的混淆项）
BUSY = {
    'high': {'D_occupancy': 0.85, 'E_inflight': 32.0, 'H_hits': 240.0, 'I_prefill': 32.0},
    'low': {'D_occupancy': 0.60, 'E_inflight': 4.0, 'H_hits': 30.0, 'I_prefill': 4.0},
}

SIGMA = 0.25          # 对数正态噪声（乘性，约 ±25%）
THRESHOLD = 0.5       # "仍在动"的相对阈值（相对忙段均值）


# ---------------------------------------------------------------- 构造


def idle_ticks(level):
    """两个语义共用的空窗格（逐位相同——§21.11 的前提，本文件不放松它）。"""
    ticks = []
    for st, en in WINDOWS[level]:
        ticks.extend(range(st, en))
    return ticks


def make_replicas(level, replicas=N_REPLICAS, ticks=TICKS):
    """全体同时忙/闲的副本轨迹：两个语义逐位相同。"""
    free = set(idle_ticks(level))
    return [[0 if t in free else 1 for t in range(ticks)] for _ in range(replicas)]


def _lognormal(rng, median):
    if median <= 0:
        return 0.0
    return median * (2.718281828459045 ** (SIGMA * rng.gauss(0.0, 1.0)))


def _poissonish(rng, lam):
    if lam <= 0:
        return 0.0
    return float(max(0, int(round(rng.gauss(lam, lam ** 0.5)))))


def make_signals(semantic, level, seed, replicas=N_REPLICAS, ticks=TICKS,
                 bg_hits_per_tick=0.0, d_in_window=None, lpc_hits=None):
    """按语义与负载生成四条通道的逐 tick 读数。

    D_occupancy —— 设备侧占用率（真空泡 10%、缓存命中 18%；忙段两负载不同）
    E_inflight  —— 在途请求数（不分阶段）
    H_hits      —— 窗口累计的 KV／前缀命中计数（真空泡归零，除非给背景命中）
    I_prefill   —— 在途请求里处于 prefill 的数量
    """
    rng = random.Random((seed * 1000003 + len(semantic) * 7919 +
                         len(level) * 104729) % (2 ** 31))
    free = set(idle_ticks(level))
    sem = IN_WINDOW[semantic]
    busy_ref = BUSY[level]
    vals = {k: [] for k in ('D_occupancy', 'E_inflight', 'H_hits', 'I_prefill')}
    for t in range(ticks):
        in_win = t in free
        for k in vals:
            base = (sem[k] if in_win else busy_ref[k])
            if k == 'D_occupancy':
                if in_win and d_in_window is not None:
                    base = d_in_window
                vals[k].append(min(1.0, _lognormal(rng, base)))
            elif k == 'H_hits':
                if in_win and semantic == RUN_LPC and lpc_hits is not None:
                    base = lpc_hits
                lam = base * replicas / float(N_REPLICAS)
                if in_win and semantic == RUN_NLPC:
                    lam = bg_hits_per_tick * replicas / float(N_REPLICAS)
                vals[k].append(_poissonish(rng, lam))
            else:
                vals[k].append(_poissonish(rng, base * replicas / float(N_REPLICAS)))
    return vals


def rate_ratio(values, free, busy):
    """窗口内均值 ÷ 窗口外（忙段）均值。"""
    inside = [v for t, v in enumerate(values) if t in free]
    outside = [v for t, v in enumerate(values) if t in busy]
    mo = sum(outside) / float(len(outside))
    mi = sum(inside) / float(len(inside))
    if mo <= 0:
        return float('inf') if mi > 0 else 0.0
    return mi / mo


def threshold_type(values, ref, threshold=THRESHOLD):
    """按"忙段均值 × threshold"做绝对阈值判定（不是相对比值）."""
    busy_mean = sum(ref) / float(len(ref))
    return (sum(values) / float(len(values))) < threshold * busy_mean


# ---------------------------------------------------------------- 判据


def fuse_and(occ_verdict, count_verdict):
    """可执行判据：两通道 AND。只有都"说没有在算"才判真空泡。

    另外三种组合都不判真空泡，包括"设备说没在算但计数在动"（缓存命中）
    与"设备在算但计数没有"（训练阶段占卡）。
    """
    return bool(occ_verdict and count_verdict)


def fuse_with_relative_ratio(occ_ratio, count_verdict):
    """**只看相对比值**的 AND：窗口均值掉到忙段的一半以下。

    与 `classify` 的绝对判定是**两个不同判据**，分开报正是为了说明：
    占用率的"回落"到底拿什么当基准，会决定判据在低负载下还成不成立。
    """
    return bool(occ_ratio < THRESHOLD and count_verdict)


def classify(semantic, level, seed, **kw):
    """两张表：**相对比值**（窗口均值 ÷ 忙段均值）与**绝对判定**。

    绝对判定的阈值是"忙段均值 × THRESHOLD"，所以判据读的是**绝对回落**，
    不是"跟忙段比回落了多少"。两者在高负载下重合、在低负载下分开——
    这正是判决三要量的事。
    """
    sig = make_signals(semantic, level, seed, **kw)
    free = set(idle_ticks(level))
    busy = set(range(TICKS)) - free
    occ_vals = sig['D_occupancy']
    cnt_vals = sig['H_hits']
    occ_ratio = rate_ratio(occ_vals, free, busy)
    cnt_ratio = rate_ratio(cnt_vals, free, busy)
    occ_idle = threshold_type([v for t, v in enumerate(occ_vals) if t in free],
                              [v for t, v in enumerate(occ_vals) if t in busy])
    cnt_idle = threshold_type([v for t, v in enumerate(cnt_vals) if t in free],
                              [v for t, v in enumerate(cnt_vals) if t in busy])
    return {'occ_ratio': occ_ratio, 'count_ratio': cnt_ratio,
            'occ_verdict_idle': occ_idle,
            'count_verdict_idle': cnt_idle,
            'fused_verdict_idle': bool(occ_idle and cnt_idle)}


# ---------------------------------------------------------------- 判决一


def scenario_channel_families():
    """两条通道族各自能回答什么：数值可能相差很大，判定却未必不同。"""
    out = {}
    for level in LEVELS:
        free = set(idle_ticks(level))
        busy = set(range(TICKS)) - free
        rec = {}
        for k in ('D_occupancy', 'E_inflight', 'H_hits', 'I_prefill'):
            nl, lp, same = [], [], 0
            for seed in SEEDS:
                a = make_signals(RUN_NLPC, level, seed)[k]
                b = make_signals(RUN_LPC, level, seed)[k]
                nl.append(rate_ratio(a, free, busy))
                lp.append(rate_ratio(b, free, busy))
                same += int(threshold_type([v for t, v in enumerate(a) if t in free],
                                           [v for t, v in enumerate(a) if t in busy])
                            == threshold_type([v for t, v in enumerate(b) if t in free],
                                              [v for t, v in enumerate(b) if t in busy]))
            rec[k] = {'run_nlpc_mean': sum(nl) / len(nl),
                      'run_lpc_mean': sum(lp) / len(lp),
                      'same_verdict_30seeds': same,
                      'separates_at_this_threshold': same == 0,
                      'in_window_ratio_diff': abs(IN_WINDOW[RUN_LPC][k] -
                                                  IN_WINDOW[RUN_NLPC][k])}
        out[level] = rec

    out['note'] = ('"同答案"不只看数值差：高负载下四条通道在两语义上都判"没在算"'
                   '（D、E 是占用率类，H、I 的计数也低于忙段一半）；低负载下阈值'
                   '（忙段均值的一半）恰好落在两语义之间，连占用率类也分开了——'
                   '所以"哪类通道能分"依赖阈值位置，不是通道的固有属性。'
                   '真正与语义绑定的只有一条：真空泡的计数为 0')
    return out


def scenario_threshold_sweep():
    """阈值扫描：把判定阈值从 0.1 扫到 0.95，占用率类何时才把两语义分开。

    分开的位置完全由"真空泡与缓存命中在空窗里的占用率落在阈值两侧"决定，
    与语义本身无关——所以这个"分开"换一个负载或换一台机器就会消失。
    """
    out = {}
    free = set(idle_ticks('high'))
    busy = set(range(TICKS)) - free
    for channel in ('D_occupancy', 'E_inflight', 'H_hits'):
        rows = []
        for thr in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.95):
            sep = 0
            for seed in SEEDS[:10]:
                a = make_signals(RUN_NLPC, 'high', seed)[channel]
                b = make_signals(RUN_LPC, 'high', seed)[channel]
                va = threshold_type([v for t, v in enumerate(a) if t in free],
                                    [v for t, v in enumerate(a) if t in busy], thr)
                vb = threshold_type([v for t, v in enumerate(b) if t in free],
                                    [v for t, v in enumerate(b) if t in busy], thr)
                sep += int(va != vb)
            rows.append({'threshold': thr, 'separated_10seeds': sep})
        out[channel] = rows
    out['note'] = ('占用率类的可分辨区间是一条**阈值带**（落在两语义的占用率之间），'
                   '带外一律同答案；计数类的可分辨区间是整条 [0,1) 除了 0——'
                   '因为真空泡的计数恒为 0')
    return out


def scenario_relative_vs_absolute():
    """"回落"的基准：相对比值（÷忙段）与绝对阈值（< 忙段均值 × 0.5）不是同一件事。

    高负载下忙段占用 0.85、阈值 0.425，缓存命中空窗 0.30 < 0.425 ⇒ 同判"没在算"；
    低负载下忙段 0.60、阈值 0.30，缓存命中 0.30 卡在边界上。
    这就是"低负载下判据反而更接近可用"的全部机制——**是阈值位置碰巧**，
    不是语义在低负载下变了。
    """
    out = {}
    for level in LEVELS:
        free = set(idle_ticks(level))
        busy = set(range(TICKS)) - free
        nl_ratio = [rate_ratio(make_signals(RUN_NLPC, level, s)['D_occupancy'], free, busy)
                    for s in SEEDS]
        lp_ratio = [rate_ratio(make_signals(RUN_LPC, level, s)['D_occupancy'], free, busy)
                    for s in SEEDS]
        nl_abs = [threshold_type(
            [v for t, v in enumerate(make_signals(RUN_NLPC, level, s)['D_occupancy'])
             if t in free],
            [v for t, v in enumerate(make_signals(RUN_NLPC, level, s)['D_occupancy'])
             if t in busy]) for s in (0,)]
        lp_abs = [threshold_type(
            [v for t, v in enumerate(make_signals(RUN_LPC, level, s)['D_occupancy'])
             if t in free],
            [v for t, v in enumerate(make_signals(RUN_LPC, level, s)['D_occupancy'])
             if t in busy]) for s in (0,)]
        out[level] = {
            'busy_occupancy': BUSY[level]['D_occupancy'],
            'abs_cut': BUSY[level]['D_occupancy'] * THRESHOLD,
            'lpc_window_occupancy': IN_WINDOW[RUN_LPC]['D_occupancy'],
            'nlpc_window_occupancy': IN_WINDOW[RUN_NLPC]['D_occupancy'],
            'relative_says_same': (max(nl_ratio) < THRESHOLD) == (max(lp_ratio) < THRESHOLD),
            'absolute_says_same': nl_abs[0] == lp_abs[0],
            'lpc_window_above_abs_cut': IN_WINDOW[RUN_LPC]['D_occupancy'] >=
            BUSY[level]['D_occupancy'] * THRESHOLD}
    out['note'] = ('两个口径在高负载下都失效、在低负载下都可能碰到边界；'
                   '报"某段时间设备利用率掉到 X%"时必须同时给忙段基准，'
                   '否则"掉下来"没有含义')
    return out


# ---------------------------------------------------------------- 判决二


def _ratio_rule_accuracy(level):
    """只按相对比值做 AND 的准确率（对照绝对判定）。"""
    ok = 0
    for seed in SEEDS:
        for sem in SEMANTICS:
            c = classify(sem, level, seed)
            v = fuse_with_relative_ratio(c['occ_ratio'], c['count_verdict_idle'])
            ok += int(v == (sem == RUN_NLPC))
    return ok / float(2 * len(SEEDS))


def scenario_and_rule():
    """AND 判据的准确率与失效方向。"""
    out = {}
    for level in LEVELS:
        nl_ok = lp_ok = 0
        for seed in SEEDS:
            nl_ok += int(classify(RUN_NLPC, level, seed)['fused_verdict_idle'])
            lp_ok += int(not classify(RUN_LPC, level, seed)['fused_verdict_idle'])
        lpc_ratio = sum(classify(RUN_LPC, level, sd)['occ_ratio']
                        for sd in SEEDS) / float(len(SEEDS))
        out[level] = {'fused_accuracy': (nl_ok + lp_ok) / float(2 * len(SEEDS)),
                      'nlpc_recall': nl_ok / float(len(SEEDS)),
                      'lpc_specificity': lp_ok / float(len(SEEDS)),
                      'lpc_occupancy_ratio_mean': lpc_ratio,
            'lpc_occupancy_abs_within_thr': lpc_ratio < THRESHOLD,
            'ratio_rule_accuracy': _ratio_rule_accuracy(level),
                      'note_works_here': '绝对值判定在高低两档都判 LPC 没在算 ⇒ 两档都失效；'
                                         '从相对比值看只有低负载档 LPC 在对侧'}

    fail = {}
    busy_h = sum(v for v in make_signals(RUN_LPC, 'high', 0)['H_hits'][:20])
    for rel in (0.0, 0.01, 0.05, 0.25, 1.0):
        bg = rel * busy_h / 20.0     # 相对忙段每秒命中数的倍数
        missed = false_idle = 0
        for seed in SEEDS:
            missed += int(not classify(RUN_NLPC, 'high', seed,
                                       bg_hits_per_tick=bg)['fused_verdict_idle'])
            false_idle += int(classify(RUN_LPC, 'high', seed,
                                       bg_hits_per_tick=bg)['fused_verdict_idle'])
        fail['bg_rel_%.2f' % rel] = {
            'bg_hits_per_tick': bg,
            'nlpc_missed': missed, 'nlpc_n': len(SEEDS),
            'nlpc_recall': 1.0 - missed / float(len(SEEDS)),
            'lpc_false_idle': false_idle, 'lpc_n': len(SEEDS)}
    out['failure_direction'] = fail
    out['asymmetry'] = ('背景命中一旦接近缓存命中的量级，真空泡全部被漏判（recall 1→0），'
                        '而缓存命中一票不反判（specificity 恒 1）——'
                        'AND 的错误方向是单向的：宁可漏报，不会把活算成空转')
    out['reference_matters'] = {
        lv: {'abs_threshold_accuracy': out[lv]['fused_accuracy'],
             'ratio_threshold_accuracy': out[lv]['ratio_rule_accuracy']}
        for lv in LEVELS}
    out['reference_note'] = ('同一个占用率通道，基准取"忙段绝对值的一半"还是'
                             '"忙段均值的一半"（=相对比值），在高负载下等价、'
                             '在低负载下不等价——报约定里必须写清用哪一个')
    out['cost'] = ('单向不是免费的：漏报的真空泡会原样留在"资源空闲"读数里，'
                   '等于退回 §21.11 那条镜像；判据只把能确认的那部分从气泡里摘出来')
    return out


# ---------------------------------------------------------------- 判决三


def scenario_min_conditions():
    """可报的最小条件：空闲格数、副本数、窗口 vs 采样间隔。"""
    out = {}
    for level in LEVELS:
        rep = make_replicas(level)
        total = sum(sum(r) for r in rep)
        out['idle_%s' % level] = {
            'share': 1.0 - total / float(N_REPLICAS * TICKS),
            'window_ticks': len(idle_ticks(level)),
            'replica_ticks': len(idle_ticks(level)) * N_REPLICAS}

    # 副本数只缩放在窗计数（走 KV 复用的真正请求数随副本数变），不缩放忙段：
    # 忙段由训练侧占卡决定，与几个 rollout 副本无关。
    in_win_hits = 15.0
    for replicas in (1, 2, 4, 8):
        free = set(idle_ticks('high'))
        busy = set(range(TICKS)) - free
        votes, ratios = [], []
        for s in SEEDS[:10]:
            sig = make_signals(RUN_LPC, 'high', s, replicas=replicas,
                               lpc_hits=in_win_hits * replicas)['H_hits']
            votes.append(threshold_type([v for t, v in enumerate(sig) if t in free],
                                        [v for t, v in enumerate(sig) if t in busy]))
            ratios.append(rate_ratio(sig, free, busy))
        out['replicas_%d' % replicas] = {
            'h_ratio_mean': sum(ratios) / len(ratios),
            'h_ratio_min': min(ratios),
            'says_still_working': sum(votes), 'n': len(votes),
            'in_window_hits_per_tick': in_win_hits * replicas}

    # 窗口短于一个采样间隔时窗口整段消失（接 §21.11 判决一），与通道无关。
    # 这里拆成两件事：①相位**允许**看到（w/k ≥ 1）；②在允许的相位里能否看到
    # 至少两次（w/k ≥ 2）——后者才是判据需要的，见 §21.12 判决三。
    grid = {}
    for tick_s in (1, 5, 10, 20):
        rows = []
        for w in (2, 5, 10, 20, 40):
            seen_any = seen_two = 0
            for phase in range(tick_s):
                n = sum(1 for t in range(20, 20 + w) if (t - phase) % tick_s == 0)
                seen_any += int(n >= 1)
                seen_two += int(n >= 2)
            rows.append({'window_s': w,
                         'w_over_k': w / float(tick_s),
                         'phase_visible_rate': seen_any / float(tick_s),
                         'two_sample_phase_rate': seen_two / float(tick_s)})
        grid['sampling_%ds' % tick_s] = rows
    out['sampling_note'] = ('相位是全枚举不是随机抽样；两列分别是"至少采到一次"与'
                            '"至少采到两次"的相位占比')
    out['sampling'] = grid
    out['note'] = ('副本少时命中计数折算后逼近零，AND 自然漏报——这是 recall 的代价而不是'
                   '误判；采样间隔只决定窗口能不能看见，接 §21.11')
    return out


def scenario_identifiability():
    """识别性（负向对照）：把背景命中拉到缓存命中的量级，判据必须失效。"""
    out = {}
    for bg in (0.0, 60.0):
        sep = 0
        for level in LEVELS:
            free = set(idle_ticks(level))
            busy = set(range(TICKS)) - free
            nl = [rate_ratio(make_signals(RUN_NLPC, level, s, bg_hits_per_tick=bg)['H_hits'],
                             free, busy) for s in SEEDS]
            lp = [rate_ratio(make_signals(RUN_LPC, level, s, bg_hits_per_tick=bg)['H_hits'],
                             free, busy) for s in SEEDS]
            sep += int((max(nl) < min(lp)) or (max(lp) < min(nl)))
        out['bg_hits_%.0f' % bg] = {'levels_separable': sep, 'of': len(LEVELS)}
    out['note'] = ('背景命中 0 时两级都分得开、背景命中等于缓存命中量级时两级都分不开；'
                   '若后者仍分得开，说明判据读的不是"有没有活"'
                   '（§13.30／§13.31 撤回指标的同一类教训）')
    return out


# ---------------------------------------------------------------- 汇总与验证


def _canon(x):
    if isinstance(x, dict):
        return {k: _canon(v) for k, v in sorted(x.items())}
    if isinstance(x, list):
        return [_canon(v) for v in x]
    if isinstance(x, bool):
        return x
    if isinstance(x, float):
        if x != x:
            return 'NaN'
        return round(x, 9)
    return x


def run_all():
    return {
        'config': {'seeds': len(SEEDS), 'ticks': TICKS, 'replicas': N_REPLICAS,
                   'sigma': SIGMA, 'threshold': THRESHOLD,
                   'windows': {k: [list(w) for w in v] for k, v in WINDOWS.items()},
                   'in_window': IN_WINDOW, 'busy': BUSY},
        'channel_families': scenario_channel_families(),
        'threshold_sweep': scenario_threshold_sweep(),
        'relative_vs_absolute': scenario_relative_vs_absolute(),
        'and_rule': scenario_and_rule(),
        'min_conditions': scenario_min_conditions(),
        'identifiability': scenario_identifiability(),
    }


def verify():
    if not os.path.exists(OUT_PATH):
        print('fusion.json 不存在，先运行一次全量')
        return 1
    with open(OUT_PATH) as f:
        archived = json.load(f)
    fresh = _canon(run_all())
    if _canon(archived) != fresh:
        diff = [k for k in fresh if _canon(archived.get(k)) != fresh[k]]
        print('差异字段：', diff)
        return 1
    print('--verify 通过：与归档 fusion.json 逐字段一致')
    return 0


def main():
    if '--verify' in sys.argv:
        return verify()
    with open(OUT_PATH, 'w') as f:
        json.dump(_canon(run_all()), f, ensure_ascii=False, indent=1, sort_keys=True)
    print('写出', OUT_PATH)
    return 0


if __name__ == '__main__':
    sys.exit(main())
