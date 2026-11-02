#!/usr/bin/env python3
"""§21.10 气泡归因的可辨识性：只看利用率轨迹，能分开 B2／B3／B5／B4 吗

支撑 notebook/rollout-performance-research.md §21.10。

§3.2 的五类气泡表是本文的承重结构之一，但"能不能测"从未被检验：
§3.5 边界 1 只承认"B4 与 B5 在 agentic 负载下会互相掩盖"，没量化能分开哪些。
§21.5 已证 C2 的拆分侧极脆（但那是**事件时间戳**口径），§21.7 已证报 η 的近似可分性。
本节问的是另一个问题：**只有引擎侧利用率轨迹时，空闲时间该归给哪一类？**

三类只在轨迹上留下不同"形状"的机制：

  B3  权重同步       所有副本同时被占住   → 同时性（simultaneity）
  B5  colocation 失配 各副本被占住的时刻不同 → 不同步性（overlap ratio）
  B4  环境等待       空闲**集中在少数长尾请求** → 空转分布的集中度（concentration）

墙钟是任意虚拟秒，不可与任何论文报值相除（§19.1 规则一）。只用标准库。

用法：
    python3 bubble_attribution.py             # 全量重跑并覆盖 summary.json
    python3 bubble_attribution.py --verify    # 重跑并与归档 summary.json 逐字段比对
    python3 -m unittest test_bubble_attribution -v
"""

import itertools
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY_PATH = os.path.join(HERE, 'summary.json')

SEEDS = list(range(30))
TICKS = 400          # 轨迹长度（时间片）
N_REPLICAS = 4       # 副本数（= DP rank 数）


# ---------------------------------------------------------------- 轨迹生成


def make_trace(kind, seed, *, busy_share=0.7, n_req=20, sync_lag=0,
               random_share=0.0, window=None):
    """生成 N_REPLICAS 条逐 tick 的忙/闲轨迹（1 = 该 tick 该副本在算）。

    kind:
      'busy'          全部副本始终忙（对照）
      'B3'            所有副本在同一段窗口内同时空闲
      'B5'            各副本错开空闲（每条副本各自有独立空闲段）
      'B4'            空闲由少数"卡住"的请求造成，堆在个别副本上
      'B2'            切换裂隙：所有副本同时空闲一小段（与 B3 同形但更短更频繁）
      'mixed_B3_B4'   B3 窗口里再混入一个卡住的请求
    """
    rng = random.Random(seed)
    lo, hi = window if window else (0, TICKS)
    span = hi - lo
    # 每副本空闲 tick 的平均目标数（真实空闲率由 trace 决定，测出来的值会被回报）
    n_free = int(round(span * (1.0 - busy_share) / N_REPLICAS))
    n_free = max(1, n_free)
    traces = [[1] * TICKS for _ in range(N_REPLICAS)]

    def blank(r, start, length):
        for t in range(start, min(TICKS, start + length)):
            traces[r][t] = 0

    if kind == 'busy':
        pass

    elif kind == 'B2':
        # 切换裂隙：与 B3 同样"全体同时空闲"，但窗口极短（两三个 tick）
        n_win = max(1, n_free // 2)
        for w in range(n_win):
            start = rng.randrange(lo, max(lo + 1, hi - 3))
            for r in range(N_REPLICAS):
                blank(r, start, 2)

    elif kind == 'B3':
        # 所有副本同时空闲（B3 少而长，B2 多而短）
        n_win, win_len = 2, max(1, n_free // 2)
        for w in range(n_win):
            start = rng.randrange(lo, max(lo + 1, hi - win_len))
            for r in range(N_REPLICAS):
                blank(r, start, win_len)

    elif kind == 'B5':
        # 各副本错开：每个副本一段长空闲，起点互不重叠
        seg = max(1, n_free)
        starts = []
        for r in range(N_REPLICAS):
            for _ in range(50):
                s = rng.randrange(lo, max(lo + 1, hi - seg))
                if all(abs(s - o) >= seg for o in starts):
                    break
            starts.append(s)
            blank(r, s, seg)

    elif kind == 'B4':
        # 少数"卡住"的请求：每个只占一个副本的一段，且集中在少数副本上
        n_victims = max(1, N_REPLICAS // 4)          # 4 副本 → 1 个受害者
        seg = max(1, (n_free * N_REPLICAS) // n_victims)
        seg = min(seg, TICKS - 1)
        for _ in range(n_victims):
            r = rng.randrange(N_REPLICAS)
            blank(r, rng.randrange(lo, max(lo + 1, hi - seg)), seg)

    elif kind == 'mixed_B3_B4':
        n_win, win_len = 2, max(1, n_free // 2)
        for w in range(n_win):
            start = rng.randrange(lo, max(lo + 1, hi - win_len))
            for r in range(N_REPLICAS):
                blank(r, start, win_len)
        # 再加一个卡住的请求，落在某个副本的额外空闲上
        r = rng.randrange(N_REPLICAS)
        blank(r, rng.randrange(lo, max(lo + 1, hi - 1)), max(1, n_free // 2))

    else:
        raise ValueError(kind)

    # 可选的随机空闲（模仿真实的碎片化）
    if random_share > 0:
        for r in range(N_REPLICAS):
            for t in range(TICKS):
                if rng.random() < random_share:
                    traces[r][t] = 0

    return traces


# ---------------------------------------------------------------- 三个形状统计量


def bubble_share(traces):
    """B5 式口径：1 − 平均利用率。它把所有空闲算成一类。"""
    busy = sum(sum(row) for row in traces)
    return 1.0 - busy / float(N_REPLICAS * TICKS)


def simultaneity(traces):
    """同时性：在"至少一个副本空闲"的 tick 里，全部副本都空闲的比例。

    B3／B2 结构性把所有副本钉在同一段窗口 ⇒ 接近 1；
    B5／B4 各副本错开 ⇒ 接近 0。另报空闲 tick 的绝对数，
    因为"同时性"在空闲极少时会被放大成噪声。
    """
    n = len(traces[0])
    any_free = 0
    all_free = 0
    for t in range(n):
        free = [r for r in range(len(traces)) if traces[r][t] == 0]
        if free:
            any_free += 1
            if len(free) == len(traces):
                all_free += 1
    if any_free == 0:
        return {'sim': float('nan'), 'free_ticks': 0, 'replica_free_ticks': 0}
    return {'sim': all_free / float(any_free),
            'free_ticks': any_free,
            'replica_free_ticks': sum(1 for tr in traces for v in tr if v == 0)}


def concentration(traces):
    """空转集中度：各副本空闲 tick 数的归一化 Herfindahl。

    H = Σ p_r²，p_r 为副本 r 占全部空闲 tick 的份额。
    全平摊（每副本一样多）⇒ 1/N；全压在一个副本上 ⇒ 1。
    归一到 [0,1]：(H − 1/N)/(1 − 1/N)。
    """
    per = [sum(1 for v in tr if v == 0) for tr in traces]
    tot = sum(per)
    if tot == 0:
        return float('nan')
    n = len(traces)
    h = sum((p / float(tot)) ** 2 for p in per)
    return (h - 1.0 / n) / (1.0 - 1.0 / n)


def longest_run(traces):
    """单个副本最长连续空闲段——B3/B2 的窗口长、B5 的短，但 B4 也可以很长。"""
    best = 0
    for tr in traces:
        run = 0
        for v in tr:
            run = run + 1 if v == 0 else 0
            best = max(best, run)
    return best


def classifiers(traces):
    """把三个统计量拼成可执行的判据。"""
    sim = simultaneity(traces)
    conc = concentration(traces)
    return {'bubble_share': bubble_share(traces),
            'simultaneity': sim['sim'],
            'free_ticks': sim['free_ticks'],
            'concentration': conc,
            'longest_run': longest_run(traces)}


# ---------------------------------------------------------------- 场景


KINDS = ['B3', 'B2', 'B5', 'B4', 'mixed_B3_B4']


def scenario_shape_space():
    """每一类机制在三个统计量上的位置（30 种子）。"""
    out = {}
    for kind in ['busy'] + KINDS:
        rows = []
        for seed in SEEDS:
            tr = make_trace(kind, seed)
            c = classifiers(tr)
            c['kind'] = kind
            rows.append(c)
        out[kind] = summarize(rows, ['bubble_share', 'simultaneity', 'free_ticks',
                                     'concentration', 'longest_run'])
    return out


def scenario_rule_accuracy():
    """可执行判据：先看同时性，再分开不重叠的两类。30 种子上逐类命中率。"""
    out = {}
    for kind in KINDS:
        hits = {'sim_high': 0, 'conc_high': 0, 'neither': 0}
        for seed in SEEDS:
            c = classifiers(make_trace(kind, seed))
            if c['simultaneity'] is not None and c['simultaneity'] == c['simultaneity']:
                if c['simultaneity'] > 0.5:
                    hits['sim_high'] += 1
                elif c['concentration'] == c['concentration'] and c['concentration'] > 0.5:
                    hits['conc_high'] += 1
                else:
                    hits['neither'] += 1
            else:
                hits['neither'] += 1
        out[kind] = {'n': len(SEEDS), 'hits': hits,
                     'share_sim_high': hits['sim_high'] / float(len(SEEDS)),
                     'share_conc_high': hits['conc_high'] / float(len(SEEDS))}
    return out


def scenario_confounders():
    """三个会让判据失效的混淆：空闲极少、碎片化、混合机制。"""
    out = {}
    # 1) 空闲极少：B3 只占 1%，同时性仍然高，但气泡份额本身已不可测
    for share in (0.01, 0.05, 0.2, 0.5):
        rows = [classifiers(make_trace('B3', s, busy_share=1.0 - share)) for s in SEEDS]
        blk = summarize(rows, ['bubble_share', 'simultaneity', 'free_ticks', 'longest_run'])
        blk['recovered_share_minus_true'] = blk['bubble_share']['mean'] - share
        out['B3_free%.2f' % share] = blk
    # 2) 碎片化：随机空闲混进 B3 窗口
    for frag in (0.0, 0.02, 0.05, 0.10):
        rows = [classifiers(make_trace('B3', s, random_share=frag)) for s in SEEDS]
        out['B3_frag%.2f' % frag] = summarize(rows, ['bubble_share', 'simultaneity',
                                                     'concentration', 'free_ticks'])
    # 3) 混合机制：B3 窗口 + 一个卡住请求
    rows = [classifiers(make_trace('mixed_B3_B4', s)) for s in SEEDS]
    out['mixed_B3_B4'] = summarize(rows, ['bubble_share', 'simultaneity',
                                          'concentration', 'free_ticks'])
    pure = [classifiers(make_trace('B3', s)) for s in SEEDS]
    out['mixed_minus_pure_sim'] = (out['mixed_B3_B4']['simultaneity']['mean'] -
                                   summarize(pure, ['simultaneity'])['simultaneity']['mean'])
    return out


def scenario_additivity():
    """同层可加性：两个气泡在时间上**不重叠**时，份额严格可加（正向对照）。

    构造：前一半时间窗只放 B3（全体同时空闲），后一半只放 B5（各副本错开）。
    两者支撑集不相交 ⇒ 并集 = 算术和，份额应精确可加。
    这与 §3.5 边界 2「不可跨层相加」是不同的命题：同层同口径时可以相加，
    真正不能相加的是不同量纲（B0 机器周期 vs B1–B5 墙钟）。
    """
    half = TICKS // 2
    rows = []
    for seed in SEEDS:
        t3 = make_trace('B3', seed, busy_share=0.7, window=(0, half))
        t5 = make_trace('B5', seed + 1000, busy_share=0.7, window=(half, TICKS))
        mixed = [[min(t3[r][t], t5[r][t]) for t in range(TICKS)]
                 for r in range(N_REPLICAS)]
        s3, s5, sm = bubble_share(t3), bubble_share(t5), bubble_share(mixed)
        rows.append({'share_B3': s3, 'share_B5': s5, 'share_sum': s3 + s5,
                     'share_mixed': sm,
                     'ratio': sm / (s3 + s5) if (s3 + s5) else float('nan')})
    out = {'disjoint_windows': summarize(rows, ['share_B3', 'share_B5', 'share_sum',
                                                'share_mixed', 'ratio'])}

    # 重叠情形：B3 窗口里，某个副本另有额外空闲——并集仍然只计一次
    rows2 = []
    for seed in SEEDS:
        t3 = make_trace('B3', seed, busy_share=0.85)
        extra = [row[:] for row in t3]
        r = seed % N_REPLICAS
        # 在 B3 窗口内部给副本 r 再加一段空闲（与 B3 重叠，不增加并集）
        for t in range(TICKS):
            if t3[r][t] == 0:
                for t2 in range(t, min(TICKS, t + 1)):
                    extra[r][t2] = 0
        rows2.append({'share_base': bubble_share(t3),
                      'share_with_overlap': bubble_share(extra),
                      'delta': bubble_share(extra) - bubble_share(t3)})
    out['overlapping_idle'] = summarize(rows2, ['share_base', 'share_with_overlap', 'delta'])
    out['note'] = ('不重叠时可加（ratio≈1）；重叠时并集只计一次（delta≈0）。'
                   '§3.5 边界 2 讲的是跨层量纲不同不能相加，与此处不冲突')
    return out


def scenario_tick_granularity():
    """B2 与 B3 只靠「窗口长度」区分——而窗口长度是 tick 粒度的函数。

    同一组事件（切换裂隙 2s、权重同步 8s），分别按 1s 与 10s 采样：
    1s 时 B3 的 run=8 > B2 的 run=2；10s 时两者都被量化成 1–2 个 tick，序反转或并列。
    """
    out = {}
    events = {'B2_window_s': 2, 'B3_window_s': 8, 'n_windows': 12}
    for tick_s in (1, 2, 5, 10, 20):
        rec = {}
        for kind, win_s in (('B2', events['B2_window_s']), ('B3', events['B3_window_s'])):
            win_ticks = max(1, int(round(win_s / float(tick_s))))
            n_ticks = max(1, int(round(400 * 1.0 / tick_s)))
            traces = [[1] * n_ticks for _ in range(N_REPLICAS)]
            rng = random.Random(0)
            for _ in range(events['n_windows']):
                st = rng.randrange(0, max(1, n_ticks - win_ticks))
                for r in range(N_REPLICAS):
                    for t in range(st, min(n_ticks, st + win_ticks)):
                        traces[r][t] = 0
            rec[kind] = {'win_ticks': win_ticks,
                         'longest_run': longest_run(traces),
                         'simultaneity': simultaneity(traces)['sim'],
                         'bubble_share': bubble_share(traces)}
        rec['order_B3_gt_B2'] = rec['B3']['longest_run'] > rec['B2']['longest_run']
        rec['tie'] = rec['B3']['longest_run'] == rec['B2']['longest_run']
        out['tick%d_s' % tick_s] = rec
    out['events'] = events
    return out


def summarize(items, keys):
    out = {}
    for k in keys:
        vals = [it[k] for it in items if it[k] == it[k]]
        if not vals:
            out[k] = {'mean': float('nan'), 'std': float('nan'),
                      'min': float('nan'), 'max': float('nan'), 'n': 0}
            continue
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1)
        out[k] = {'mean': m, 'std': var ** 0.5, 'min': min(vals),
                  'max': max(vals), 'n': len(vals)}
    return out


def run_all():
    return {
        'config': {'seeds': len(SEEDS), 'ticks': TICKS, 'replicas': N_REPLICAS},
        'shape_space': scenario_shape_space(),
        'rule_accuracy': scenario_rule_accuracy(),
        'confounders': scenario_confounders(),
        'additivity': scenario_additivity(),
        'tick_granularity': scenario_tick_granularity(),
    }


def _canon(x):
    if isinstance(x, dict):
        return {k: _canon(v) for k, v in sorted(x.items())}
    if isinstance(x, list):
        return [_canon(v) for v in x]
    if isinstance(x, float):
        return round(x, 9)
    return x


def _strip_nan(x):
    if isinstance(x, dict):
        return {k: _strip_nan(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_strip_nan(v) for v in x]
    if isinstance(x, float) and x != x:
        return 'NaN'
    return x


def verify():
    if not os.path.exists(SUMMARY_PATH):
        print('summary.json 不存在，先运行一次全量')
        return 1
    with open(SUMMARY_PATH) as f:
        archived = json.load(f)
    fresh = _strip_nan(_canon(run_all()))
    if _strip_nan(_canon(archived)) != fresh:
        print('差异字段：', [k for k in fresh if _strip_nan(_canon(archived.get(k))) != fresh[k]])
        return 1
    print('--verify 通过：与归档 summary.json 逐字段一致')
    return 0


def main():
    if '--verify' in sys.argv:
        return verify()
    with open(SUMMARY_PATH, 'w') as f:
        json.dump(_canon(run_all()), f, ensure_ascii=False, indent=1, sort_keys=True)
    print('写出', SUMMARY_PATH)
    return 0


if __name__ == '__main__':
    sys.exit(main())
