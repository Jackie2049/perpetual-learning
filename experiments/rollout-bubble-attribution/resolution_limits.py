#!/usr/bin/env python3
"""§21.11 气泡归因的分辨率与镜像不可分

支撑 notebook/rollout-performance-research.md §21.11（补强 §21.10）。

§21.10 证明了四类机制的气泡份额可以完全重合，且 B2 与 B3 在 10s／20s 采样下并列。
本文件回答它留下的四个问题：

  A. **τ 的可分辨界**：固定物理负载（每 100 秒一个窗口、B3 窗长 = 4×B2 窗长），
     只扫"采样间隔 × 窗口长度"，给出**最小可分辨窗口**与**混叠区间**。
     要点：采样是二值的，窗口 [s, s+w) 落在 k 秒栅格上占几格不是 w/k，
     而是 #{m : s/k ≤ m < (s+w)/k}——它随偏移在 ℤ 上跳变，支撑集为 {0,1} 时
     **整个窗口可能一个采样都没被采到**。

  B. **镜像不可分（一）**：同时性、集中度逐位相同，气泡份额差 5 倍——
     形状统计量不含份额信息，判据无法据此排序。

  C. **镜像不可分（二，最要紧）**：同一份逐格轨迹，只换"重叠段空闲从哪来"的标签
     （运行时驱动重叠 vs 请求命中前缀缓存），**每个统计量逐位相同**。
     于是"全体同时空闲"既可能是上下文切换、也可能是缓存命中，而结论正好相反。

  D. **跨口径**：同一个物理占空比在不同采样间隔下的读数偏差；以及
     §21.10 Q3 那一档 trace 的 n_ticks 随 tick 变化，不能与主表横比。

墙钟是任意虚拟秒，不可与任何论文报值相除（§19.1 规则一）。只用标准库。

用法：
    python3 resolution_limits.py             # 全量重跑并覆盖 resolution.json
    python3 resolution_limits.py --verify    # 重跑并与归档 resolution.json 逐字段比对
    python3 -m unittest test_resolution_limits -v
"""

import json
import math
import os
import random
import sys

from bubble_attribution import classifiers, longest_run

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, 'resolution.json')

SEEDS = list(range(30))
N_REPLICAS = 4
T_GRID = 400          # 冻结负载的格子数（与 §21.10 的 TICKS 一致）
RATIO_REF = 4.0       # B3 窗口是 B2 的若干倍（§21.10 用的是 8s / 2s = 4）


# ---------------------------------------------------------------- 采样二值化


def support_spectrum(w_sec, k_sec, samples=4000):
    """一个窗口 [s, s+w) 落在 k 秒栅格上占**几个格**：精确枚举，不是 ceil 也不是 round。

    支撑数 = #{m∈ℤ : s/k ≤ m < (s+w)/k}，随偏移 s 只有两个相邻取值（长度 1 的模 1 量）。
    支撑集只有单元素 ⇒ 位移不变、窗口必然可见；有 {n, n+1} 两元素 ⇒ 窗口可能整段
    落在两个采样点之间（采样点只在格边界看，二值化后信息不可恢复）。
    """
    out = set()
    for i in range(samples):
        s0 = i * k_sec / float(samples)
        lo = math.ceil(s0 / k_sec - 1e-12)
        hi = math.ceil((s0 + w_sec) / k_sec - 1e-12)
        out.add(hi - lo)
    return sorted(out)


def _window_trace(w_sec, k_sec, rng, load_period=100.0, replicas=N_REPLICAS,
                  ticks=T_GRID, p_visible=0.25):
    """把"每 load_period 秒一次、长 w_sec 的全体同时窗口"按 k 秒采样落地。

    支撑集单元素 n ⇒ 每次必可见、占 n 格；
    支撑集 {n, n+1} ⇒ 以概率 p_visible 可见、占 n 格，否则**整段消失**。
    周期按物理负载固定，不随支撑数伸缩（否则份额会被构造偷偷改写）。
    """
    specs = support_spectrum(w_sec, k_sec, samples=64)
    n = min(specs)
    wobble = len(specs) > 1
    per = max(1, int(math.ceil(load_period / k_sec)))
    traces = [[1] * ticks for _ in range(replicas)]
    st = rng.randrange(0, per)
    while st < ticks:
        visible = (not wobble) or (rng.random() < p_visible)
        if visible and n > 0:
            for r in range(replicas):
                for t in range(st, min(ticks, st + n)):
                    traces[r][t] = 0
        st += per
    return traces


def _positive_fraction_sum(a, b, ticks=T_GRID):
    """开区间可加性：r[f] = |{(ω,f) : a(ω)>inf_a, b(ω)>inf_b}| / 长度(ω)，取全 ω 平均。"""
    tot = pos = 0
    for f in a:
        la, lb = a[f], b[f]
        tol = 1e-9
        for i in range(ticks):
            if la[i] > tol and lb[i] > tol:
                pos += 1
            tot += 1
    return pos / float(tot)


# ---------------------------------------------------------------- A：τ 的可分辨界


def scenario_tau_threshold():
    """固定物理负载，扫 (采样间隔, τ_B2)：B2 与 B3 何时分得开、何处混叠、何处全丢。

    B3 窗长恒为 RATIO_REF×B2。判据沿用 §21.10：逐种子比较两者 longest_run。
    """
    out = {}
    for tick in (1, 2, 5, 10, 20):
        rows = []
        for tau_b2 in range(1, 31):
            hit, lost = 0, 0
            for seed in SEEDS:
                rng = random.Random(seed)
                t2 = _window_trace(tau_b2, tick, rng)
                t3 = _window_trace(tau_b2 * RATIO_REF, tick, rng)
                l2, l3 = longest_run(t2), longest_run(t3)
                if l2 == 0 or l3 == 0:
                    lost += 1
                    continue
                if l3 > l2:
                    hit += 1
            rows.append({'tau_b2_s': tau_b2,
                         'tau_b3_s': tau_b2 * RATIO_REF,
                         'support_b2': support_spectrum(tau_b2, tick),
                         'support_b3': support_spectrum(tau_b2 * RATIO_REF, tick),
                         'hit_rate': hit / float(len(SEEDS)),
                         'lost_rate': lost / float(len(SEEDS))})
        full = [r for r in rows if r['hit_rate'] >= 1.0]
        out['tick%d_s' % tick] = {
            'sweep': rows,
            'min_tau_b2_full_accuracy': full[0]['tau_b2_s'] if full else None,
            'min_tau_b3_full_accuracy': full[0]['tau_b3_s'] if full else None,
            'ambiguous_band': [r['tau_b2_s'] for r in rows
                               if 0.0 < r['hit_rate'] < 1.0],
            'never_separable': [r['tau_b2_s'] for r in rows if r['hit_rate'] == 0.0],
            'alias_free': [r['tau_b2_s'] for r in rows if r['support_b2'] == r['support_b3']],
        }
    out['ratio_ref'] = RATIO_REF
    return out


def scenario_ratio_curve():
    """采样间隔固定为 10 秒、τ_B2 = 10 秒，只扫 B3/B2 窗长比：分得开的最小比值。"""
    tick, tau_b2 = 10, 10
    rows = []
    for ratio in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0):
        hit = 0
        for seed in SEEDS:
            rng = random.Random(seed)
            t2 = _window_trace(tau_b2, tick, rng)
            t3 = _window_trace(tau_b2 * ratio, tick, rng)
            if longest_run(t3) > longest_run(t2) > 0:
                hit += 1
        rows.append({'ratio': ratio,
                     'support_b2': support_spectrum(tau_b2, tick),
                     'support_b3': support_spectrum(tau_b2 * ratio, tick),
                     'hit_rate': hit / float(len(SEEDS))})
    return {'tick_s': tick, 'tau_b2_s': tau_b2, 'sweep': rows}


# ---------------------------------------------------------------- B／C：镜像不可分


def _mirror_block(ma, mb):
    out = {
        'A': {'bubble_share': ma['bubble_share'], 'simultaneity': ma['simultaneity'],
              'concentration': ma['concentration'], 'free_ticks': ma['free_ticks'],
              'longest_run': ma['longest_run']},
        'B': {'bubble_share': mb['bubble_share'], 'simultaneity': mb['simultaneity'],
              'concentration': mb['concentration'], 'free_ticks': mb['free_ticks'],
              'longest_run': mb['longest_run']},
    }
    out['same_sim'] = out['A']['simultaneity'] == out['B']['simultaneity']
    out['same_conc'] = out['A']['concentration'] == out['B']['concentration']
    out['same_free_ticks'] = out['A']['free_ticks'] == out['B']['free_ticks']
    out['same_share'] = abs(out['A']['bubble_share'] - out['B']['bubble_share']) < 1e-12
    out['share_ratio'] = (out['B']['bubble_share'] / out['A']['bubble_share']
                          if out['A']['bubble_share'] else float('nan'))
    return out


def scenario_mirror_length():
    """同时性与集中度逐位相同、气泡份额差 5 倍：形状统计量不含份额信息。"""
    def trace(n):
        tr = [[1] * T_GRID for _ in range(N_REPLICAS)]
        step = max(1, T_GRID // max(1, n))
        for i in range(n):
            t = min(T_GRID - 1, i * step)
            for r in range(N_REPLICAS):
                tr[r][t] = 0
        return tr
    return _mirror_block(classifiers(trace(8)), classifiers(trace(40)))


def context_trace(walk=(0, 150), calib=(150, 250), hold=(200, 250),
                  replicas=N_REPLICAS, ticks=T_GRID):
    """B5 型的可加分解：两段互不重叠的"全体同时窗口" + 一段压在第二段里的尾部空闲。

    参数都是 (起点, 终点)，越界按 ticks 截断。
    walk  —— 运行时驱动重叠（各副本同时走 kernel，谁都没在算）：**真气泡**
    calib —— 请求命中前缀缓存（部分副本在 KV 重叠上工作）：**不是气泡**
    hold  —— 尾部请求挂在图／工具上，压在 calib 窗口里 ⇒ 并集贡献为零
    于是并集份额 = |walk| + |calib|，而其中真气泡只有 |walk|。
    """
    traces = [[1] * ticks for _ in range(replicas)]
    for r in range(replicas):
        for st, en in (walk, calib, hold):
            for t in range(max(0, st), min(ticks, en)):
                traces[r][t] = 0
    return traces


def context_trace_misaligned(shift=40, replicas=N_REPLICAS, ticks=T_GRID):
    """同一个 B5 轨迹，但尾部窗口挪开 shift 格、不再压在 calib 窗口里。

    指标不变而物理归属改变——用来量"对齐假设被打破"时判据错多远。
    """
    return context_trace(hold=(200 + shift, 250 + shift), replicas=replicas, ticks=ticks)


def scenario_component_mass():
    """五类机制**只有一个成分**（全体同时、单个长窗口）时的份额与形状。

    §21.10 主表的四类机制还差了一个成分——运行时的重叠窗口（stream trainer）：
    只做一个全体同时的长窗口，气泡份额与 B3 同量级，同时性 1.0，其余统计量不变。
    """
    def one_window(w, replicas=N_REPLICAS, ticks=T_GRID):
        tr = [[1] * ticks for _ in range(replicas)]
        for r in range(replicas):
            for t in range(w):
                tr[r][t] = 0
        return tr
    out = {}
    for w in (30, 60, 120):
        c = classifiers(one_window(w))
        out['window_%d' % w] = {'bubble_share': c['bubble_share'],
                                'simultaneity': c['simultaneity'],
                                'concentration': c['concentration'],
                                'longest_run': c['longest_run']}
    out['note'] = ('全体同时的长窗口在三个统计量上与 B3 逐位相同——'
                   '§21.10 的判据把"运行时驱动重叠"也归进 B2/B3 族')
    return out


def scenario_additivity_mass():
    """可加性的**质量核算**：份额读数可加，但份额里真正是气泡的那一部分不可加。

    walk（0–150 格，真气泡）+ calib（150–250 格，KV 重叠上的正常工作，份额里不该算）
    + 尾部（200–250 格，压在 calib 里 ⇒ 并集贡献为零）。
    """
    def only(segs):
        tr = [[1] * T_GRID for _ in range(N_REPLICAS)]
        for r in range(N_REPLICAS):
            for st, en in segs:
                for t in range(st, min(T_GRID, en)):
                    tr[r][t] = 0
        return classifiers(tr)
    a = only([(0, 0 + 150)])
    b = only([(150, 150 + 100)])
    c = classifiers(context_trace())
    return {'walk_only': {'share': a['bubble_share'], 'is_bubble': True},
            'calib_only': {'share': b['bubble_share'], 'is_bubble': False},
            'union': {'share': c['bubble_share'],
                      'sum_of_parts': a['bubble_share'] + b['bubble_share']},
            'true_bubble_share_in_union': a['bubble_share'],
            'tail_contribution': 0.0,
            'additive_reads_not_mass': True}


def scenario_mirror_context():
    """同一份逐格轨迹、只换标签：每个统计量逐位相同，归因结论相反。"""
    base = context_trace()
    ma, mb = classifiers(base), classifiers(base)   # 同一实例，两种语义
    out = _mirror_block(ma, mb)
    out['identical_instance'] = True
    out['label_A'] = '空闲全来自运行时重叠 —— 照 §21.10 判据属 B2/B3 族（同时性=1）'
    out['label_B'] = '同一轨迹，若重叠段的空闲来自请求命中前缀缓存，则是正常的计算，不是气泡'
    return out


def scenario_mirror_misaligned():
    """"对齐"这个前提被打破时错多远：同一条 B5 轨迹，尾部窗口挪开 40 格。

    对齐时尾部整段落在 calib 窗口里（并集贡献 0）；挪开后它自己有 40 格露在外面，
    份额与最长段一起变——但判据的三个桶一个都没动，读者看不出差别从哪来。
    """
    a = classifiers(context_trace())
    b = classifiers(context_trace_misaligned())
    return {'aligned': {'bubble_share': a['bubble_share'],
                        'simultaneity': a['simultaneity'],
                        'concentration': a['concentration'],
                        'longest_run': a['longest_run']},
            'shifted_by_40': {'bubble_share': b['bubble_share'],
                              'simultaneity': b['simultaneity'],
                              'concentration': b['concentration'],
                              'longest_run': b['longest_run']},
            'share_delta': b['bubble_share'] - a['bubble_share'],
            'note': '两条物理上都是"资源空闲 + 正常工作"，只有对齐关系不同；'
                    '判据读不出对齐，读数却随对齐变化'}


def scenario_mirror_boundary():
    """份额比值的上界：要让集中度保持 1，空闲格总数被副本数封顶。"""
    return {'replicas_%d' % n: {'share_ratio_upper_bound': float(n)}
            for n in (2, 4, 8, 16)}


# ---------------------------------------------------------------- D：跨口径


def scenario_same_scale():
    """同一物理占空比（30 秒窗 / 100 秒周期 = 0.30）在不同采样间隔下的读数。"""
    out = {}
    w_sec, period = 30.0, 100.0
    for tick in (1, 2, 5, 10, 20, 50):
        specs = support_spectrum(w_sec, tick, samples=512)
        span = min(specs)
        per = max(1, int(math.ceil(period / tick)))
        out['tick%d_s' % tick] = {
            'support_set': specs,
            'min_span_ticks': span,
            'period_ticks': per,
            'measured_share': span / float(per),
            'true_duty': w_sec / period,
            'window_lost': span == 0,
        }
    out['inverse_formula'] = ('读到的份额 = min({#m : s/k ≤ m < (s+w)/k}) / ⌈period/k⌉；'
                              '反解 w 不单调——短窗整段消失（支撑 0），长窗被钉成 ⌈period/k⌉ 分之一取整')
    return out


def scenario_cross_scale():
    """§21.10 Q3 那一档的 n_ticks 随采样间隔变化，份额与主表不同口径。

    复现其构造：窗口 2s／8s、各 12 次、副本恒忙、n_ticks = ⌈400/tick⌉。"""
    out = {}
    for tick in (1, 2, 5, 10, 20):
        rec = {}
        for kind, win_s in (('B2', 2), ('B3', 8)):
            win_ticks = max(1, int(round(win_s / float(tick))))
            n_ticks = max(1, int(round(400.0 / tick)))
            traces = [[1] * n_ticks for _ in range(N_REPLICAS)]
            rng = random.Random(0)
            for _ in range(12):
                st = rng.randrange(0, max(1, n_ticks - win_ticks))
                for r in range(N_REPLICAS):
                    for t in range(st, min(n_ticks, st + win_ticks)):
                        traces[r][t] = 0
            c = classifiers(_reshaped(traces))
            rec[kind] = {'win_ticks': win_ticks, 'n_ticks': n_ticks,
                         'bubble_share': c['bubble_share'],
                         'longest_run': c['longest_run']}
        rec['tie'] = rec['B2']['longest_run'] == rec['B3']['longest_run']
        rec['share_ratio_B3_over_B2'] = (rec['B3']['bubble_share'] /
                                         rec['B2']['bubble_share']
                                         if rec['B2']['bubble_share'] else float('nan'))
        out['tick%d_s' % tick] = rec
    out['note'] = ('该档 n_ticks = ⌈400/tick⌉、副本恒忙，份额是"trace 内空转格占比"，'
                   '随 tick 一起变；不能与主表 0.075（同一 400 格口径）横比')
    return out


def _reshaped(traces, ticks=T_GRID):
    """把任意长度的轨迹截断／补齐到统一格数，供 classifiers 使用（保持各副本同步）。"""
    out = []
    for tr in traces:
        row = tr[:ticks] + [1] * max(0, ticks - len(tr))
        out.append(row)
    return out


# ---------------------------------------------------------------- E：判据自指


def scenario_rule_circularity():
    """§21.10 的判据没有"以上都不是"这一项：任何轨迹都会被分进某一桶。"""
    rng = random.Random(7)
    traces = [[1] * T_GRID for _ in range(N_REPLICAS)]
    for r in range(N_REPLICAS):
        for t in range(T_GRID):
            if rng.random() < 0.25:
                traces[r][t] = 0
    c = classifiers(traces)
    if c['simultaneity'] > 0.5:
        bucket = 'B2/B3'
    elif c['concentration'] > 0.5:
        bucket = 'B4'
    else:
        bucket = 'B5'
    return {'bubble_share': c['bubble_share'], 'simultaneity': c['simultaneity'],
            'concentration': c['concentration'], 'free_ticks': c['free_ticks'],
            'assigned_bucket': bucket,
            'note': '四类都不是的轨迹同样得到确定答案——判据的输出不是"识别"，是"强制归类"'}


# ---------------------------------------------------------------- 汇总与验证


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


def run_all():
    return {
        'config': {'seeds': len(SEEDS), 'ticks': T_GRID, 'replicas': N_REPLICAS,
                   'ratio_ref': RATIO_REF},
        'tau_threshold': scenario_tau_threshold(),
        'ratio_curve': scenario_ratio_curve(),
        'mirror_length': scenario_mirror_length(),
        'component_mass': scenario_component_mass(),
        'additivity_mass': scenario_additivity_mass(),
        'mirror_context': scenario_mirror_context(),
        'mirror_misaligned': scenario_mirror_misaligned(),
        'mirror_boundary': scenario_mirror_boundary(),
        'same_scale': scenario_same_scale(),
        'cross_scale': scenario_cross_scale(),
        'rule_circularity': scenario_rule_circularity(),
    }


def verify():
    if not os.path.exists(OUT_PATH):
        print('resolution.json 不存在，先运行一次全量')
        return 1
    with open(OUT_PATH) as f:
        archived = json.load(f)
    fresh = _strip_nan(_canon(run_all()))
    if _strip_nan(_canon(archived)) != fresh:
        diff = [k for k in fresh if _strip_nan(_canon(archived.get(k))) != fresh[k]]
        print('差异字段：', diff)
        return 1
    print('--verify 通过：与归档 resolution.json 逐字段一致')
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
