#!/usr/bin/env python3
"""§21.9 会计恒等式吃的是「被测分项」：C1 计时三件套的误差在 §13.25 恒等式上怎么传播

支撑 notebook/rollout-performance-research.md §21.9。

§13.25 把「阶段占比中介」升级为**精确会计恒等式**：
    整步× = 1 / ( Σ_k s_k/G_k + Σ_m a_m/T )
它是代数恒等式——**只要分项与整步来自同一份测量，重建必然精确**。
但真实计时块不是分项真值：§21.1 的 C1 列出三类必须声明的边界
（含策略等待 / 分项重叠 / 未命名块）。本实验问：**把这三类误差注进分项，恒等式会怎么骗人？**

  Q1  恒等式在「喂被测分项」时是否仍自洽？→ 是，且这正是危险所在（它不自我校验）
  Q2  把子阶段藏进一个块、再声称"该块加速 G 倍"，会多报多少？
  Q3  丢掉未计时的墙钟（分母缩水）会多报多少？
  Q4  阶段占比差 1pp 对最终倍数的影响有多大？解析式是什么？
  Q5  论文真实分项（MTP-RL 7B）上，把「训练块含策略等待」注进去会怎样？

墙钟是任意虚拟秒，不可与任何论文报值相除（§19.1 规则一）。只用标准库。

用法：
    python3 identity_measurement_bias.py             # 全量重跑并覆盖 summary.json
    python3 identity_measurement_bias.py --verify    # 重跑并与归档 summary.json 逐字段比对
    python3 -m unittest test_identity_measurement_bias -v
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY_PATH = os.path.join(HERE, 'summary.json')


# ---------------------------------------------------------------- 恒等式


def identity_ratio(parts, g):
    """§13.25 恒等式：整步× = Σt_k / Σ(t_k/G_k)。parts 与 g 用同一组键。"""
    total = sum(parts.values())
    denom = sum(parts[k] / g[k] for k in parts)
    return total / denom


def analytic_share_sensitivity(s, g):
    """阶段占比误差对倍数的放大系数。

    R = 1/(1 − a·s)，a = 1 − 1/G
    ⇒ dR/ds = a·R²（绝对斜率）
    ⇒ dR/R = a·R·ds（相对弹性，乘 ds 得相对误差）
    """
    a = 1.0 - 1.0 / g
    r = 1.0 / (1.0 - a * s)
    return {'R': r,
            'absolute_slope': a * r * r,
            'relative_elasticity': a * r}


# 基线：一个训练步的 10 秒。生成 4s / logp 1s / refit 0.5s / 训练 3s / GPU 空转 1.5s
BASE = {'gen': 4.0, 'logp': 1.0, 'refit': 0.5, 'train': 3.0, 'idle': 1.5}
TOTAL = sum(BASE.values())


# ---------------------------------------------------------------- Q1 自洽性


def q1_self_consistency():
    """恒等式喂被测分项仍然自洽——这不是精度，是同一个等式两边。

    情形：
      clean          分项与整步都完整，没有加速 ⇒ 必然精确
      idle_folded    空转被并进"训练"块，但仍未给任何块加速 ⇒ 仍然精确
      idle_lost      一部分空转根本没被计时（分母缩水）⇒ 即使 G 全为 1 也会偏
    """
    out = {}
    g_none = {k: 1.0 for k in BASE}
    # clean
    out['clean'] = {'truth': identity_ratio(BASE, g_none),
                    'rebuilt': identity_ratio(BASE, g_none),
                    'rel_err_pct': 0.0}
    # idle 并进 train（未加速任何块时，合并与否不改变等式）
    folded = {'gen': 4.0, 'logp': 1.0, 'refit': 0.5, 'train': 4.5, 'idle': 0.0}
    out['idle_folded_no_accel'] = {
        'truth': identity_ratio(BASE, g_none),
        'rebuilt': identity_ratio(folded, g_none),
        'rel_err_pct': 0.0}
    # 空转有 50% 从未被计时
    lost = {'gen': 4.0, 'logp': 1.0, 'refit': 0.5, 'train': 3.75, 'idle': 0.0}
    g_gen2 = {'gen': 2.0, 'logp': 1.0, 'refit': 1.0, 'train': 1.0, 'idle': 1.0}
    truth = identity_ratio(BASE, g_gen2)          # 用完整真值算的真值倍数
    rebuilt = identity_ratio(lost, g_gen2)        # 用缩水后的读数重建
    out['idle_half_unmeasured'] = {
        'truth_total': TOTAL, 'measured_total': sum(lost.values()),
        'truth': truth, 'rebuilt': rebuilt,
        'rel_err_pct': (rebuilt - truth) / truth * 100.0}
    return out


# ---------------------------------------------------------------- Q2 藏起来的子阶段


def q2_hidden_substage():
    """把子阶段藏进一个块、再声称"该块加速 G 倍"，会多报多少。

    train 块真实由两部分构成：训练 3.0（可加速到 G）＋空转 1.5（由调度决定，不随机制加速）。
    论文把两者合成一个 4.5 的"训练"块，并声称该块达到 G 倍。
    """
    out = {}
    g_truth = {'gen': 1.0, 'logp': 1.0, 'refit': 1.0, 'train': 3.0, 'idle': 1.0}
    for gval in (1.5, 2.0, 3.0):
        truth_parts = dict(BASE)
        truth_g = dict(g_truth)
        truth_g['train'] = gval
        truth = identity_ratio(truth_parts, truth_g)
        # 错误做法：把整个"训练"块（含空转）都按 gval 加速
        wrong_parts = {'gen': 4.0, 'logp': 1.0, 'refit': 0.5,
                       'train': BASE['train'] + BASE['idle'], 'idle': 0.0}
        wrong_g = {'gen': 1.0, 'logp': 1.0, 'refit': 1.0, 'train': gval, 'idle': 1.0}
        wrong = identity_ratio(wrong_parts, wrong_g)
        out['train_G%.1f' % gval] = {
            'truth': truth, 'over_credited': wrong,
            'rel_err_pct': (wrong - truth) / truth * 100.0,
            'hidden_share_of_block': BASE['idle'] / (BASE['train'] + BASE['idle'])}
    # 隐藏份额扫描（G=2 固定）
    scan = []
    for hidden in (0.0, 0.1, 0.2, 0.333, 0.5):
        train_true = BASE['train']
        idle_hidden = train_true * hidden / (1.0 - hidden) if hidden < 1 else 0.0
        parts = {'gen': BASE['gen'], 'logp': BASE['logp'], 'refit': BASE['refit'],
                 'train': train_true, 'idle': idle_hidden}
        g_true = {'gen': 1.0, 'logp': 1.0, 'refit': 1.0, 'train': 2.0, 'idle': 1.0}
        truth = identity_ratio(parts, g_true)
        wrong_parts = {'gen': parts['gen'], 'logp': parts['logp'], 'refit': parts['refit'],
                       'train': train_true + idle_hidden, 'idle': 0.0}
        wrong = identity_ratio(wrong_parts, {'gen': 1.0, 'logp': 1.0, 'refit': 1.0,
                                            'train': 2.0, 'idle': 1.0})
        scan.append({'hidden_share': hidden, 'truth': truth, 'over_credited': wrong,
                     'rel_err_pct': (wrong - truth) / truth * 100.0})
    out['hidden_share_scan'] = scan
    return out


# ---------------------------------------------------------------- Q3 分母缩水


def q3_denominator_shrink():
    """丢掉未计时的墙钟：加速块的占比被抬高，倍数被多报。

    真值 10s，生成 4s 加速 2× ⇒ 8s ⇒ 1.25×。
    如果 0.75s 的空转从未进入任何计时块，读数 9.25s、
    重建分母 4/2 + 5.25 = 7.25 ⇒ 1.2759×（偏高 2.07%）。
    """
    out = {}
    g = {'gen': 2.0, 'logp': 1.0, 'refit': 1.0, 'train': 1.0, 'idle': 1.0}
    truth = identity_ratio(BASE, g)
    scan = []
    for lost in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
        p = dict(BASE)
        take = min(lost, p['idle'])
        p['idle'] -= take
        r = identity_ratio(p, g)
        scan.append({'lost_seconds': lost, 'measured_total': sum(p.values()),
                     'rebuilt': r, 'rel_err_pct': (r - truth) / truth * 100.0})
    out['truth'] = truth
    out['scan'] = scan
    # 解析校验：丢掉的份额 s_lost（相对真值）使倍数变为 R/(1 - s_lost·(1-1/G))·… 直接数值核
    return out


# ---------------------------------------------------------------- Q4 占比敏感性


def q4_share_sensitivity():
    """阶段占比每差 1pp，最终倍数偏多少——解析式 dR/R = R(1−1/G)·ds。"""
    out = {}
    s_true = BASE['gen'] / TOTAL
    for gval in (1.5, 2.0, 3.0, 5.0):
        a = analytic_share_sensitivity(s_true, gval)
        # 数值核：占比 ±1pp / ±5pp
        rows = []
        for dpp in (-5, -2, -1, 0, 1, 2, 5):
            s = s_true + dpp / 100.0
            r = 1.0 / (s / gval + (1.0 - s))
            rows.append({'delta_pp': dpp, 'ratio': r,
                         'rel_err_pct': (r - a['R']) / a['R'] * 100.0})
        out['G%.1f' % gval] = {'s_true': s_true, 'R': a['R'],
                               'absolute_slope': a['absolute_slope'],
                               'relative_elasticity': a['relative_elasticity'],
                               'rows': rows}
    return out


# ---------------------------------------------------------------- Q5 真实分项


def q5_reported_case():
    """MTP-RL 7B 的真实分项（§13.25 逐字）：MTP 50.74 / Rollout 223.93 / Training 295.47。

    假设论文声称"训练块加速 1.5×"。真值分解是 rollout 与 training 两块；
    若 training 计时块里含了 5% 总时长的策略等待（本该算 rollout 尾），
    重建倍数会偏高多少？
    """
    gen, train = 223.93, 295.47
    total = gen + train
    g_train = 1.5
    out = {'base_total': total, 'g_train': g_train}
    truth = total / (gen + train / g_train)
    out['truth_ratio'] = truth
    scan = []
    for leak_share in (0.0, 0.02, 0.05, 0.10, 0.20):
        leak = total * leak_share
        measured_gen = gen - leak
        measured_train = train + leak
        r = total / (measured_gen + measured_train / g_train)
        scan.append({'leak_share': leak_share, 'leak_seconds': leak,
                     'rebuilt': r,
                     'rel_err_pct': (r - truth) / truth * 100.0})
    out['train_block_leak'] = scan
    # 对照：若加速的恰是被泄漏进来的那部分（即泄漏高估的是加速块）
    out['note'] = ('泄漏把时间从 gen 挪进 train；当 train 加速而 gen 不加速时，'
                   '被挪过去的时间也被"加速"了，倍数虚高')
    return out


def run_all():
    return {
        'config': {'base': BASE, 'total': TOTAL},
        'q1_self_consistency': q1_self_consistency(),
        'q2_hidden_substage': q2_hidden_substage(),
        'q3_denominator_shrink': q3_denominator_shrink(),
        'q4_share_sensitivity': q4_share_sensitivity(),
        'q5_reported_case': q5_reported_case(),
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
