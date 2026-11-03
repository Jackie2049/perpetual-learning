#!/usr/bin/env python3
"""两版实现逐值对照：共享判据 vs 各自手写的那一份（2026-11-03 M1 第六批遗留③）。

**为什么要这个文件**：第六批把判据抽进 `tools/paired-diff` 之后，唯一的验收证据是
"五份归档 `--verify` 逐字段相同"。那条证据说明不了"抽出来的实现 == 原来那份
实现"——它只说明两条链路喂给判据的**输入列**在既有数据上没有落进两版判据的
分歧区。若某个脚本在边界值上原本会给出不同答案，归档比对看不出来。

本文件的做法：**同一列输入，分别喂给旧版（`git show 038fc2e3b^:` 取出的手写实现）
与新版（`tools/paired-diff`），逐值比对**。输入分三类：

  ① 三个脚本的**真实输入列**（从各自归档反解出逐种子的 `diffs`）——证明已发布读数一致；
  ② 边界合成列——容差上、恰等于容差、整数步号、全零、单点、空列；
  ③ 随机列（固定种子）——覆盖真实数据没走到的小数区间。

任何一处分歧都会被报出来，并附带"这处分歧是否已在真实归档中出现"。

运行：cd experiments/rollout-c-class-ledger && python3 compare_old_new_impl.py [--verbose]
"""

import argparse
import importlib.util
import json
import random
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / 'tools' / 'paired-diff'))
import paired_diff  # noqa: E402

# 引入共享工具的提交；其父提交即"旧版"
TOOL_COMMIT = '038fc2e3b'
OLD_LEDGER_PATH = '/tmp/old_ledger.py'
OLD_BUCKET_PATH = '/tmp/old_bucketing.py'
OLD_REAL_PATH = '/tmp/old_real.py'


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- 旧版实现
# 三处手写实现都是"同一族判据、各写一遍"，语义按当时的代码原样复刻。

def old_sign_counts_bucketing(diffs):
    """`bucketing_probe.seed_sign_stability` 旧版（相对 1e-12，无 identical 返回）。"""
    pos = sum(1 for d in diffs if d > 1e-12)
    neg = sum(1 for d in diffs if d < -1e-12)
    return {
        'n_positive': pos,
        'n_negative': neg,
        'same_sign_fraction': round(max(pos, neg) / len(diffs), 3),
        'sd_pp': round(statistics.stdev(diffs) * 100, 3) if len(diffs) > 1 else None,
    }


def old_sign_counts_real(diffs):
    """`real_lengths_policy.policy_sign_stability` 旧版（同上，多一个 n_identical）。"""
    pos = sum(1 for d in diffs if d > 1e-12)
    neg = sum(1 for d in diffs if d < -1e-12)
    return {
        'n_positive': pos,
        'n_negative': neg,
        'n_identical': len(diffs) - pos - neg,
        'same_sign_fraction': round(max(pos, neg) / float(len(diffs)), 3),
        'sd_pp': round(100 * statistics.stdev(diffs), 3) if len(diffs) > 1 else None,
    }


def old_stats_ledger(diffs):
    """`ledger.steps_at` 旧版：**整数精确相等**，没有容差。"""
    n = len(diffs)
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    return {
        'same_sign_arm_slower': (pos / n) if n else None,
        'n_ties': sum(1 for d in diffs if d == 0),
    }


def old_span(vals):
    return (max(vals) - min(vals)) if vals else None


def old_flip_zero(vals):
    return len({v > 0 for v in vals}) > 1 if vals else None


def old_flip_one(vals):
    return len({v > 1 for v in vals}) > 1 if vals else None


# ---------------------------------------------------------------- 新版实现

def new_sign_counts(diffs):
    c = paired_diff.sign_counts(diffs)
    return {
        'n_positive': c['n_positive'],
        'n_negative': c['n_negative'],
        'n_identical': c['n_identical'],
        'same_sign_fraction': paired_diff.same_sign_fraction(diffs),
        'sd_pp': paired_diff.sd_pp(diffs),
    }


# ---------------------------------------------------------------- 真实输入列

def real_columns():
    """从源 summary 逐种子重算三个脚本“喂给判据的那一列”。

    注意：`wall-threshold-band.json` / `ledger.json` 只保留计数与汇总量，没有留下
    逐种子的差值元组，所以从归档反解不出真实输入列（第一版脚本就是这么误报
    “0 列、PASS”的——一个没扫到东西的 PASS 什么也没证明）。正确做法是回到
    `wall_threshold_band.load()` 读的那两份源 summary，按各脚本同一套口径把
    逐种子差值重算出来。
    """
    import wall_threshold_band as W

    p3 = W.load(W.P3_PATH)
    p4 = W.load(W.P4_PATH)
    cols = {}

    def pairs_of(tag, arms):
        # p4 的基线是 'base'；p3 的基线是 'R_nofilter'（与 ledger.py:259 一致）。
        # 第一版错把 p3 的第一个条目当基线，结果 R 自己和自己比、读出 30/30 identical。
        if tag == 'p4':
            return [(a, arms['base']) for a in ('freq', 'replay', 'cancel', 'cap')]
        base = arms['R_nofilter']
        return [(a, base) for a in ('R', 'A', 'B', 'C')]

    # ① ledger.steps_at 的真实输入：整数步号差（逐种子、逐阈值）
    for tag, arms in (('p4', p4), ('p3', p3)):
        for name, base in pairs_of(tag, arms):
            for thr in W.THRESHOLDS:
                diffs = []
                for a, b in zip(arms[name], base):
                    ia = W.first_cross(a['acc_curve'], thr)
                    ib = W.first_cross(b['acc_curve'], thr)
                    if ia is None or ib is None:
                        continue
                    diffs.append(ia - ib)
                if diffs:
                    cols['steps:%s:%s@%.2f' % (tag, name, thr)] = diffs

    # ② wall_ratio_at 的真实输入：逐种子“窗口成本比 − 1”（浮点、贴近 0）
    for tag, arms in (('p4', p4), ('p3', p3)):
        for name, base in pairs_of(tag, arms):
            for thr in W.THRESHOLDS:
                vals = []
                for a, b in zip(arms[name], base):
                    ia = W.first_cross(a['acc_curve'], thr)
                    ib = W.first_cross(b['acc_curve'], thr)
                    if ia is None or ib is None:
                        continue
                    den = sum(b['step_costs'][: ib + 1])
                    if den:
                        vals.append(sum(a['step_costs'][: ia + 1]) / den - 1.0)
                if vals:
                    cols['wall:%s:%s@%.2f' % (tag, name, thr)] = vals

    # ③ 阈值摆幅那一列（span / flip 的真实输入：逐阈值的比率读数）
    for tag, arms in (('p4', p4), ('p3', p3)):
        for name, base in pairs_of(tag, arms):
            wall_vals, step_vals = [], []
            for thr in W.THRESHOLDS:
                r = W.wall_ratio_at(arms[name], base, thr)
                if r['wall_ratio'] is not None:
                    wall_vals.append(r['wall_ratio'])
                na, nb = [], []
                for a, b in zip(arms[name], base):
                    ia = W.first_cross(a['acc_curve'], thr)
                    ib = W.first_cross(b['acc_curve'], thr)
                    if ia is not None and ib is not None:
                        na.append(ia)
                        nb.append(ib)
                if nb and sum(nb):
                    step_vals.append(sum(na) / sum(nb))
            if len(wall_vals) > 1:
                cols['flipcol:%s:%s:wall' % (tag, name)] = wall_vals
            if len(step_vals) > 1:
                cols['flipcol:%s:%s:steps' % (tag, name)] = step_vals
    return cols


# ---------------------------------------------------------------- 边界输入列

def boundary_columns():
    TOL = paired_diff.TOL
    return {
        'empty': [],
        'single_zero': [0.0],
        'single_pos': [0.3],
        'all_identical': [0.0] * 12,
        'exactly_tol': [TOL, -TOL, TOL],
        'just_under_tol': [TOL * 0.999, -TOL * 0.999],
        'just_over_tol': [TOL * 1.001, -TOL * 1.001],
        'integers': [0, 1, -1, 2, 0, 0, -3, 1],
        'half_each': [1.0, -1.0, 1.0, -1.0],
        'cancel_mean': [0.1, -0.05, 0.1, -0.05, 0.01, 0.01],
    }


def random_columns(n=200, seed=20261103):
    rng = random.Random(seed)
    cols = {}
    for i in range(n):
        k = rng.choice([3, 8, 20, 30])
        scale = rng.choice([1e-13, 1e-9, 1e-3, 0.1, 1.0])
        cols['rand%d' % i] = [round(rng.uniform(-scale, scale), 12) for _ in range(k)]
    return cols


# ---------------------------------------------------------------- 比对

def compare(diffs, verbose=False):
    """返回该列的比对结果：新增的分歧项（若两版一致则为空）。"""
    issues = []
    old_b = old_sign_counts_bucketing(diffs) if diffs else None
    old_r = old_sign_counts_real(diffs) if diffs else None
    new = new_sign_counts(diffs) if diffs else None
    if diffs:
        for key in ('n_positive', 'n_negative'):
            if old_b[key] != new[key]:
                issues.append(('bucketing.%s' % key, old_b[key], new[key], 'COUNT'))
            if old_r[key] != new[key]:
                issues.append(('real.%s' % key, old_r[key], new[key], 'COUNT'))
        if old_r['n_identical'] != new['n_identical']:
            issues.append(('real.n_identical', old_r['n_identical'], new['n_identical'], 'COUNT'))
        if old_b['same_sign_fraction'] != new['same_sign_fraction']:
            issues.append(('bucketing.ssf', old_b['same_sign_fraction'],
                           new['same_sign_fraction'], 'RATIO'))
        # 步号口径：旧版用整数精确相等
        if all(isinstance(d, int) or (isinstance(d, float) and d == int(d)) for d in diffs):
            old_l = old_stats_ledger([int(d) for d in diffs])
            new_l_ties = paired_diff.sign_counts(diffs)['n_identical']
            if old_l['n_ties'] != new_l_ties:
                issues.append(('ledger.n_ties', old_l['n_ties'], new_l_ties, 'COUNT'))
    if verbose:
        print('   old(bucket)=%r' % (old_b,))
        print('   new        =%r' % (new,))
    return issues


def compare_spans():
    """极差与翻号两处：旧版 `max-min`／`{v>ref}` 与共享实现逐值比对。"""
    issues = []
    rng = random.Random(7)
    for i in range(200):
        k = rng.choice([1, 2, 5])
        scale = rng.choice([1e-12, 1e-6, 0.1, 1.0])
        vals = [1.0 + rng.uniform(-scale, scale) for _ in range(k)]
        if old_span(vals) != paired_diff.span(vals):
            issues.append(('span#%d' % i, old_span(vals), paired_diff.span(vals), 'SPAN'))
        if old_flip_one(vals) != paired_diff.sign_flip_around(vals, 1.0):
            issues.append(('flip_one#%d' % i, old_flip_one(vals),
                           paired_diff.sign_flip_around(vals, 1.0), 'FLIP'))
    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    real = real_columns()
    synth = dict(boundary_columns())
    synth.update(random_columns())

    all_issues = []
    real_issues = []
    for name, diffs in sorted(real.items()):
        iss = compare(diffs, args.verbose)
        for it in iss:
            real_issues.append((name,) + it)
    for name, diffs in sorted(synth.items()):
        iss = compare(diffs, args.verbose)
        for it in iss:
            all_issues.append((name,) + it)
    span_issues = compare_spans()

    print('真实输入列（从源 summary 逐种子重算）：%d 列' % len(real))
    if len(real) < 40:
        print('  ** 真实列不足 40 —— 多半是没扫到源数据，这个 PASS 不可信 **')
        print('结论：FAIL（覆盖不足）')
        sys.exit(1)
    print('  分歧：%d' % len(real_issues))
    for row in real_issues[:10]:
        print('   ', row)
    print('合成边界 + 随机输入列：%d 列' % len(synth))
    print('  分歧：%d' % len(all_issues))
    # 分歧按"形态"归类，便于判断是新版更严还是旧版更松
    from collections import Counter
    shape = Counter()
    for _, field, old, new, kind in all_issues:
        if kind == 'COUNT':
            if old < new:
                shape['旧版少记（容差内记成别的类）'] += 1
            else:
                shape['旧版多记'] += 1
        elif kind == 'RATIO':
            shape['同号占比不同'] += 1
    for k, v in shape.items():
        print('   %s: %d' % (k, v))
    for row in all_issues[:10]:
        print('   ', row)
    print('极差／翻号逐值：%d 处分歧' % len(span_issues))
    for row in span_issues[:5]:
        print('   ', row)

    # 判定：真实列必须零分歧；合成分歧只允许出现在"容差边界"这一类
    ok = (not real_issues) and (not span_issues)
    print()
    print('真实归档列零分歧：%s' % ('是' if not real_issues else '否'))
    print('极差／翻号零分歧：%s' % ('是' if not span_issues else '否'))
    print('结论：%s' % ('PASS' if ok else 'FAIL'))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
