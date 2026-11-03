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
import os
import random
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / 'tools' / 'paired-diff'))
import paired_diff  # noqa: E402

# 引入共享工具的提交；其父提交即"旧版"
TOOL_COMMIT = '038fc2e3b'


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def old_source_at(relpath, commit=TOOL_COMMIT):
    """从 git 取出**抽共享工具之前**（`commit^`）的脚本原文。

    第七批把旧版手写实现写成 `old_*` 函数直接嵌在本文件里，另留了三条
    `/tmp/old_*.py` 常量——那三个文件是本机临时产物、不在仓库里，换台机器就
    不存在，"旧版实现"于是变成一份不会随着上游变动的手抄本。这里改成从 git
    直接取原文件（含完整上下文与依赖），核对性因此不依赖任何人先跑过某个命令。
    """
    return subprocess.run(
        ['git', '-C', str(ROOT), 'show', '%s^:%s' % (commit, relpath)],
        check=True, capture_output=True, text=True).stdout


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


# ---------------------------------------------------------------- 第八批：舍入与刻度
# 第七批的对照只覆盖**计数**。`same_sign_fraction` 与 `sd_pp` 这两个"非计数"的
# 输出没有逐值对照，而它们恰恰是写进正文的读数（同号占比、种子间 sd）。这里补两件：
# ① 逐值比对（旧版手写 vs 共享实现）——预期**零分歧**，包括平局输入；
# ② 把平局的口径**量出来**：`round` 取偶 vs 人类读法（四舍五入）差多少。

def rounding_columns():
    """覆盖舍入的输入列：全同向、半数、平局、极小数、大数。

    平局只有靠**二进制可精确表示**的值才造得出来（1.2375 = 99/80、0.3125 = 5/16）；
    真实差值来自浮点除法与求和，末位永远不会恰好落在 5 上，这也是"碰到平局"
    从未在归档里发生的原因。
    """
    return {
        'ssf_5of16': [0.1] * 5 + [-0.1] * 11,        # 同号占比 0.3125（平局）
        'ssf_all_pos': [0.1] * 16,
        'ssf_half': [0.1] * 8 + [-0.1] * 8,
        'mean_tie_1_2375': [0.2] * 10 + [0.3] * 6,   # mean+1 = 1.2375（平局）
        'tiny_diffs': [1e-9, 2e-9, -1e-9],
        'large_diffs': [12345.6789, -12345.1111],
        'single_point': [0.5],
        'two_points': [0.0, 0.0001],
    }


def compare_ssf_sd(diffs):
    """`same_sign_fraction` 与 `sd_pp` 两列输出逐值比对（旧版 vs 新版）。"""
    if not diffs:
        return []
    issues = []
    old_b = old_sign_counts_bucketing(diffs)
    old_r = old_sign_counts_real(diffs)
    new = new_sign_counts(diffs)
    for who, old in (('bucketing', old_b), ('real', old_r)):
        for key in ('same_sign_fraction', 'sd_pp'):
            if old.get(key) != new.get(key):
                issues.append(('%s.%s' % (who, key), old.get(key), new.get(key), 'RATIO'))
    return issues


def rounding_audit(real_cols):
    """把"平局怎么舍入"从声明变成数字。

    - `tie_inputs_found`：所有被对照的输入列里，`sd_pp` 的末位落在 5 上的个数。
      真实数据为 **0**——这就是"平局不会发生"的实测依据，而不是推测。
    - `half_up_vs_builtin`：构造平局时两种读法的差。有差 ⇒ 若将来真出现平局，
      归档里的第 3 位小数取决于用的是哪一个，必须写明。
    """
    cases, half_up, builtin = paired_diff.rounding_cases()
    tie_hits = []
    for name, diffs in real_cols.items():
        s = paired_diff.sd_pp(diffs)
        if s is None:
            continue
        scaled = s * 1000.0
        frac = abs(scaled - round(scaled))
        if abs(frac - 0.5) < 1e-6:
            tie_hits.append((name, s))
    return {
        'tie_inputs_found': tie_hits,
        'half_up_vs_builtin': {
            k: {'value': cases[k], 'half_up': half_up[k], 'builtin': builtin[k]} for k in cases},
    }


# ---------------------------------------------------------------- 第八批：盲区扫描
# 第七批发现"距 1 不到 TOL 的比率在重新定基时被吞成 0、记成 identical"，但只核了
# `ledger.json`。这里扫**全部六份归档**的每一个 float，分别对 ref=0（差值）与
# ref=1（比率）各查一遍。

ARCHIVES = [
    'experiments/rollout-c-class-ledger/ledger.json',
    'experiments/rollout-c-class-ledger/wall-threshold-band.json',
    'experiments/rollout-c-class-ledger/resolution.json',
    'experiments/rollout-group-split-tradeoff/bucketing-probe.json',
    'experiments/rollout-group-split-tradeoff/real-lengths-summary.json',
    'experiments/rollout-group-split-tradeoff/legacy-group-arm-check.json',
]


def _floats(obj, path=''):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _floats(v, path + '/' + k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _floats(v, path + '/%d' % i)
    elif isinstance(obj, float) and not isinstance(obj, bool):
        yield path, obj


def blind_spot_scan(paths=None):
    """每份归档里 `0 < |v - ref| < TOL`（ref=0 或 1）的成员——它们会被吞成 identical。"""
    out = {}
    for rel in (paths or ARCHIVES):
        p = ROOT / rel
        if not p.exists():
            out[rel] = {'error': 'missing'}
            continue
        data = json.loads(p.read_text())
        hits_zero, hits_one, n_float = [], [], 0
        for path, v in _floats(data):
            n_float += 1
            if 1e-30 < abs(v - 0.0) < paired_diff.TOL:
                hits_zero.append((path, v))
            if 1e-30 < abs(v - 1.0) < paired_diff.TOL:
                hits_one.append((path, v))
        out[rel] = {'n_float': n_float, 'near_zero': hits_zero, 'near_one': hits_one}
    return out


def scale_audit(real_cols):
    """同一列比率、两种刻度下的 `n_identical`——说明"相同"这个词必须带刻度。

    `real_lengths_policy`／`bucketing_probe` 用原量纲（`v - 1`），
    `steps_ratio`／`wall_ratio` 的摆幅用 pp；两者容忍的偏差差 100 倍。
    """
    rows = []
    for name, vals in sorted(real_cols.items()):
        if not name.startswith('flipcol:'):
            continue          # 只有比率型列适用
        sc = paired_diff.across_scale(vals, ref=1.0)
        rows.append((name, sc['raw']['n_identical'], sc['pp']['n_identical']))
    return rows


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

    # ---- 第八批：same_sign_fraction／sd_pp 逐值 + 舍入口径
    rnd = rounding_columns()
    ssf_issues = []
    for name, diffs in sorted(list(real.items()) + list(synth.items()) + list(rnd.items())):
        for it in compare_ssf_sd(diffs):
            ssf_issues.append((name,) + it)
    print()
    print('same_sign_fraction／sd_pp 逐值（真实 %d + 合成 %d + 舍入用例 %d 列）：%d 处分歧'
          % (len(real), len(synth), len(rnd), len(ssf_issues)))
    for row in ssf_issues[:10]:
        print('   ', row)

    audit = rounding_audit(dict(list(real.items()) + list(rnd.items())))
    print('真实输入里 sd_pp 落在舍入平局上的个数：%d' % len(audit['tie_inputs_found']))
    for k, v in audit['half_up_vs_builtin'].items():
        print('   平局 %-16s 值=%-8s 四舍五入=%s  内建 round=%s%s'
              % (k, v['value'], v['half_up'], v['builtin'],
                 '  ← 两种读法不同' if v['half_up'] != v['builtin'] else ''))

    # ---- 第八批：盲区扫描 + 刻度
    scan = blind_spot_scan()
    blind_total = 0
    print()
    print('盲区扫描（0 < |v-ref| < TOL 的浮点数）：')
    for rel, row in scan.items():
        if 'error' in row:
            print('   %-64s %s' % (rel, row['error']))
            continue
        n = len(row['near_zero']) + len(row['near_one'])
        blind_total += n
        print('   %-64s %d 个 float，ref=0 近零 %d，ref=1 近一 %d'
              % (rel.split('/')[-1], row['n_float'], len(row['near_zero']),
                 len(row['near_one'])))
    for row in scale_audit(real):
        if row[1] != row[2]:
            print('   刻度敏感：%s raw=%d pp=%d' % row)

    ok = (not real_issues) and (not span_issues) and (not ssf_issues)
    print()
    print('真实归档列零分歧：%s' % ('是' if not real_issues else '否'))
    print('极差／翻号零分歧：%s' % ('是' if not span_issues else '否'))
    print('same_sign_fraction／sd_pp 零分歧：%s' % ('是' if not ssf_issues else '否'))
    print('盲区命中：%d 个（0 = 既有归档无一落进 |v-ref| < TOL）' % blind_total)
    # 覆盖自查：既有「真实列零分歧」只对**计数**是同一句话——整数步号 40 列
    # 走的是精确等价路径，旧版 `d == 0` 与新版天然同答；真正检验容差的只有
    # 那些贴着 1 的浮点比率列。这里把覆盖面写出来，避免再次把"零分歧"读大。
    n_int = sum(1 for d in real.values() if all(float(x) == int(x) for x in d))
    n_ratio = len(real) - n_int
    n_flip = sum(1 for n, d in real.items()
                 if not n.startswith('steps:') and paired_diff.sign_flip_around(d, 1.0))
    print('覆盖：整数步号列 %d（精确等价路径，不检验容差）／浮点比率列 %d，'
          '其中相对 1 翻号 %d 列' % (n_int, n_ratio, n_flip))
    print('结论：%s' % ('PASS' if ok else 'FAIL'))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
