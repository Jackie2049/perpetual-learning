#!/usr/bin/env python3
"""两版实现逐值对照：共享判据 vs 各自手写的那一份（2026-11-03 M1 第六批遗留③）。

**为什么要这个文件**：第六批把判据抽进 `tools/paired-diff` 之后，唯一的验收证据是
"五份归档 `--verify` 逐字段相同"。那条证据说明不了"抽出来的实现 == 原来那份
实现"——它只说明两条链路喂给判据的**输入列**在既有数据上没有落进两版判据的
分歧区。若某个脚本在边界值上原本会给出不同答案，归档比对看不出来。

本文件的做法：**同一列输入，分别喂给旧版（`git show 038fc2e3b^:` 取出的手写实现）
与新版（`tools/paired-diff`），逐值比对**。输入分三类：

  ① 五个脚本的**真实输入列**（从源 summary 逐种子重算出喂给判据的 `diffs`）——证明已发布读数一致；
  ② 边界合成列——容差上、恰等于容差、整数步号、全零、单点、空列；
  ③ 随机列（固定种子）——覆盖真实数据没走到的小数区间。

任何一处分歧都会被报出来，并附带"这处分歧是否已在真实归档中出现"。

**旧版实现的取材方式（第九批改）**：① 沿用的三处手写实现是 `git show 038fc2e3b^:` 取出
原文件后**逐字抽出**函数体并 `exec`，不再由本文件手抄一份；② 分桶探针与真实负载脚本的旧版
直接从 `git show 038fc2e3b^:` 加载整个模块。手抄本一旦与上游漂移，"零分歧"证明的就是
两份抄本一致，而不是"抽工具没有改变语义"。

运行：cd experiments/rollout-c-class-ledger && python3 compare_old_new_impl.py [--verbose]
"""

import argparse
import ast
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


def load_old_module(relpath, commit=TOOL_COMMIT, name=None):
    """从 git 取出 `commit^` 时刻的**整个脚本**并 import（第九批）。

    第七／八批只把"旧版实现"从 git 取出来看，实际对照的仍是本文件里的一份手抄
    副本——抄错一处、或者上游后来又改了这个函数，对照照样全绿。这里把旧版脚本
    本身加载进来当被测对象：`bucketing_probe.seed_sign_stability`、
    `bucketing_probe.summarize`、`real_lengths_policy.fifo_reorder_band` 三处
    直接取旧版对象，不再手抄。
    """
    src = old_source_at(relpath, commit)
    modname = name or ('old_' + Path(relpath).stem)
    spec = importlib.util.spec_from_loader(modname, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__dict__['__file__'] = str(ROOT / relpath)
    sys.modules[modname] = mod
    exec(compile(src, str(ROOT / relpath) + ' @ %s^' % commit, 'exec'), mod.__dict__)
    return mod


def _old_module(relpath, commit=TOOL_COMMIT):
    """从 `commit^` 取出**整个脚本**并 import（第九批）。

    分桶探针与真实负载脚本被抽走判据的，直接换成旧版对象——本文件不再手抄一份
    "旧版实现"：手抄本一旦与上游漂移，"零分歧"证明的只是两份抄本一致。
    """
    src = old_source_at(relpath, commit)
    name = 'old_' + Path(relpath).stem
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__dict__['__file__'] = str(ROOT / relpath)
    sys.modules[name] = mod
    sys.path.insert(0, str(ROOT / os.path.dirname(relpath)))
    exec(compile(src, '<old %s @ %s^>' % (relpath, commit), 'exec'), mod.__dict__)
    return mod


def _extract_old(relpath, funcs, commit=TOOL_COMMIT):
    """从 `commit^` 的原文件里抽出指定顶层 def（连同模块级常量）并 `exec`。

    旧源码里的 import、模块级常量与被抽的函数体都保留——`ledger.steps_at` 依赖
    `first_cross`，靠这个把它一起带进来；整份文件又带 argparse 主程序与相对路径
    常量，所以不能整份 exec。
    """
    src = old_source_at(relpath, commit)
    tree = ast.parse(src)
    keep = [n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom, ast.Assign))
            and not (isinstance(n, ast.FunctionDef) and n.name not in funcs)]
    ns = {}
    ns['__file__'] = str(ROOT / relpath)
    sys.path.insert(0, str(ROOT / os.path.dirname(relpath)))
    exec(compile(ast.Module(body=keep, type_ignores=[]), '<old %s>' % relpath, 'exec'), ns)
    return {name: ns[name] for name in funcs}


# ---------------------------------------------------------------- 旧版实现
# 三处手写实现都是"同一族判据、各写一遍"，语义直接取自抽工具之前的那版源码。

def _old_handwritten():
    lg = _extract_old('experiments/rollout-c-class-ledger/ledger.py',
                      {'steps_at', 'first_cross', '_cross_or_complete'})
    bl_mod = _old_module('experiments/rollout-group-split-tradeoff/bucketing_probe.py')
    rl_mod = _old_module('experiments/rollout-group-split-tradeoff/real_lengths_policy.py')
    return (lg, bl_mod.seed_sign_stability, rl_mod.policy_sign_stability)


(_OLD_LEDGER_MOD, _OLD_SEED_SIGN_STABILITY, _OLD_POLICY_SIGN_STABILITY) = _old_handwritten()
_OLD_STEPS_AT = _OLD_LEDGER_MOD['steps_at']
_OLD_FIRST_CROSS = _OLD_LEDGER_MOD['first_cross']
OLD_DONE = 10.0        # 造出来的曲线里"早已到点"的那一段长度
_OLD_BUCKETING = None    # 惰性加载，见 old_bucketing()
_OLD_REAL_LENGTHS = None


def old_bucketing():
    """旧版 `bucketing_probe` 模块（含 `summarize`／`seed_sign_stability`）。"""
    global _OLD_BUCKETING
    if _OLD_BUCKETING is None:
        _OLD_BUCKETING = _old_module('experiments/rollout-group-split-tradeoff/bucketing_probe.py')
    return _OLD_BUCKETING


def old_real_lengths():
    """旧版 `real_lengths_policy` 模块（含 `fifo_reorder_band`）。"""
    global _OLD_REAL_LENGTHS
    if _OLD_REAL_LENGTHS is None:
        _OLD_REAL_LENGTHS = _old_module('experiments/rollout-group-split-tradeoff/real_lengths_policy.py')
    return _OLD_REAL_LENGTHS


def old_sign_counts_bucketing(diffs):
    """`bucketing_probe.seed_sign_stability` 旧版（相对 1e-12，无 identical 返回）。"""
    rows = [{'abs': d, 'rel': 1.0} for d in diffs]
    return _OLD_SEED_SIGN_STABILITY(rows)


def old_sign_counts_real(diffs):
    """`real_lengths_policy.policy_sign_stability` 旧版（同上，多一个 n_identical）。

    旧版签名带排程参数（`groups, n_groups, backlog, policy`）、内部自己抽种子，
    喂不了任意列；但它那一段计数逻辑与分桶探针的旧版**逐行相同**，所以这里沿用
    抽取出来的分桶旧版。旧版没有"容忍差值"这个键，`n_identical` 恒为 0——这本身
    就是第五批修好的那个分辨率刹车。
    """
    rows = [{'abs': d, 'rel': 1.0} for d in diffs]
    out = dict(_OLD_SEED_SIGN_STABILITY(rows))
    # 旧版把 |d| < 1e-12 的种子既不算正也不算负，`n_identical` 本来是可以由此推出的；
    # 早先写成恒 0，把它读成了"旧版没有分辨率刹车"，于是每个含并列的列都被报成分歧。
    out['n_identical'] = len(diffs) - out['n_positive'] - out['n_negative']
    return out


def _old_stats_ledger(diffs):
    """旧版 `ledger.steps_at` 读的"同号比例／并列数"，用旧版 `steps_at` 本身算出来。

    旧 `steps_at` 收的是**准确率曲线**，我们要的是它读出的那列配对差；所以把差值
    编码进曲线（见下面的 `_old_curve_for`），交给旧函数，再把它的两个计数键取回来：
    `same_sign_arm_slower`（这里换算成 `same_sign_arm_slower` 的原义 = 正号占比）
    与 `n_ties`（`d == 0` 的个数）。
    """
    if not diffs:
        return {'same_sign_arm_slower': None, 'n_ties': 0, 'n_pairs': 0}
    curves = [_old_curve_for(d) for d in diffs]
    base = [_old_curve_for(0) for _ in diffs]
    out = _OLD_STEPS_AT(curves, base, PROBE_THR)
    return {
        'same_sign_arm_slower': out['same_sign_arm_slower'],
        'n_ties': out['n_ties'],
        'n_pairs': out['n_pairs'],
    }


def _all_integral(diffs):
    """整列差值是否**逐个**都是整数步号（浮点 2.0 也算、-1.0 也算）。"""
    return bool(diffs) and all(float(d).is_integer() for d in diffs)


# ---------------------------------------------------------------- 步号差值的曲线编码
# 旧 `steps_at` 的入参是准确率曲线，取 `first_cross(curve, thr)` 当步号。差值可正
# 可负，而步号必须非负——所以对每个差值造一对曲线：
#   臂：`[0.0]*K + [PROBE_THR]`，在第 K 步到点；
#   基线：`[0.0]*K0 + [PROBE_THR]`，K0 固定。
# 于是 `first_cross(臂) − first_cross(基线) = K − K0`，取 `K = K0 + d` 即可还原差值。
# 曲线元素一律 ≥ PROBE_THR 的只会出现在第 K 步，前面全是 0.0 —— **首元素必须低于
# 阈值**，这是本函数前两版踩的坑（把 0.0 当首元素、结果恒首达 0）。

PROBE_THR = 0.90
K0 = 50          # 基线到点的步号；差值 d 对应臂在第 K0+d 步到点


def _old_curve_for(d, k0=K0):
    """差值 `d` → 曲线：在第 `k0 + d` 步才跨过 `PROBE_THR`。"""
    k = k0 + int(round(d))
    if k < 0:
        raise ValueError('差值 %r 超出可编码范围（步号不能为负）' % (d,))
    return [0.0] * k + [PROBE_THR]


def recover_old_step(curve):
    """`first_cross(curve, PROBE_THR)` —— 用旧版函数取回点在第几步。"""
    return _OLD_FIRST_CROSS(curve, PROBE_THR)


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


# ---------------------------------------------------------------- 第九批：分桶探针与真实负载
# 第七／八批的对照只覆盖 C 类账那两个脚本。分桶探针（`bucketing_probe`）与真实负载
# 脚本（`real_lengths_policy`）的**真实输入列**当时没纳入，而 §13.30 第四十一／四十二批
# 的读数正是从这两处来的。这里补两件事：
#   ① 真实输入列：重跑这两个脚本的负载产生过程，拿到逐种子的配对差，喂给新旧两版；
#   ② 非计数输出：`summarize` 里的 identical 与同号占比，逐格比对。

class ProbeCache:
    """惰性、只算一次的分桶探针真实输入（跑一次约数分钟）。"""

    def __init__(self):
        self._cells = None

    def cells(self):
        if self._cells is None:
            import bucketing_probe as BP
            self._cells = list(BP.iter_rows())
        return self._cells

    def columns(self):
        """逐格逐种子配对差：`(格子, 每种子 FIFO−LOF)`。"""
        return {'probe:%s' % cell: [r['abs'] for r in rows]
                for cell, rows, _ in self.cells()}


_PROBE = ProbeCache()


def new_summarize(rows):
    import bucketing_probe as BP
    return BP.summarize(rows)


def compare_probe_cells(cells):
    """分桶探针的**每格输出**新旧比对：`summarize` 的非计数部分 + 真实输入列。

    比对的是归档里真实存在的键：`identical`（旧版手写 `abs(d) < 1e-12`）、
    `abs_pp_mean`／`abs_pp_median`、`sign_stability` 里的同号比例与 sd。计数部分
    由上一条 `compare()` 负责，这里只管"非计数输出"。
    """
    old_mod = old_bucketing()
    issues, n_cells, n_rows = [], 0, 0
    keys = ('identical', 'abs_pp_mean', 'abs_pp_median',
            'sign_stability.n_positive', 'sign_stability.n_negative',
            'sign_stability.same_sign_fraction', 'sign_stability.sd_pp',
            'sign_stability.min_pp', 'sign_stability.max_pp')

    def _get(d, path):
        for part in path.split('.'):
            if not isinstance(d, dict) or part not in d:
                return '<missing>'
            d = d[part]
        return d

    for cell, rows, _ in cells:
        n_cells += 1
        n_rows += len(rows)
        old = old_mod.summarize(rows)
        new = new_summarize(rows)
        for k in keys:
            if _get(old, k) != _get(new, k):
                issues.append((cell, k, _get(old, k), _get(new, k), 'PROBE'))
    return issues, n_cells, n_rows


def compare_real_lengths_band():
    """真实负载脚本 `fifo_reorder_band` 新旧比对（同一组集合、同一随机序）。

    `policy_sign_stability` 旧版签名要求排程参数、内部自抽种子，喂不了任意列，
    它的计数段与分桶探针旧版逐行相同（已由 `old_sign_counts_real` 覆盖）；真正
    只在 `real_lengths_policy` 里出现的、能被单独重跑的是 `fifo_reorder_band`。
    """
    import real_lengths_policy as R
    old_mod = old_real_lengths()
    issues, n_cells = [], 0
    for name in [a[0] for a in R.REAL_ARMS]:
        g = R.load_real_groups(os.path.join(R.REAL_DIR, dict(R.REAL_ARMS)[name]))
        for backlog in (4, 16):
            n_cells += 1
            a = old_mod.fifo_reorder_band(g, R.N_GROUPS_CONSTRUCT, backlog)
            b = R.fifo_reorder_band(g, R.N_GROUPS_CONSTRUCT, backlog)
            for k in a:
                if a[k] != b.get(k):
                    issues.append(('band:%s@%d' % (name, backlog), k, a[k], b.get(k), 'BAND'))
    return issues, n_cells


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

def compare(diffs, verbose=False, probe=False):
    """返回该列的比对结果（两版一致则为空）。

    `probe=True` 用于分桶探针那一侧的格子：旧版 `seed_sign_stability` 报的是
    "相对 1e-12 的正负计数"，**没有 identical 这个键**——两者在新版里的差别正好就是
    第五批修掉的分辨率刹车，不是抽工具引入的回归。所以那一侧只比 `summarize`
    里真实存在的键（见 `compare_probe_cells`），这里不比计数——**但正负计数仍要
    比**，否则"探针这一侧零分歧"会变成一句没内容的话。
    """
    issues = []
    if diffs:
        old_b = old_sign_counts_bucketing(diffs)
        new = new_sign_counts(diffs)
        for key in ('n_positive', 'n_negative'):
            if old_b[key] != new[key]:
                issues.append(('bucketing.%s' % key, old_b[key], new[key], 'COUNT'))
    if probe:
        return issues
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
        if _all_integral(diffs):
            old_l = _old_stats_ledger(diffs)
            new_l_ties = paired_diff.sign_counts(diffs)['n_identical']
            if old_l['n_ties'] != new_l_ties:
                issues.append(('ledger.n_ties', old_l['n_ties'], new_l_ties, 'COUNT'))
            new_pos_frac = paired_diff.sign_counts(diffs)['n_positive'] / float(len(diffs))
            if old_l['same_sign_arm_slower'] != new_pos_frac:
                issues.append(('ledger.same_sign_arm_slower',
                               old_l['same_sign_arm_slower'], new_pos_frac, 'RATIO'))
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

    # ---- 第九批：分桶探针与真实负载的真实输入列 + 非计数输出
    # 只算一次，后面三处（输入列、每格输出、平局审计）复用同一批行。
    probe_cells = _PROBE.cells()
    probe_cols = _PROBE.columns()
    probe_issues = []
    for name, diffs in sorted(probe_cols.items()):
        for it in compare(diffs, args.verbose, probe=True):
            probe_issues.append((name,) + it)
        for it in compare_ssf_sd(diffs):
            probe_issues.append((name,) + it)
    cell_issues, n_cells, n_rows = compare_probe_cells(probe_cells)
    band_issues, n_band = compare_real_lengths_band()
    print()
    print('分桶探针真实输入列（%d 格／%d 个种子对）：%d 处分歧（正负计数 + ssf/sd_pp）'
          % (len(probe_cols), n_rows, len(probe_issues)))
    for row in probe_issues[:10]:
        print('   ', row)
    print('分桶探针每格输出（%d 格，含 identical／同号占比／sd／极值）：%d 处分歧'
          % (n_cells, len(cell_issues)))
    for row in cell_issues[:10]:
        print('   ', row)
    # `identical` 两版都有，只是**出处不同**：旧版由 `summarize` 自己数
    # （`abs(d) < 1e-12`），新版调 `paired_diff.sign_counts` 的容差档；旧版
    # `seed_sign_stability` 内部**没有**这个键（那是它的返回字段，不是 summarize 的）。
    # 所以这里按各自判据分别数，比的仍然是同一个量：落在容差内的种子数。
    new_ident = sum(new_summarize(rows)['identical'] for _, rows, _ in probe_cells)
    old_ident = sum(1 for _, rows, _ in probe_cells for r in rows if abs(r['abs']) < 1e-12)
    print('identical 这一侧：旧版 %d、新版 %d（两版都有此档，必须相等）'
          % (old_ident, new_ident))
    print('真实负载 fifo_reorder_band（%d 格）：%d 处分歧' % (n_band, len(band_issues)))
    for row in band_issues[:10]:
        print('   ', row)

    # 平局审计扩到分桶探针：sd_pp 是在 pp 刻度上做第 3 位小数舍入的
    probe_audit = rounding_audit(probe_cols)
    print('分桶探针里 sd_pp 落在舍入平局上的个数：%d' % len(probe_audit['tie_inputs_found']))

    ok = (not real_issues) and (not span_issues) and (not ssf_issues) \
        and (not probe_issues) and (not cell_issues) and (not band_issues) \
        and old_ident == new_ident
    print()
    print('真实归档列零分歧：%s' % ('是' if not real_issues else '否'))
    print('极差／翻号零分歧：%s' % ('是' if not span_issues else '否'))
    print('same_sign_fraction／sd_pp 零分歧：%s' % ('是' if not ssf_issues else '否'))
    print('分桶探针零分歧：%s' % ('是' if not (probe_issues or cell_issues or band_issues) else '否'))
    # 两版都有 identical 这一档（出处不同：旧版由 summarize 自己数、新版取容差档），
    # 所以它必须**相等**并进入 PASS/FAIL——上一版只打印不判定。
    print('分桶探针 identical 一致：%s（旧版 %d／新版 %d）'
          % ('是' if old_ident == new_ident else '否', old_ident, new_ident))
    print('盲区命中：%d 个（0 = 既有归档无一落进 |v-ref| < TOL）' % blind_total)
    # 覆盖自查：既有「真实列零分歧」只对**计数**是同一句话——整数步号 40 列
    # 走的是精确等价路径，旧版 `d == 0` 与新版天然同答；真正检验容差的只有
    # 那些贴着 1 的浮点比率列。这里把覆盖面写出来，避免再次把"零分歧"读大。
    n_int = sum(1 for d in real.values() if all(float(x) == int(x) for x in d))
    n_ratio = len(real) - n_int
    n_flip = sum(1 for n, d in real.items()
                 if not n.startswith('steps:') and paired_diff.sign_flip_around(d, 1.0))
    print('覆盖：C 类账整数步号列 %d（精确等价路径，不检验容差）／浮点比率列 %d，'
          '其中相对 1 翻号 %d 列' % (n_int, n_ratio, n_flip))
    # 分桶探针那一侧同样要报覆盖面：2 级以下积压的配对差经常整列落在容差内，
    # 那些格只检验"两版是否都把它读成 identical"，检验不了正负分类。
    _n_tiny = sum(1 for d in probe_cols.values()
                  if all(abs(x) <= paired_diff.TOL for x in d) and d)
    _n_both = sum(1 for d in probe_cols.values()
                  if any(x > paired_diff.TOL for x in d)
                  and any(x < -paired_diff.TOL for x in d))
    print('覆盖：分桶探针 %d 格里 %d 格整列落在容差内（只检验 identical 一侧）、'
          '%d 格同时含正负（检验正负分类）、其余 %d 格单侧'
          % (len(probe_cols), _n_tiny, _n_both, len(probe_cols) - _n_tiny - _n_both))
    n_identical_total = sum(paired_diff.sign_counts(d)['n_identical']
                            for d in list(real.values()) + list(probe_cols.values()))
    print('identical 种子合计（C 类账 %d 列 + 分桶探针 %d 列）：%d'
          % (len(real), len(probe_cols), n_identical_total))
    print('结论：%s' % ('PASS' if ok else 'FAIL'))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
