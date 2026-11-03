#!/usr/bin/env python3
"""`wall_ratio` 的阈值拆解：L4 墙钟读数也建在 0.9 上（2026-11-03 M1 第五批补遗之二）。

本批前一步做了 `steps_ratio` 的阈值带宽，结论是"方向由形状定、量级由阈值定"。
但 `wall_ratio` 依赖**同一个** 0.9——它把两臂各自"到 0.9 的步"之前的累计步成本
相加再相比，换阈值两臂各自截断的步号都会挪。所以"L3 稳不稳"与"L4 稳不稳"是
两个问题，这一轮补后者。

**做法**：不 import `ledger`（它只保留了 `prefix_cost`，没有逐种子 `step_costs`），
直接从两份源归档的 `per_seed[].{step_costs, acc_curve}` 重算：

    wall_ratio(t) = Σ_seeds Σ_{k≤steps_to_t(arm)}  cost_arm[k]
                  / Σ_seeds Σ_{k≤steps_to_t(base)} cost_base[k]

口径与 `ledger.arm_row` 一致——在 t=0.9 上重算必须逐位复现归档里的 `wall_ratio`，
这是本脚本的自校验（复现不了就说明我把口径读错了，不是归档错了）。

另外两个读数：
  * `gap_ratio(t) = wall_ratio(t) / cost_ratio`：`cost_ratio` 与阈值无关，所以它
    的摆幅**全部**来自 `wall_ratio`；§13.43 的 gap 是靠这个比值算的。
  * `level_matched_wall(t)`：同一水平上的"窗口内成本"差——把两臂都截到**基线**的
    步号再比（而不是各按自己的步号），看有多少漂移来自"两臂到点步号不同"。

用法：
    python3 wall_threshold_band.py            # 重算并写 wall-threshold-band.json
    python3 wall_threshold_band.py --verify   # 与归档逐字段比对
"""

import json
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / 'tools' / 'paired-diff'))
import paired_diff  # noqa: E402  配对差判读的共享判据（翻号／极差）

P3_PATH = HERE.parent.parent / 'experiments' / 'rollout-p3-dapo-group-filter' / 'summary.json'
P4_PATH = HERE.parent.parent / 'experiments' / 'rollout-p4-c-mechanisms' / 'summary.json'
OUT = HERE / 'wall-threshold-band.json'

# 与 ledger.py 同一组阈值；改阈值必须两边一起改，否则两批读数不同口径。
THRESHOLDS = (0.85, 0.87, 0.9, 0.93, 0.95)
REFERENCE = 0.9


def first_cross(curve, thr):
    for i, v in enumerate(curve):
        if v >= thr:
            return i
    return None


def load(path):
    """直接读源归档的 per_seed，保留 step_costs（ledger.load_* 没有保留）。"""
    d = json.loads(path.read_text())
    container = d.get('arms') or d.get('modes')
    out = {}
    for name, v in container.items():
        out[name] = v['per_seed']
    return out


def wall_ratio_at(arm_seeds, base_seeds, thr):
    """某一阈值下的整批 wall_ratio，以及有多少对种子未达阈值被跳过。"""
    num = den = 0.0
    n_pairs = 0
    n_miss = 0
    for a, b in zip(arm_seeds, base_seeds):
        ia, ib = first_cross(a['acc_curve'], thr), first_cross(b['acc_curve'], thr)
        if ia is None or ib is None:
            n_miss += 1
            continue
        num += sum(a['step_costs'][: ia + 1])
        den += sum(b['step_costs'][: ib + 1])
        n_pairs += 1
    return {
        'wall_ratio': (num / den) if den else None,
        'n_pairs': n_pairs,
        'n_miss': n_miss,
    }


def level_matched_wall(arm_seeds, base_seeds, thr):
    """两臂都截到**各自**到点步号之外，再截到同一水平比较。

    做法：取基线在该阈值的步号 s_b，臂按同一 **步号** 截断——两者都在"第 s_b 步
    之前"的窗口上，因此比较的是同一段时间内的相对成本，而不是两条不同长度窗口。
    这是"同水平"的近似（同一阈值下基线到点时刻相同），与 `level_matched_steps`
    的逐种子步号差是互补的两个读数。
    """
    diffs = []
    for a, b in zip(arm_seeds, base_seeds):
        ib = first_cross(b['acc_curve'], thr)
        if ib is None:
            continue
        diffs.append(sum(a['step_costs'][: ib + 1]) / sum(b['step_costs'][: ib + 1])
                     if sum(b['step_costs'][: ib + 1]) else None)
    vals = [d for d in diffs if d is not None]
    if not vals:
        return {'median_ratio': None, 'n_pairs': 0}
    return {
        'median_ratio': sorted(vals)[len(vals) // 2],
        'n_pairs': len(vals),
    }


def per_seed_wall_ratio(arm_seeds, base_seeds, thr):
    """逐种子的 `wall_ratio`（臂窗口成本 / 基线窗口成本）及其符号分布。

    总量比会说谎的方向在这一维最清楚：`replay` 的总量比在 1.0 上下穿（0.85 上
    1.0085、0.95 上 0.9913），看着像"方向未定"；逐种子看却是 18–20/30 为 >1，
    方向稳定，只是被个别种子的窗口长度差异放大。**总量比与逐种子比必须一起报**
    （§13.47.1 的判读纪律）。`n_identical` 大（如 `cap` 多数种子恰为 1.0000）
    说明该格没有分辨率。
    """
    vals = []
    for a, b in zip(arm_seeds, base_seeds):
        ia, ib = first_cross(a['acc_curve'], thr), first_cross(b['acc_curve'], thr)
        if ia is None or ib is None:
            continue
        den = sum(b['step_costs'][: ib + 1])
        if den:
            vals.append(sum(a['step_costs'][: ia + 1]) / den)
    counts = paired_diff.sign_counts([v - 1.0 for v in vals])
    n = len(vals)
    return {
        'median': sorted(vals)[n // 2] if n else None,
        'n_slower': counts['n_positive'],    # >1 ⇒ 臂更慢
        'n_faster': counts['n_negative'],
        'n_identical': counts['n_identical'],
        'n_pairs': n,
        'faster_frac': (counts['n_negative'] / n) if n else None,
    }


def arm_band(arm_seeds, base_seeds, cost_ratio, thresholds=THRESHOLDS):
    rows = {('%.2f' % t): wall_ratio_at(arm_seeds, base_seeds, t) for t in thresholds}
    vals = {t: rows['%.2f' % t]['wall_ratio'] for t in thresholds
            if rows['%.2f' % t]['wall_ratio'] is not None}
    ratio_vals = [vals[t] for t in thresholds if t in vals]
    # gap 口径：gap 由 wall/cost 决定，cost 与阈值无关 ⇒ gap 的摆幅全部来自 wall
    gap_vals = [v / cost_ratio for v in ratio_vals] if cost_ratio else []
    matched = {('%.2f' % t): level_matched_wall(arm_seeds, base_seeds, t)
               for t in thresholds}
    per_seed = {('%.2f' % t): per_seed_wall_ratio(arm_seeds, base_seeds, t)
                for t in thresholds}
    # 逐种子口径下的方向：各阈值都以"多数方向"报，再看这个多数方向有没有翻号
    dirs = {t: per_seed['%.2f' % t]['faster_frac'] for t in thresholds}
    dirs_ok = [v for v in dirs.values() if v is not None]
    return {
        'by_threshold': rows,
        'per_seed_by_threshold': per_seed,
        'wall_span': paired_diff.span(ratio_vals),
        'wall_min': min(ratio_vals) if ratio_vals else None,
        'wall_max': max(ratio_vals) if ratio_vals else None,
        # 翻号参考点是 1.0：wall_ratio<1 表示墙钟变快，>1 变慢
        'wall_sign_flip': bool(paired_diff.sign_flip_around(ratio_vals, 1.0))
        if ratio_vals else None,
        # 逐种子多数方向是否在所有阈值上一致
        'per_seed_direction_flip': bool(paired_diff.sign_flip_around(dirs_ok, 0.5))
        if dirs_ok else None,
        'per_seed_faster_frac_min': min(dirs_ok) if dirs_ok else None,
        'per_seed_faster_frac_max': max(dirs_ok) if dirs_ok else None,
        'gap_span': paired_diff.span(gap_vals),
        'level_matched_wall': matched,
    }


def build():
    p3 = load(P3_PATH)
    p4 = load(P4_PATH)
    out = {'meta': {'p3': str(P3_PATH.relative_to(HERE.parent.parent)),
                    'p4': str(P4_PATH.relative_to(HERE.parent.parent)),
                    'thresholds': list(THRESHOLDS),
                    'reference': REFERENCE}, 'p4': {}, 'p3': {}}

    def cost_ratio_of(arm_seeds, base_seeds):
        return (sum(s['total_cost'] for s in arm_seeds)
                / sum(s['total_cost'] for s in base_seeds))

    for arm, seeds in p4.items():
        if arm == 'base':
            continue
        out['p4'][arm] = arm_band(seeds, p4['base'], cost_ratio_of(seeds, p4['base']))
    for a, b in (('C', 'B'), ('B', 'R_nofilter'), ('A', 'R_nofilter'),
                 ('C', 'R_nofilter')):
        out['p3']['%s_vs_%s' % (a, b)] = arm_band(
            p3[a], p3[b], cost_ratio_of(p3[a], p3[b]))

    # 自校验：在参考阈值上重算必须复现 ledger 归档里的 wall_ratio
    ledger_archive = json.loads((HERE / 'ledger.json').read_text())
    checks = []
    for arm, blk in out['p4'].items():
        recomputed = blk['by_threshold']['%.2f' % REFERENCE]['wall_ratio']
        archived = ledger_archive['p4_arms'][arm]['wall_ratio']
        checks.append({'arm': 'p4:' + arm, 'recomputed': recomputed,
                       'archived': archived,
                       'match': abs(recomputed - archived) < 1e-12})
    for key, blk in out['p3'].items():
        recomputed = blk['by_threshold']['%.2f' % REFERENCE]['wall_ratio']
        archived = ledger_archive['p3_pairs'][key]['wall_ratio']
        checks.append({'arm': 'p3:' + key, 'recomputed': recomputed,
                       'archived': archived,
                       'match': abs(recomputed - archived) < 1e-12})
    out['reference_reproduction'] = checks
    out['all_reproduced'] = all(c['match'] for c in checks)
    return out


def main():
    data = build()
    if '--verify' in sys.argv:
        old = json.loads(OUT.read_text())
        diffs = []
        flat = lambda d, p='': {p + k: v for k, v in d.items()
                                if not isinstance(v, dict)} if isinstance(d, dict) else {}
        def walk(a, b, path=''):
            if isinstance(a, dict) and isinstance(b, dict):
                for k in sorted(set(a) | set(b)):
                    if k not in a:
                        diffs.append('%s: 新增' % (path + '/' + k))
                    elif k not in b:
                        diffs.append('%s: 缺失' % (path + '/' + k))
                    else:
                        walk(a[k], b[k], path + '/' + k)
            elif a != b:
                diffs.append('%s: %r ≠ %r' % (path, a, b))
        walk(data, old)
        if diffs:
            print('VERIFY FAILED:')
            for d in diffs[:20]:
                print('  -', d)
            sys.exit(1)
        print('VERIFY OK: 归档与重算逐字段一致 (%d 对参考阈值复现)' %
              len(data['reference_reproduction']))
        if not data['all_reproduced']:
            print('WARNING: 参考阈值未复现 ledger.json 的 wall_ratio')
            sys.exit(1)
        return
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1) + '\n')
    print('all_reproduced:', data['all_reproduced'])
    for arm, blk in out_iter(data):
        print('%-24s wall %.4f–%.4f span %.4f flip %-5s  逐种子 faster %.2f–%.2f flip %s' % (
            arm, blk['wall_min'], blk['wall_max'], blk['wall_span'],
            blk['wall_sign_flip'], blk['per_seed_faster_frac_min'],
            blk['per_seed_faster_frac_max'], blk['per_seed_direction_flip']))


def out_iter(data):
    for grp in ('p4', 'p3'):
        for k, v in data[grp].items():
            yield '%s:%s' % (grp, k), v


if __name__ == '__main__':
    main()
