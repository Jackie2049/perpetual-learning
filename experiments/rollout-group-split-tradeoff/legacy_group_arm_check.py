#!/usr/bin/env python3
"""第四十批（2026-11-03 M1）：`grouping_tax` 的组臂归还粒度曾经退化，L1–L2 的 `R` 偏低

支撑 notebook/rollout-performance-research.md §13.30 第四十批。

**发现**：`grouping_tax` 把整组当成"n_per 个任务"送进 `_makespan(group_level=True)`，
于是每个任务各自成组（`group_left` 每个 key 都是 1），一完成就把 n_per 个槽位全还回去
——组臂的归还粒度退化成逐条，`R` 被系统性压低。§13.30 第二／三／四批用的是
`_makespan_by_policy`（组号 `i // n_per`，口径正确），所以**只有第一批的
0–54.7%（cv 0.2–1.5 的扫描）取自这条退化路径**。

**做法**：用正确组号的组臂重算同一批构造负载（16 组 × 8 条，30 种子，cv 0.2/0.6/1.2，
积压 1/4/16），与 §13.30 第一批的两条口径并排。
并给出**守护用例**：一组恒权负载上旧路径量出负税（不可能），新路径为 0。

用法：
    python3 legacy_group_arm_check.py            # 重算并写归档
    python3 legacy_group_arm_check.py --verify   # 与归档逐字段比对
"""
import json
import os
import random
import statistics
import sys

from group_split_tradeoff import (POLICY_FIFO, _group_makespan, _makespan,
                                  _makespan_by_policy, sample_group_lens)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, 'legacy-group-arm-check.json')

SEEDS = list(range(30))
N_GROUPS, N_PER = 16, 8
# cv=1.5 是 §13.30 首批扫描的端点（`group_split_tradeoff.py` 的 L2 扫描也是 0.2/0.6/1.5），
# 2026-11-04 补齐后 sweep 成为全连续扫描；此前只到 1.2，端点 44.9% 只能在台账里手工重算。
CVS = (0.2, 0.6, 1.2, 1.5)
BACKLOGS = (1, 4, 16)


def legacy_group_arm(group_lens, backlog_groups):
    """旧路径：`_makespan([n_per]*k, dur, True)`——每个任务各自成组。"""
    n_per = len(group_lens[0])
    cap = len(group_lens) * n_per
    dur = [max(g) for b in range(backlog_groups) for g in group_lens]
    t_group = _makespan([n_per] * len(dur), dur, True, cap=cap)
    flat = [x for b in range(backlog_groups) for g in group_lens for x in g]
    t_traj = _makespan([1] * len(flat), flat, False, cap=cap)
    return (t_group - t_traj) / t_group if t_group else 0.0


def fixed_group_arm(group_lens, backlog_groups):
    """正确路径：组号按整组取。"""
    n_per = len(group_lens[0])
    cap = len(group_lens) * n_per
    dur = [max(g) for b in range(backlog_groups) for g in group_lens]
    t_group = _group_makespan(dur, n_per, cap)
    flat = [x for b in range(backlog_groups) for g in group_lens for x in g]
    t_traj = _makespan([1] * len(flat), flat, False, cap=cap)
    return (t_group - t_traj) / t_group if t_group else 0.0


def guard_case():
    """负税在"整组归还只会更慢"下不可能——旧路径却能量出来。

    **必须是不可整除的容量**：4 组 × 4 条时 cap=16 恰好等于工作量，
    整组一起放下，两种粒度同 makespan，谁也量不出负税（第一版就是这么写的，
    守护用例给了 0/0/0，等于没守）。换成 4 组 × 8 条（cap=32、2 批工作量 64），
    排队的才有得看：组间无差别时正确税必须为 0，旧路径给出 **−100%**
    （组臂 200 vs 轨迹臂 400）——这就是"归还粒度退化成逐条"的直接证据。
    """
    groups = [[100.0] * 8] * 4
    return {'backlog2': {'legacy': legacy_group_arm(groups, 2),
                         'fixed': fixed_group_arm(groups, 2),
                         'policy': _makespan_by_policy(groups, 2, POLICY_FIFO)['tax']}}


def build():
    out = {'guard': guard_case(), 'sweep': {}, 'paper_batch1': {}}
    for cv in CVS:
        for b in BACKLOGS:
            legacy, fixed, policy = [], [], []
            for s in SEEDS:
                lens = sample_group_lens(random.Random(s), n_groups=N_GROUPS,
                                         n_per=N_PER, cv=cv)
                legacy.append(legacy_group_arm(lens, b))
                fixed.append(fixed_group_arm(lens, b))
                policy.append(_makespan_by_policy(lens, b, POLICY_FIFO)['tax'])
            out['sweep']['cv%.1f_backlog%d' % (cv, b)] = {
                'legacy_mean': statistics.mean(legacy),
                'fixed_mean': statistics.mean(fixed),
                'policy_mean': statistics.mean(policy),
                'fixed_minus_policy': statistics.mean(fixed) - statistics.mean(policy),
            }
    # §13.30 第一批正文的两条口径：
    #   无积压 0.0%、4 批 43.7%、16 批 54.7%（cv 0.6 档）
    #   cv 0.2→1.5 时 20.6%→56.6%（4 批档）
    # 端点登记名原先写作 `cv1.2_...`，与正文的 1.5 不符（键名只是引用标签，
    # 读数始终是 unpack 扫描端点 1.5 的值）；2026-11-04 改名并补 1.2 档做中间点核对。
    out['paper_batch1'] = {
        'cv0.6_backlog4_claimed': 0.437,
        'cv0.6_backlog16_claimed': 0.547,
        'cv0.2_backlog4_claimed': 0.206,
        'cv1.2_backlog4_claimed': 0.543,
        'cv1.5_backlog4_claimed': 0.566,
        'cv0.2_max_backlog4_claimed': 0.183,
        'cv1.5_max_backlog4_claimed': 0.527,
    }
    # 与"换了先验、换了一批随机数"的申报隔离：`_group_makespan` 与 `_makespan_by_policy`
    # 是两条独立实现，逐种子必须给出同一个税（原先只比均值）。
    for cv in CVS:
        for b in BACKLOGS:
            diffs = []
            for s in SEEDS:
                lens = sample_group_lens(random.Random(s), n_groups=N_GROUPS,
                                         n_per=N_PER, cv=cv)
                diffs.append(abs(fixed_group_arm(lens, b)
                                 - _makespan_by_policy(lens, b, POLICY_FIFO)['tax']))
            out.setdefault('arm_crosscheck', {})['cv%.1f_backlog%d' % (cv, b)] = {
                'max_abs_diff': max(diffs),
                'all_within_1e-12': max(diffs) < 1e-12,
            }

    # 正文引用的六个订正数必须有归档字段托底。2026-11-03 第四十批把它们写进台账，
    # 但 44.9%（cv 1.5、4 批）与 56.3%（cv 1.5、16 批）当时**不在任何归档里**——
    # 脚本 cv 扫描只到 1.2，读者无从复核。这里逐值对账，容差 5e-4（正文四位小数）。
    claimed = {
        'cv0.6_backlog4': 0.340, 'cv0.6_backlog16': 0.424,
        'cv0.2_backlog4': 0.147, 'cv1.5_backlog4': 0.449,
        'cv1.5_backlog16': 0.563,
    }
    for key, want in claimed.items():
        got = out['sweep'][key]['fixed_mean']
        out.setdefault('claims_vs_sweep', {})[key] = {
            'claimed': want, 'archived': got,
            'match_within_5e-4': abs(got - want) <= 5e-4,
        }
    return out


def main(argv):
    out = build()
    # 交叉核对必须是发布的前置条件，不是打印出来给人看的附注：
    # 两条组臂实现对不上时，`sweep` 里的 fixed 列就不能当成"正确口径"。
    bad = sorted(k for k, v in out['arm_crosscheck'].items() if not v['all_within_1e-12'])
    bad += sorted(k for k, v in out['claims_vs_sweep'].items() if not v['match_within_5e-4'])
    text = json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True)
    if '--verify' in argv:
        with open(OUT_PATH) as f:
            ok = json.load(f) == json.loads(text)
        print('VERIFY OK: 逐字段相同' if ok else 'VERIFY FAIL: 与归档不一致')
        return 0 if ok else 1
    if bad:
        print('AUDIT FAIL：%s' % ', '.join(bad))
        return 1
    with open(OUT_PATH, 'w') as f:
        f.write(text + '\n')
    g = out['guard']['backlog2']
    print('守护用例（恒权 4×8×100、积压 2 批）：旧 %.4f  新 %.4f  对照组 %.4f'
          % (g['legacy'], g['fixed'], g['policy']))
    print('cv    backlog  旧 R     新 R     口径正确对照')
    for k, v in sorted(out['sweep'].items()):
        cv, b = k.split('_')
        print('%s %-7s %.4f   %.4f   %.4f'
              % (cv, b, v['legacy_mean'], v['fixed_mean'], v['policy_mean']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
