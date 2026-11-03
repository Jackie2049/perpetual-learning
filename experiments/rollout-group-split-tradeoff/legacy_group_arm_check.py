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
CVS = (0.2, 0.6, 1.2)
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
    #   cv 0.2→1.5 时 20.6%→56.6%（4 批档，按正文引用值登记）
    out['paper_batch1'] = {
        'cv0.6_backlog4_claimed': 0.437,
        'cv0.6_backlog16_claimed': 0.547,
        'cv0.2_backlog4_claimed': 0.206,
        'cv1.2_backlog4_claimed': 0.566,
    }
    return out


def main(argv):
    out = build()
    text = json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True)
    if '--verify' in argv:
        with open(OUT_PATH) as f:
            ok = json.load(f) == json.loads(text)
        print('VERIFY OK: 逐字段相同' if ok else 'VERIFY FAIL: 与归档不一致')
        return 0 if ok else 1
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
