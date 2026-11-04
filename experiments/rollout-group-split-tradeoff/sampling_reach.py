#!/usr/bin/env python3
"""`no_reading` 是"这个负载问不出来"，还是"这一次抽样问不出来"？

**为什么要单独问这一句（2026-11-04 第十四批）。** 第十一批把带宽为 0 分成三档，
其中 `no_reading`（每个种子的 FIFO `R` 都恒为 0）只写了"这一格问不出来"。那句话
把**两件不同的事**压成了一句：

  - 负载上**每一条路上**都排不出先后（所有可抽的组里组最大都相同）⇒ 换抽样也没用；
  - 负载上还有能排的组，只是**这次的 n_groups** 太小、每次都抽不到它们 ⇒ 加大
    `n_groups` 就问得出来。

两者的判词不同：前者是这个排程器问不了这个问题（要换负载档或改造模型），后者是
实验设计的问题。本脚本用三种口径把它们分开，**全部在真实负载上做，不新建模型**：

  1. **`n_groups` 扫描**：把 16 换成 4／8／32／64，看 `band_floor` 怎么变。
  2. **抽样内自由组计数**：逐种子数"抽到的组里有几组的组最大 < 上限"，把"问不出来"
     归因到"抽到的组全贴顶"。`free_share_min_for_reading` 给出使"至少一半抽样抽到
     非贴顶组"的最小 `n_groups`（超几何，解析值，不依赖随机数）。
  3. **全拟合对照**：把贴顶的组换成一様从**非贴顶组池**抽出的组（其余不动），重算
     带宽。若带宽显著离开 0，说明**上限本身**是把读数抹掉的原因，而不是这些负载
     结构上不可分辨。

**这三条各自能证明什么、不能证明什么**（都写进输出，避免被读大）：

  - 口径 1 证明 `no_reading` **依赖 n_groups**——它不是一个与抽样无关的负载属性；
  - 口径 2 说明依赖的**机制**（贴顶组占比 × n_groups）；
  - 口径 3 是**反事实**，不是任何一种真实配置下的读数：真实推理确实有 256 上限，
    贴顶组不会被换掉。它只用来回答"如果上限不存在会怎样"。

    python3 sampling_reach.py            # 打印三条口径的表
    python3 sampling_reach.py --json     # 同上，另写 sampling-reach.json

运行时间约 4 分钟（n=16／32／64 三档各要重算 30 种子 × 12 次换序）。
"""

import json
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import real_lengths_policy as RL  # noqa: E402  （它已把 tools/paired-diff 放进 sys.path）
import paired_diff  # noqa: E402  过半数判据的唯一出处
from group_split_tradeoff import POLICY_FIFO, _makespan_by_policy  # noqa: E402

OUT_PATH = os.path.join(HERE, 'sampling-reach.json')
SEED_COUNT = len(RL.SEEDS)
TRIALS = 12
N_SWEEP = (4, 8, 16, 32, 64)
BACKLOGS = (2, 4)          # `no_reading` 只出现在低积压，深积压已可读，不必全扫
NULL_REPS = 200            # 全拟合对照的重复次数


def _band_of_pick(pick, backlog, s):
    """同一条抽样在 `backlog` 积压下的 FIFO 带宽（pp）——与 `RL.fifo_reorder_band` 同序列。"""
    vals = []
    for t in range(TRIALS):
        sh = list(pick)
        random.Random(s * 7919 + t).shuffle(sh)
        vals.append(_makespan_by_policy(sh, backlog, POLICY_FIFO)['tax'])
    return (max(vals) - min(vals)) * 100.0, (max(vals) == 0.0)


def sampling_floor(groups, cap, n_groups, backlog, rng_seed=0):
    """按 `n_groups` 复算三档，并逐种子数"抽到的非贴顶组"。口径与归档一致。"""
    rng = random.Random(rng_seed)
    bands, n_free, n_r_zero = [], [], 0
    for s in range(SEED_COUNT):
        pick = rng.sample(groups, n_groups)
        n_free.append(sum(1 for g in pick if max(g) < cap))
        band, r_identically_zero = _band_of_pick(pick, backlog, s)
        bands.append(band)
        n_r_zero += r_identically_zero
    n_band_zero = sum(1 for b in bands if b == 0.0)
    if n_r_zero == SEED_COUNT:
        verdict = 'no_reading'
    elif paired_diff.band_floor_rule('degenerate')(n_band_zero, SEED_COUNT):
        verdict = 'degenerate'
    else:
        verdict = 'usable'
    return {
        'band_floor': verdict,
        'n_band_zero': n_band_zero,
        'n_r_identically_zero': n_r_zero,
        'band_median_pp': round(statistics.median(bands), 3),
        'free_per_pick_min': min(n_free),
        'free_per_pick_median': statistics.median(n_free),
        'free_per_pick_max': max(n_free),
    }


def free_share_min(n_total, n_free, share):
    """超几何：抽 `n` 组、其中非贴顶组数 ≥1 的概率首次达到 `share` 的最小 `n`。

    解析值，不依赖随机数。`n_free=0`（负载里没有一组的组最大 < 上限）返回 None——
    这时**任何** `n_groups` 都问不出来，属"负载上排不出先后"那一类。
    """
    if n_free <= 0:
        return None
    p_none = 1.0
    for n in range(1, n_total + 1):
        p_none *= (n_total - n_free - (n - 1)) / float(n_total - (n - 1))
        if (1.0 - p_none) >= share:
            return n
    return n_total


def all_fits_null(groups, cap, free_pool, backlog, n_groups=RL.N_GROUPS_CONSTRUCT,
                  reps=NULL_REPS):
    """全拟合反事实：抽 16 组，把贴顶组替换成从非贴顶池抽的一组，只量 FIFO 带宽。

    这不是任何真实配置下的读数——真实推理里 256 上限就是 256。它只回答"如果上限
    不存在（同样的组内结构、同样的组数），这一格还会不会问不出来"。
    """
    rng = random.Random(1)
    bands = []
    for rep in range(reps):
        pick = rng.sample(groups, n_groups)
        fixed = []
        for i, g in enumerate(pick):
            if max(g) < cap:
                fixed.append(g)
            else:
                fixed.append(free_pool[random.Random(9000 + rep * 131 + i).randrange(
                    len(free_pool))])
        bands.append(_band_of_pick(fixed, backlog, rep)[0])
    return {
        'band_median_pp': round(statistics.median(bands), 3),
        'frac_zero_band': round(sum(1 for b in bands if b == 0.0) / float(reps), 3),
        'n_reps': reps,
    }


def diagnose():
    arms = {}
    for label, filename in RL.REAL_ARMS:
        groups = RL.load_real_groups(os.path.join(RL.REAL_DIR, filename))
        cap = RL.arm_cap(label)
        free_pool = [g for g in groups if max(g) < cap]
        cells = {}
        for backlog in BACKLOGS:
            base = sampling_floor(groups, cap, RL.N_GROUPS_CONSTRUCT, backlog)
            sweep = {'n_groups=%d' % n: sampling_floor(groups, cap, n, backlog)
                     for n in N_SWEEP}
            cells['backlog%d' % backlog] = {
                'at_n16': base,
                'sweep': sweep,
                'truncated_group_fraction': round(1.0 - len(free_pool) / float(len(groups)), 4),
                'free_share_min_for_reading': {
                    'share0.50': free_share_min(len(groups), len(free_pool), 0.50),
                    'share0.90': free_share_min(len(groups), len(free_pool), 0.90),
                },
                'all_fits_null': all_fits_null(groups, cap, free_pool, backlog),
            }
        arms[label] = {'cap': cap, 'free_pool_size': len(free_pool), 'cells': cells}
    return {
        'n_seeds': SEED_COUNT,
        'trials': TRIALS,
        'n_groups_archive': RL.N_GROUPS_CONSTRUCT,
        'selfcheck': 'n_groups=16 的扫描项必须与 at_n16 逐字段相同（含 free_per_pick_*）',
        'counting': ('at_n16 与归档 `band_floor` 同口径、同 rng 种子；`sweep` 的 n=16 一项'
                     '必须与 at_n16 逐字段相同（脚本自检）。`all_fits_null` 是反事实对照，'
                     '不是真实配置下的读数。'),
        'arms': arms,
    }


def main(argv):
    out = diagnose()
    bad = 0
    print('%-16s %-4s %-6s %-13s %-7s %-8s %s'
          % ('arm', 'bl', 'n', 'band_floor', 'R≡0', 'band0', 'free/pick(median)'))
    for label, arm in out['arms'].items():
        for backlog, c in arm['cells'].items():
            rows = [('16', c['at_n16'])] + [(k.split('=')[1], v)
                                            for k, v in c['sweep'].items()]
            n16_agrees = True
            for n, r in rows:
                print('%-16s %-4s %-6s %-13s %-7d %-8d %s'
                      % (label, backlog.replace('backlog', ''), n, r['band_floor'],
                         r['n_r_identically_zero'], r['n_band_zero'],
                         r['free_per_pick_median']))
                # 自检：n=16 的扫描项必须与 at_n16 逐字段相同（含 median 的舍入）
                if n == '16':
                    # 全字段比较（此前排除掉了一个本该相等的键，判据写松了 = 没检查）
                    agree = all(r[k] == c['at_n16'][k] for k in r)
                    n16_agrees = n16_agrees and agree
                    if not agree:
                        bad += 1
                        print('   MISMATCH: n=16 扫描项与 at_n16 不一致')
            print('   %-16s 截断组占比 %.3f ｜ 非贴顶组池 n=%d ｜ 全拟合带宽中位 %.3fpp（零带宽占比 %.2f）'
                  % ('', c['truncated_group_fraction'], arm['free_pool_size'],
                     c['all_fits_null']['band_median_pp'], c['all_fits_null']['frac_zero_band']))
            print('   %-16s 使"至少一半抽样抽到非贴顶组"的最小 n_groups：%s；90%%：%s'
                  % ('', c['free_share_min_for_reading']['share0.50'],
                     c['free_share_min_for_reading']['share0.90']))
            print('   %-16s n=16 扫描项与 at_n16 逐字段一致：%s' % ('', n16_agrees))
    text = json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True)
    if '--json' in argv:
        with open(OUT_PATH, 'w') as f:
            f.write(text + '\n')
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
