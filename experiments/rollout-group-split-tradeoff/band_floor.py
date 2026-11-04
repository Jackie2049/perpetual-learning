#!/usr/bin/env python3
"""带宽退化诊断：`fifo_band == 0` 到底是不是"基线很稳"。

**为什么单独一个脚本。** `exceeds_band(|效应|, 带宽)` 的判词是"`|效应| > 带宽`
才可引用"。带宽为 0 时这个式子对所有非零效应一律通过，于是 `qwen15b_cap256`
积压 2 那格会被读成"效应超出带宽、方向稳定"。实际是**这一格根本没有读数**：
16 组里组最大全部等于 `max_response`，`R` 恒等 0（2026-11-03 第四十二批已记
过一句"噪声为零的带宽不构成门槛"，但没有把它变成一个能查的判据）。

**三档**（只看带宽取到 0 的原因，不看效应方向）：

  - `no_reading`：抽样里**每个种子**的 FIFO `R` 都恒为 0。组最大全部贴顶时，
    组臂与轨迹臂都只受最慢那一条约束，两臂 makespan 相等 ⇒ `R = 0` 是**结构
    必然**，打乱顺序也不会变。这不是"基线稳"，是这一格问不出来。
    **（2026-11-04 第十四批订正）**：上面这句原写"这一格问不出来"，没说清是"这个
    负载"还是"这一次抽样"。实测以后者为主——同一负载换 `n_groups`（4／8 反而读得
    出来）或换积压档都会变档，见 [sampling_reach.py](sampling_reach.py)。
  - `degenerate`：带宽 > 0，但**过半种子**的带宽是 0（同一批组换顺序读数不动）。
    有效带宽由剩下的少数种子撑起，`|效应| > 带宽` 只能当"部分种子看得出"。
  - `usable`：过半种子的带宽 > 0，带宽是个真门槛。

**为什么截断档会这样（机制，逐档量过）**：cap256 的组最大 **86.0%（1.5B）／
94.2%（0.5B）** 恰好等于 256；cap1024 只有 0.2%／4.8%。组最大贴顶 ⇒ 这 16 组
的 `max(g)` 完全相同 ⇒ 任何顺序都无事可排。

    python3 band_floor.py            # 打印四档诊断表，不写文件
    python3 band_floor.py --json     # 同上，另把结果写 band-floor.json
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

SEEDS = RL.SEEDS
N_GROUPS = RL.N_GROUPS_CONSTRUCT
BACKLOGS = (2, 4, 8, 16)
TRIALS = 12
OUT_PATH = os.path.join(HERE, 'band-floor.json')


def band_per_seed(groups, backlog, n_groups=None, seeds=SEEDS, trials=TRIALS, rng_seed=0):
    """**逐种子**的带宽（pp，未汇总）与"整格 `R` 恒为 0"的种子数。

    抽样与汇总口径（中位／均值／p90／最大）不在这里算：调用方直接用
    `RL.fifo_band_floor` 拿汇总值——两处各算一份会在某一天漂成两个数。
    """
    n_groups = N_GROUPS if n_groups is None else n_groups
    rng = random.Random(rng_seed)
    bands, zero_reading = [], 0
    for s in range(len(seeds)):
        pick = rng.sample(groups, n_groups)
        vals = []
        for t in range(trials):
            sh = list(pick)
            # 与 `RL.fifo_band_floor` 同一条 shuffle 种子序列（本文件曾写死
            # `s * 7919 + t`，与归档用的 `rng_seed * 100003 + s * 7919 + t`
            # 在 rng_seed=0 时恰好相同，非零时就会错位——已改为同一式子）。
            random.Random(rng_seed * 100003 + s * 7919 + t).shuffle(sh)
            vals.append(_makespan_by_policy(sh, backlog, POLICY_FIFO)['tax'])
        bands.append(max(vals) - min(vals))
        if max(vals) == 0.0:
            zero_reading += 1
    return bands, zero_reading


def band_floor_verdict(bands, zero_reading, n_total):
    """把一格带宽分成三档。阈值 `0.5` 是判读门槛，不是显著性检验。

    **只作自检用，归档里的 `band_floor` 不取这里的返回值**——真正写进
    `real-lengths-summary.json` 的是 `RL.fifo_band_floor`。两者口径若哪天分家，
    本脚本的 `--json` 会在 `crosscheck` 里把差异报出来（见 `diagnose`）。
    """
    if n_total and zero_reading == n_total:
        return 'no_reading'
    n_zero = sum(1 for b in bands if b == 0.0)
    # 过半数这一行取自共享判据（2026-11-04 第十六批）；此前本文件与
    # `RL.fifo_band_floor` 各写一遍，`crosscheck` 比的其实是两份抄本。
    if n_total and paired_diff.band_floor_rule('degenerate')(n_zero, n_total):
        return 'degenerate'
    return 'usable'


def truncation_pinch(groups, cap):
    """组最大恰好贴到上限的组占比——`no_reading` 的直接成因。"""
    mx = [max(g) for g in groups]
    return round(sum(1 for m in mx if m == cap) / float(len(mx)), 4)


def diagnose():
    arms = {}
    for label, filename in RL.REAL_ARMS:
        groups = RL.load_real_groups(os.path.join(RL.REAL_DIR, filename))
        cap = RL.arm_cap(label)
        cells = {}
        for backlog in BACKLOGS:
            bands, zero_reading = band_per_seed(groups, backlog)
            # 归档用的就是这个函数；本脚本不再另算一份汇总，只核"逐种子分解
            # 能不能复现它"。
            official = RL.fifo_band_floor(groups, N_GROUPS, backlog)
            verdict = band_floor_verdict(bands, zero_reading, len(bands))
            cells['backlog%d' % backlog] = {
                'band_median_pp': (RL.fifo_reorder_band(groups, N_GROUPS, backlog)
                                   ['median_pp']),
                'n_band_zero': sum(1 for b in bands if b == 0.0),
                'n_r_identically_zero': zero_reading,
                'n_seeds': len(bands),
                'official_n_band_zero': official['n_band_zero'],
                'official_n_r_identically_zero': official['n_r_identically_zero'],
                'band_floor': official['band_floor'],
                'crosscheck': ('ok' if (official['band_floor'] == verdict
                                        and official['n_band_zero'] ==
                                        sum(1 for b in bands if b == 0.0)
                                        and official['n_r_identically_zero'] == zero_reading)
                               else 'mismatch'),
            }
        arms[label] = {
            'cap': cap,
            'truncation_pinch': truncation_pinch(groups, cap),
            'cells': cells,
        }
    return {
        'n_groups': N_GROUPS,
        'n_seeds': len(SEEDS),
        'trials': TRIALS,
        'band_floor_threshold': '过半种子带宽为 0 ⇒ degenerate',
        'arms': arms,
    }


def main(argv):
    out = diagnose()
    text = json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True)
    if '--json' in argv:
        with open(OUT_PATH, 'w') as f:
            f.write(text + '\n')
    print('%-16s %-7s %-4s %-9s %-8s %-9s %-13s %s'
          % ('arm', 'pinch', 'bl', 'median_pp', 'band==0', 'R≡0种子', 'band_floor', 'crosscheck'))
    bad = 0
    for label, arm in out['arms'].items():
        for backlog, c in arm['cells'].items():
            bad += c['crosscheck'] != 'ok'
            print('%-16s %-7.3f %-4s %-9.3f %-8d %-9d %-13s %s'
                  % (label, arm['truncation_pinch'], backlog.replace('backlog', ''),
                     c['band_median_pp'], c['n_band_zero'], c['n_r_identically_zero'],
                     c['band_floor'], c['crosscheck']))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
