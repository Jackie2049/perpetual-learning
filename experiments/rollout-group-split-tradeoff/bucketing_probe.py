#!/usr/bin/env python3
"""分桶探针：组的排队顺序能改变 R 多少（接 TM-011）

复用 grouping_tax，只换组的入队顺序。**不是分桶模型**——它只量"给定组的集合
改顺序"的效应，不建模"哪些组被放进同一批"。

用法：python3 bucketing_probe.py
"""
import importlib.util
import json
import pathlib
import random
import statistics

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('g', HERE / 'group_split_tradeoff.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

SEEDS = 20
N_GROUPS = 16
N_PER = 8
CV = 0.07          # 真实 1.5B cap256 的组内 cv（§13.30 第四批）
BACKLOG = 4


def make_load(seed):
    rng = random.Random(seed)
    out = []
    for _ in range(N_GROUPS):
        base = rng.uniform(100, 400)
        out.append([base * (1 + rng.gauss(0, CV)) for _ in range(N_PER)])
    return out


def main():
    wins, gains = 0, []
    for s in range(SEEDS):
        groups = make_load(s)
        rng = random.Random(s + 1000)
        lof = M.grouping_tax(sorted(groups, key=lambda g: -max(g)),
                             backlog_groups=BACKLOG)['grouping_tax']
        rnd = []
        for _ in range(10):
            g = groups[:]
            rng.shuffle(g)
            rnd.append(M.grouping_tax(g, backlog_groups=BACKLOG)['grouping_tax'])
        med = statistics.median(rnd)
        if lof < med:
            wins += 1
        gains.append(med - lof)

    out = {
        'seeds': SEEDS, 'n_groups': N_GROUPS, 'n_per': N_PER,
        'cv': CV, 'backlog': BACKLOG,
        'wins': wins,
        'gain_pp_mean': round(statistics.mean(gains) * 100, 2),
        'gain_pp_median': round(statistics.median(gains) * 100, 2),
        'gain_pp_min': round(min(gains) * 100, 2),
        'gain_pp_max': round(max(gains) * 100, 2),
    }
    print(json.dumps(out, indent=1, ensure_ascii=False))
    (HERE / 'bucketing-probe.json').write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
