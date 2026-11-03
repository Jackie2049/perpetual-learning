#!/usr/bin/env python3
"""门槛校准：`sign_regime` 的 0.5／0.8 与 `band_floor` 的"过半"各自误判多少。

**为什么要做。** 第十批把四档 `sign_regime()` 写进两份归档，三处文档都注明
"门槛是判读门槛、不是显著性检验"。**这句话只交代了性质，没交代价位**：一个
门槛若在"真相是同向"的列上也有三成概率报 `mixed`，那它给的档名就不能当结论
用。本脚本用**已知真相**的合成列把误判率量出来。

**与"注入"的区别。** 注入（把 `TOL` 改成 1e-3、把判据改成恒真）检验的是
"改了会不会被测红"，要求的是测试灵敏度；这里量的是"判据本身在正常输入上出
错的概率"，与测试无关。两者不能互相代替。

**三条判词分开量**，因为它们错的方向不同：

  - `sign_regime` 的假阳：真值同号（或全 zero），却被报成 `mixed`。
    直接用这个档名当"方向混杂"的证据就会误报。
  - `sign_regime` 的假阴：真值确是 50/50（每个种子独立掷硬币），却被报成
    `stable`。此时 `stable` 意味着"这一列的符号没有信息"，不能写方向。
  - `band_floor` 的误导：真值是"FIFO 臂确定不动"（带宽恒 0），却因为抽样只
    抽到几个非零种子被判 `usable`——于是一个噪声为零的格子拿到了"能用的门槛"。

    python3 calibrate_thresholds.py            # 打印校准表
    python3 calibrate_thresholds.py --json     # 另写 threshold-calibration.json
    python3 calibrate_thresholds.py --verify   # 与归档逐字段比对
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'tools', 'paired-diff'))
import paired_diff  # noqa: E402

OUT_PATH = os.path.join(HERE, 'threshold-calibration.json')
TRIALS = 4000
SQRT12 = 12 ** 0.5


def _draw_signed(rng, n, mean, sd):
    """截断对称分布：均值 mean、尺度 sd，取值落在 ±1 内（与 `R` 的 pp 刻度量级相当）。"""
    out = []
    for _ in range(n):
        while True:
            v = mean + sd * rng.uniform(-SQRT12 / 2, SQRT12 / 2)
            if -1.0 < v < 1.0:
                out.append(v)
                break
    return out



def calibrate_regime(n=20, trials=TRIALS, seed=7):
    """`sign_regime` 在六种已知真相上的档位分布（每格 `trials` 列，每列 `n` 个种子）。"""
    rng = random.Random(seed)
    scenes = {
        'stable':      lambda: _draw_signed(rng, n, 0.30, 0.05),
        'mixed':       lambda: _draw_signed(rng, n, 0.05, 0.30),
        'thin_same':   lambda: [_draw_signed(rng, 1, 0.30, 0.05)[0] if i < max(1, n // 6)
                                else 0.0 for i in range(n)],
        'coin':        lambda: [rng.choice((1.0, -1.0)) * rng.uniform(0.01, 0.2) for _ in range(n)],
        'all_zero':    lambda: [0.0] * n,
        'thin_zero_mix': lambda: [_draw_signed(rng, 1, 0.0, 0.3)[0] if i < max(1, n // 6)
                                  else 0.0 for i in range(n)],
    }
    out = {}
    for name, gen in scenes.items():
        dist = {'stable': 0, 'mixed': 0, 'consistent_but_thin': 0, 'no_resolution': 0}
        for _ in range(trials):
            dist[paired_diff.sign_regime(gen())['regime']] += 1
        out[name] = {k: round(v / float(trials), 4) for k, v in dist.items()}
        out[name]['n'] = n
        out[name]['trials'] = trials
    # 判词（本脚本量出来的那一句）
    out['_verdict'] = {
        'false_mixed_rate': out['stable']['mixed'],
        'false_stable_on_coin_rate': out['coin']['stable'],
        'note': ('门槛 0.5／0.8 是判读门槛；上表给出它们在已知真相上的档位分布。'
                 'stable 行的 mixed 列 = 真同向却被报方向混杂；coin 行的 stable 列 = '
                 '真无信息却被报稳定。两者都应≤0.05 才配当结论句的依据。'),
    }
    return out


def calibrate_band_floor(n_seeds=30, trials=1000, seed=11, n_trials_per_seed=12):
    """`band_floor` 的"过半"在两种已知真相上的误判率。

    真相 A（"恒零"）：每个种子的 `R` 都是 0 ⇒ 正确判词必须是 `no_reading`。
    真相 B（"真带宽"）：每个种子的 `R` 在 [0, b] 上均匀 ⇒ 带宽恒为 b>0，
    正确判词必须是 `usable`（`degenerate` 也算**保守误判**：不构成假阳，
    因为它禁止引用；这里仍分开报）。

    **关键点**：`no_reading` 的定义是"**所有**种子的 max R 都 == 0"。只要有一
    个种子因为抽样拿到非零，整格就不是 `no_reading`——真相 A 的真阳性率因此
    随"非零种子出现的概率"急剧下降。本函数把"非零种子比例 p"扫一遍。
    """
    rng = random.Random(seed)

    def verdict_of(vals):
        # 直接复算 `real_lengths_policy.fifo_band_floor` 的三行判词，不 import
        # 那个模块（它会读真实负载，几十秒起步）。
        if all(max(v) == 0.0 for v in vals):
            return 'no_reading'
        if sum(1 for v in vals if max(v) - min(v) == 0.0) * 2 > len(vals):
            return 'degenerate'
        return 'usable'

    p_rows = {}
    for p in (0.0, 0.05, 0.1, 0.2, 0.5):
        dist = {'no_reading': 0, 'degenerate': 0, 'usable': 0}
        n_nonzero_seeds = []
        for _ in range(trials):
            vals, nz = [], 0
            for _ in range(n_seeds):
                if rng.random() < p:
                    nz += 1
                    vals.append([rng.uniform(0.0, 0.01) for _ in range(n_trials_per_seed)])
                else:
                    vals.append([0.0] * n_trials_per_seed)
            n_nonzero_seeds.append(nz)
            dist[verdict_of(vals)] += 1
        p_rows['p_nonzero=%.2f' % p] = {
            'dist': {k: round(v / float(trials), 4) for k, v in dist.items()},
            'expected_nonzero_seeds': round(p * n_seeds, 2),
        }
    return {
        'n_seeds': n_seeds,
        'trials_per_seed': n_trials_per_seed,
        'trials': trials,
        'truth_all_zero': p_rows,
        '_verdict': ('真相"每个种子 R 恒 0"时，只要有 1 个种子抽到非零，整格就'
                     '不再判 no_reading。实测见 p_nonzero 各行——这是结构性上限，'
                     '不是门槛调松能解决的；引用 band_floor 时必须同时看 '
                     'n_r_identically_zero 的计数，不能只看档名。'),
    }


def real_cells(n_groups=16, backlog_range=(2, 4, 8, 16)):
    """真实档 16 格的 `n_r_identically_zero` 落在"全零"连续统的哪个位置（只读归档）。

    **为什么要单列一张。** 上面那个 p 扫描是抽象的；真实的格子必须对号入座。
    关键是把"档名"和"零种子计数"并排报——`usable` 只说明**过半种子带宽非零**，
    不说明基线稳；而 `no_reading` 要求 30 个种子全零，宽松到只要有一个种子抽到
    非零就翻档。本函数直接读 `real-lengths-summary.json`，不重算。
    """
    path = os.path.join(HERE, '..', 'rollout-group-split-tradeoff', 'real-lengths-summary.json')
    with open(path) as f:
        arch = json.load(f)
    out = {}
    for label, arm in arch['real_arms'].items():
        cells = {}
        for b in backlog_range:
            c = arm['sign_stability']['backlog%d' % b]
            bf = c['band_floor']
            cells['backlog%d' % b] = {
                'band_floor': bf['band_floor'],
                'n_r_identically_zero': bf['n_r_identically_zero'],
                'n_band_zero': bf['n_band_zero'],
                'band_median_pp': c['fifo_band']['median_pp'],
            }
        out[label] = cells
    return out


def build():
    return {
        'doc': ('门槛校准：已知真相下的档位分布，量的是判据误判率而不是测试灵敏度。'
                'real_cells 是同一判据在真实归档 16 格上的落点，只读不重算。'),
        'sign_regime': calibrate_regime(),
        'band_floor': calibrate_band_floor(),
        'real_cells': real_cells(),
    }


def main(argv):
    out = build()
    text = json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True)
    if '--verify' in argv:
        with open(OUT_PATH) as f:
            ref = json.load(f)
        ok = json.dumps(ref, ensure_ascii=False, indent=1, sort_keys=True) == text
        print('VERIFY OK: 逐字段相同' if ok else 'VERIFY FAIL: 与归档不一致')
        return 0 if ok else 1
    if '--json' in argv:
        with open(OUT_PATH, 'w') as f:
            f.write(text + '\n')
    print('== sign_regime：已知真相 → 档位分布（n=20 种子/列）==')
    for name, row in out['sign_regime'].items():
        if name.startswith('_'):
            continue
        print('  %-14s %s' % (name, '  '.join(
            '%s=%.3f' % (k, row[k]) for k in
            ('stable', 'mixed', 'consistent_but_thin', 'no_resolution'))))
    print('== band_floor：真相"每种子 R 恒 0"，按非零种子比例 p 扫 ==')
    for name, row in out['band_floor']['truth_all_zero'].items():
        print('  %-16s no_reading=%.3f degenerate=%.3f usable=%.3f  (期望非零种子 %s)'
              % (name, row['dist']['no_reading'], row['dist']['degenerate'],
                 row['dist']['usable'], row['expected_nonzero_seeds']))
    print('== 真实档 16 格：档名与零种子计数并排（只读归档）==')
    for label, cells in out['real_cells'].items():
        print('  %-16s %s' % (label, '  '.join(
            '%s:%s(%d/30零,%.2fpp)' % (b.replace('backlog', 'bl'),
                                       c['band_floor'], c['n_r_identically_zero'],
                                       c['band_median_pp'])
            for b, c in sorted(cells.items(), key=lambda kv: int(kv[0][7:])))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
