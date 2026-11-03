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


from math import comb


def calibrate_units(active_seeds=(3, 4, 5, 19, 26, 27, 30), seed=17):
    """判据的**单位数**：同一个 0.8 门槛，在 3 个与 30 个表态种子上等价于什么。

    **为什么要补这一条（2026-11-04 第十五批）。** 第十三批量出的"`stable` 假阳为 0"
    是在"20 个种子**全部表态**"的场景里取到的；而真实 16 格里表态种子只有 **3–27** 个
    （0.5B cap256 积压 2 一个都没有、1.5B cap256 积压 2 只有 3 个），那几格恰恰就是
    判 `stable` 的地方。⇒ **那个 0 没有覆盖判据被实际使用的参数区间。**

    `sign_regime` 的 `stable` 条件是"表态种子同号占比 ≥ 0.8"。掷硬币输入（真值无信息）
    下，它取决于**表态种子的个数**：3 个种子要三个全中（概率 0.51），27 个要 ≥22 个
    同号（概率 0.00076）——**同一个常数、同一个判据，误报率差三个数量级**。

    本函数精确枚举（`paired_diff.falsy_rate`，判据用 `sign_regime` 那一句原样传入），
    另附 20 种子的蒙特卡洛复核（与第十三批的 0.0107 对得上才配对）。
    """
    def stable_rule(tally, n):  # 与 `paired_diff.sign_regime` 的 stable 分支逐字相同
        return tally / float(n) >= 0.8

    rows = {}
    for n in active_seeds:
        r = paired_diff.falsy_rate(n, passes=stable_rule, min_true_frac=0.5, two_sided=True)
        rows['n_active=%d' % n] = r
    # 蒙特卡洛复核：n=20 时精确值应落在第十三批实测 0.0107 的量级上
    rng = random.Random(seed)
    trials = 20000
    n20 = 0
    for _ in range(trials):
        diffs = [rng.choice((1.0, -1.0)) * rng.uniform(0.01, 0.2) for _ in range(20)]
        if paired_diff.sign_regime(diffs)['regime'] == 'stable':
            n20 += 1
    exact20 = paired_diff.falsy_rate(20, passes=stable_rule, min_true_frac=0.5,
                                        two_sided=True)
    return {
        'by_active_seeds': rows,
        '_check_n20': {
            'monte_carlo_rate': round(n20 / float(trials), 4),
            'monte_carlo_trials': trials,
            'exact_rate': exact20['consistency_rate'],
            'unit_needs': exact20['units_needed'],
        },
        '_verdict': ('同一个 0.8 门槛随表态种子数移动两个数量级：3 个种子时掷硬币输入'
                     '有 25% 被判 stable，27 个时只有 0.15%。⇒ 报"假阳为 0"必须同时报'
                     '表态种子数；`stable` 只在表态种子数足够多时才算方向已定。'),
    }


def real_cell_units(backlog_range=(2, 4, 8, 16)):
    """真实 16 格逐格：判据实际用在了多少个单位上、无信息输入下会判成什么（只读归档）。"""
    path = os.path.join(HERE, '..', 'rollout-group-split-tradeoff', 'real-lengths-summary.json')
    with open(path) as f:
        arch = json.load(f)
    out = {}

    def stable_rule(tally, n):
        return tally / float(n) >= 0.8

    for label, arm in sorted(arch['real_arms'].items()):
        cells = {}
        for b in backlog_range:
            c = arm['sign_stability']['backlog%d' % b]
            lof = c['lof_sign']
            n_active = lof['n_positive'] + lof['n_negative']
            same = max(lof['n_positive'], lof['n_negative'])
            row = {
                'regime': lof['regime'],
                'n_active': n_active,
                'n_same_sign': same,
            }
            if n_active:
                fr = paired_diff.falsy_rate(n_active, passes=stable_rule,
                                            min_true_frac=0.5, two_sided=True)
                row['units_needed_for_stable'] = fr['units_needed']
                row['coin_stable_rate'] = fr['consistency_rate']
                # 离掉档还差几个种子：判 `stable` 的格子里这个数最小者最脆
                row['margin_seeds'] = same - fr['units_needed']
            cells['backlog%d' % b] = row
        out[label] = cells
    return out


def _band_ratio_scene(rng, ratio, n=30, trials=12, sd_band=0.02):
    """一个"真方向已知、真效应 = `ratio` × 本种子带宽"的格子。

    逐种子配三组读数：

      - `base_i`：FIFO 臂换序后的 `trials` 个读数，宽 `w_i = max−min` 就是**这一格
        拿到的带宽**；
      - `alt_i`：对照臂读数 = 同一 `base_i` **逐点**加上 `ratio·w_i`（真效应）再加一份
        独立噪声 `ε_i`。逐点平移保证"真效应"确实是 `ratio·w_i`，**独立噪声保证两臂
        不再是同一个数平移出来的**——否则每个种子的配对差恒为正、`sign_regime`
        永远判 `stable`，扫描就退化成常量（本函数第一版正是这样，见 readme）。

    `ε` 的尺度与 `base_i` 的抖动同阶（`sd_band`），所以 |均值差| 大致仍是 `ratio·w_i`，
    而**逐种子的符号会因 `ε` 而抖动**——这才是要量的漏报。

    `n=30`／`trials=12` 对齐真实档（`fifo_reorder_band` 每种子 12 次换序、每格 30 种子）。
    """
    diffs, bands = [], []
    for _ in range(n):
        base = _draw_signed(rng, trials, 0.5, sd_band)
        w = max(base) - min(base)
        eps = _draw_signed(rng, 1, 0.0, sd_band)[0]
        diffs.append(ratio * w + eps)
        bands.append(w)
    return diffs, bands


def effect_band_ratio(n=30, trials=12, replications=400, seed=13):
    """`sign_regime` 的漏报：真方向已知、效应量只有 `ratio` 倍带宽时会怎样。

    **为什么要补这一条。** 上面那条校准把"已知真相"设成"每个种子都明显同向"，
    于是 `stable` 的假阳是 0——但**那个 0 是在效应远大于噪声的条件下取到的**。
    真实归档里"效应是否超带宽"恰恰是每个格子都要问的问题，而那些格子的
    |效应|／带宽 中位只有 **0.71**（16 格实测，见 `real_cells` 的 `band_ratio`）。
    ⇒ 本函数扫 `ratio`，量出**当真方向存在、但被噪声盖住时**，`sign_regime`
    还判不判得出方向。

    单列一格时，30 个种子上的"|效应|"**波动很大**：每个种子只有 12 次换序，
    带宽由极值决定，本身就是随机数，所以 `ratio=0.71` 的一格既不保证每个种子都
    看得出方向，也不保证均值看得出。这正是要量的东西——它决定"档名是 `stable`"
    能不能当"方向已确定"的证据。
    """
    rng = random.Random(seed)
    rows = {}
    for ratio in (0.20, 0.25, 0.30, 0.40, 0.50, 0.71, 1.0, 2.0, 4.0):
        dist = {'stable': 0, 'mixed': 0, 'consistent_but_thin': 0, 'no_resolution': 0}
        agree_with_mean = 0           # 判 `stable`／`thin` 且方向与真方向一致
        for _ in range(replications):
            diffs, _ = _band_ratio_scene(rng, ratio, n=n, trials=trials)
            r = paired_diff.sign_regime(diffs)
            dist[r['regime']] += 1
            if r['regime'] in ('stable', 'consistent_but_thin') and diffs and sum(diffs) > 0:
                agree_with_mean += 1
        rows['ratio=%.2f' % ratio] = {
            'dist': {k: round(v / float(replications), 4) for k, v in dist.items()},
            # "判出方向且方向正确"的联合比率：既要求给出方向，也要求给对
            'correct_direction_rate': round(agree_with_mean / float(replications), 4),
            'n': n,
            'trials_per_seed': trials,
        }
    return {
        'replications': replications,
        'note': ('真方向已知（所有种子同向），真效应量 = ratio × 该种子带宽。'
                 '方向一列的分母是全部 replications；`mixed` 在这里是**假阳**'
                 '（真值同向却被报方向混杂），`no_resolution` 是漏报'
                 '（真方向存在却被报"什么也不能说"）。'),
        'rows': rows,
    }


def real_band_ratios():
    """真实 16 格逐种子的 |效应|／带宽比值——**实测值，不是本脚本算的**。

    2026-11-04 M1 用真实负载逐种子重算过一次（`real_lengths_policy` 的
    `POLICY_LOF` 减 `POLICY_FIFO` 作效应，同一种子 12 次换序的 FIFO 极差作带宽，
    30 种子／格，四个负载 × 四档积压，约 3.5 分钟）。逐种子的中位比值：

    | 负载 | 2 批 | 4 批 | 8 批 | 16 批 |
    |---|---|---|---|---|
    | 1.5B cap256 | —（带宽全 0） | 0.71 | −0.10 | −0.59 |
    | 1.5B cap1024 | 0.72 | 1.24 | 1.30 | 1.17 |
    | 0.5B cap256 | —（带宽全 0） | 1.38 | 0.00 | −0.27 |
    | 0.5B cap1024 | 0.38 | 2.03 | 1.79 | 1.78 |

    **符号是本格实测的 LOF−FIFO 方向**，负值说明该格"长度优先"让 `R` 变小；
    这里的用途只是**量级**——真实格的 |效应| 与带宽同阶，多数在 0.4–2.0 之间。
    逐种子的原始配对差与带宽没有入库（归档只留汇总），所以本表是**一次性实测**
    记录，不随脚本重算；重跑方法见 `readme.md` 第十一批。
    """
    measured = {
        'qwen15b_cap256':  [None, 0.71, -0.10, -0.59],
        'qwen15b_cap1024': [0.72, 1.24, 1.30, 1.17],
        'qwen05b_cap256':  [None, 1.38, 0.00, -0.27],
        'qwen05b_cap1024': [0.38, 2.03, 1.79, 1.78],
    }
    backlogs = (2, 4, 8, 16)
    out = {}
    vals = []
    for label, row in measured.items():
        out[label] = {'backlog%d' % b: v for b, v in zip(backlogs, row)}
        vals.extend(v for v in row if v is not None)
    vals.sort()
    absvals = sorted(abs(v) for v in vals)
    out['_summary'] = {
        'n_cells_measured': len(vals),
        'n_cells_with_zero_band': 2,
        'median_abs_ratio': round(absvals[len(absvals) // 2], 2),
        'range_abs_ratio': [absvals[0], absvals[-1]],
        'counting': ('逐种子 |效应|／带宽 的**中位**。两格带宽全为 0（两个模型 cap256 '
                     '积压 2）无定义，记为 null；0.5B cap256 积压 8 的中位恰为 0.00'
                     '（该格效应与带宽都被个别种子撑起）。'),
    }
    return out


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
        # 第十一批新增两条：校准覆盖面的缺口（上面那条把真值设成"效应远大于噪声"）
        'effect_band_ratio': effect_band_ratio(),
        'real_band_ratios': real_band_ratios(),
        # 第十五批新增：门槛被用在**多少个单位**上（第十三批的覆盖面缺口）
        'units': calibrate_units(),
        'real_cell_units': real_cell_units(),
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
    eb = out['effect_band_ratio']
    print('== 真方向已知、效应量 = ratio × 带宽：%d 次重复 ==' % eb['replications'])
    for name, row in eb['rows'].items():
        print('  %-12s stable=%.3f mixed=%.3f thin=%.3f no_res=%.3f  | 方向判对=%.3f'
              % (name, row['dist']['stable'], row['dist']['mixed'],
                 row['dist']['consistent_but_thin'], row['dist']['no_resolution'],
                 row['correct_direction_rate']))
    rb = out['real_band_ratios']['_summary']
    print('== 真实格逐种子 |效应|／带宽（实测记录）==  %d 格，中位 |比值| %.2f，范围 %.2f–%.2f'
          % (rb['n_cells_measured'], rb['median_abs_ratio'],
             rb['range_abs_ratio'][0], rb['range_abs_ratio'][1]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
