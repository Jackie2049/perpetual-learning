#!/usr/bin/env python3
"""配对差判读的共享工具：符号分布、基线带宽、判读门槛（2026-11-03 M1）。

这个文件只有一个目的——把同一族判据从各实验脚本里收拢到一处。此前同类判据在
四个脚本里各写一遍，容差（1e-12）、"相等等于什么"、字段名都各写各的，改正一处
不会带动另一处。第①到第⑧条是把它们原样搬过来，**语义与数值逐位不变**；第⑨条
是搬的过程中才看清的坑（见下）。

为什么要一起报：一个配对差 `d_i = 臂_i − 基线_i` 只报均值，会把"每个种子都同向"
和"一半正一半负正好抵消"读成同一个数。判读分三步，次序不能换：

  ① `n_identical` 有没有吃掉大半种子 —— 两臂读数完全相同不是"一致"，是这一格
     **没有分辨率**（低积压时 `R` 本来就接近 0，两臂一起落在 0 上）；
  ② `same_sign_fraction` —— 均值是不是靠少数种子撑起来的；
  ③ `|效应| > 带宽` —— 效应是否大到无法用**基线自身抖动**（同一个设置换输入顺序
     就能动的那部分）解释。

三步任一不过就只能写"方向未定"。`exceeds_band` 是**判读门槛，不是显著性检验**：
它没做 CI 或置换，回答的只是"能不能用基线抖动解释"。

**容差是口径的一部分。** `TOL = 1e-12` 是绝对容差，不是相对容差：R 在 [0,1]、步数
在 [0,300] 这种量纲下它等价于"精确相等"，而换成 1e-9 或相对容差会让"两臂完全
相同"的种子从 `n_identical` 掉进 `n_positive`／`n_negative`——那正是第⑨条的事故。

用法（作为库）：
    from paired_diff import summarize_paired, exceeds_band
    st = summarize_paired(diffs, scale=100)        # 百分数口径
    ok = exceeds_band(effect, band)                 # 判读门槛

**容差必须和刻度一起引用。** 第⑩条是 2026-11-03 第八批补上的：`TOL` 是绝对容差，
而比率型读数（`v` 与 1 比较）在不同刻度上有**不同的零**——`steps_ratio` 差 1e-13 与
`wall_ratio` 差 1e-13×100 是两个不同的容差口径。直接把 `v` 喂给 `sign_counts` 会把
"恰好等于 1"记成正号；必须先用 `around(v, ref)` 平移成 `v - ref` 再归类。

自检：
    python3 paired_diff.py --selftest
"""

import math
import statistics
import sys
from math import comb

# 绝对容差。R／准确率在 [0,1]，步数在 [0,300]：1e-12 相当于"精确相等"。
# 不改这个值，除非同时重算所有引用它的归档（见 readme 的「口径锁定」）。
TOL = 1e-12

# 归档落盘的小数位数。ratio／pp 一律 3 位，swing（极差）也 3 位。
_NDIGITS = 3


def _tol(tol):
    """容差在**调用时**解析。写成 `def f(tol=TOL)` 会把定义那一刻的值绑进默认参数，
    之后改 `TOL` 不生效——这个坑于 2026-11-03 抽共享工具时实测到：把 TOL 注入成
    1e-3 后 `sign_counts([1e-6])` 仍记正，且无法从外部收紧容差。"""
    return TOL if tol is None else tol


def sign_counts(diffs, tol=None):
    """把配对差分成正／负／**相同**三类。返回的是计数，不做任何舍入。

    `n_identical` 与 `n_ties` 不是同一件事：前者在两处代码里都指"两臂读数落在
    容差内"，后者在 `steps_at` 里指"到阈值步号完全相同"（整数，精确 ==）。
    两者都照原样保留，不合并。
    """
    tol = _tol(tol)
    pos = sum(1 for d in diffs if d > tol)
    neg = sum(1 for d in diffs if d < -tol)
    return {
        'n_positive': pos,
        'n_negative': neg,
        'n_identical': len(diffs) - pos - neg,
    }


def same_sign_fraction(diffs, tol=None):
    """多数方向的占比（**全列做分母**，identical 的种子也留在分母里）。

    这个口径**只会把读数压向 0**：identical 的种子既不算正也不算负，却占着分母。
    20 个种子里 18 个相同、余下 2 个同向 ⇒ 给 0.1，看着像"方向混杂"，其实是
    "没分辨率"。真实例：`qwen15b_gsm8k_n8` 积压 2（18 个相同，表态的 2 个全同向，
    表态口径给 1.0）。

    所以 ssf **不能单独引用**：必须同时报 `resolve_active_fraction()`（有多少种子
    在表态）与 `same_sign_fraction_active()`（只在表态种子里的占比）。全列 ssf 恒
    ≤ 表态口径（分母 `n_active ≤ n`、分子相同），两者之差就是"被 identical 压掉的
    那部分"。三者一起看，"没分辨率"与"方向混杂"才分得开。
    """
    tol = _tol(tol)
    c = sign_counts(diffs, tol)
    n = len(diffs)
    if not n:
        return None
    return round(max(c['n_positive'], c['n_negative']) / float(n), 3)


def resolve_active_fraction(diffs, tol=None):
    """有多少比例的种子**真的在表态**（非 identical）。配套 `same_sign_fraction` 用。

    同一个 ssf 可能来自"方向混杂"也可能来自"没分辨率"，两者解读相反。本函数给出
    `n_active / n`：小于约 0.5 时 ssf 一律不应单独引用（分母里大半种子没表态）。
    """
    c = sign_counts(diffs, _tol(tol))
    n = len(diffs)
    if not n:
        return None
    return round((c['n_positive'] + c['n_negative']) / float(n), 3)


def same_sign_fraction_active(diffs, tol=None):
    """只在**表态的种子**里算多数方向占比；全为 identical 时返回 None。

    `same_sign_fraction` 用全列做分母，本函数用 `n_active` 做分母。因为
    `n_active ≤ n` 且分子相同，**`active ≥ ssf` 恒成立**（已穷举验证）——两者之差
    度量"这一格被 identical 吃掉了多少"，不会出现反向修正。归档里仍写全列口径的
    ssf（既有字段不动），引用时两者一起看。
    """
    c = sign_counts(diffs, _tol(tol))
    active = c['n_positive'] + c['n_negative']
    if not active:
        return None
    return round(max(c['n_positive'], c['n_negative']) / float(active), 3)


def sign_regime(diffs, tol=None, min_active=0.5, min_same_active=0.8):
    """把一格配对差分成四档，**取代"先看 n_identical"这一步**。

    旧的三步把"重复比例高"直接读成"没分辨率"。这只在"表态的种子方向也混杂"时
    成立；表态种子若方向一致，高重复恰恰说明**方向稳定、只是分辨率薄**。两者对
    引用的要求不同（都没法主张效应量，但后者不该被写成"方向混杂"或不稳）。

    `min_active`／`min_same_active` 是**判读门槛不是显著性检验**（未做 CI／置换），
    与 `exceeds_band` 同性质。判词：

      - `no_resolution`：表态占比 < `min_active`，且表态种子内也不一致（< `min_same_active`）。
        两臂大多相等、表态的少数又各说各话 —— 什么也不能说。
      - `consistent_but_thin`：表态占比 < `min_active`，但表态种子内一致。
        方向一致，**分辨率不足**，不能主张效应量，也不能写"方向混杂"。
      - `mixed`：表态占比够，但表态种子内不同号 —— 真正的方向混杂。
      - `stable`：表态占比够，且表态种子内一致。

    返回 `regime` 与判读所需的三个数，便于归档与复核时直接引用而不是手读。
    """
    c = sign_counts(diffs, _tol(tol))
    n = len(diffs)
    n_active = c['n_positive'] + c['n_negative']
    frac_active = round(n_active / float(n), 3) if n else None
    frac_same_active = same_sign_fraction_active(diffs, tol)
    if not n_active:
        regime = 'no_resolution'
    elif frac_active < min_active:
        regime = 'consistent_but_thin' if frac_same_active >= min_same_active else 'no_resolution'
    else:
        regime = 'stable' if frac_same_active >= min_same_active else 'mixed'
    return {
        'regime': regime,
        'resolve_active_fraction': frac_active,
        'same_sign_fraction_active': frac_same_active,
        'n_identical': c['n_identical'],
    }


def sd_pp(diffs, scale=100.0):
    """种子间标准差乘 scale。样本少于 2 时返回 None（单点没有离散度可言）。"""
    if len(diffs) < 2:
        return None
    return round(scale * statistics.stdev(diffs), 3)


def summarize_paired(diffs, scale=100.0, tol=None, with_mean=True):
    """一次给出判读三步要的全部量：计数、同号占比、种子间 sd、极值。

    `scale` 把 R（[0,1]）换成百分数：`100` 给 pp，`1` 给原始量纲。调用方原先把
    这些量分别装箱（`seed_sign_stability` 报 6 个键、`policy_sign_stability`
    报 7 个、`steps_at` 报 5 个），字段集合不同是因为各处需要不同，**不是口径
    不同**；这里只提供底层量，各脚本保留自己的键名与舍入，避免归档字段漂移。
    """
    tol = _tol(tol)
    out = {}
    if with_mean:
        out['mean'] = round(statistics.mean(diffs) * scale, 3) if diffs else None
    c = sign_counts(diffs, tol)
    out.update(c)
    out['same_sign_fraction'] = same_sign_fraction(diffs, tol)
    # 与 ssf 配套的三个量：分母里有多少种子在表态、只在表态种子里算的同号占比、
    # 以及 ssf 被 identical 压低了多少（前者减后者）。
    out['n_active'] = c['n_positive'] + c['n_negative']
    out['resolve_active_fraction'] = resolve_active_fraction(diffs, tol)
    out['same_sign_fraction_active'] = same_sign_fraction_active(diffs, tol)
    out['sd_pp'] = sd_pp(diffs, scale)
    out['min_pp'] = round(min(diffs) * scale, 3) if diffs else None
    out['max_pp'] = round(max(diffs) * scale, 3) if diffs else None
    return out


def across_scale(samples, ref=1.0):
    """把"比率型"读数换到各处实际使用的**刻度**上，与 `ref` 相减后归类。

    第⑩条：比率读数的"相同"依赖刻度。同一列 `v`（`R` 的比值、窗口成本比）在
    三处被喂进判据时口径不同：

      - `real_lengths_policy`／`bucketing_probe`：`v - 1.0`，刻度 1（原量纲）；
      - `steps_ratio`／`wall_ratio` 的摆幅：`pp` 刻度，`100 * (v - 1)`；
      - `wall_ratio_at`：先 `v - 1` 再乘 100，与上一条等价。

    `1e-12` 在"原量纲"上只吞掉 1e-14 的偏差，在 `pp` 刻度上却吞掉 1e-12 的偏差——
    **两者相差 100 倍**。这不是 bug（各脚本各自归档、各自 `--verify`），但不能拿
    一个刻度的 `n_identical` 去解释另一个刻度，也不能把"某刻度上 0 个相同"读成
    "另一刻度上也是 0 个"。

    返回两个刻度下的计数，供对照脚本逐字段比对。
    """
    out = {}
    for label, scale in (('raw', 1.0), ('pp', 100.0)):
        shifted = [scale * (v - ref) for v in samples]
        c = sign_counts(shifted)
        out[label] = {
            'scale': scale,
            'n_positive': c['n_positive'],
            'n_negative': c['n_negative'],
            'n_identical': c['n_identical'],
            'tol_effective': TOL / scale,   # 原量纲上相当于容忍多大的偏差
        }
    return out


def rounding_cases():
    """构造两组"舍入会分岔"的输入，用来验证 `half_up` 与内建 `round` 的差异确实存在。

    存在的理由是：`round(x, 3)` 对**精确的 x.xxx5** 取偶（1.0125→1.012、
    0.3125→0.312），而人写的"保留三位小数的百分数"读作 `1.013`／`0.313`。
    在真实数据上碰不到——差值本身是浮点计算结果，最后一位永远不会恰好落在
    5 上——所以这是**口径声明**而不是缺陷修复。

    返回 `(inputs, half_up, builtin)` 三元组；差值集合为
    `[0.2]*10 + [0.3]*6` 时，`mean + 1.0 == 1.2375`（二进制可精确表示，是真正的平局）。
    """
    from decimal import Decimal

    def _half_up(x, ndigits):
        q = Decimal(1).scaleb(-ndigits)
        return float(Decimal(repr(x)).quantize(q, rounding='ROUND_HALF_UP'))

    cases = {
        'ratio_mean': 1.2375,
        'ssf': 5 / 16,          # 0.3125，5 正 11 负
        'small': 0.0625,
    }
    return (cases,
            {k: _half_up(v, _NDIGITS) for k, v in cases.items()},
            {k: round(v, _NDIGITS) for k, v in cases.items()})


def unresolved(diffs, tol=None, identical_floor=0.5):
    """这一格是否根本没有分辨率：`n_identical` 占掉了 `identical_floor` 以上的种子。

    **注意方向**（2026-11-03 第十批订正）：identical 留在分母里会让
    `same_sign_fraction` **偏低**，不是"虚高"——分母变大、分子不变，读数只会被
    往下压（`qwen15b_gsm8k_n8` 积压 2 给了 0.1，而表态的 2 个种子全同向）。原
    docstring 写的是"结构性虚高"，方向写反了。

    这里把"刹车"变成判定：超过一半种子两臂读数相同，就不该报方向。但**不该报方向
    不等于"方向混杂"**——那两件事由 `sign_regime()` 分开判（`consistent_but_thin`
    与 `mixed`）。本函数保留为"有没有分辨率"的布尔问句，档名请用 `sign_regime()`。
    """
    tol = _tol(tol)
    if not diffs:
        return True
    c = sign_counts(diffs, tol)
    return c['n_identical'] >= identical_floor * len(diffs)


def exceeds_band(effect, band, strict=True):
    """效应绝对值是否超出基线自身抖动（带宽）。**判读门槛，不是显著性检验。**

    只需判断"能不能用基线抖动解释"，不回答"效应是否真实存在"。`strict=True`
    取严格大于：效应恰等于带宽时按"解释得了"处理，不能算过门槛。

    **带宽恰为 0 时抛 `ValueError`**（2026-11-04 第十四批）。带宽为 0 有两种成因：
    确实观察到了非零读数、只是每次换序结果相同（`baseline_stable`），与每个种子
    的读数恒为 0、这一格根本没有读数（`no_reading`）——判词相反，而本函数只看得到
    那个 0。要断言一个为 0 的带宽，请用 `band_floor_pass(..., zero_kind=...)`。
    """
    if abs(band) == 0.0:
        _reject_zero_band(None)
    return abs(effect) > abs(band) if strict else abs(effect) >= abs(band)


def _reject_zero_band(zero_kind):
    raise ValueError(
        'band=0.0 时不能直接调 exceeds_band——带宽为 0 有两种成因，判词相反。'
        '确认后二选一：band_floor_pass(effect, 0.0, zero_kind="baseline_stable")'
        '（确实观察到了非零读数、只是换序不动）或 zero_kind="no_reading"'
        '（每个种子读数恒为 0，这一格问不出来）。实际传入：%r' % (zero_kind,))


def band_floor_pass(effect, band, zero_kind=None, strict=True):
    """`exceeds_band` 的带宽为 0 情形的显式版本。

    带宽 > 0：与 `exceeds_band` 逐位相同。
    带宽 == 0：只有声明 `zero_kind='baseline_stable'` 才按"非零效应即超门槛"作答；
    `'no_reading'`、未声明或未知成因一律返回 **False**（问不出来不是证据）。
    """
    if abs(band) != 0.0:
        return abs(effect) > abs(band) if strict else abs(effect) >= abs(band)
    if zero_kind == 'baseline_stable':
        return abs(effect) > 0.0 if strict else True
    if zero_kind == 'no_reading':
        return False
    _reject_zero_band(zero_kind)
    return False  # 不可达：_reject_zero_band 必然抛出


def band_floor_rule(kind):
    """`n_r_zero * 2 > n` ⇔ 过半数种子带宽为 0 ⇔ `degenerate`（2026-11-04 第十六批）。

    这是 `band_floor` 那三行判词里与 `units` 有关的一行，判据本身**只有这一处**：
    `real_lengths_policy.fifo_band_floor`、`band_floor.py` 的 `band_floor_verdict`、
    `sampling_reach.py` 的同名副本，以及 `calibrate_thresholds` 的合成场景，都从这里取
    （此前是四处各写一遍 `* 2 >`）。它同时是 `falsy_rate(n, passes=..., two_sided=True)`
    的现成输入——"过半"两侧都算，这一点与 `stable` 的同号占比相同。

    `kind='degenerate'` 与原式逐字相同；`kind='usable'` 是它的补集（**不是**取反后的
    另一份式子——补集由同一个 `passes` 取反得到，两档之和恒为 1）。
    """
    over_half = lambda tally, n: tally * 2 > n  # noqa: E731
    if kind == 'degenerate':
        return over_half
    if kind == 'usable':
        return lambda tally, n: not over_half(tally, n)
    raise ValueError("band_floor_rule 只认 'degenerate' / 'usable'，收到 %r" % (kind,))


def falsy_rate(n_truth, passes=None, min_true_frac=0.5, two_sided=False):
    """判据取"**多少单位满足**"时，它在无信息输入上的判真概率（2026-11-04 第十五批）。

    门槛都写成常数（`min_same_active=0.8`、`tally * 2 > n`），而**判据真正被用在
    多少个单位上逐格不同**：真实 16 格里表态种子 3–27 个、带宽非零种子 0–30 个。
    0.8 在 3 个单位上等价于"三个全中"（概率 0.51），在 27 个单位上等价于
    "≤5 个输"（概率 0.0024）——**常数本身不交代这一点，单位数才交代**。

    **另一条触发条件（2026-11-04 第十六批）**：判据的单位总数本身随格变化时，
    同一个"过半"在 3 个单位上只是"两个赢"——掷硬币输入下 `degenerate` 有 **100%**
    （门槛取 2，2/3 与 3/3 全落在过半数且双侧都算），退到 `usable` 只要 **1 个**种子
    不同号。这一类"单位数小到判据失去分辨力"的格子，**别名的误判率比档名更有信息**。

    `n_truth`：真值下全部符合条件的单位数（如真相全同号时的表态种子数）。
    `passes(tally, n)`：调用方的**真实判据**，传进来而不是在这里复写一份——
    复写会漂，正如 `band_floor` 与 `fifo_reorder_band` 各写一份种子式子那次。
    `tally` 指**某一指定侧**的计数（同号数、带宽为零的种子数）；缺省按"过半数"。

    `two_sided`：判据对 `tally` 与 `n − tally` 谁大谁小都成立（`stable` 的"同号占比"、
    `band_floor` 的"过半"都是）。双侧判据在两侧各算一次再相加；单侧（"全部为零"）只
    算一侧。**必须由调用方声明**——本函数第一版想自动判，结果 `stable` 的镜像侧
    在 n=20 时恰好差一个种子不通，自动判判成单侧，把误判率少算了一半
    （精确值 0.0059 vs 蒙特卡洛 0.0117）。

    无信息输入模型：每个单位独立以 `min_true_frac`（掷硬币取 0.5）落在判据想要的
    那一侧。返回 `units_needed`（最小的 tally）、`all_units_rate`（全部满足，幂运算
    精确）、`consistency_rate`（判真概率，**精确枚举**、不做正态近似——本函数的全部
    意义就是小 n 的尾巴）。
    """
    n = int(n_truth)
    if n <= 0:
        raise ValueError('n_truth 必须为正整数（判据至少要有一个单位才谈得上误报率）')
    if passes is None:
        passes = lambda tally, total: tally * 2 > total  # noqa: E731
    p = float(min_true_frac)
    needed = next((i for i in range(1, n + 1) if passes(i, n)), n + 1)
    if needed > n:
        raise ValueError('判据在 n=%d 个单位上恒假，无法给出判真率' % n)
    pmf = [comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i)) for i in range(n + 1)]
    rate = sum(pmf[i] for i in range(n + 1) if passes(i, n))
    if two_sided:
        rate *= 2
    return {
        'n_units': n,
        'units_needed': needed,
        'two_sided': bool(two_sided),
        'all_units_rate': round(p ** n, 6),
        'consistency_rate': round(min(rate, 1.0), 6),
    }


def span(values):
    """一列读数的极差（序列为空返回 None）。用于"同一个设置换个阈值／顺序差多少"。"""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return max(vals) - min(vals)


def sign_flip(values):
    """一列读数是否在零的两侧都出现过（含恰等于零）。`values` 为空返回 None。

    注意这是**相对零**的翻号；比率类读数要问的是"相对 1"（`steps_ratio` 越过 1 表示
    臂更慢），用 `sign_flip_around(values, 1.0)`。两者把同一列判成相反结论并不矛盾——
    0.98 与 1.005 都在零的同侧、却在 1 的两侧。
    """
    return sign_flip_around(values, 0.0)


def sign_flip_around(values, ref):
    """一列读数是否在 `ref` 两侧都出现过（含恰等于 `ref`）。`values` 为空返回 None。

    比率读数的"翻号"是相对 1 的：`steps_ratio` 从 0.978 变到 1.005 就是"从更慢变成
    更快"。把这一列直接喂 `sign_flip` 会一律返回 False（它们都在零以上），这正是
    抽共享判据时差点丢掉的语义。
    """
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return len({v > ref for v in vals}) > 1


def resolution_limit(diffs, tol=None):
    """这一格能分辨的最小步长：非降解种子间差值的**最小非零绝对值**。

    R／pp 口径没有这一维，步号口径有——两个种子差 0.5 步在步号上不该写成
    "快半步"。没有可分辨的非零差值时返回 None（等同于"无分辨率"）。
    """
    tol = _tol(tol)
    live = [abs(d) for d in diffs if abs(d) > tol]
    return min(live) if live else None


def _selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        if not cond:
            print('FAIL %s' % name)
            ok = False

    # ① 均值相同、符号分布不同：这正是"均值不能代替符号分布"
    #    a、b 的均值都是 +0.01；a 全部种子同向，b 只有一半种子有方向、另一半在
    #    容差内。只报均值的读法会给出同一个数，实际前者稳定、后者无分辨率。
    a = [0.01, 0.01, 0.01, 0.01]
    b = [0.02, 0.02, 0.00, 0.00]
    check('same mean', abs(statistics.mean(a) - statistics.mean(b)) < 1e-12)
    check('same mean differs in same_sign',
          same_sign_fraction(a) == 1.0 and same_sign_fraction(b) < 1.0)

    # ② 容差内一律记 identical（绝对容差，1e-13 与 0 同等对待）
    check('tol bucket', sign_counts([1e-13, -1e-13, 0.5])['n_identical'] == 2)
    # ③ 恰好等于容差时不进位（严格大于才算正）
    check('tol strict', sign_counts([TOL])['n_identical'] == 1)
    # ④ 全零种子：同号占比为 0，但该格无分辨率
    check('degenerate', same_sign_fraction([0.0] * 10) == 0.0)
    check('degenerate unresolved', unresolved([0.0] * 10))
    # ⑤ 只有一半种子有分辨率时**恰好**触发（>= 0.5×n 即无分辨率），差一个种子就翻转
    check('half boundary', unresolved([0.0] * 5 + [0.1] * 5))
    check('just under half', not unresolved([0.0] * 4 + [0.1] * 6))
    # ⑥ 判读门槛：恰等于带宽不过门槛
    check('band strict', not exceeds_band(1.0, 1.0))
    check('band over', exceeds_band(1.5, 1.0))
    # ⑥b 带宽恰为 0 时**拒绝作答**：两种成因判词相反，而这个函数分不出是哪一种。
    try:
        exceeds_band(0.02, 0.0)
        check('zero band raises', False)
    except ValueError:
        check('zero band raises', True)
    check('zero band stable', band_floor_pass(0.02, 0.0, zero_kind='baseline_stable'))
    check('zero band unreadable', not band_floor_pass(0.02, 0.0, zero_kind='no_reading'))
    try:
        band_floor_pass(0.02, 0.0)
        check('zero band undeclared raises', False)
    except ValueError:
        check('zero band undeclared raises', True)
    # 非零带宽时 band_floor_pass 与 exceeds_band 逐位相同（两个分支不能各行其是）
    check('nonzero band delegates',
          band_floor_pass(1.5, 1.0) == exceeds_band(1.5, 1.0)
          and band_floor_pass(1.0, 1.0) == exceeds_band(1.0, 1.0))
    # ⑦ 极差与翻号
    check('span', span([1.0, 1.2, 0.9]) == 0.3 or abs(span([1.0, 1.2, 0.9]) - 0.3) < 1e-12)
    check('flip', sign_flip([1.0, -0.1, 0.5]))
    check('no flip', not sign_flip([1.0, 0.1, 0.5]))
    # ⑦b 比率读数的翻号**相对 1**：0.978 → 1.005 是翻号，但相对零不是。
    #     这一条是实际事故——把 `steps_ratio` 交给 sign_flip 会把 p3:A 的翻号读丢。
    ratio_col = [0.978, 1.005]
    check('ratio flips around 1', sign_flip_around(ratio_col, 1.0))
    check('ratio does not flip around 0', not sign_flip(ratio_col))
    check('all above ref', not sign_flip_around([1.024, 1.206], 1.0))
    # ⑧ 步号分辨率
    check('resolution', resolution_limit([2, 2, 1]) == 1)
    check('resolution none', resolution_limit([0, 0]) is None)
    check('sd needs two points', sd_pp([1.0]) is None)

    # ⑨ 舍入与刻度（本文件第八批补）：这两条**不是**在修 bug，是在声明口径。
    #    它们保证：① `round(x,3)` 对精确平局取偶，所以"1.2375 → 1.238"是巧合、
    #    不是规则；② 同一列比率在原量纲与 pp 刻度上容忍的偏差相差 100 倍。
    cases, half_up, builtin = rounding_cases()
    check('half_up differs from builtin on exact tie',
          half_up['ssf'] != builtin['ssf'])
    check('ratio tie happens to agree',
          half_up['ratio_mean'] == builtin['ratio_mean'])
    # ⑩ 刻度：同一列 `v` 在 raw／pp 两个刻度上的 n_identical 可以不同
    near1 = [1.0 + 5e-13, 1.0 + 5e-11]   # raw: 一个 identical 一个正；pp: 两个都正
    sc = across_scale(near1, ref=1.0)
    check('raw sees one identical', sc['raw']['n_identical'] == 1)
    check('pp sees none identical', sc['pp']['n_identical'] == 0)
    check('tol scales with the ruler',
          abs(sc['raw']['tol_effective'] - 1e-12) < 1e-24
          and abs(sc['pp']['tol_effective'] - 1e-14) < 1e-26)

    # ⑪ 四档分档（本文件第十批补）：同样的"重复很多"，方向一致与方向混杂必须分开。
    #    这两格全列 ssf 分别是 0.1 与 0.25，旧的三步都会读成"没分辨率"。
    thin = sign_regime([0.1] * 2 + [0.0] * 18)
    check('thin but consistent', thin['regime'] == 'consistent_but_thin'
          and thin['same_sign_fraction_active'] == 1.0)
    check('high duplicate + mixed active is no_resolution',
          sign_regime([0.1] * 5 + [-0.1] * 4 + [0.0] * 11)['regime'] == 'no_resolution')
    check('high duplicate but enough active stays mixed',
          sign_regime([0.1] * 7 + [-0.1] * 6 + [0.0] * 7)['regime'] == 'mixed')
    check('all identical is no_resolution',
          sign_regime([0.0] * 20)['regime'] == 'no_resolution')
    check('mixed when active fraction is enough',
          sign_regime([0.1] * 10 + [-0.1] * 10)['regime'] == 'mixed')
    check('stable', sign_regime([0.1] * 18 + [-0.1] * 2)['regime'] == 'stable')

    if not ok:
        sys.exit(1)
    print('selftest: 全部通过')


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        _selftest()
    else:
        print(__doc__)
