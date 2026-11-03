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

import statistics
import sys

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
    """多数方向的占比。**注意它在 n_identical 很大时会虚高**——见 unresolved()。

    两臂读数完全相同的种子既不是正也不是负，却留在分母里；30 个种子里 20 个相同、
    剩下 10 个里 9 正 1 负，这个函数给 0.9，看着像"9 成种子同向"。所以引用它
    之前必须先看 `n_identical`。
    """
    tol = _tol(tol)
    c = sign_counts(diffs, tol)
    n = len(diffs)
    if not n:
        return None
    return round(max(c['n_positive'], c['n_negative']) / float(n), 3)


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
    out.update(sign_counts(diffs, tol))
    out['same_sign_fraction'] = same_sign_fraction(diffs, tol)
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

    第三条判读下来才知道，`same_sign_fraction` 在低积压档会**结构性虚高**：
    两臂 `R` 都落在 0 附近，差值在容差内，全部记为 `n_identical`，同号占比看似
    稳定。§13.30 第五批的原话是"`n_identical` 是分辨率刹车"——这里把刹车变成
    判定：超过一半种子两臂读数相同，就不该报方向。
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
    """
    return abs(effect) > abs(band) if strict else abs(effect) >= abs(band)


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

    if not ok:
        sys.exit(1)
    print('selftest: 全部通过')


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        _selftest()
    else:
        print(__doc__)
