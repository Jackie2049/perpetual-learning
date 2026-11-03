#!/usr/bin/env python3
"""C 类机制账：省下的每步成本，有多少被吃掉。

用途
----
主文档 §18.9 的 7.6 节要一句话论点。现有两批实验（§13.28 DAPO 组过滤、§13.29 四个 C
机制）各给结论表，但没放进同一个算式，且 §13.29 的"到 0.9 墙钟比"列**分母混用**
（见下）。本脚本只读两批归档的 per_seed，重算并给出统一口径的账。

账分三层，缺一层就会读错
----
  L1 单步口径   gen_ratio  = 生成 token 总量比        ← 机制改的是采样量
  L2 每步口径   cost_ratio = 步成本总量比              ← 步成本 = max(生成, 更新)，更新恒 8 单位
  L3 收敛口径   steps_ratio = 到 0.9 的步数总量比      ← 改训练集合会改收敛速度
  L4 墙钟       wall_ratio = 到 0.9 的累计步成本总量比

缺口（L2→L4）：kappa = 1−cost_ratio、r = 1−wall_ratio，
      gap = 1 − (1−r)/(1−kappa) = 1 − x，   x = r/kappa（墙钟省的比例 ÷ 每步省的比例）。

**gap 本身不带收敛方向**（2026-11-02 补强批修订）：x ≥ 0 恒成立 ⇒ gap ≤ 1 恒成立，所以
"gap<0 表示收敛变快"这条读法**在本算式里不成立**——它是 x>1 的另一种写法。收敛方向要看
步数比 s，以及自动算出的 sigma_steps = cost_ratio − wall_ratio：
  sigma_steps > 0  收敛变快（s<1）：每步成本比墙钟比高，差额是步数带来的
  sigma_steps < 0  收敛变慢（s>1）：吃掉每步省下的一部分
三个量分读：
  gap = 0           每步省的完整兑现。**此时 wall_ratio 恰等于 cost_ratio，同样与 s 无关**
  0 < gap < 1       只兑现了一部分，原因看 sigma_steps 的符号
  gap = 1（钉住）   墙钟没省（r≤0）：把 r 钳在 0，只说明"至少全部被吃掉"，不是算出来的数
  gap 无定义        kappa ≤ 0（每步一点没省，或反变慢）必须单独归类，不能给一个数

**gap 与步数比的关系（本批新增，是上面那条误标的根源）**
----
  cost_ratio 与 wall_ratio 都是**累计总量比**，s 已经在里面：
      W_09 = P × s（P = 每步成本之和，s = 到 0.9 步数），C = P × n（n = 总步数）
      ⇒ wall_ratio / cost_ratio ≡ s（比率恒等式，对任意 s 成立；见 gap_pieces 的断言）
  ⇒ x = r/kappa 只在 s≈1 附近才看得见 s；一旦 s 偏离 1，x 主要由 (1+s)−1/s 决定：
      s=0.95 → x≈21.7（读起来像"几乎全部被吃掉"）
      s=1.07 → x≈0.26（读起来像"收敛贡献了一大块"）
  两者都远大于 |s−1|。**gap 不能单独当"收敛代价"引用**：它必须与 s（或 sigma_steps）
  一起报，否则同一个数会被读成两种相反的机制。

分母问题（本脚本存在的直接原因）
----
§13.29 第 3104 行表的"到 0.9 墙钟比"列：freq 0.813 / replay 1.001 / cap 0.937 是
**臂/基线**（种子均值之比），而 cancel 0.660 是**基线/臂**（若按臂/基线应为 1.515）。
同列混用两种分母 ⇒ cancel 的"墙钟比 0.660"与同行"配对 Δ墙钟 −4510（±528）"看起来自洽，
实际是两次取倒数后仍然同号，属巧合。本脚本对四臂统一算三种分母并断言其关系：

    diff_denom_base     = mean(A) − mean(B)          归档"配对 Δ"的口径
    ratio_denom_base    = mean(A) / mean(B)
    ratio_of_totals     = ΣA / ΣB
    ratio_base_over_arm = mean(B) / mean(A)          ← cancel 那一格误用的口径

数据来源（只读，不重新训练）
----
    --p3  experiments/rollout-p3-dapo-group-filter/summary.json
    --p4  experiments/rollout-p4-c-mechanisms/summary.json

用法
----
    python3 ledger.py                       # 重算并写 ledger.json
    python3 ledger.py --verify ledger.json  # 与归档逐字段比对
    python3 -m unittest test_ledger -v      # 单测（含负向对照）
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
UPD_FIXED = 8.0  # 两批实验共用的墙钟模型常量（归档 meta 逐字）


# ---------------------------------------------------------------- 基础统计


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def paired_ci(diffs: List[float], z: float = 1.96) -> float:
    n = len(diffs)
    if n < 2:
        return 0.0
    m = mean(diffs)
    var = sum((d - m) ** 2 for d in diffs) / (n - 1)
    return z * math.sqrt(var / n)


def paired(arm: List[float], base: List[float]) -> Dict[str, float]:
    diffs = [a - b for a, b in zip(arm, base)]
    return {
        "n": len(diffs),
        "mean_diff": mean(diffs),
        "ci95": paired_ci(diffs),
        "n_zero": sum(1 for d in diffs if d == 0),
        "mean_arm": mean(arm),
        "mean_base": mean(base),
    }


def denominators(arm: List[float], base: List[float]) -> Dict[str, float]:
    """同一组配对数据的四种写法，单位为原始单位。"""
    m_a, m_b = mean(arm), mean(base)
    return {
        "diff_denom_base": m_a - m_b,
        "ratio_denom_base": m_a / m_b,
        "ratio_of_totals": sum(arm) / sum(base),
        "ratio_base_over_arm": m_b / m_a,
    }


def gap(ratio_wall: float, ratio_cost: float) -> Dict[str, Optional[float]]:
    """缺口与放大系数。两个比值必须取自同一口径（都用总量或都用种子均值）。"""
    k = 1.0 - ratio_cost
    r = 1.0 - ratio_wall
    if k <= 1e-12:
        # 每步成本一点没省：缺口算式无定义（分母≥1、分子≤1 会让覆盖率>100%）
        return {"kappa": k, "r": r, "gap": None, "amplification": None}
    if r <= 1e-12:
        # 省了每步成本但墙钟没有省/变慢：按"全部被吃掉"记 1，并给出还回去的倍数
        return {"kappa": k, "r": r, "gap": 1.0, "amplification": 1.0 / (1.0 - k)}
    consumed = 1.0 - (1.0 - r) / (1.0 - k)
    return {"kappa": k, "r": r, "gap": consumed, "amplification": consumed / k}


def gap_pieces(
    ratio_wall: float, ratio_cost: float, steps_ratio: float, prefix_ratio: Optional[float] = None
) -> Dict:
    """把 gap 拆成能直接读的两项，并显式标出"gap 不是收敛方向"。

    x = r/kappa 是 gap 的唯一来源（gap = 1−x）。收敛方向要单独从 s 读：
      sigma_steps = cost_ratio − wall_ratio = kappa − r   （带符号；>0 = 收敛变快，s<1）

    归档里两批模拟都满足：walls_to_09 ≡ Σ step_costs[:steps_to_09+1]（逐种子核过 10 个运行）。
    于是 wall_ratio 的分子是"到 0.9 那一步为止的成本"、cost_ratio 的分子是"全部 300 步的成本"，
    两者分母不同 ⇒ 相除是窗口占比之比，混着"S 与 300 的相对位置"与"窗口内每步成本"两个因子；
    它随 s 同向但**不是** s 的函数。故本函数不把 x 当收敛量，只给带符号的 sigma_steps。
    """
    k = 1.0 - ratio_cost
    r = 1.0 - ratio_wall
    sigma = ratio_cost - ratio_wall
    x = (r / k) if k > 1e-12 else None
    return {
        "sigma_steps": sigma,
        "convergence_faster": (sigma > 0) if abs(sigma) > 1e-12 else None,
        "steps_ratio": steps_ratio,
        "prefix_ratio": prefix_ratio,
        "x_ratio_over_cost": x,
    }


def bootstrap_ci(
    arm: List[float], base: List[float], n_boot: int = 2000, seed: int = 12345
) -> Optional[Dict[str, float]]:
    """配对 bootstrap：重采样种子对，给墙钟比与步数比的 95% 分位区间。

    为什么需要它：本批新增的"收敛代价分解"用到步数比与墙钟比的差，而归档只存了总量比。
    配对 bootstrap 用同一批种子对做重采样，给出比值本身的波动范围——若区间跨 1，
    则该比"省没省"本身就不显著，后续分解不必解读。数据不足 3 对时返回 None（不可得就说
    不可得，不能给一个 0 宽的假区间）。
    """
    import random

    n = len(arm)
    if n < 3 or len(base) != n:
        return None
    if sum(base) == 0 or sum(arm) == 0:
        return None
    rng = random.Random(seed)
    idx = list(range(n))
    walls: List[float] = []
    for _ in range(n_boot):
        pick = [rng.choice(idx) for _ in idx]
        den = sum(base[i] for i in pick)
        if den == 0:
            continue
        walls.append(sum(arm[i] for i in pick) / den)
    if not walls:
        return None
    walls.sort()

    def q(p: float) -> float:
        i = min(len(walls) - 1, max(0, int(round(p * (len(walls) - 1)))))
        return walls[i]

    return {
        "n_boot": len(walls),
        "wall_ratio_p025": q(0.025),
        "wall_ratio_p975": q(0.975),
        "ratio_includes_one": q(0.025) <= 1.0 <= q(0.975),
    }


# ---------------------------------------------------------------- 数据装载


def load_p3(path: Path) -> Dict[str, Dict[str, List[float]]]:
    d = json.loads(path.read_text())
    out = {}
    for mode, v in d["modes"].items():
        ps = v["per_seed"]
        out[mode] = {
            "walls_to_09": [s["walls_to_09"] for s in ps],
            "steps_to_09": [s["steps_to_09"] for s in ps],
            "total_cost": [s["total_cost"] for s in ps],
            "prefix_cost": [sum(s["step_costs"][: s["steps_to_09"] + 1]) for s in ps],
            "final_acc": [s["final_acc"] for s in ps],
            "acc_curve": [s["acc_curve"] for s in ps],
            # p3 归档无生成量／轨迹数／每步计数三字段：测不了就不给，也不用别处现算值代替
            # （p3 的轨迹数由模式长度常数决定，不是可观测的量）。
        }
    return out


def load_p4(path: Path) -> Dict[str, Dict[str, List[float]]]:
    d = json.loads(path.read_text())
    out = {}
    for arm, v in d["arms"].items():
        ps = v["per_seed"]
        out[arm] = {
            "walls_to_09": [s["walls_to_09"] for s in ps],
            "steps_to_09": [s["steps_to_09"] for s in ps],
            "total_cost": [s["total_cost"] for s in ps],
            "total_gen": [s["total_gen"] for s in ps],
            "n_traj": [s["total_traj"] for s in ps],
            "mean_counts": [s["mean_counts"] for s in ps],
            "prefix_cost": [sum(s["step_costs"][: s["steps_to_09"] + 1]) for s in ps],
            "final_acc": [s["final_acc"] for s in ps],
            "acc_curve": [s["acc_curve"] for s in ps],
        }
    return out


# ---------------------------------------------------------------- 口径检查
#
# 第三批（2026-11-02）加的四条检查。它们不改任何比值，只回答"这份账自己能不能站住"：
# 一个机制账可以在数字全对的情况下，把不可测的差读成显著、把跟随量读成独立证据。
# 每条都配一个能在单测里真的被抓到的负向对照（见 test_ledger.TestAccountChecks）。


def ceiling_rate(values: List[float], thr: float = 0.9999) -> float:
    """落在评测上限（≈1.0）附近的种子比例。上限饱和时准确率差不可读。"""
    if not values:
        return 0.0
    return sum(1 for v in values if v >= thr) / len(values)


def ceiling_checks(p3: Dict[str, List[float]], p4: Dict[str, List[float]]) -> Dict:
    """① 天花板饱和：准确率是主结论之一（"没观察到显著退化"）的唯一来源。"""
    rows = {}
    for name, arms, base in (("p4", p4, "base"), ("p3", p3, "R_nofilter")):
        for arm, v in arms.items():
            if arm == base:
                continue
            rows[f"{name}:{arm}"] = {
                "base_at_ceiling": ceiling_rate(arms[base]["final_acc"]),
                "arm_at_ceiling": ceiling_rate(v["final_acc"]),
                "acc_paired": paired(v["final_acc"], arms[base]["final_acc"]),
            }
    return rows


def sign_checks(rows: Dict[str, Dict]) -> Dict:
    """② 离散指标（步数、准确率）的配对数：非零对多少、符号是否 30/30 同向。

    连续量（成本、轨迹、生成量）在逐对配对里会有微小的系统性差；步数是整数，
    准确率是 1/800 的网格，两者都能给出"零方差"区间——那种区间不说明显著，
    只说明几乎所有种子打平。
    """
    out = {}
    for name, row in rows.items():
        acc = row["acc_paired"]
        steps = row["steps_paired"]
        out[name] = {
            "acc_n_nonzero": acc["n"] - acc["n_zero"],
            "acc_sign": _uniform_sign([acc["mean_diff"]]),
            "steps_n_nonzero": steps["n"] - steps["n_zero"],
            "steps_sign": _uniform_sign([steps["mean_diff"]]),
        }
    return out


def _uniform_sign(diffs: List[float]) -> Optional[int]:
    """所有差同号则有符号（+1/−1），否则 None。"""
    if not diffs:
        return None
    pos = all(d > 0 for d in diffs)
    neg = all(d < 0 for d in diffs)
    return 1 if pos else (-1 if neg else None)


def collinear_checks(ledger_rows: Dict[str, Dict]) -> Dict:
    """③ 共线量的回退一致性：当 A ≡ B 时，一切 |A−B| 恒为 0。

    p4 的墙钟模型里生成量恒等于步成本（`total_gen == total_cost` 逐种子；
    更新恒 8 单位、非空步的步成本 = max(gen, 8)、而本模型非空步的生成恒 > 8），
    于是 `paired_total_gen` 与 `paired_total_cost` 逐位相同。这不是两条独立证据，
    是一条证据被数了两遍。同理 `cap` 的墙钟差恒为 0（生成与成本都没变）——
    它在墙钟上"省"的 6.3% 只能经收敛路径到达。
    """
    out = {}
    for name, row in ledger_rows.items():
        cost_diff = row["_cost_diff"]
        # 逐对比较：哪些"独立字段"的逐种子差与成本差逐位相同。
        same = {
            "traj": row["_traj_diff"] == cost_diff,
            "gen": row["_gen_diff"] == cost_diff,
            "mean_counts": row["_counts_diff"] == cost_diff,
        }
        out[name] = {
            "gen_equals_cost_total": abs(row["gen_ratio"] - row["cost_ratio"]) < 1e-12
            if row.get("gen_ratio") is not None
            else None,
            "diff_equals_cost_diff": {k: v for k, v in same.items() if v is not None},
            "identity_columns": sorted(
                k for k, v in (("traj_ratio", row["_traj_eq"]), ("gen_ratio", row["_gen_eq"])) if v
            ),
            "wall_independent_of_gen_and_cost": (
                abs(row["wall_ratio"] - 1.0) > 1e-12
                and abs(row["wall_ratio"] - row["cost_ratio"]) > 1e-12
            ),
        }
    return out


def degeneracy_checks(rows: Dict[str, Dict]) -> Dict:
    """④ 独立字段退化：cost_ratio 与 traj_ratio 同时 <1 时，两个字段在这四臂里退化成一个。

    §13.43 的判据把"省每步成本"与"省训练样本"当两条独立读数，但 `freq` 的
    traj_ratio ≡ cost_ratio（都是 0.277）——每份轨迹长度固定，生成量、轨迹数与
    步成本同比例缩放。本模型能分开二者的只有 `cap`（traj 0.282 / cost 1.000）。
    """
    pairs = ("freq", "replay", "cancel", "cap")
    out = {"traj_equals_cost": {}, "separating_arms": [], "degenerate_arms": []}
    for name in pairs:
        row = rows.get(name)
        if not row or row.get("traj_ratio") is None:
            continue
        equal = abs(row["traj_ratio"] - row["cost_ratio"]) < 1e-9
        out["traj_equals_cost"][name] = equal
        if equal:
            out["degenerate_arms"].append(name)
        else:
            out["separating_arms"].append(name)
    return out


def accuracy_resolution(p3_meta: Dict, p4_meta: Dict) -> Dict:
    """准确率的最小可分辨步长：最终评测的轨迹数决定网格。

    p3 的 final_acc 用 n_eval*4 条、p4 用 n_eval 条；网格粗到 1/800 时，
    "分布重度集中在平局"与"有真实差异"在单种子上不可分辨（两个值相差 0.005
    就是网格上相邻的两格）。
    """
    return {
        "p3_eval_trajs_final": p3_meta.get("eval_size", 0) * 4,
        "p4_eval_trajs_final": p4_meta.get("eval_size", 0),
        "p3_grid": 1.0 / max(1, p3_meta.get("eval_size", 0) * 4),
        "p4_grid": 1.0 / max(1, p4_meta.get("eval_size", 0)),
    }


# ---------------------------------------------------------------- 第五批：L3 收敛层的阈值
#
# `steps_ratio` 与 `wall_ratio` 都读"累计成本到某一刻"，而**那个时刻由阈值决定**。
# 第五批把阈值当基线要素之一：同一份账挪一下阈值，读数动多少、谁翻号、到点那一步
# 还剩多少余量。检查只报读数，不改任何比值——判读规则见 ledger.py 顶部与 readme 第五批。


DEFAULT_THRESHOLDS = (0.85, 0.87, 0.9, 0.93, 0.95)
REFERENCE_THRESHOLD = 0.9


def _key(t: float) -> str:
    """阈值在归档里的键名固定两位小数：0.9 → "0.90"，与 0.85／0.95 同形。"""
    return f"{t:.2f}"


def first_cross(curve: List[float], thr: float) -> Optional[int]:
    """首次达到 thr 的步号（0 基）；整条都没达到返回 None。

    未达阈值**不能当 0 步**：那会把"没收敛"读成"收敛极快"。调用方必须先看
    `zero_margin_arms`，再在 `n_miss` 非零时换 `limit_completed` 口径（见 threshold_probe）。
    """
    for i, v in enumerate(curve):
        if v >= thr:
            return i
    return None


def _flatten_after_cross(curve: List[float], thr: float) -> List[float]:
    """仅用于记录错误判据的单测对照；不参与任何读数（见 curve_tail 的说明）。"""
    i = first_cross(curve, thr)
    if i is None:
        return list(curve)
    return list(curve[: i + 1]) + [curve[i]] * (len(curve) - i - 1)


def _cross_or_complete(curve: List[float], thr: float) -> int:
    """未达阈值时按"跑满全程"计（阈值越高，未达标的曲线占比越高，读数随之上移）。"""
    i = first_cross(curve, thr)
    return len(curve) if i is None else i


def steps_at(arm_curves: List[List[float]], base_curves: List[List[float]],
             thr: float, complete_at_end: bool = False) -> Dict:
    """某一阈值下 `steps_ratio` 的按种子配对读数。

    返回的是**比率**（Σ臂步数 / Σ基线步数），与 `arm_row` 的 `steps_ratio` 同口径；
    另附配对中位数与同号比例——只报比率会把"一半种子正、一半负"读成一次干净位移。
    """
    f = _cross_or_complete if complete_at_end else first_cross
    if complete_at_end:
        num = sum(f(c, thr) for c in arm_curves)
        den = sum(f(c, thr) for c in base_curves)
        diffs = [f(a, thr) - f(b, thr) for a, b in zip(arm_curves, base_curves)]
        missing = 0
    else:
        na = [first_cross(c, thr) for c in arm_curves]
        nb = [first_cross(c, thr) for c in base_curves]
        missing = sum(1 for x in (*na, *nb) if x is None)
        num = sum(x for x in na if x is not None)
        den = sum(x for x in nb if x is not None)
        diffs = [a - b for a, b in zip(na, nb) if a is not None and b is not None]
    ratio = (num / den) if den else None
    n = len(diffs)
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    return {
        "steps_ratio": ratio,
        "n_pairs": n,
        "n_miss": missing,
        "same_sign_arm_slower": (pos / n) if n else None,
        "n_ties": sum(1 for d in diffs if d == 0),
        "median_paired_diff": sorted(diffs)[n // 2] if n else None,
    }


def threshold_band(arm_curves: List[List[float]], base_curves: List[List[float]],
                   thresholds=DEFAULT_THRESHOLDS) -> Dict:
    """`steps_ratio` 随阈值移动的摆幅、是否翻号，以及阈值的"余量"。

    摆幅（`span_pp`）是**基线带宽**，不是显著性：它量的是"同一个机制账换个阈值读
    会差多少"，与种子抽样误差无关（后者见 `steps_at` 的 `n_pairs`）。
    """
    rows = {_key(t): steps_at(arm_curves, base_curves, t) for t in thresholds}
    vals = [r["steps_ratio"] for r in rows.values() if r["steps_ratio"] is not None]
    signs = {v > 1 for v in vals}
    # 余量：每个种子恰跨过阈值的两条曲线，到点那一步的读数离阈值还有多远
    margins = []
    for a, b in zip(arm_curves, base_curves):
        for c in (a, b):
            i = first_cross(c, REFERENCE_THRESHOLD)
            if i is not None:
                margins.append(c[i] - REFERENCE_THRESHOLD)
    margins.sort()
    n = len(margins)
    return {
        "by_threshold": rows,
        "span_pp": round(100 * (max(vals) - min(vals)), 3) if vals else None,
        "sign_flip": len(signs) > 1,
        "min_margin": margins[0] if margins else None,
        "median_margin": margins[n // 2] if n else None,
        "max_margin": margins[-1] if margins else None,
        "margin_n": n,
    }


def level_matched_steps(arm_curves: List[List[float]], base_curves: List[List[float]],
                        thresholds=DEFAULT_THRESHOLDS) -> Dict:
    """同一**水平**上的步数差：`steps_ratio` 漂移里有多少是"水平一致、步号换算"。

    `steps_ratio = Σ步数(臂) / Σ步数(基线)` 的关键性质：**分子分母在同一阈值上**，
    所以它比较的从来不是两臂同一个准确率水平——阈值抬高时两臂各自往后挪，而挪多少
    取决于各自曲线的斜率。把差值按阈值逐档报出来、再看同号比例与零重叠种子数，
    就能把"曲线形状不同"与"阈值选择"分开：同号比例在所有档上都一样高 ⇒ 差的符号
    是形状决定的；绝对值随阈值近似单调放大 ⇒ 量级是阈值决定的。
    """
    rows = {}
    for t in thresholds:
        diffs = []
        for a, b in zip(arm_curves, base_curves):
            ia, ib = first_cross(a, t), first_cross(b, t)
            if ia is None or ib is None:
                continue
            diffs.append(ia - ib)
        n = len(diffs)
        rows[_key(t)] = {
            "mean_steps_arm_minus_base": mean(diffs) if n else None,
            "n_pairs": n,
            "arm_slower_frac": (sum(1 for d in diffs if d > 0) / n) if n else None,
            "n_ties": sum(1 for d in diffs if d == 0),
        }
    vals = [r["mean_steps_arm_minus_base"] for r in rows.values()
            if r["mean_steps_arm_minus_base"] is not None]
    return {
        "by_threshold": rows,
        "span_steps": (max(vals) - min(vals)) if vals else None,
        "sign_flip": len({v > 0 for v in vals}) > 1 if vals else None,
    }


def curve_tail(arm_curves: List[List[float]], base_curves: List[List[float]],
               base_thr: float = REFERENCE_THRESHOLD) -> Dict:
    """到点之后的曲线还有多长、走到什么水平——**只做描述，不参与任何计数**。

    第五批原想用"把到点后的曲线压平，摆幅应塌到 0"作负向对照，**判据本身是错的**：
    曲线上任何一个高于阈值的新高点都会把首次跨过时刻往后推，压平到"到点那一步的值"
    等于抹掉此后所有新高，读数必然移动。压平只能在**水平**上取值而不能在**步号**上，
    这不是"读的是别的东西"。本轮记下这件事，用 `accuracy_at` 的换读法代替压平对照。
    """
    lengths, rise = [], []
    for c in arm_curves + base_curves:
        i = first_cross(c, base_thr)
        if i is None:
            continue
        lengths.append(len(c) - i - 1)
        rise.append(max(c[i:]) - c[i])
    lengths.sort()
    rise.sort()
    n = len(lengths)
    return {
        "tail_len_median": lengths[n // 2] if n else None,
        "tail_rise_at_thr_median": rise[n // 2] if n else None,
        "n_curves": n,
    }


def accuracy_at(curve: List[float], step: int, fallback: float) -> float:
    """第 step 步的轨迹累计准确率；超出曲线长度时退回 final_acc（两批曲线长度不一）。"""
    return curve[step] if 0 <= step < len(curve) else fallback


def threshold_transfer(arm_curves: List[List[float]], base_curves: List[List[float]],
                       arm_final: List[float], base_final: List[float],
                       thresholds=DEFAULT_THRESHOLDS) -> Dict:
    """把"0.9 步数"换到曲线上真实的第 k 步去读：各阈值的 `steps_ratio` 能否被同一个步号复现。

    每一步的累计准确率在曲线上只取一个值，所以不同阈值落在不同步号上；如果"到 0.9"
    这条读数的漂移**纯粹来自曲线还在往上爬**，那么每个阈值对应到的步区间的准确率水平
    应当一致。本函数只做这一件事：报每个阈值对应的**两臂到点步区间的准确率中位数**，
    以及它们与阈值本身的距离——距离随阈值单调放大、而准确率区间不随阈值单调移动，
    说明漂移是"阈值挪了、步号跟着挪"，不是"收敛变快了"。
    """
    rows = {}
    for t in thresholds:
        lo_a, hi_a, lo_b, hi_b = [], [], [], []
        for a, b, fa, fb in zip(arm_curves, base_curves, arm_final, base_final):
            ia, ib = first_cross(a, t), first_cross(b, t)
            if ia is None or ib is None:
                continue
            lo_a.append(accuracy_at(a, ia - 1 if ia else 0, fa))
            hi_a.append(accuracy_at(a, ia, fa))
            lo_b.append(accuracy_at(b, ib - 1 if ib else 0, fb))
            hi_b.append(accuracy_at(b, ib, fb))
        med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None
        rows[_key(t)] = {
            "arm_band_acc": [med(lo_a), med(hi_a)],
            "base_band_acc": [med(lo_b), med(hi_b)],
            "n": len(lo_a),
        }
    return rows


def convergence_checks(p3: Dict, p4: Dict, thresholds=DEFAULT_THRESHOLDS) -> Dict:
    """⑤ L3 收敛层的阈值敏感性：`steps_ratio` 的摆幅、符号与余量。

    第五批做的事是把"到点余量"从"读数会不会漂"里分开——两者方向可以相反：
    余量大只说明扫参不改变到点那一步，不说明整列读数稳。四臂与 p3 四对一并报，
    引用 `steps_ratio` 时必须带上阈值（§13.47.1）。
    """
    out = {}
    for name, arms, base in (("p4", p4, "base"), ("p3", p3, "R_nofilter")):
        for arm, v in arms.items():
            if arm == base:
                continue
            a, b = v["acc_curve"], arms[base]["acc_curve"]
            key = arm if name == "p4" else f"{arm}_vs_{base}"
            band = threshold_band(a, b, thresholds)
            band["level_matched"] = level_matched_steps(a, b, thresholds)
            band["tail"] = curve_tail(a, b)
            band["band_accuracy"] = threshold_transfer(
                a, b, v["final_acc"], arms[base]["final_acc"], thresholds
            )
            out[f"{name}:{key}"] = band
    return out


# ---------------------------------------------------------------- 主计算


def arm_row(arm: Dict[str, List[float]], base: Dict[str, List[float]]) -> Dict:
    # p3 归档不存生成量（只有步成本），故 gen_ratio 在两批之间不统一，缺则记 None。
    gen_r = (
        sum(arm["total_gen"]) / sum(base["total_gen"])
        if "total_gen" in arm and "total_gen" in base
        else None
    )
    cost_r = sum(arm["total_cost"]) / sum(base["total_cost"])
    wall_r = sum(arm["walls_to_09"]) / sum(base["walls_to_09"])
    steps_r = sum(arm["steps_to_09"]) / sum(base["steps_to_09"])
    # 轨迹总数比：§13.43 初版把 cancel 的 34% 说成"收敛造成"，其实是它砍掉了每轮固定
    # 业务成本（traj 少 30%），同一批步骤推进得快。这个比值是区分两者的直接证据。
    traj_r = (
        sum(arm["n_traj"]) / sum(base["n_traj"])
        if "n_traj" in arm and "n_traj" in base
        else None
    )
    # 窗口成本占比比：walls_to_09 = Σ step_costs[:steps_to_09+1]，total_cost = Σ 全 300 步
    # ⇒ wall_ratio / cost_ratio 恰好等于这个比值（对任意 s 成立，见 gap_pieces 的 identity 校验）。
    prefix_r = (
        sum(arm["prefix_cost"]) / sum(base["prefix_cost"])
        if "prefix_cost" in arm and "prefix_cost" in base
        else None
    )
    # 逐种子差值只在内部用于口径检查，不进归档（归档字段与上一版保持可比）。
    cost_diff = [a - b for a, b in zip(arm["total_cost"], base["total_cost"])]

    def _diff(key):
        if key not in arm or key not in base:
            return None
        return [a - b for a, b in zip(arm[key], base[key])]

    return {
        "_cost_diff": cost_diff,
        "_traj_diff": _diff("n_traj"),
        "_gen_diff": _diff("total_gen"),
        "_counts_diff": _diff("mean_counts"),
        "_traj_eq": traj_r is not None and abs(traj_r - cost_r) < 1e-9,
        "_gen_eq": gen_r is not None and abs(gen_r - cost_r) < 1e-9,
        "gen_ratio": gen_r,
        "cost_ratio": cost_r,
        "wall_ratio": wall_r,
        "steps_ratio": steps_r,
        "traj_ratio": traj_r,
        "prefix_ratio": prefix_r,
        "gen_paired": paired(arm["total_gen"], base["total_gen"])
        if gen_r is not None
        else None,
        "wall_denominators": denominators(arm["walls_to_09"], base["walls_to_09"]),
        "steps_paired": paired(arm["steps_to_09"], base["steps_to_09"]),
        "wall_paired": paired(arm["walls_to_09"], base["walls_to_09"]),
        "acc_paired": paired(arm["final_acc"], base["final_acc"]),
        "gap_from_ratios": gap(wall_r, cost_r),
        "gap_pieces": gap_pieces(wall_r, cost_r, steps_r, prefix_r),
        "wall_bootstrap": bootstrap_ci(arm["walls_to_09"], base["walls_to_09"]),
        "acc_at_ceiling": ceiling_rate(arm["final_acc"]),
        "base_acc_at_ceiling": ceiling_rate(base["final_acc"]),
    }


def build(p3: Dict, p4: Dict, meta: Optional[Dict] = None) -> Dict:
    rows = {a: arm_row(v, p4["base"]) for a, v in p4.items() if a != "base"}
    p3_pairs = (("C", "B"), ("B", "R_nofilter"), ("A", "R_nofilter"), ("C", "R_nofilter"))
    p3_rows = {f"{a}_vs_{b}": arm_row(p3[a], p3[b]) for a, b in p3_pairs}
    meta = meta or {}
    # 第三批：口径检查（在剔除内部字段之前做，它们要读逐种子差）
    account = {
        "ceiling": ceiling_checks(p3, p4),
        "signs": sign_checks({**rows, **p3_rows}),
        "collinear": collinear_checks(rows),
        "degeneracy": degeneracy_checks(rows),
        "accuracy_grid": accuracy_resolution(meta.get("p3_meta", {}), meta.get("p4_meta", {})),
        "threshold_band": convergence_checks(p3, p4),
    }
    for row in (*rows.values(), *p3_rows.values()):
        for k in [k for k in row if k.startswith("_")]:
            row.pop(k)
    return {
        "meta": {
            "experiment": "C 类机制账：省下的每步成本有多少被吃掉（分层口径＋分母核验）",
            "update_cost_fixed": UPD_FIXED,
            "pass_cost_model": "步成本 = max(生成, 更新)；更新恒 8 单位 ⇒ 更新吃掉省额是下界",
            "sources": {
                "p3": "experiments/rollout-p3-dapo-group-filter/summary.json",
                "p4": "experiments/rollout-p4-c-mechanisms/summary.json",
            },
            "caveat": "全部数字来自两批进程内模拟归档，非 GPU、非端到端；不可与论文报值相除",
        },
        "p4_arms": rows,
        "p3_pairs": p3_rows,
        # 只回答"这份账自己能不能站住"，不改上面任何比值。
        "account_checks": account,
    }


# ---------------------------------------------------------------- 校验


def _close(a, b, tol=1e-9):
    if a is None or b is None:
        return a is b
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


CHECK_SECTIONS = ("account_checks",)


def _deep_diff(prefix: str, want, got, problems: List[str]) -> None:
    """递归比对两层以上嵌套（bool 用同一性，数值用相对容差）。"""
    if isinstance(want, dict):
        if not isinstance(got, dict):
            problems.append(f"{prefix}: 归档缺子表")
            return
        for k, v in want.items():
            _deep_diff(f"{prefix}.{k}", v, got.get(k), problems)
    elif isinstance(want, list):
        if not isinstance(got, list) or len(want) != len(got):
            problems.append(f"{prefix}: 归档 {got} ≠ 重算 {want}")
            return
        for i, (a, b) in enumerate(zip(want, got)):
            _deep_diff(f"{prefix}[{i}]", a, b, problems)
    elif isinstance(want, bool) or isinstance(got, bool):
        if want is not got:
            problems.append(f"{prefix}: 归档 {got} ≠ 重算 {want}")
    elif not _close(want, got):
        problems.append(f"{prefix}: 归档 {got} ≠ 重算 {want}")


def verify(archive_path: Path, fresh: Dict) -> List[str]:
    old = json.loads(archive_path.read_text())
    problems: List[str] = []
    for section in CHECK_SECTIONS:
        if section not in old:
            problems.append(f"归档缺 section {section}")
            continue
        _deep_diff(section, fresh[section], old[section], problems)
    for section in ("p4_arms", "p3_pairs"):
        if section not in old:
            problems.append(f"归档缺 section {section}")
            continue
        for key, row in fresh[section].items():
            if key not in old[section]:
                problems.append(f"归档缺 {section}.{key}")
                continue
            for field, val in row.items():
                if isinstance(val, dict):
                    got_obj = old[section][key].get(field)
                    if not isinstance(got_obj, dict):
                        problems.append(f"{section}.{key}.{field}: 归档缺子表")
                        continue
                    for f2, v2 in val.items():
                        got = got_obj.get(f2)
                        if isinstance(v2, bool) or isinstance(got, bool):
                            if v2 is not got:
                                problems.append(
                                    f"{section}.{key}.{field}.{f2}: 归档 {got} ≠ 重算 {v2}"
                                )
                            continue
                        if not _close(v2, got):
                            problems.append(
                                f"{section}.{key}.{field}.{f2}: 归档 {got} ≠ 重算 {v2}"
                            )
                else:
                    got = old[section][key].get(field)
                    if not _close(val, got):
                        problems.append(f"{section}.{key}.{field}: 归档 {got} ≠ 重算 {val}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p3", default=str(HERE.parent / "rollout-p3-dapo-group-filter" / "summary.json"))
    ap.add_argument("--p4", default=str(HERE.parent / "rollout-p4-c-mechanisms" / "summary.json"))
    ap.add_argument("--out", default=str(HERE / "ledger.json"))
    ap.add_argument("--verify", metavar="ARCHIVE", default=None)
    a = ap.parse_args()

    result = build(
        load_p3(Path(a.p3)),
        load_p4(Path(a.p4)),
        {
            "p3_meta": json.loads(Path(a.p3).read_text())["meta"],
            "p4_meta": json.loads(Path(a.p4).read_text())["meta"],
        },
    )

    if a.verify:
        problems = verify(Path(a.verify), result)
        if problems:
            print("VERIFY FAILED:")
            for p in problems:
                print("  -", p)
            return 1
        print("VERIFY OK: 归档与重算逐字段一致")
        return 0

    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for arm, r in result["p4_arms"].items():
        g = r["gap_from_ratios"]
        p = r["gap_pieces"]
        print(
            f"{arm:7s} gen {r['gen_ratio']:.3f} traj {(r['traj_ratio'] or float('nan')):.3f} "
            f"cost {r['cost_ratio']:.3f} steps {r['steps_ratio']:.3f} wall {r['wall_ratio']:.3f} | "
            f"kappa {g['kappa']:+.4f} r {g['r']:+.4f} "
            f"gap {g['gap'] if g['gap'] is None else round(g['gap'], 4)} | "
            f"prefix_ratio {(p['prefix_ratio'] or float('nan')):.3f} "
            f"sigma_steps {p['sigma_steps']:+.4f}"
        )
        assert abs(p["prefix_ratio"] - r["wall_ratio"]) < 1e-9, f"{arm}: prefix_ratio ≠ wall_ratio"
    for pair, r in result["p3_pairs"].items():
        g = r["gap_from_ratios"]
        p = r["gap_pieces"]
        print(
            f"{pair:16s} cost {r['cost_ratio']:.3f} steps {r['steps_ratio']:.3f} "
            f"wall {r['wall_ratio']:.3f} | gap "
            f"{g['gap'] if g['gap'] is None else round(g['gap'], 4)} | "
            f"sigma_steps {p['sigma_steps']:+.4f} "
            f"(convergence {'faster' if p['convergence_faster'] else 'slower'})"
        )
    checks = result["account_checks"]
    print("\n口径检查（第三批）")
    # 共线假设是本批归类的前提：若生成量不再恒等于步成本，下面所有"×_ratio 是独立读数"
    # 的说法都要重做。断言放在打印前，失效就立刻停。
    for name, c in checks["collinear"].items():
        assert c["gen_equals_cost_total"], f"{name}: 生成量与步成本不再相等，共线假设失效"
    for name, c in checks["ceiling"].items():
        print(
            f"  ceiling {name:18s} base {c['base_at_ceiling']:.0%} arm {c['arm_at_ceiling']:.0%} "
            f"Δacc {c['acc_paired']['mean_diff']:+.4f} ± {c['acc_paired']['ci95']:.4f} "
            f"(非零对 {c['acc_paired']['n'] - c['acc_paired']['n_zero']}/{c['acc_paired']['n']})"
        )
    for name, c in checks["collinear"].items():
        same = ",".join(k for k, v in c["diff_equals_cost_diff"].items() if v)
        print(
            f"  collinear {name:7s} 恒等列 {c['identity_columns']} 逐对同差 [{same}] "
            f"独立墙钟 {c['wall_independent_of_gen_and_cost']}"
        )
    dg = checks["degeneracy"]
    print(f"  degeneracy traj≡cost: {dg['traj_equals_cost']} 可分开的臂 {dg['separating_arms']}")
    print(f"  accuracy_grid {checks['accuracy_grid']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
