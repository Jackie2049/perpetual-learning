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
            "prefix_cost": [sum(s["step_costs"][: s["steps_to_09"] + 1]) for s in ps],
            "final_acc": [s["final_acc"] for s in ps],
        }
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
    return {
        "gen_ratio": gen_r,
        "cost_ratio": cost_r,
        "wall_ratio": wall_r,
        "steps_ratio": steps_r,
        "traj_ratio": traj_r,
        "prefix_ratio": prefix_r,
        "wall_denominators": denominators(arm["walls_to_09"], base["walls_to_09"]),
        "steps_paired": paired(arm["steps_to_09"], base["steps_to_09"]),
        "wall_paired": paired(arm["walls_to_09"], base["walls_to_09"]),
        "acc_paired": paired(arm["final_acc"], base["final_acc"]),
        "gap_from_ratios": gap(wall_r, cost_r),
        "gap_pieces": gap_pieces(wall_r, cost_r, steps_r, prefix_r),
        "wall_bootstrap": bootstrap_ci(arm["walls_to_09"], base["walls_to_09"]),
    }


def build(p3: Dict, p4: Dict) -> Dict:
    rows = {a: arm_row(v, p4["base"]) for a, v in p4.items() if a != "base"}
    p3_pairs = (("C", "B"), ("B", "R_nofilter"), ("A", "R_nofilter"), ("C", "R_nofilter"))
    p3_rows = {f"{a}_vs_{b}": arm_row(p3[a], p3[b]) for a, b in p3_pairs}
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
    }


# ---------------------------------------------------------------- 校验


def _close(a, b, tol=1e-9):
    if a is None or b is None:
        return a is b
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def verify(archive_path: Path, fresh: Dict) -> List[str]:
    old = json.loads(archive_path.read_text())
    problems: List[str] = []
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

    result = build(load_p3(Path(a.p3)), load_p4(Path(a.p4)))

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
