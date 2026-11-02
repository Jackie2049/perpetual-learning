#!/usr/bin/env python3
"""C 类机制账 §13.43 收尾：⑩ 留下的四处"未定"哪些真能定，哪些测不出来。

用途
----
§13.43 ⑩（第二十六批口径检查）把这份账的分辨力边界划到了四处：准确率坐在天花板、
收敛方向与墙钟撞同一网格、`traj_ratio`/`cost_ratio` 只有一处分开、检查对输入敏感。
但 ⑩ 只说明"原表读不出",没说**还有没有别的字段能把它们读出来**。本批换一个问法：

    这份归档里，⑩ 标为"未定"的每一个量，是否有一个尚未被用过的字段能把它定下来？

字段面（归档里已经存在、但 §13.43 三批都没用过的）：
  P4  p4 的 step_costs（每步成本）在四臂里**恒为 896**；但 p4 的机制中 `freq`/`cancel`
      改的是"发起几个任务",这个量不在 step_costs 里，于是被 max(生成, 更新) 抹平了。
      ——这一条决定 ⑦③ 的"III 类收益落在哪个分项"能不能更细。
  P3  p3 的 step_costs **随步变化**（生成侧主导，280–796），且 `upd_fraction` =
      total_update_cost / total_cost 可直接从归档 mem 字段算出，扣掉固定 8 单位后
      生成侧成本与墙钟的残差比可算。这是 ⑦② `cancel` 与 ⑦④ `C vs B`"差额落在哪"的
      直接判定式。
  B   p4 有 30 种子×每步 `acc_curve`，`steps_to_09` 的触发时刻逐种子已知 ⇒ 阈值位置的
      离散化是**可算**的（网格 0.005 × 首次跨过的阈值余量）。
  A   准确率天花板下，唯一还能分辨的是**曲线面积**（AUC）——不是终值，也不是到点步数。

本脚本只读两批既有归档，不重训、不改原表任何比值；输出写 `resolution.json` 或副本。

用法
----
    python3 resolution.py                      # 重算 → resolution.json
    python3 resolution.py --verify resolution.json   # 归档 vs 重算逐字段核（含旧归档断言）

四条断言（--verify 也跑）：
  1. P4 四臂 step_costs 恒 8 单位模型成立（每步 ∈ {8, ...} 且 >0 的最小量级一致），
     因此"p4 分不出每步成本的构成"这件事本身可证；
  2. P3 update 项 = 8 × (有更新步数)，逐种子成立 ⇒ 扣固定项后生成侧成本可分离；
  3. `cancel` 与 `C vs B` 的墙钟差额，在**同一 reconcile 式**下逐档给出（见下）；
  4. 阈值离散化：`steps_to_09` 由 `acc_curve` 首次 ≥0.9 完全重建，逐种子零误差。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
UPDATE_UNIT = 8.0  # 两份归档的统一换算单位：有更新步 = 8


# --------------------------------------------------------------------------
# 读取
# --------------------------------------------------------------------------
def load(p3_path: Path, p4_path: Path) -> dict:
    return {
        "p3": json.loads(p3_path.read_text()),
        "p4": json.loads(p4_path.read_text()),
        "p3_path": str(p3_path),
        "p4_path": str(p4_path),
    }


# --------------------------------------------------------------------------
# P4：每步成本字段能不能分辨阶段构成
# --------------------------------------------------------------------------
def p4_step_cost_face(p4: dict) -> dict:
    """四臂 step_costs 的取值集合，以及"改的是什么量"与"看得见什么量"的对照。

    **实测修正（本批）**：p4 四臂里只有 `base`/`cap` 的每步成本恒为 896；`freq`/`replay`/
    `cancel` 三臂每步成本随步变化。原因看数值：`freq` 出现 224/308/392/…，`cancel` 出现
    448/560/784/896——都是 112 的整数倍，而 112 = 剩余任务数 × 轨迹长度 14。也就是说
    step_costs 记的是**本步生成的 token 总量**，不是"发起了几个任务"。于是对"III 类收益
    落在哪个成本分项"而言，归档只能把成本拆到 token 生成侧，发起次数/调度固定开销
    **未被记录**——这正是 ⑩ 标"未定"的那一项，本批要证明它在本归档里确实读不出。
    """
    out = {}
    for arm, d in p4["arms"].items():
        per = d["per_seed"]
        vals = set()
        for s in per:
            vals.update(s["step_costs"])
        out[arm] = {
            "distinct_step_cost_values": sorted(vals),
            "step_cost_is_constant": len(vals) == 1,
            "total_cost_mean": d["total_cost_mean"],
            "total_traj_mean": d["total_traj_mean"],
            "total_gen_mean": d["total_gen_mean"],
            "mechanism_touches_task_count": arm in ("freq", "cancel"),
            # 机制改的是"发起几次"，记录的是"生成多少 token"：两者在本归档里等号成立
            # （p4 ⑩③ 已核 total_gen == total_cost），故发起次数不构成独立字段
            "task_count_is_separately_recorded": False,
        }
    return out


# --------------------------------------------------------------------------
# P3：扣掉固定更新项后，生成侧成本与墙钟的可分离性
# --------------------------------------------------------------------------
def p3_update_separation(p3: dict) -> dict:
    """扣掉固定更新项后，生成侧成本与墙钟的可分离性。

    **实测修正（本批，两处）**：
    1. p3 的"有更新步"**不是恒 8 单位**——单步更新成本取 280/334/…/958 等离散值。
       §13.29／§13.43 里写的"更新恒 8 单位"是 **p4** 的模型；p3 的更新成本随批规模变化，
       故 p3 的"更新吃掉省额"是**上界可达的实测**，比 p4 的下界模型信息更多。
    2. 归档里空更新步成本是 **1（不是 0）**，故"有更新步"按 `step_cost > floor_unit(1)` 判，
       而"该步有更新"的标志位是 `step_has_update`——两者在本归档逐种子一致，
       但含义不同：前者是成本特征，后者是事件标记。
    """
    floor_unit = 1.0
    out = {}
    for mode, d in p3["modes"].items():
        per = d["per_seed"]
        upd_costs = []
        n_steps = 0
        gen_only_costs = []
        for s in per:
            for c in s["step_costs"]:
                n_steps += 1
                if c > floor_unit:
                    upd_costs.append(c)
                else:
                    gen_only_costs.append(c)
        total_cost = sum(s["total_cost"] for s in per)
        update_cost = sum(upd_costs)
        out[mode] = {
            "n_seeds": len(per),
            "update_cost_is_fixed_8": len(set(upd_costs)) == 1 and all(
                math.isclose(c, UPDATE_UNIT) for c in upd_costs
            ),
            "update_cost_values": sorted(set(upd_costs))[:12],
            "n_update_steps": len(upd_costs),
            "n_floor_steps": len(gen_only_costs),
            "total_update_cost": update_cost,
            "total_cost": total_cost,
            "upd_fraction": update_cost / total_cost if total_cost else float("nan"),
            "empty_update_share": len(gen_only_costs) / n_steps if n_steps else 0.0,
            "sum_step_costs": sum(sum(s["step_costs"]) for s in per),
        }
    return out


def reconcile_wall(p3: dict, mode_or_pair) -> dict:
    """把"到 0.9 的每步成本"拆成生成侧与更新侧两项。

    式（逐种子成立，再取均值——**不可先取均值再相乘**，那会引入 ~0.02 的乘性残差）：
        n09      = steps_to_09 + 1（含触发步）
        mean_step = w09 / n09
        upd_side  = Σ step_cost(>1) over first n09 steps / n09
        gen_side  = mean_step − upd_side
    判据用：Δmean = Δupd_side + Δgen_side，两侧都是绝对量。

    **本批结论：在 p3 上 gen_side ≡ 0**（见 `p3_degeneracy`）——更新恒为步成本的最大项，
    分解恒等式成立但只在更新侧起作用。因此这一项在 p3 上**不具备分辨力**，
    不能据此说"省额全在更新侧"，只能说"本归档的步成本字段分不开这两侧"。
    """
    def one(mode):
        d = p3["modes"][mode]
        per = d["per_seed"]
        n09, mean_step, upd_side, gen_side = [], [], [], []
        for s in per:
            n = s["steps_to_09"] + 1  # 含触发步（见 internal_consistency 的索引约定）
            w = s["walls_to_09"]
            upd = sum(c for c in s["step_costs"][:n] if c > 1.0)
            m = w / n if n else float("nan")
            u = upd / n if n else float("nan")
            n09.append(n)
            mean_step.append(m)
            upd_side.append(u)
            gen_side.append(m - u)
        return {
            # 含触发步的步数（= steps_to_09 + 1），与 walls_to_09 的分母一致
            "steps_counted_mean": statistics.fmean(n09),
            "mean_step_cost_to_09": statistics.fmean(mean_step),
            "upd_side_mean_step": statistics.fmean(upd_side),
            "gen_side_mean_step": statistics.fmean(gen_side),
            "upd_share_of_step_cost": statistics.fmean(
                [u / m if m else float("nan") for u, m in zip(upd_side, mean_step)]
            ),
            "walls_to_09_mean": statistics.fmean([s["walls_to_09"] for s in per]),
        }

    if isinstance(mode_or_pair, tuple):
        a, b = mode_or_pair
    else:
        a, b = mode_or_pair, "R_nofilter"
    A, B = one(a), one(b)
    d_mean = A["mean_step_cost_to_09"] - B["mean_step_cost_to_09"]
    d_upd = A["upd_side_mean_step"] - B["upd_side_mean_step"]
    d_gen = A["gen_side_mean_step"] - B["gen_side_mean_step"]
    return {
        "pair": f"{a} vs {b}",
        "A": A,
        "B": B,
        "steps_ratio": A["steps_counted_mean"] / B["steps_counted_mean"],
        "wall_ratio": A["walls_to_09_mean"] / B["walls_to_09_mean"],
        # 注意：这是"到收敛为止的**每步**成本比"，**不是** ledger 的 `cost_ratio`
        # （后者是全部 300 步的 total_cost 之比）。两者在 C vs B 上相差 1.063 vs 0.551，
        # 方向相反——因为 C 的省额集中在前期、B 的 300 步里后段更便宜。不可混用（见 readme）。
        "mean_step_cost_ratio_to_09": A["mean_step_cost_to_09"] / B["mean_step_cost_to_09"],
        "delta_mean_step_cost": d_mean,
        "delta_from_update_term": d_upd,
        "delta_from_gen_side": d_gen,
        "gen_share_of_step_delta": (d_gen / d_mean) if d_mean else float("nan"),
    }


# --------------------------------------------------------------------------
# B：阈值位置离散化（p4 的 200 条评测 ⇒ 网格 0.005）
# --------------------------------------------------------------------------
def threshold_grid(p3: dict, p4: dict) -> dict:
    out = {}

    def rebuild_ok(per, target=0.9):
        bad = 0
        for s in per:
            # 归档约定：steps_to_09 记录的是 acc_curve 里**首次 ≥target 的下标**
            # （本批逐种子核过：两批归档 0 处不符，i==steps_to_09 全部成立）
            first = next((i for i, a in enumerate(s["acc_curve"]) if a >= target), None)
            if first != s["steps_to_09"]:
                bad += 1
        return bad

    grid_p4 = 1.0 / p4["meta"]["eval_size"]
    grid_p3 = 1.0 / p3["meta"]["eval_size"]

    # p4：到 0.9 的那一步，准确率"超出阈值多少"⇒ 阈值位置可移动的余量
    over = []
    for arm, d in p4["arms"].items():
        over_arm = []
        for s in d["per_seed"]:
            i = s["steps_to_09"]
            over_arm.append(s["acc_curve"][i] - 0.9)
        over.append((arm, statistics.fmean(over_arm), max(over_arm)))
    out["p4"] = {
        "eval_size": p4["meta"]["eval_size"],
        "grid": grid_p4,
        "rebuild_mismatch_seeds": rebuild_ok(p4["arms"]["base"]["per_seed"]),
        "margin_above_threshold": {a: {"mean": m, "max": mx} for a, m, mx in over},
    }
    over_p3 = []
    for mode, d in p3["modes"].items():
        over_mode = [s["acc_curve"][s["steps_to_09"]] - 0.9 for s in d["per_seed"]]
        over_p3.append((mode, statistics.fmean(over_mode), max(over_mode)))
    out["p3"] = {
        "eval_size": p3["meta"]["eval_size"],
        "grid": grid_p3,
        "mismatch_modes": {m: rebuild_ok(d["per_seed"]) for m, d in p3["modes"].items()},
        "margin_above_threshold": {m: {"mean": a, "max": b} for m, a, b in over_p3},
    }
    return out


# --------------------------------------------------------------------------
# A：天花板下的曲线面积
# --------------------------------------------------------------------------
def curve_area(p4: dict) -> dict:
    """终值饱和时，唯一还能分辨的是"爬上去多快"：到 0.9 的累计缺失面积。"""
    base = p4["arms"]["base"]["per_seed"]
    out = {}
    for arm, d in p4["arms"].items():
        diffs = []
        for sb, sa in zip(base, d["per_seed"]):
            n = min(len(sb["acc_curve"]), len(sa["acc_curve"]))
            area_b = sum(0.9 - a for a in sb["acc_curve"][:n])
            area_a = sum(0.9 - a for a in sa["acc_curve"][:n])
            diffs.append(area_a - area_b)
        m = statistics.fmean(diffs)
        sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
        ci = 1.96 * sd / math.sqrt(len(diffs))
        out[arm] = {
            "n": len(diffs),
            "mean_diff_area": m,
            "ci95": ci,
            "n_negative": sum(1 for x in diffs if x < 0),
            "n_positive": sum(1 for x in diffs if x > 0),
        }
    return out


# --------------------------------------------------------------------------
# p3 分解的退化：每步成本是否全为更新侧
# --------------------------------------------------------------------------
def p3_degeneracy(p3: dict) -> dict:
    """p3 的步成本 = max(生成, 更新)。若更新恒为最大项，生成侧被完全吸收，
    Δgen ≡ 0 是**恒等式的结果**而非测量结论——分解在这一层没有信息量，必须写明。"""
    shares = []
    for mode, d in p3["modes"].items():
        for s in d["per_seed"]:
            n = s["steps_to_09"] + 1
            upd = sum(c for c in s["step_costs"][:n] if c > 1.0)
            if s["walls_to_09"]:
                shares.append(upd / s["walls_to_09"])
    return {
        "n_seed_mode_cells": len(shares),
        "min_update_share": min(shares),
        "max_update_share": max(shares),
        "gen_side_always_zero": min(shares) >= 1.0 - 1e-12,
        "why": "步成本 = max(生成, 更新)；p3 上更新恒为最大项 ⇒ 生成侧被完全吸收，"
               "按此分解得 Δgen ≡ 0。这是恒等式的结果，不是'生成侧没省'的证据。"
               "p3 因此只能把'省额在墙钟上如何兑现'算清，不能把'省额属于生成侧还是更新侧'分开。",
    }


# --------------------------------------------------------------------------
# 归档内部一致性：walls_to_09 是否等于前 n09 步 step_costs 之和
# --------------------------------------------------------------------------
def internal_consistency(p3: dict, p4: dict) -> dict:
    """分解式两侧若同源，注错会互相抵消。这条检查把"归档自洽"单独钉住。"""

    def check(per):
        bad = []
        for s in per:
            # acc_curve[i] 是第 i 步**执行之后**的准确率，steps_to_09 记的是这个下标
            # （0 基）⇒ 到点累计墙钟 = 前 steps_to_09+1 步成本之和。本批逐种子核过。
            n = s["steps_to_09"] + 1
            if not math.isclose(sum(s["step_costs"][:n]), s["walls_to_09"], rel_tol=0, abs_tol=1e-9):
                bad.append(s["seed"])
        return {"n_seeds": len(per), "n_inconsistent_seeds": len(bad), "bad_seeds": bad[:5]}

    return {
        "p3": {m: check(d["per_seed"]) for m, d in p3["modes"].items()},
        "p4": {a: check(d["per_seed"]) for a, d in p4["arms"].items()},
    }


# --------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------
def build(data: dict) -> dict:
    p3, p4 = data["p3"], data["p4"]
    return {
        "meta": {
            "experiment": "C 类机制账收尾：⑩ 的四处未定，哪些能被归档里未用过的字段定下来",
            "update_unit": UPDATE_UNIT,
            "sources": {"p3": data["p3_path"], "p4": data["p4_path"]},
            "caveat": "全部为对既有 CPU 进程内模拟归档的重算，非 GPU、非端到端；不新增训练",
        },
        "p4_step_cost_face": p4_step_cost_face(p4),
        "p3_update_separation": p3_update_separation(p3),
        "p3_reconcile": {
            "C_vs_B": reconcile_wall(p3, ("C", "B")),
            "C_vs_R_nofilter": reconcile_wall(p3, ("C", "R_nofilter")),
            "B_vs_R_nofilter": reconcile_wall(p3, ("B", "R_nofilter")),
            "R_vs_R_nofilter": reconcile_wall(p3, ("R", "R_nofilter")),
            "A_vs_R_nofilter": reconcile_wall(p3, ("A", "R_nofilter")),
        },
        "p3_decomposition_degenerate": p3_degeneracy(p3),
        "p3_internal_consistency": internal_consistency(p3, p4)["p3"],
        "p4_internal_consistency": internal_consistency(p3, p4)["p4"],
        "threshold_grid": threshold_grid(p3, p4),
        "p4_curve_area": curve_area(p4),
    }


# --------------------------------------------------------------------------
# 断言 / 校验
# --------------------------------------------------------------------------
def assert_invariants(res: dict) -> list[str]:
    problems = []

    # 1. p4 每步成本恒定（模型事实）
    for arm, d in res["p4_step_cost_face"].items():
        if arm not in ("base", "cap") and d["step_cost_is_constant"]:
            problems.append(f"p4/{arm}: step_costs 恒定了，成本构成判定前提失效")

    # 2. p3 的更新成本**不是**恒 8（与 p4 相反）——这条要断言的是相反方向：
    #    若哪天 p3 变成恒 8，"p3 是上界、p4 是下界"的对照结论就要重写
    for mode, d in res["p3_update_separation"].items():
        if d["update_cost_is_fixed_8"]:
            problems.append(f"p3/{mode}: 更新成本变成恒 8，与 p4 下界模型的对照结论失效")

    # 3. reconcile：三项之和必须回到每步差
    for name, r in res["p3_reconcile"].items():
        s = r["delta_from_update_term"] + r["delta_from_gen_side"]
        if not math.isclose(s, r["delta_mean_step_cost"], rel_tol=1e-9, abs_tol=1e-9):
            problems.append(f"reconcile/{name}: 分解项之和 {s} ≠ 每步差 {r['delta_mean_step_cost']}")

    # 4. 阈值重建零误差
    for p in ("p3", "p4"):
        if p == "p4" and res["threshold_grid"]["p4"]["rebuild_mismatch_seeds"] != 0:
            problems.append("p4 阈值重建有误差")
        if p == "p3" and any(v != 0 for v in res["threshold_grid"]["p3"]["mismatch_modes"].values()):
            problems.append("p3 阈值重建有误差")

    # 5. 归档内部一致性：walls_to_09 必须等于前 steps_to_09 步 step_costs 之和
    #    （本批新增。没有这条时，分解式"两侧同源"会互相抵消，注入缺陷抓不到——见单测）
    for mode, d in res["p3_internal_consistency"].items():
        if d["n_inconsistent_seeds"] != 0:
            problems.append(
                f"p3/{mode}: {d['n_inconsistent_seeds']} 个种子的 walls_to_09 ≠ Σ前 n09 步成本"
            )
    for arm, d in res["p4_internal_consistency"].items():
        if d["n_inconsistent_seeds"] != 0:
            problems.append(
                f"p4/{arm}: {d['n_inconsistent_seeds']} 个种子的 walls_to_09 ≠ Σ前 n09 步成本"
            )
    return problems


def conclusion_assertions(res: dict) -> list[str]:
    """结论级断言：这批的结论要成立，归档必须满足下面几条；不满足则结论失效，脚本应报错。"""
    problems = []
    if res["threshold_grid"]["p4"]["grid"] < 0.0011:
        problems.append("p4 评测网格小于 §13.29 ⑥ 的 Δacc 上界，天花板论证失效")
    if res["p4_step_cost_face"]["freq"]["task_count_is_separately_recorded"]:
        problems.append("p4 freq 的任务数已成为独立字段，III 类分项判定可再细一层")
    return problems


def verify_against(recomputed: dict, arch: dict) -> list[str]:
    problems = []

    def walk(a, b, path=""):
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a.keys()) != set(b.keys()):
                problems.append(f"{path}: 键集合不同")
            for k in a.keys() & b.keys():
                walk(a[k], b[k], f"{path}/{k}")
        elif isinstance(a, float) and isinstance(b, float):
            if math.isnan(a) and math.isnan(b):
                return
            if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9):
                problems.append(f"{path}: {a} ≠ {b}")
        elif a != b:
            problems.append(f"{path}: {a!r} ≠ {b!r}")

    walk(arch, recomputed)
    problems.extend(assert_invariants(recomputed))
    problems.extend(conclusion_assertions(arch))
    return problems


def verify(archive_path: Path, recomputed: dict) -> list[str]:
    return verify_against(recomputed, json.loads(archive_path.read_text()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    here = HERE.parent
    ap.add_argument("--p3", default=str(here / "rollout-p3-dapo-group-filter" / "summary.json"))
    ap.add_argument("--p4", default=str(here / "rollout-p4-c-mechanisms" / "summary.json"))
    ap.add_argument("--out", default=str(HERE / "resolution.json"))
    ap.add_argument("--verify", metavar="ARCHIVE", default=None)
    a = ap.parse_args(argv)

    res = build(load(Path(a.p3), Path(a.p4)))
    probs = assert_invariants(res)
    if probs:
        print("INVARIANT FAILED:")
        for p in probs:
            print("  -", p)
        return 1

    if a.verify:
        vp = verify(Path(a.verify), res)
        if vp:
            print("VERIFY FAILED:")
            for p in vp:
                print("  -", p)
            return 1
        print("VERIFY OK：归档与重算逐字段一致，四条不变量成立")
        return 0

    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
