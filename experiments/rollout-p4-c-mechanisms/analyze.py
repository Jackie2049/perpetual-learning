#!/usr/bin/env python3
"""P-4 延伸的离线分析：把归档的 per_seed 曲线换算成「净墙钟」对照。

为什么需要它
-----------
四种 C 类机制都省生成量，但都改变训练集合，因而收敛变慢。只看"总成本"或只看
"到 0.9 的墙钟"都会给出片面答案：

  - 只看总成本  → 机制看起来一律省钱（回放甚至略贵，因为多了两条样本），但没算收敛变慢；
  - 只看 convergence wall clock → 惩罚过度，因为它默认"变慢就要一直跑到最后"。

本脚本按「固定训练预算」公平比较：在每条种子的前 T 步内，取**首次达到目标准确率**
所需墙钟；到不了的按该 T 步的累计墙钟计。这样"省生成但学得慢"的机制在 T 足够大时
可以追平，在 T 小时会输，净效应由 T 决定——T 是本章唯一的外生选择，故做 T 扫描。

    python3 analyze.py                       # 读 summary.json，打印净墙钟表
    python3 analyze.py --json net_wallclock.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

HERE = Path(__file__).parent
TARGET = 0.9
BUDGETS = (10, 20, 30, 50, 100, 200, 300)


def cumulative(step_costs: list[int]) -> list[int]:
    out, run = [], 0
    for c in step_costs:
        run += c
        out.append(run)
    return out


def wall_at_target(seedrec: dict, budget: int) -> tuple[int, bool]:
    """前 budget 步内首次达到 TARGET 的墙钟；未达到则记为 budget 步末尾的累计墙钟。"""
    costs = cumulative(seedrec["step_costs"][:budget])
    for i, acc in enumerate(seedrec["acc_curve"][:budget]):
        if acc >= TARGET:
            return costs[i], True
    return costs[-1], False


def final_acc_at(seedrec: dict, budget: int) -> float:
    return seedrec["acc_curve"][min(budget, len(seedrec["acc_curve"])) - 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(HERE / "summary.json"))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    data = json.loads(Path(args.summary).read_text())
    arms = data["arms"]
    base = arms["base"]["per_seed"]
    n = len(base)
    result = {"target": TARGET, "budgets": {}}

    for budget in BUDGETS:
        row = {}
        b_walls, b_reach = zip(*(wall_at_target(s, budget) for s in base))
        b_mean = statistics.mean(b_walls)
        for arm, d in arms.items():
            rs = d["per_seed"]
            walls, reach = zip(*(wall_at_target(s, budget) for s in rs))
            paired = [w - b for w, b in zip(walls, b_walls)]
            sd = statistics.pstdev(paired) if len(paired) > 1 else 0.0
            half = 1.96 * sd / (n ** 0.5) if n > 1 else 0.0
            row[arm] = {
                "net_wall_mean": statistics.mean(walls),
                "ratio_vs_base": statistics.mean(walls) / b_mean,
                "reached": sum(reach),
                "paired_diff_mean": statistics.mean(paired),
                "paired_diff_ci95": half,
                "final_acc_mean_at_budget": statistics.mean(final_acc_at(s, budget) for s in rs),
            }
        row["_base_reached"] = sum(b_reach)
        row["_base_wall_mean"] = b_mean
        result["budgets"][str(budget)] = row

    print(f"{'budget':>7} {'arm':<8} {'净墙钟':>10} {'vs基线':>8} {'达标数':>6} {'到预算末准确率':>12}")
    for budget in BUDGETS:
        r = result["budgets"][str(budget)]
        for arm in ("base", "freq", "replay", "cancel", "cap"):
            e = r[arm]
            print(f"{budget:>7} {arm:<8} {e['net_wall_mean']:>10.0f} {e['ratio_vs_base']:>8.3f} "
                  f"{e['reached']:>6} {e['final_acc_mean_at_budget']:>12.4f}")
        print()
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
