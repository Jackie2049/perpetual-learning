#!/usr/bin/env python3
"""学习率敏感性：三个臂在 lr = 0.1／0.2／0.4 下各跑一遍，看结论方向是否随 lr 变化。

模型与统计全部来自 `c_mechanisms_ablation.py`，本文件只是驱动器（同 P-3 的 lr_sensitivity.py 做法）。

    python3 lr_sensitivity.py            # 三档 lr × 5 臂 × 30 种子 × 300 步
    python3 lr_sensitivity.py --verify   # 复跑并与归档逐字段比对
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import c_mechanisms_ablation as m

LRS = (0.1, 0.2, 0.4)


def build(seeds: int, steps: int, n_eval: int) -> dict:
    by_lr = {}
    for lr in LRS:
        cfg = m.Cfg(lr=lr, seeds=seeds, steps=steps, n_eval=n_eval)
        s = m.run_all(cfg)
        row = {}
        for arm, v in s["arms"].items():
            st = [x["steps_to_09"] for x in v["per_seed"] if x["steps_to_09"] is not None]
            row[arm] = {
                "final_acc": v["final_acc_mean"],
                "delta_acc_vs_base": v["paired_final_acc_mean"],
                "cost_ratio_vs_base": v["cost_ratio_vs_base"],
                "traj_per_step": v["mean_counts"],
                "reached_09": v["reached_09"],
                "steps_to_09_mean": statistics.mean(st) if st else None,
                "walls_to_09_ratio_vs_base": (
                    v["walls_to_09_mean"] / s["arms"]["base"]["walls_to_09_mean"]
                    if v["walls_to_09_mean"] and s["arms"]["base"]["walls_to_09_mean"] else None
                ),
            }
        by_lr[str(lr)] = row
    return {
        "note": f"lr 敏感性：每档 {seeds} 种子 × {steps} 步；口径同 summary.json 的 wallclock_model",
        "by_lr": by_lr,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--eval", type=int, default=200, dest="n_eval")
    ap.add_argument("--out", default=str(Path(__file__).parent / "lr_sensitivity.json"))
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    result = build(args.seeds, args.steps, args.n_eval)
    if args.verify:
        archived = json.loads(Path(args.out).read_text())
        if archived == result:
            print(f"复跑核对通过：与 {args.out} 逐字段相同")
            return
        for lr in result["by_lr"]:
            for arm, v in result["by_lr"][lr].items():
                if archived["by_lr"].get(lr, {}).get(arm) != v:
                    print(f"  差异 lr={lr} {arm}: {archived['by_lr'].get(lr, {}).get(arm)} != {v}")
        raise SystemExit(1)
    out = Path(args.out)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(out)
    for lr, row in result["by_lr"].items():
        print(lr, {a: (round(v["cost_ratio_vs_base"], 3), round(v["walls_to_09_ratio_vs_base"], 3),
                     round(v["traj_per_step"], 1)) for a, v in row.items()})


if __name__ == "__main__":
    main()
