#!/usr/bin/env python3
"""P-3 的学习率敏感性：把主脚本的五种模式在三档 lr 上各跑一遍。

原先 `lr_sensitivity.json` 没有对应脚本，任何读者都无法复算这张表。本文件只是驱动器，
模型与统计全部来自 `p3_group_filter_net_effect.py`，不复制一份逻辑——所以两处结论永远同源。

    python3 lr_sensitivity.py             # 三档 lr × 30 种子 × 300 步，约 4 分钟
    python3 lr_sensitivity.py --verify    # 复跑并与归档逐字段比对
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import p3_group_filter_net_effect as p3

LRS = (0.05, 0.1, 0.2)
RATIO_KEYS = {
    "walls_to_09_ratio": "walls_to_09_mean",
    "update_walls_to_09_ratio": "update_walls_to_09_mean",
    "final_cost_ratio": "cost_mean",
    "final_update_cost_ratio": "total_update_cost_mean",
}


def build(seeds: int, steps: int, prompts: int, n_eval: int) -> dict:
    by_lr = {}
    for lr in LRS:
        summary = p3.run_all(seeds, steps, lr, prompts, n_eval)
        base = summary["modes"]["R_nofilter"]
        row = {}
        for m in p3.MODES:
            d = summary["modes"][m]
            entry = {
                "final_acc": d["acc_mean"],
                "delta_acc_vs_base": d["acc_delta_vs_R_nofilter"],
            }
            for out_key, src_key in RATIO_KEYS.items():
                b, v = base[src_key], d[src_key]
                entry[out_key] = (v / b) if b else None
            row[m] = entry
        by_lr[f"{lr}"] = row
    return {
        "note": f"lr 敏感性：每档 {seeds} seed × {steps} 步；口径同 summary.json 的 wallclock_model；"
                f"比值均为相对同 lr 下 R_nofilter 的均值比",
        "by_lr": by_lr,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--eval", type=int, default=400, dest="n_eval")
    ap.add_argument("--out", default=str(Path(__file__).parent / "lr_sensitivity.json"))
    ap.add_argument("--verify", action="store_true", help="与归档逐字段比对（不写文件）")
    args = ap.parse_args()
    result = build(args.seeds, args.steps, args.prompts, args.n_eval)
    if args.verify:
        archived = json.loads(Path(args.out).read_text())
        if archived == result:
            print(f"复跑核对通过：与 {args.out} 逐字段相同")
            return
        for lr in result["by_lr"]:
            for m in result["by_lr"][lr]:
                a, b = archived["by_lr"].get(lr, {}).get(m), result["by_lr"][lr][m]
                if a != b:
                    print(f"  差异 lr={lr} {m}: archived={a} fresh={b}")
        raise SystemExit(1)
    out = Path(args.out)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(out)
    for lr, row in result["by_lr"].items():
        print(lr, {m: (round(v["walls_to_09_ratio"], 3), round(v["update_walls_to_09_ratio"], 3))
                   for m, v in row.items()})


if __name__ == "__main__":
    main()
