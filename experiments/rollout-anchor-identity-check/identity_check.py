#!/usr/bin/env python3
"""同栈锚：§20.3.1 恒等式在 M5 三档锚点上的独立复算。

只读 experiments/rollout-g8-candidate-g/reanchor-summary.json 与 ab-summary.json，
不重算任何原始计时。做法：从 G1（1 卡）的逐阶段均值与 G2/G4 的逐阶段加速
重建整步加速，与实测整步加速对比，并把差值逐项归到"跨阶段残差"。

用法：
    python3 identity_check.py            # 打印读数
    python3 identity_check.py --verify   # 断言 + 退出码
    python3 identity_check.py --json     # 机读输出
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
G_DIR = REPO / "experiments" / "rollout-g8-candidate-g"
REANCHOR = G_DIR / "reanchor-summary.json"
AB = G_DIR / "ab-summary.json"

# 阶段顺序即 §13.45 表序；adv 是优势计算，不在恒等式的"阶段"里单列为可观测量时
# 仍作为加项保留（它确实花墙钟）。
STAGES = ["gen", "old_log_prob", "ref", "adv", "update_actor", "update_weights"]


def load():
    reanchor = json.loads(REANCHOR.read_text())
    ab = json.loads(AB.read_text())
    return reanchor, ab


def direct_model(base, target, stages=STAGES):
    """投影整步时间 = Σ_k t_k^base / G_k，G_k 逐阶段加速（adv 不加速，G=1）。"""
    t_proj = 0.0
    terms = {}
    for s in stages:
        g = target.get(s, 1.0) if s != "adv" else 1.0
        terms[s] = base[s] / g
        t_proj += terms[s]
    return t_proj, terms


def check_anchor(reanchor):
    modes = reanchor["modes"]
    scaling = reanchor["scaling"]
    base = modes["g1p"]
    base_step = base["step"]
    rows = []
    for key in ("g2p", "g4p"):
        target = modes[key]
        st = scaling[key]
        t_proj, terms = direct_model(base, st["stage_scaling"])
        stage_sum = target["stage_sum"]
        half_residual = t_proj - stage_sum          # 半程误差：投影 vs 同一份分项
        full_residual = stage_sum - target["step"]  # 全程误差：分项之和 vs 整步
        rows.append({
            "mode": key,
            "n_gpus": st["n_gpus"],
            "step_measured_s": target["step"],
            "step_projected_s": t_proj,
            "speedup_measured": st["speedup"],
            "speedup_projected": base_step / t_proj,
            "speedup_residual_pct": (base_step / t_proj) / st["speedup"] - 1.0,
            "speedup_residual_abs": base_step / t_proj - st["speedup"],
            "stage_sum_s": stage_sum,
            "half_residual_s": half_residual,
            "half_residual_pct_of_step": half_residual / target["step"],
            "full_residual_s": full_residual,
            "full_residual_pct_of_step": full_residual / target["step"],
            "share_of_total_half_residual": (
                half_residual / (half_residual + full_residual)
                if (half_residual + full_residual) != 0 else None
            ),
            "per_stage_projected_s": terms,
        })
    return rows


def full_identity(reanchor):
    """§20.3.1 完整式：整步× = 1 / (1 - Σ s_k(1-1/G_k) + Σ a_m/T)。

    **基线口径是这里的全部要害**：份额 s_k 必须以"各阶段墙钟之和" Σt_k 为分母。
    若用整步 T 作分母，而 T 与 Σt_k 不相等（本摊位的分项和约 0.28%），
    公式会退化成 1/Σ(s_k/G_k)，与"逐阶段时间相加"差出 (T-Σt)/T 这一项——
    那不是舍入，是基线用错。两种口径都算出来对照。
    """
    modes = reanchor["modes"]
    scaling = reanchor["scaling"]
    base = modes["g1p"]
    T = base["step"]
    t_sum = sum(base[s] for s in STAGES)
    out = []
    for key in ("g2p", "g4p"):
        st = scaling[key]
        G = {s: (st["stage_scaling"].get(s, 1.0) if s != "adv" else 1.0) for s in STAGES}
        t_proj, _ = direct_model(base, st["stage_scaling"])
        # 正确口径：份额以分项和为分母
        s_ok = {s: base[s] / t_sum for s in STAGES}
        x_ok = 1.0 / (1.0 - sum(s_ok[s] * (1.0 - 1.0 / G[s]) for s in STAGES))
        x_add = t_sum / t_proj
        # 错误口径：份额以整步为分母（T != Σt 时结果偏移）
        s_bad = {s: base[s] / T for s in STAGES}
        x_bad = 1.0 / (1.0 - sum(s_bad[s] * (1.0 - 1.0 / G[s]) for s in STAGES))
        out.append({
            "mode": key,
            "speedup_from_formula_stagesum": x_ok,
            "speedup_from_time_addition": x_add,
            "formula_vs_addition_abs_diff": abs(x_ok - x_add),
            "speedup_from_formula_stepsum": x_bad,
            "stepsum_vs_addition_abs_diff": abs(x_bad - x_add),
            "stage_sum_s": t_sum,
            "step_s": T,
            "unaccounted_share": (T - t_sum) / T,
        })
    return out


def check_placement(ab):
    """同一配置两种放置的逐阶段差：验证"阶段和 = 整步"在多档放置上同样成立。"""
    modes = ab["modes"]
    rows = []
    pairs = [("pA1", "pB1"), ("pA2", "pB2"), ("pA3", "pB3"), ("pA4", "pB4")]
    for a, b in pairs:
        if a not in modes or b not in modes:
            continue
        for tag in (a, b):
            m = modes[tag]
            ss = sum(m[s] for s in STAGES)
            rows.append({
                "mode": tag,
                "placement": "A(0,1,3,4)" if tag.startswith("pA") else "B(4,5,6,7)",
                "step_s": m["step"],
                "stage_sum_s": ss,
                "residual_s": ss - m["step"],
                "residual_pct": (ss - m["step"]) / m["step"],
            })
    return rows


def verify(reanchor, ab):
    """断言：核心声称必须成立，否则退出码非零。"""
    problems = []

    # 1) 完整式（分项和口径）与逐阶段相加必须恒等
    for row in full_identity(reanchor):
        if row["formula_vs_addition_abs_diff"] > 1e-9:
            problems.append(f"分项和口径下公式与相加不等价: {row['mode']} "
                            f"差={row['formula_vs_addition_abs_diff']:.2e}")

    # 1b) 整步口径必须**确实不同**——若这一条不成立，"基线口径"的提醒就是空话
    for row in full_identity(reanchor):
        if row["stepsum_vs_addition_abs_diff"] < 1e-4:
            problems.append(f"整步口径未产生可观测偏移，基线口径结论失效: {row['mode']}")

    # 2) 分项之和与整步之差（全程误差）必须在 §13.45 报的 0.040–0.076s 量级内
    for row in check_anchor(reanchor):
        if abs(row["full_residual_s"]) > 0.5:
            problems.append(f"全程误差超出量级: {row['mode']} {row['full_residual_s']:.3f}s")

    # 3) 实测加速必须与 §13.45 一致（1.526 / 1.942）
    expect = {"g2p": 1.5264, "g4p": 1.9419}
    for row in check_anchor(reanchor):
        if abs(row["speedup_measured"] - expect[row["mode"]]) > 0.01:
            problems.append(f"实测加速与 §13.45 不符: {row}")

    # 4) 放置对照的每档都应给出小的残差
    for row in check_placement(ab):
        if abs(row["residual_pct"]) > 0.02:
            problems.append(f"放置档残差过大: {row}")

    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    reanchor, ab = load()
    payload = {
        "inputs": {
            "reanchor": str(REANCHOR.relative_to(REPO)),
            "ab": str(AB.relative_to(REPO)),
        },
        "anchor": check_anchor(reanchor),
        "identity": full_identity(reanchor),
        "placement": check_placement(ab),
    }

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print("=== 同栈锚：恒等式分层复算（G1 -> G2/G4）===")
    print(f"{'档':6} {'实测s':>8} {'投影s':>8} {'实测×':>7} {'投影×':>7} "
          f"{'×残差%':>8} {'半程s':>7} {'全程s':>7} {'半程占比':>8}")
    for r in payload["anchor"]:
        print(f"{r['mode']:6} {r['step_measured_s']:8.3f} {r['step_projected_s']:8.3f} "
              f"{r['speedup_measured']:7.4f} {r['speedup_projected']:7.4f} "
              f"{r['speedup_residual_pct']*100:8.2f} {r['half_residual_s']:7.3f} "
              f"{r['full_residual_s']:7.3f} {r['share_of_total_half_residual']*100:7.1f}%")

    print("\n=== 完整式 vs 逐阶段相加（正确口径下应恒等）===")
    for r in payload["identity"]:
        print(f"{r['mode']:6} 分项和口径 公式×={r['speedup_from_formula_stagesum']:.6f}  "
              f"相加×={r['speedup_from_time_addition']:.6f}  "
              f"差={r['formula_vs_addition_abs_diff']:.2e}")
        print(f"{'':6} 整步口径 公式×={r['speedup_from_formula_stepsum']:.6f}  "
              f"相对相加差={r['stepsum_vs_addition_abs_diff']:.4f}  "
              f"（未归属份额 {r['unaccounted_share']*100:.2f}%）")

    print("\n=== 放置对照：每档分项和 vs 整步 ===")
    print(f"{'档':6} {'放置':14} {'整步s':>8} {'分项和s':>9} {'残差s':>8} {'残差%':>7}")
    for r in payload["placement"]:
        print(f"{r['mode']:6} {r['placement']:14} {r['step_s']:8.3f} "
              f"{r['stage_sum_s']:9.3f} {r['residual_s']:8.3f} {r['residual_pct']*100:7.2f}%")

    if args.verify:
        problems = verify(reanchor, ab)
        if problems:
            print("\n--verify 失败：")
            for p in problems:
                print("  -", p)
            return 1
        print("\n--verify 通过：四条断言全部成立。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
