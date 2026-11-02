#!/usr/bin/env python3
"""复算新意锚② 的多阶段恒等式：用已入库归档的分阶段墙钟重建整步比，与实测整步比对照。

只读归档，不跑 GPU、不改任何原读数。两个检查：

  A. 两套放置的 `G_k` 是否被混用（§20.3.5 的引用面）
  B. 恒等式 `整步× = 1/(Σ_k s_k/G_k)` 在可复算的臂上能否重建实测整步比

用法：
    python3 check_identity.py            # 打印全部核对并写 summary.json
    python3 check_identity.py --verify   # 只断言本脚本记录的事实仍然成立（失败 exit 1）
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
STAGES = ["gen", "old_log_prob", "ref", "adv", "update_actor", "update_weights"]


def load(rel):
    with open(os.path.join(EXP, rel)) as f:
        return json.load(f)


def means_mode(modes, key):
    """candidate-g 的两份归档都是 {stage: float} 平表。"""
    m = modes[key]
    return {s: m[s] for s in STAGES}, m["step"]


def means_timing(timing):
    """poll 系列归档是 {stage: {n, mean, sd, ...}} 且含 'step'。"""
    t = {s: timing[s]["mean"] for s in STAGES if s in timing}
    return t, timing["step"]["mean"]


# ---------------------------------------------------------------- A. 两套放置

def placement_check():
    re_a = load("rollout-g8-candidate-g/reanchor-summary.json")["modes"]
    old = load("rollout-g8-candidate-g/summary.json")["modes"]
    gA, _ = means_mode(re_a, "g1p")
    gO, _ = means_mode(old, "g1")
    out = {}
    for label, src, base, two, four in [
        ("re-anchor 2+2 轨（§13.45）", re_a, "g1p", "g2p", "g4p"),
        ("旧锚点 2+0／3+1 轨（§13.36）", old, "g1", "g2", "g4"),
    ]:
        b, _ = means_mode(src, base)
        row = {"2卡": {s: round(b[s] / means_mode(src, two)[0][s], 3) for s in STAGES},
               "4卡": {s: round(b[s] / means_mode(src, four)[0][s], 3) for s in STAGES}}
        out[label] = row
    # §20.3.5 实际引用的五对
    cited = {"gen": (1.322, 1.933), "old_log_prob": (1.608, 2.116), "ref": (2.023, 3.147),
             "update_actor": (1.628, 2.087), "update_weights": (0.882, 0.770)}
    mismatches = []
    for s, (c2, c4) in cited.items():
        r2, r4 = out["re-anchor 2+2 轨（§13.45）"]["2卡"][s], out["re-anchor 2+2 轨（§13.45）"]["4卡"][s]
        o2, o4 = out["旧锚点 2+0／3+1 轨（§13.36）"]["2卡"][s], out["旧锚点 2+0／3+1 轨（§13.36）"]["4卡"][s]
        src = []
        if abs(c2 - r2) < 0.002:
            src.append("2卡=re-anchor")
        elif abs(c2 - o2) < 0.002:
            src.append("2卡=旧锚点")
        else:
            src.append("2卡=对不上")
        if abs(c4 - r4) < 0.002:
            src.append("4卡=re-anchor")
        elif abs(c4 - o4) < 0.002:
            src.append("4卡=旧锚点")
        else:
            src.append("4卡=对不上")
        mismatches.append({"stage": s, "cited": [c2, c4], "source": "／".join(src)})
    return out, mismatches


# ---------------------------------------------------------------- B. 恒等式

def identity(s1, t1, sN, tN, label):
    """s1/t1 是单卡配置的 {stage: t} 与整步 T，sN/tN 是目标配置。

    按 §13.25／§20.3.1 写下的形式取 s_k = t_k^target / Σt^target、G_k = t_k^single / t_k^target。
    t1/tN 一并带进来只为报表：本脚本要检验的正是"只用 s_k 与 G_k 就够"这句话。

    另外给出按"整步比 = (T1/TN)·Σ_k s_k²·G_k"（s_k = t_k^N/Σt^N、G_k = t_k^1/t_k^N、分项和口径）
    的重建值——这是把各阶段省下的时间折算成"整步秒数"后逐项相加的形状，用于和上面的公式对照。
    """
    S1 = sum(s1.values())
    g = {k: s1[k] / sN[k] for k in s1}
    s_sum = {k: sN[k] / sum(sN.values()) for k in s1}      # 分项和作分母
    fam_sum = 1.0 / sum(s_sum[k] / g[k] for k in s1)
    lower = sum(s_sum[k] for k in s1)                      # G_k≡1 时的公式值
    quad = (t1 / tN) * sum(s_sum[k] ** 2 * g[k] for k in s1)
    return {
        "label": label,
        "recon_sum_denominator": round(fam_sum, 4),
        "recon_seamless_quadratic": round(quad, 4),
        "formula_value_if_no_acceleration": round(lower, 4),
        # 实测整步比：从归档里同级取出的 step 均值的比值，不由重建式推出
        "actual_step_ratio": round(t1 / tN, 4),
        "stage_sum_single": round(S1, 4),
        "stage_sum_target": round(sum(sN.values()), 4),
        "stage_sum_over_step_single": round(S1 / t1, 5),
        "stage_sum_over_step_target": round(sum(sN.values()) / tN, 5),
        "g_k": {k: round(g[k], 3) for k in s1},
    }


def identity_checks():
    rows = []

    # 1) 候选 G：同一份归档内的 1 卡 → 2 卡 / 4 卡（re-anchor 2+2 轨，放置内部一致）
    re_a = load("rollout-g8-candidate-g/reanchor-summary.json")["modes"]
    b, Tb = means_mode(re_a, "g1p")
    for key, name in [("g2p", "g1p→g2p（1+1）"), ("g4p", "g1p→g4p（2+2）")]:
        t, Tt = means_mode(re_a, key)
        r = identity(b, Tb, t, Tt, name)
        r["error_pct"] = round((r["recon_sum_denominator"] / r["actual_step_ratio"] - 1) * 100, 2)
        rows.append(r)

    # 2) 跨放置对照：2 卡基线 → 4 卡基线（同一份 poll 归档，各阶段 G_k 同向）
    p = load("verl-v1-poll-quantum/summary.json")
    b, Tb = means_timing(p["pbase"]["timing"])
    t, Tt = means_timing(p["p4base"]["timing"])
    r = identity(b, Tb, t, Tt, "pbase(2卡)→p4base(4卡)")
    # 跨放置两档的整步比不是"同一负载加卡"，只作对照，不进结论
    rows.append(r)

    # 3) 同一配置下的单轴干预（poll 2.0→0.1，每阶段 G_k 名义为 1）
    pairs = [("verl-v1-poll-quantum/summary.json", "pbase", "p010", "poll 2.0→0.1（2 卡）"),
             ("verl-v1-poll-quantum/summary.json", "p4base", "p4p010", "poll 2.0→0.1（4 卡）")]
    for rel, base, mod, name in pairs:
        d = load(rel)
        b, Tb = means_timing(d[base]["timing"])
        t, Tt = means_timing(d[mod]["timing"])
        r = identity(b, Tb, t, Tt, name)
        r["error_pct"] = round((r["recon_sum_denominator"] / r["actual_step_ratio"] - 1) * 100, 2)
        rows.append(r)

    d = load("verl-v1-poll-quantum-dapo-g4/summary-rerun.json")["arms"]
    b, Tb = means_timing(d["g4base"]["timing"])
    for mod, name in [("g4p050", "poll 2.0→0.5（4 卡 DAPO）"), ("g4p010", "poll 2.0→0.1（4 卡 DAPO）")]:
        t, Tt = means_timing(d[mod]["timing"])
        r = identity(b, Tb, t, Tt, name)
        r["error_pct"] = round((r["recon_sum_denominator"] / r["actual_step_ratio"] - 1) * 100, 2)
        rows.append(r)

    d = load("verl-v1-poll-quantum-dapo/summary-inflight1.json")
    b, Tb = means_timing(d["di1base"]["timing"])
    t, Tt = means_timing(d["di1p010"]["timing"])
    r = identity(b, Tb, t, Tt, "poll 2.0→0.1（2 卡 DAPO inflight=1）")
    r["error_pct"] = round((r["recon_sum_denominator"] / r["actual_step_ratio"] - 1) * 100, 2)
    rows.append(r)
    return rows


def main(verify):
    placement, mism = placement_check()
    idr = identity_checks()

    if verify:
        ok = True
        # A：§20.3.5 引用的五对 G_k 同源于一套放置（2026-11-03 已订正混用）
        for m in mism:
            if m["source"] not in ("2卡=re-anchor／4卡=re-anchor", "2卡=旧锚点／4卡=旧锚点"):
                ok = False
                print("[verify] 引用错配仍在：%s → %s" % (m["stage"], m["source"]))
        # B：同一归档内的两档重建值都不等于实测整步比
        for r in idr[:2]:
            if abs(r["error_pct"]) < 1.0:
                ok = False
                print("[verify] 恒等式在 %s 上已能重建实测比，结论需复核" % r["label"])
        # B2：公式在"没有加速"时给出 1，而不是基线/目标的整步比
        for r in idr[:2]:
            if r["formula_value_if_no_acceleration"] != 1.0:
                ok = False
                print("[verify] %s 的 G_k≡1 值不是 1" % r["label"])
        print("[verify]", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    out = {"placement": placement, "cited_checks": mism, "identity": idr}
    with open(os.path.join(HERE, "summary.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print("== A. 两套放置的分阶段 G_k ==")
    for label, row in placement.items():
        print(" ", label)
        for arm in ("2卡", "4卡"):
            print("   ", arm, row[arm])
    print("\n== A2. §20.3.5 实际引用的五对来自哪一套 ==")
    for m in mism:
        print("  %-15s cited=%s  → %s" % (m["stage"], m["cited"], m["source"]))
    print("\n== B. 恒等式重建 ==")
    for r in idr:
        print("  %-28s 公式=%.4f 二次型=%.4f 实测=%s 差=%s%%  G≡1时=%.4f  T/S=%s→%s"
              % (r["label"], r["recon_sum_denominator"], r["recon_seamless_quadratic"],
                 r["actual_step_ratio"], r.get("error_pct", "—"),
                 r["formula_value_if_no_acceleration"],
                 r["stage_sum_over_step_single"], r["stage_sum_over_step_target"]))
    return 0


if __name__ == "__main__":
    sys.exit(main("--verify" in sys.argv))
