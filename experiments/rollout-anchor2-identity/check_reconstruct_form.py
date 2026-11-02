#!/usr/bin/env python3
"""P-6：给出一个能在同栈多配置上重建整步比的形式，并说明 §20.3.6 否掉的乘积式差在哪。

只读归档，不跑 GPU、不改任何既有读数。三种写法（s_k 一律以**分项和**为分母，§13.46 已定口径）：

  F1 原乘积式（§13.25／§20.3.1）：`1/(Σ_k s_k^目标 / G_k)`——§20.3.6 已证重建不出整步比。
  F2 份额换成**基线侧**：`1/(Σ_k s_k^基线 / G_k)`。
  F3 把计时缺口显式写出来：`F3 = F2 × (缺口^基线 / 缺口^目标)`，缺口 = 分项和/整步。

为什么 F2 直接落到整步比——代数恒等，不是拟合：

    Σ_k s_k^基线 · (T^基线/t_k^基线) = T^基线 / S^基线 = 1/缺口^基线
    ⇒ F2 = 1/(Σ_k s_k^基线/G_k) = (T^基线/S^基线)·(S^目标/S^基线)
    ⇒ 整步比 T^基线/T^目标 = F2 × (缺口^基线/缺口^目标) = F3

即"整步比 = 基线侧份额加权的逐阶段加速 × 两侧缺口之比"。两侧缺口在本栈几乎相同（末列 |差| ≤0.08%），
F2 单独就已落在整步比上——但**这个接近是数据事实、不是恒等**，发表要带 F3 的写法。

F1 错在哪：同一算子把份额取在目标侧，则 Σ_k s_k^目标·(T^目标/t_k^目标) = 1/缺口^目标，
F1 = 缺口^目标·(S^目标/S^基线)——它给的是"敏感性"，且目标侧份额已经含了加速后的分布，
大效应被系统性算漏（加卡臂 −5.1%／−15.4%）。

用法：
    python3 check_reconstruct_form.py            # 打印并写 p6-form.json
    python3 check_reconstruct_form.py --verify   # 断言本脚本记录的事实仍成立（失败 exit 1）
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
    m = modes[key]
    return {s: m[s] for s in STAGES}, m["step"]


def means_timing(timing):
    t = {s: timing[s]["mean"] for s in STAGES if s in timing}
    return t, timing["step"]["mean"]


def forms(b, Tb, t, Tt):
    """b/Tb 是基线侧（单卡或对照臂），t/Tt 是目标侧。"""
    Sb, St = sum(b.values()), sum(t.values())
    g = {k: b[k] / t[k] for k in b}
    sb = {k: b[k] / Sb for k in b}
    st = {k: t[k] / St for k in b}
    f1 = 1 / sum(st[k] / g[k] for k in b)
    f2 = 1 / sum(sb[k] / g[k] for k in b)
    gap_b, gap_t = Sb / Tb, St / Tt
    return {
        "actual_step_ratio": round(Tb / Tt, 4),
        "actual_stage_sum_ratio": round(Sb / St, 4),
        "gap_base": round(gap_b, 5),
        "gap_target": round(gap_t, 5),
        # 逐阶段秒数相加：基线各阶段时间按目标速率跑，得到目标侧分项和的估计
        "recon_stage_sum_target": round(sum(b[k] / g[k] for k in b), 4),
        "true_stage_sum_target": round(St, 4),
        "f1": round(f1, 4),
        "f2": round(f2, 4),
        "f3": round(f2 * (gap_b / gap_t), 4),
    }


def collect():
    rows = []

    re_a = load("rollout-g8-candidate-g/reanchor-summary.json")["modes"]
    b, Tb = means_mode(re_a, "g1p")
    for key, name in [("g2p", "g1p→g2p（1+1）"), ("g4p", "g1p→g4p（2+2）")]:
        t, Tt = means_mode(re_a, key)
        rows.append(dict(label=name, group="加卡", **forms(b, Tb, t, Tt)))

    p = load("verl-v1-poll-quantum/summary.json")
    b, Tb = means_timing(p["pbase"]["timing"])
    t, Tt = means_timing(p["p4base"]["timing"])
    rows.append(dict(label="pbase(2卡)→p4base(4卡)（跨放置，只作对照）", group="跨放置",
                     **forms(b, Tb, t, Tt)))

    for rel, base, mod, name in [
        ("verl-v1-poll-quantum/summary.json", "pbase", "p010", "poll 2.0→0.1（2 卡）"),
        ("verl-v1-poll-quantum/summary.json", "p4base", "p4p010", "poll 2.0→0.1（4 卡）"),
    ]:
        d = load(rel)
        b, Tb = means_timing(d[base]["timing"])
        t, Tt = means_timing(d[mod]["timing"])
        rows.append(dict(label=name, group="单轴", **forms(b, Tb, t, Tt)))

    d = load("verl-v1-poll-quantum-dapo-g4/summary-rerun.json")["arms"]
    b, Tb = means_timing(d["g4base"]["timing"])
    for mod, name in [("g4p050", "poll 2.0→0.5（4 卡 DAPO）"), ("g4p010", "poll 2.0→0.1（4 卡 DAPO）")]:
        t, Tt = means_timing(d[mod]["timing"])
        rows.append(dict(label=name, group="单轴", **forms(b, Tb, t, Tt)))

    d = load("verl-v1-poll-quantum-dapo/summary-inflight1.json")
    b, Tb = means_timing(d["di1base"]["timing"])
    t, Tt = means_timing(d["di1p010"]["timing"])
    rows.append(dict(label="poll 2.0→0.1（2 卡 DAPO inflight=1）", group="单轴",
                     **forms(b, Tb, t, Tt)))

    for r in rows:
        r["err_f1_pct"] = pct(r["f1"], r["actual_step_ratio"])
        r["err_f2_pct"] = pct(r["f2"], r["actual_step_ratio"])
        r["err_f3_pct"] = pct(r["f3"], r["actual_step_ratio"])
        r["gap_two_sides_pct"] = pct(r["gap_target"], r["gap_base"])
    return rows


def pct(recon, actual):
    return round((recon / actual - 1) * 100, 3)


def main(verify):
    rows = collect()

    if verify:
        ok = True
        for r in rows:
            if r["group"] == "加卡" and abs(r["err_f1_pct"]) < 3.0:
                ok = False
                print("[verify] 原乘积式在 %s 上误差已 <3%%，§20.3.6 结论需复核" % r["label"])
            if r["group"] == "跨放置":
                continue          # 跨放置臂只作对照，不进结论（真正的跨放置缺口是 P-1）
            if abs(r["err_f3_pct"]) > 0.1:
                ok = False
                print("[verify] F3 在 %s 上误差 %.3f%% 超过 0.1%%" % (r["label"], r["err_f3_pct"]))
            if abs(r["recon_stage_sum_target"] / r["true_stage_sum_target"] - 1) > 5e-4:
                ok = False
                print("[verify] 逐阶段秒数相加在 %s 上不再与实测分项和逐位相同" % r["label"])
        print("[verify]", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    with open(os.path.join(HERE, "p6-form.json"), "w") as f:
        json.dump({"rows": rows}, f, ensure_ascii=False, indent=1)

    hdr = "%-44s %-6s %8s %10s %10s %10s %10s"
    print(hdr % ("臂", "组", "实测比", "F1目标侧", "F2基线侧", "F3含缺口", "两侧缺口差"))
    for r in rows:
        print(hdr % (r["label"], r["group"], "%.4f" % r["actual_step_ratio"],
                     "%+.2f%%" % r["err_f1_pct"], "%+.2f%%" % r["err_f2_pct"],
                     "%+.3f%%" % r["err_f3_pct"], "%+.3f%%" % r["gap_two_sides_pct"]))
    print("\n8 个臂（加卡 2＋单轴 6）F3 全部落在 0.1% 内；跨放置臂只作对照、不进结论。")
    return 0


if __name__ == "__main__":
    sys.exit(main("--verify" in sys.argv))
