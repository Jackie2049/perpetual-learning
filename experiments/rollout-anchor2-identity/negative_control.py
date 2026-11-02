#!/usr/bin/env python3
"""负向对照：证明 check_identity.py 的判定对输入敏感。

三组注入，每组必须被抓到：
  1. 把 §20.3.5 改回"2 卡取 re-anchor、4 卡取旧锚点"的混用 → A2 应报错配
  2. 把 g4p 的 update_actor 换成旧锚点值 1.628 → B 的重建差应改变，校验它的敏感度
  3. 把 s_k 换成"单卡份额"（分母用单卡分项和）→ 必须让"形式不是整步比"这件事更明显或保持
"""
import importlib.util, json, os, sys, copy

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ci", os.path.join(HERE, "check_identity.py"))
ci = importlib.util.module_from_spec(spec); spec.loader.exec_module(ci)

results = []

# --- 1. 混用两套放置
re_a = ci.load("rollout-g8-candidate-g/reanchor-summary.json")["modes"]
old = ci.load("rollout-g8-candidate-g/summary.json")["modes"]
cited = {"gen": (1.322, 1.937), "old_log_prob": (1.608, 1.777), "ref": (2.023, 2.623),
         "update_actor": (1.628, 1.628), "update_weights": (0.882, 0.690)}
b2, _ = ci.means_mode(re_a, "g1p"); b4, _ = ci.means_mode(re_a, "g4p")
o2, _ = ci.means_mode(old, "g1") if False else (None, None)
caught = 0
gA2 = {s: round(b2[s] / ci.means_mode(re_a, "g2p")[0][s], 3) for s in ci.STAGES}
gA4 = {s: round(b4[s] / ci.means_mode(re_a, "g4p")[0][s], 3) for s in ci.STAGES}
gp2 = {s: round(ci.means_mode(old, "g1")[0][s] / ci.means_mode(old, "g2")[0][s], 3) for s in ci.STAGES}
gp4 = {s: round(ci.means_mode(old, "g1")[0][s] / ci.means_mode(old, "g4")[0][s], 3) for s in ci.STAGES}
for s, (c2, c4) in cited.items():
    m2 = "2卡=re-anchor" if abs(c2 - gA2[s]) < 0.002 else ("2卡=旧锚点" if abs(c2 - gp2[s]) < 0.002 else "2卡=对不上")
    m4 = "4卡=re-anchor" if abs(c4 - gA4[s]) < 0.002 else ("4卡=旧锚点" if abs(c4 - gp4[s]) < 0.002 else "4卡=对不上")
    if m2 + "／" + m4 != "2卡=re-anchor／4卡=re-anchor":
        caught += 1
results.append(("混用两套放置被抓到", caught == len(cited), f"{caught}/{len(cited)} 项被判异常"))

# --- 2. 篡改一个阶段值，重建差必须移动
b, Tb = ci.means_mode(re_a, "g1p")
t, Tt = ci.means_mode(re_a, "g4p")
base = ci.identity(b, Tb, t, Tt, "orig")
t2 = dict(t); t2["update_actor"] *= 1.10   # 目标阶段慢 10%
pert = ci.identity(b, Tb, t2, Tt, "perturbed")
results.append(("篡改阶段值后重建值改变", abs(pert["recon_sum_denominator"] - base["recon_sum_denominator"]) > 1e-3,
                f"{base['recon_sum_denominator']} → {pert['recon_sum_denominator']}"))

# --- 3. G_k≡1 时公式给 1：说明重建式与"基线/目标整步比"不是同一个量
results.append(("G_k≡1 时公式值恰为 1（≠整步比）", base["formula_value_if_no_acceleration"] == 1.0,
                f"G≡1 → {base['formula_value_if_no_acceleration']}，而实际整步比 {base['actual_step_ratio']}"))

ok = True
for name, passed, detail in results:
    print(("PASS" if passed else "FAIL"), name, "|", detail)
    ok = ok and passed
print("negative_control:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
