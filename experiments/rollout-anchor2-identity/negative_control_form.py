#!/usr/bin/env python3
"""负向对照：证明 check_reconstruct_form.py 的判定对输入敏感，不是"怎么算都对"。

四组注入，每组必须被抓到（退出码 0 表示全部抓到）：
  1. 把 F2 的份额换回**目标侧**（即 F1）→ 加卡臂必须重新出现 ≥3% 的偏差（§20.3.6 结论仍成立）
  2. 把缺口修正**方向写反**（乘缺口^目标/缺口^基线 而不是 缺口^基线/缺口^目标）：本栈两侧缺口只差
     0.006%，预期几乎无感——此组用于记录"缺口因子在本栈≈1"，真正的方向检验放在第 4 组
  3. 把某一阶段的 G_k 人为改成 1（声称"该阶段没加速"）→ F3 必须偏离
  4. 把两侧缺口人为改成差 20%（模拟"计时缺口随配置变化"这个反例）→ F3 必须偏离，
     说明 F2≈整步比 只是本栈数据事实，F3 才是可迁移的写法
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("crf", os.path.join(HERE, "check_reconstruct_form.py"))
crf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(crf)

def forms_at(gaps, b, t, acc):
    """按给定两侧缺口构造 F2/F3，用于构造反例。"""
    gap_b, gap_t = gaps
    Sb, St = sum(b.values()), sum(t.values())
    Tb, Tt = Sb / gap_b, St / gap_t
    # 按恒等式，整步比 = acc × (gap_b/gap_t)；反解 Tt 使构造自洽
    Tt = Tb / (acc * (gap_b / gap_t))
    return {"f2": acc, "f3": acc * (gap_b / gap_t),
            "actual_step_ratio": Tb / Tt, "gap_base": gap_b, "gap_target": gap_t}


re_a = crf.load("rollout-g8-candidate-g/reanchor-summary.json")["modes"]
b, Tb = crf.means_mode(re_a, "g1p")
t4, Tt4 = crf.means_mode(re_a, "g4p")
results = []

st = sum(t4.values())
sb_ = sum(b.values())

g = {k: b[k] / t4[k] for k in b}
f1 = 1 / sum((t4[k] / st) / g[k] for k in b)
f2 = 1 / sum((b[k] / sb_) / g[k] for k in b)
gp, gq = sb_ / Tb, st / Tt4

# 1. 目标侧份额（F1）
err = (f1 / (Tb / Tt4) - 1) * 100
results.append(("① 份额换回目标侧", abs(err) > 3.0, "g1p→g4p 误差 %.2f%%" % err))

# 2. 缺口方向写反
wrong = f2 * (gq / gp)
err2 = (wrong / (Tb / Tt4) - 1) * 100
# 本栈两侧缺口只差 0.006%，方向写反的偏差也因此只有 0.006%——这正是"F3 的缺口因子≈1"的含义；
# 把这一项挪到第 4 组（缺口差 20% 时方向错必须被抓到），此处只记录本栈上的实际偏差。
results.append(("② 缺口方向写反（本栈缺口差 0.006%，预期几乎无感）", abs(err2) < 0.05,
                "误差 %+.4f%%" % err2))

# 3. 谎报某阶段没加速
t_bad = dict(t4)
t_bad["update_actor"] = b["update_actor"]
r_bad = crf.forms(b, Tb, t_bad, Tt4)
err3 = (r_bad["f3"] / r_bad["actual_step_ratio"] - 1) * 100
results.append(("③ 谎报 update_actor 无加速", abs(err3) > 0.1, "F3 误差 %+.3f%%" % err3))

# 4. 两侧缺口差 20%（构造反例：F2 不再等于整步比，F3 仍准、写反被抓）
gap_b4, gap_t4 = 0.95, 0.76
acc = 1 / sum((b[k] / sb_) / g[k] for k in b)
r4 = forms_at((gap_b4, gap_t4), b, t4, acc)
err4 = (r4["f2"] / r4["actual_step_ratio"] - 1) * 100
err4b = (r4["f3"] / r4["actual_step_ratio"] - 1) * 100
err4_rev = (acc * (gap_t4 / gap_b4) / r4["actual_step_ratio"] - 1) * 100
results.append(("④ 缺口差 20%：F2 偏离、F3 仍准、F3 写反被抓",
                abs(err4) > 5 and abs(err4b) < 0.1 and abs(err4_rev) > 5,
                "F2 %+.2f%%，F3 %+.3f%%，写反 %+.2f%%" % (err4, err4b, err4_rev)))

# 5. 配对错档：同配置重复臂的基线若取成 re-anchor 的 g4p（4 卡加卡轨），实测比从 0.8372 变 0.7030。
#    这一组同时说明两件事：错误配对会静默产出一个"看着像真的"数（所以检查器必须锁同一份归档）；
#    而 F3 对这种错配对照样闭合（−0.06% → 记录值），即**形式的闭合不能代替配对正确**。
g_old = crf.load("rollout-g8-candidate-g/summary.json")["modes"]
b_ok, Tb_ok = crf.means_mode(g_old, "g4")
t_ok, Tt_ok = crf.means_mode(g_old, "g4n")
r_ok = crf.forms(b_ok, Tb_ok, t_ok, Tt_ok)
r_wrong = crf.forms(b, Tb, t_ok, Tt_ok)      # b/Tb 是上面的 g1p —— 连卡数都对不上，更该被抓
r_g4p = crf.forms(crf.means_mode(re_a, "g4p")[0], crf.means_mode(re_a, "g4p")[1], t_ok, Tt_ok)
results.append(("⑤ 同配置重复臂配对错档（g4p vs g4n）",
                abs(r_ok["actual_step_ratio"] - 0.8372) < 1e-3
                and abs(r_g4p["actual_step_ratio"] - 0.8372) > 0.1
                and abs(r_wrong["actual_step_ratio"] - 0.8372) > 0.5,
                "正确 %.4f，g4p 配错 %.4f，g1p 配错 %.4f"
                % (r_ok["actual_step_ratio"], r_g4p["actual_step_ratio"],
                   r_wrong["actual_step_ratio"])))

ok = True
for name, caught, detail in results:
    print("[%-40s] %s  %s" % (name, "抓到" if caught else "漏过", detail))
    ok = ok and caught
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
