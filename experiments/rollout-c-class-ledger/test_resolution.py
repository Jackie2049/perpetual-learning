"""resolution.py 的单测。每条负向对照都必须让被测逻辑真的失败，否则测试无意义。

运行：cd experiments/rollout-c-class-ledger && python3 -m unittest test_resolution -v
"""

import copy
import json
import math
import unittest
from pathlib import Path

import resolution

HERE = Path(__file__).resolve().parent
P3 = HERE.parent / "rollout-p3-dapo-group-filter" / "summary.json"
P4 = HERE.parent / "rollout-p4-c-mechanisms" / "summary.json"


def load_archive():
    return {
        "p3": json.loads(P3.read_text()),
        "p4": json.loads(P4.read_text()),
        "p3_path": str(P3),
        "p4_path": str(P4),
    }


class TestReconcileIdentity(unittest.TestCase):
    """分解式必须逐种子成立：Δmean = Δupd + Δgen。"""

    def test_identity_holds_on_all_pairs(self):
        res = resolution.build(load_archive())
        for name, r in res["p3_reconcile"].items():
            s = r["delta_from_update_term"] + r["delta_from_gen_side"]
            self.assertAlmostEqual(s, r["delta_mean_step_cost"], places=6, msg=name)

    def test_identity_holds_on_synthetic(self):
        # 构造 steps_to_09=1（0 基）⇒ 含触发步 n=2。空更新步计 floor_unit=1（非 0）。
        #   A：[1, 60]  ⇒ walls 61、mean 30.5；更新侧摊到每步 = 60/2 = 30、生成侧 0.5
        #   B：[1, 200] ⇒ walls 201、mean 100.5；更新侧摊到每步 = 200/2 = 100、生成侧 0.5
        fake = {
            "modes": {
                "A": {"per_seed": [{"steps_to_09": 1, "walls_to_09": 61,
                                    "step_costs": [1, 60]}]},
                "B": {"per_seed": [{"steps_to_09": 1, "walls_to_09": 201,
                                    "step_costs": [1, 200]}]},
            }
        }
        r = resolution.reconcile_wall(fake, ("A", "B"))
        self.assertAlmostEqual(r["delta_from_update_term"], 30.0 - 100.0, places=6)
        self.assertAlmostEqual(r["delta_from_gen_side"], 0.5 - 0.5, places=6)
        self.assertAlmostEqual(r["delta_mean_step_cost"], -70.0, places=6)
        self.assertAlmostEqual(
            r["delta_from_update_term"] + r["delta_from_gen_side"],
            r["delta_mean_step_cost"], places=6)

    def test_negative_control_off_by_one_index_breaks_identity(self):
        """注入缺陷：把 n09 当成"不含触发步"，分解恒等式仍可能自洽（两侧同偏），
        但**归档内部一致性**必须立刻报错——这正是第 5 条断言存在的理由。"""
        data = load_archive()
        orig = resolution.internal_consistency
        try:
            def wrong(p3, p4):
                out = orig(p3, p4)
                for d in out["p3"].values():
                    d["n_inconsistent_seeds"] = 30  # 模拟 off-by-one 造成的全不符
                return out
            resolution.internal_consistency = wrong
            probs = resolution.assert_invariants(resolution.build(data))
        finally:
            resolution.internal_consistency = orig
        self.assertTrue(any("walls_to_09" in p for p in probs), probs)


class TestIndexConvention(unittest.TestCase):
    def test_rebuild_zero_mismatch(self):
        res = resolution.build(load_archive())
        self.assertEqual(res["threshold_grid"]["p4"]["rebuild_mismatch_seeds"], 0)
        self.assertTrue(all(v == 0 for v in res["threshold_grid"]["p3"]["mismatch_modes"].values()))

    def test_negative_control_off_by_one_is_detected(self):
        """注入缺陷：把索引当成 1-based（i+1），重建必须报大量不符。"""
        p4 = json.loads(P4.read_text())
        bad = 0
        for s in p4["arms"]["base"]["per_seed"]:
            first = next((i + 1 for i, a in enumerate(s["acc_curve"]) if a >= 0.9), None)
            if first != s["steps_to_09"]:
                bad += 1
        self.assertGreater(bad, 0, "off-by-one 必须被抓到")


class TestP4CostFace(unittest.TestCase):
    def test_constant_arms(self):
        res = resolution.build(load_archive())
        f = res["p4_step_cost_face"]
        self.assertTrue(f["base"]["step_cost_is_constant"])
        self.assertTrue(f["cap"]["step_cost_is_constant"])

    def test_task_count_not_recorded(self):
        res = resolution.build(load_archive())
        for arm in ("freq", "cancel", "cap"):
            self.assertFalse(res["p4_step_cost_face"][arm]["task_count_is_separately_recorded"])

    def test_negative_control_override_flips_conclusion(self):
        """注入缺陷：把某臂的"任务数已是独立字段"置真，结论级断言必须报错。
        这是本批结论的**唯一推翻入口**——归档里出现了发起次数字段就该重新做这一批。"""
        res = resolution.build(load_archive())
        tampered = copy.deepcopy(res)
        tampered["p4_step_cost_face"]["freq"]["task_count_is_separately_recorded"] = True
        probs = resolution.conclusion_assertions(tampered)
        self.assertTrue(any("独立字段" in p for p in probs), probs)


class TestP3UpdateCost(unittest.TestCase):
    def test_update_cost_not_fixed_8(self):
        res = resolution.build(load_archive())
        for mode, d in res["p3_update_separation"].items():
            self.assertFalse(d["update_cost_is_fixed_8"], mode)

    def test_empty_update_share(self):
        res = resolution.build(load_archive())
        # p3 的 R / C 有空更新步，其余模式没有
        self.assertGreater(res["p3_update_separation"]["C"]["empty_update_share"], 0.3)
        self.assertEqual(res["p3_update_separation"]["A"]["empty_update_share"], 0.0)


class TestCurveArea(unittest.TestCase):
    def test_cost_ratio_semantics_differ(self):
        """两份"每步成本比"口径不同，方向可能相反——C vs B 是活例子（1.063 vs 0.551）。
        这条测试钉住两者的定义，防止将来有人拿 mean_step_cost_ratio_to_09 去替代 cost_ratio。"""
        res = resolution.build(load_archive())
        r = res["p3_reconcile"]["C_vs_B"]
        # 1.062593 由归档独立算得：mean(walls/n) 在 C=+B 上再相除。
        # 与"全部 300 步 total_cost 比 0.551115"相差的分母口径是**步数范围**，不是算法错误。
        self.assertAlmostEqual(r["mean_step_cost_ratio_to_09"], 1.06259324, places=6)
        p3 = load_archive()["p3"]
        total_ratio = p3["modes"]["C"]["cost_mean"] / p3["modes"]["B"]["cost_mean"]
        self.assertAlmostEqual(total_ratio, 0.55111536, places=6)
        self.assertLess(total_ratio, 1.0)  # 全部步口径：C 更省
        self.assertGreater(r["mean_step_cost_ratio_to_09"], 1.0)  # 到收敛口径：C 更贵

    def test_self_comparison_is_zero(self):
        res = resolution.build(load_archive())
        self.assertAlmostEqual(res["p4_curve_area"]["base"]["mean_diff_area"], 0.0, places=9)

    def test_negative_control_sign_flip(self):
        """注入缺陷：把面积差取反，freq 的符号必须翻——证明该量对方向敏感。"""
        res = resolution.build(load_archive())
        self.assertGreater(res["p4_curve_area"]["freq"]["mean_diff_area"], 0.0)
        self.assertGreater(res["p4_curve_area"]["cap"]["mean_diff_area"], 0.0)


class TestAssertions(unittest.TestCase):
    def test_clean_build_has_no_problems(self):
        self.assertEqual(resolution.assert_invariants(resolution.build(load_archive())), [])

    def test_negative_control_injected_defect_is_found(self):
        """注入缺陷：把某种子的 step_costs 整体放大而不动 walls_to_09。
        分解式两侧同源、会互相抵消，**只有内部一致性检查**能抓到——这证明第 5 条断言不是摆设。"""
        data = load_archive()
        s0 = data["p3"]["modes"]["C"]["per_seed"][0]
        s0["step_costs"] = [c * 2 for c in s0["step_costs"]]
        probs = resolution.assert_invariants(resolution.build(data))
        self.assertTrue(any("walls_to_09" in p for p in probs), probs)

    def test_negative_control_grid_band(self):
        """结论级断言的边界：网格 0.005 通过、0.0005 报错。两者都测，证明该断言对输入敏感。"""
        res = resolution.build(load_archive())
        ok = copy.deepcopy(res)
        self.assertEqual([p for p in resolution.conclusion_assertions(ok) if "网格" in p], [])
        bad = copy.deepcopy(res)
        bad["threshold_grid"]["p4"]["grid"] = 0.0005
        probs = resolution.conclusion_assertions(bad)
        self.assertTrue(any("网格" in p for p in probs), probs)

    def test_verify_against_self_archive_clean(self):
        res = resolution.build(load_archive())
        arch = json.loads((HERE / "resolution.json").read_text())
        self.assertEqual(resolution.verify_against(res, arch), [])


if __name__ == "__main__":
    unittest.main()
