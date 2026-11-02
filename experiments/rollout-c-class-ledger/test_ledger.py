"""ledger.py 的单测。两条负向对照必须让被测逻辑真的失败，否则测试无意义。

运行：cd experiments/rollout-c-class-ledger && python3 -m unittest test_ledger -v
"""

import json
import math
import unittest
from pathlib import Path

import ledger

HERE = Path(__file__).resolve().parent
P3 = HERE.parent / "rollout-p3-dapo-group-filter" / "summary.json"
P4 = HERE.parent / "rollout-p4-c-mechanisms" / "summary.json"


class TestGapAlgebra(unittest.TestCase):
    def test_gap_zero_when_wall_tracks_cost(self):
        # 每步省 50%、墙钟也省 50% ⇒ 缺口 0
        g = ledger.gap(0.5, 0.5)
        self.assertAlmostEqual(g["gap"], 0.0)
        self.assertAlmostEqual(g["amplification"], 0.0)

    def test_gap_reduces_to_one_minus_ratio_of_ratios(self):
        """关键代数事实（初版遗漏）：`1−r = wall_ratio`、`1−kappa = cost_ratio`，
        故 **gap = 1 − wall_ratio/cost_ratio** —— 它只由两个比值的商决定，与步数无关。
        这一条同时证明：gap 不是"收敛代价"，收敛方向必须另读 steps_ratio / sigma_steps。"""
        for wall_r, cost_r in [(0.8128, 0.2771), (0.66, 0.984), (0.5, 0.5), (0.937, 1.0)]:
            g = ledger.gap(wall_r, cost_r)
            if g["gap"] is not None and cost_r > 0:
                self.assertAlmostEqual(g["gap"], 1 - wall_r / cost_r, places=6)

    def test_gap_one_when_wall_not_improved(self):
        # 每步省 10%、墙钟一点没省 ⇒ 缺口按 1 记，还回去的倍数 = 1/(1−k) = 1/0.9
        g = ledger.gap(1.0, 0.9)
        self.assertEqual(g["gap"], 1.0)
        self.assertAlmostEqual(g["amplification"], 1 / 0.9)

    def test_gap_none_when_no_per_step_saving(self):
        self.assertIsNone(ledger.gap(0.94, 1.0)["gap"])
        self.assertIsNone(ledger.gap(0.90, 1.001)["gap"])

    def test_negative_control_naive_ratio(self):
        """负向对照：把缺口简化成 1 − r/k（丢掉 (1−k) 分母）时，必须与正确式不同。"""
        r, k = 0.1872, 0.7229
        correct = 1 - (1 - r) / (1 - k)
        naive = 1 - r / k
        self.assertNotAlmostEqual(correct, naive, places=3)


class TestDenominators(unittest.TestCase):
    def test_four_forms_consistent_on_same_shape(self):
        # 同一形状（臂 = 0.8×基线，逐种子同比例）时四种写法应给出相互一致的关系
        base = [100.0, 200.0, 300.0]
        arm = [0.8 * b for b in base]
        d = ledger.denominators(arm, base)
        self.assertAlmostEqual(d["ratio_denom_base"], 0.8)
        self.assertAlmostEqual(d["ratio_of_totals"], 0.8)
        self.assertAlmostEqual(d["ratio_base_over_arm"], 1 / 0.8)
        self.assertAlmostEqual(d["diff_denom_base"], -40.0)

    def test_negative_control_inverted_denominator(self):
        """负向对照：取倒数（1/0.8=1.25）永远不等于原比值，除非比值是 1。"""
        d = ledger.denominators([80.0, 160.0], [100.0, 200.0])
        self.assertNotAlmostEqual(d["ratio_denom_base"], d["ratio_base_over_arm"])


class TestAgainstArchives(unittest.TestCase):
    """以下断言全部写死常数，不取被测输出自证。"""

    @classmethod
    def setUpClass(cls):
        cls.result = ledger.build(
            ledger.load_p3(P3), ledger.load_p4(P4)
        )

    def test_cancel_wall_ratio_is_0_66_not_1_515(self):
        # §13.29 第 3104 行把 cancel 的墙钟比写了 0.660（= 臂/基线），需确认这是正确方向
        r = self.result["p4_arms"]["cancel"]
        self.assertAlmostEqual(r["wall_ratio"], 0.6599, places=3)
        self.assertAlmostEqual(r["wall_denominators"]["ratio_base_over_arm"], 1.5154, places=3)

    def test_freq_gap_is_large_negative(self):
        # 每步省 72.3%，墙钟只省 18.7% ⇒ 缺口 −1.93（负数：墙钟省得比每步省多）
        g = self.result["p4_arms"]["freq"]["gap_from_ratios"]
        self.assertAlmostEqual(g["kappa"], 0.7229, places=4)
        self.assertAlmostEqual(g["r"], 0.1872, places=4)
        self.assertAlmostEqual(g["gap"], -1.9329, places=4)

    def test_cap_and_replay_have_no_gap(self):
        for arm in ("cap", "replay"):
            self.assertIsNone(self.result["p4_arms"][arm]["gap_from_ratios"]["gap"])

    def test_p3_c_vs_b_wall_effect_not_negative(self):
        # §13.28 的表把 C 的墙钟写成 0.499× vs B 0.486×（即 C 略差），归档复算后是 1.0267（C 略慢 2.7%）
        r = self.result["p3_pairs"]["C_vs_B"]
        self.assertAlmostEqual(r["wall_ratio"], 1.0267, places=3)
        self.assertAlmostEqual(r["cost_ratio"], 0.5511, places=3)
        self.assertFalse(r["gap_from_ratios"]["gap"] is None)

    def test_negative_control_gap_sign_does_not_encode_convergence(self):
        """负向对照：gap 的符号只说明"墙钟比每步成本省得多还是少"，**不说明收敛快慢**。
        freq：gap −1.93（墙钟相对每步省得更多），但 steps_ratio 1.070 是**收敛变慢**——
        若按初版读法（gap<0 ⇒ 收敛变快），方向会被读反。
        反向例子：cancel 的 gap 是 +0.33 而 steps_ratio 0.949（收敛变快）——同样同号反例。"""
        freq = self.result["p4_arms"]["freq"]
        self.assertAlmostEqual(freq["gap_from_ratios"]["gap"], -1.9329, places=4)
        self.assertGreater(freq["steps_ratio"], 1.0)  # 收敛变慢
        cancel = self.result["p4_arms"]["cancel"]
        self.assertGreater(cancel["gap_from_ratios"]["gap"], 0.0)  # 正 gap
        self.assertLess(cancel["steps_ratio"], 1.0)  # 却也是收敛变快
        # 有意思的是 freq 与 cancel 在"gap 符号"和"steps 是否 <1"上同向（负/正 对应 慢/快），
        # 真正的反例来自 §13.28：C vs B 的 gap 钉在 1（"全部被吃掉"），而 steps_ratio 0.965
        # 是**收敛变快**——这是 gap 无法承载收敛方向的直接证据。
        c_b = self.result["p3_pairs"]["C_vs_B"]
        self.assertEqual(c_b["gap_from_ratios"]["gap"], 1.0)
        self.assertLess(c_b["steps_ratio"], 1.0)

    def test_p3_c_vs_r_nofilter_ratio(self):
        r = self.result["p3_pairs"]["C_vs_R_nofilter"]
        self.assertAlmostEqual(r["cost_ratio"], 0.3144, places=4)
        self.assertAlmostEqual(r["wall_ratio"], 0.4986, places=4)

    def test_negative_control_wrong_denominator_is_caught(self):
        """负向对照：若把 cancel 的墙钟比按基线/臂写成 1.515，gap 会变成负值且符号反转。
        断言"错误写法算出来的 gap 与正确写法符号不同"，证明这一格确实会改变结论。"""
        # 正确口径：每步省 1.57%，墙钟省 34% ⇒ 缺口 +0.33（墙钟收益超过每步省，非"被吃掉"而是收敛变快）
        # 错误口径（写成基线/臂 1.515）：墙钟变成"变慢 51.5%" ⇒ 缺口被记成 1.0（全部被吃掉）
        cost_r = self.result["p4_arms"]["cancel"]["cost_ratio"]
        right = ledger.gap(0.6599, cost_r)["gap"]
        wrong = ledger.gap(1.5154, cost_r)["gap"]
        self.assertGreater(right, 0)
        self.assertEqual(wrong, 1.0)
        self.assertNotAlmostEqual(right, wrong, places=2)

    def test_verify_roundtrip(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            arch = Path(td) / "ledger.json"
            arch.write_text(json.dumps(self.result, ensure_ascii=False))
            self.assertEqual(ledger.verify(arch, self.result), [])

    def test_verify_detects_tampering(self):
        """负向对照：改动归档里一个数，verify 必须报错而不是静默通过。"""
        import copy
        import tempfile

        bad = copy.deepcopy(self.result)
        bad["p4_arms"]["freq"]["kappa" if "kappa" in bad["p4_arms"]["freq"] else "cost_ratio"] = 0.999
        with tempfile.TemporaryDirectory() as td:
            arch = Path(td) / "ledger.json"
            arch.write_text(json.dumps(bad, ensure_ascii=False))
            problems = ledger.verify(arch, self.result)
            self.assertTrue(problems, "篡改未被 verify 发现")


if __name__ == "__main__":
    unittest.main()
