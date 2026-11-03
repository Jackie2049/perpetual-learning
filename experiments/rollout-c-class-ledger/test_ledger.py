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


class TestAccountChecks(unittest.TestCase):
    """第三批的四条口径检查。每条都配一个负向对照：把检查关掉或换成错的口径时，
    同一条断言必须失败——否则断言没有约束力。"""

    @classmethod
    def setUpClass(cls):
        p3 = json.loads(P3.read_text())
        p4 = json.loads(P4.read_text())
        cls.result = ledger.build(
            ledger.load_p3(P3),
            ledger.load_p4(P4),
            {"p3_meta": p3["meta"], "p4_meta": p4["meta"]},
        )

    # ---- ① 天花板饱和

    def test_p4_accuracy_sits_at_ceiling(self):
        """p4 的 base 有 28/30 种子最终准确率 = 1.0。此时 Δacc 的区间宽度
        （±0.0006–0.0011）比评测网格（0.005）还窄——只可能来自"少数种子差一格、多数打平"。"""
        ceil = self.result["account_checks"]["ceiling"]
        self.assertAlmostEqual(ceil["p4:freq"]["base_at_ceiling"], 28 / 30, places=6)
        self.assertAlmostEqual(ceil["p4:freq"]["arm_at_ceiling"], 22 / 30, places=6)
        self.assertAlmostEqual(ceil["p4:cap"]["arm_at_ceiling"], 24 / 30, places=6)

    def test_p4_nonzero_accuracy_pairs_are_few(self):
        """freq 只有 8/30 对非零、replay 2/30、cancel 5/30、cap 4/30。"""  # noqa: E501
        signs = self.result["account_checks"]["signs"]
        self.assertEqual(signs["freq"]["acc_n_nonzero"], 8)
        self.assertEqual(signs["replay"]["acc_n_nonzero"], 2)
        self.assertEqual(signs["cancel"]["acc_n_nonzero"], 5)
        self.assertEqual(signs["cap"]["acc_n_nonzero"], 4)

    def test_negative_control_ceiling_blind(self):
        """负向对照：阈值放到 1.0 以上时天花板比例恒为 0，"饱和"就看不见了。"""
        vals = [0.995, 1.0, 1.0, 0.99]
        self.assertEqual(ledger.ceiling_rate(vals, thr=1.01), 0.0)
        self.assertGreater(ledger.ceiling_rate(vals), 0.0)

    # ---- ② 离散指标的符号一致性

    def test_discrete_separation_beats_continuous(self):
        """p3 的步数差 26–29/30 非零，且与成本差**同向**（同一步数口径），故"删组收敛更快"
        在会话内是稳的。p4 四臂的步数差 16–24/30 非零、方向一致但更少，且**与墙钟差撞在
        同一个网格上**——这才是 p4 上不能把"收敛略快"当结论的原因。"""
        signs = self.result["account_checks"]["signs"]
        self.assertEqual(signs["C_vs_B"]["steps_sign"], -1)
        self.assertEqual(signs["B_vs_R_nofilter"]["steps_sign"], -1)
        self.assertGreaterEqual(signs["C_vs_B"]["steps_n_nonzero"], 25)
        for arm, expect in {"replay": 19, "cancel": 22, "cap": 16}.items():
            self.assertEqual(signs[arm]["steps_n_nonzero"], expect)

    def test_negative_control_sign_mix_returns_none(self):
        """负向对照：混符号时 _uniform_sign 必须返回 None（不能强行给方向）。"""
        self.assertIsNone(ledger._uniform_sign([1.0, -1.0]))
        self.assertEqual(ledger._uniform_sign([-2.0, -0.5]), -1)
        self.assertIsNone(ledger._uniform_sign([]))

    # ---- ③ 共线量的回退一致性

    def test_gen_equals_cost_in_all_p4_arms(self):
        """p4 的生成量恒等于步成本（逐种子），所以 paired_total_gen 与 paired_total_cost
        是同一个数——引用时不能当两条独立证据。"""
        coll = self.result["account_checks"]["collinear"]
        for arm in ("freq", "replay", "cancel", "cap"):
            self.assertTrue(coll[arm]["gen_equals_cost_total"])
            self.assertTrue(coll[arm]["diff_equals_cost_diff"]["gen"])
        cap = self.result["p4_arms"]["cap"]
        self.assertEqual(cap["gen_paired"]["mean_diff"], 0.0)
        self.assertEqual(
            self.result["p4_arms"]["freq"]["gen_paired"]["mean_diff"],
            -194308.8,
        )

    def test_cap_wall_zero_pairs_are_not_a_separation(self):
        """cap 的生成量与步成本逐对全同（Δ=0），但墙钟差**不是**全零：30 对里只有 14 对
        打平，其余 16 对差在 ±1.5 个步单位量级。所以它不是"墙钟没变"的证据，
        而是"离到 0.9 的距离在一个网格上下"——本批因此**不归因**。"""
        cap = self.result["p4_arms"]["cap"]
        self.assertEqual(cap["gen_paired"]["n_zero"], cap["gen_paired"]["n"])
        self.assertEqual(cap["wall_paired"]["n_zero"], 14)
        self.assertAlmostEqual(cap["wall_ratio"], 0.9369, places=3)
        self.assertAlmostEqual(cap["cost_ratio"], 1.0, places=9)

    def test_negative_control_collinear_breaks_when_inputs_differ(self):
        """负向对照：给一个生成量 ≠ 成本的臂，检查必须报 False。"""
        arm = {"total_gen": [22.0, 33.0], "total_cost": [11.0, 22.0], "walls_to_09": [5.0, 6.0],
               "steps_to_09": [1.0, 2.0], "n_traj": [1.0, 2.0], "final_acc": [0.5, 0.6],
               "prefix_cost": [5.0, 6.0]}
        base = {"total_gen": [10.0, 20.0], "total_cost": [10.0, 20.0], "walls_to_09": [5.0, 6.0],
                "steps_to_09": [1.0, 2.0], "n_traj": [1.0, 2.0], "final_acc": [0.5, 0.6],
                "prefix_cost": [5.0, 6.0]}
        rows = {"x": ledger.arm_row(arm, base)}
        self.assertFalse(ledger.collinear_checks(rows)["x"]["gen_equals_cost_total"])

    # ---- ④ 独立字段退化

    def test_traj_and_cost_degenerate_for_three_arms(self):
        """freq／replay／cancel 的 traj_ratio ≡ cost_ratio；只有 cap 能把两者分开
        （traj 0.282 vs cost 1.000）。故"省样本"这条读法在本模型里只有一个可用实例。"""
        dg = self.result["account_checks"]["degeneracy"]
        self.assertEqual(sorted(dg["separating_arms"]), ["cap"])
        self.assertEqual(sorted(dg["degenerate_arms"]), ["cancel", "freq", "replay"])
        self.assertAlmostEqual(self.result["p4_arms"]["cap"]["traj_ratio"], 0.2823, places=3)

    def test_negative_control_degeneracy_needs_split_arm(self):
        """负向对照：若所有臂都 traj≡cost，separating_arms 必须为空。"""
        rows = {a: {"traj_ratio": 0.5, "cost_ratio": 0.5} for a in ("freq", "cap")}
        dg = ledger.degeneracy_checks(rows)
        self.assertEqual(dg["separating_arms"], [])
        self.assertEqual(sorted(dg["degenerate_arms"]), ["cap", "freq"])

    # ---- 准确率网格

    def test_accuracy_grid_is_coarse_for_p4(self):
        """p4 最终评测 200 条 ⇒ 网格 0.005；p3 是 1600 条 ⇒ 0.000625。
        p4 的 Δacc 量级（0.0003–0.0013）低于一个网格。"""
        grid = self.result["account_checks"]["accuracy_grid"]
        self.assertAlmostEqual(grid["p4_grid"], 0.005, places=9)
        self.assertAlmostEqual(grid["p3_grid"], 0.000625, places=9)
        self.assertLess(abs(self.result["p4_arms"]["freq"]["acc_paired"]["mean_diff"]), grid["p4_grid"])


class TestThresholdBand(unittest.TestCase):
    """第五批的 L3 阈值检查。断言全部写死常数，不取被测输出自证。

    两条来自本批自查的错误判据都留在这里当负向对照：① 把 `_cross_or_complete`
    当成默认口径（未达阈值记全程）；② 用"压平到点之后"当摆幅应为 0 的对照。
    """

    @classmethod
    def setUpClass(cls):
        cls.th = ledger.build(ledger.load_p3(P3), ledger.load_p4(P4))["account_checks"][
            "threshold_band"
        ]

    def test_freq_span_matches_probe(self):
        """`freq` 五档 1.024→1.206、摆幅 18.2pp——与独立脚本 threshold_probe.py 同值。"""
        f = self.th["p4:freq"]
        self.assertAlmostEqual(f["by_threshold"]["0.85"]["steps_ratio"], 1.0243, places=3)
        self.assertAlmostEqual(f["by_threshold"]["0.95"]["steps_ratio"], 1.2064, places=3)
        self.assertAlmostEqual(f["span_pp"], 18.174, places=2)
        self.assertFalse(f["sign_flip"])

    def test_only_p3_A_flips_sign(self):
        flips = sorted(k for k, v in self.th.items() if v["sign_flip"])
        self.assertEqual(flips, ["p3:A_vs_R_nofilter"])
        a = self.th["p3:A_vs_R_nofilter"]["by_threshold"]
        self.assertLess(a["0.85"]["steps_ratio"], 1.0)
        self.assertGreater(a["0.87"]["steps_ratio"], 1.0)
        self.assertAlmostEqual(a["0.87"]["steps_ratio"], 1.0053, places=4)

    def test_level_matched_span_is_narrow_for_p4(self):
        """四臂在同一水平上的步数差跨阈值只摆 0.3–3.1 步（比率口径是 1.7–18.2pp）。

        即：`freq` 的"慢 2.4%→20.6%"里，形状只贡献 0.3→3.4 步，其余是阈值挪动本身。
        """
        for arm, want in (("freq", 3.13), ("replay", 0.33), ("cancel", 0.40), ("cap", 0.60)):
            lm = self.th[f"p4:{arm}"]["level_matched"]
            self.assertAlmostEqual(lm["span_steps"], want, places=2)
        lm = self.th["p4:freq"]["level_matched"]["by_threshold"]
        self.assertAlmostEqual(lm["0.85"]["mean_steps_arm_minus_base"], 0.30, places=2)
        self.assertAlmostEqual(lm["0.95"]["mean_steps_arm_minus_base"], 3.43, places=2)

    def test_level_matched_sign_is_shape_determined(self):
        """方向不是阈值给的：`freq` 五档都 ≥0、`cap` 五档都 <0，与 `steps_ratio` 的符号一致。"""
        for key in ("p4:freq", "p4:cap", "p3:R_vs_R_nofilter"):
            self.assertFalse(self.th[key]["level_matched"]["sign_flip"], key)
        self.assertGreaterEqual(
            min(
                r["mean_steps_arm_minus_base"]
                for r in self.th["p4:freq"]["level_matched"]["by_threshold"].values()
            ),
            0.0,
        )
        self.assertLess(
            max(
                r["mean_steps_arm_minus_base"]
                for r in self.th["p4:cap"]["level_matched"]["by_threshold"].values()
            ),
            0.0,
        )

    def test_negative_control_cross_or_complete_shifts_ratios(self):
        """负向对照①：两种"未达阈值"的口径给出不同读数 ⇒ 不能混用；默认口径遇到
        未达阈值的曲线必须留下 `n_miss`，而不是把它当 0 步。"""
        arms, base = [[0.1, 0.5, 0.95, 1.0]], [[0.2, 0.6, 0.98, 1.0]]
        self.assertAlmostEqual(ledger.steps_at(arms, base, 0.9)["steps_ratio"], 2 / 2)
        self.assertAlmostEqual(
            ledger.steps_at(arms, base, 0.9, complete_at_end=True)["steps_ratio"], 3 / 3
        )
        miss = ledger.steps_at([[0.1, 0.2, 0.3]], [[0.1, 0.2, 0.3]], 0.9)
        self.assertEqual(miss["n_miss"], 2)
        self.assertIsNone(miss["steps_ratio"])

    def test_negative_control_flatten_after_cross_moves_crossings(self):
        """负向对照②（记录本批的错误判据）：曲线 0.80→0.88→0.91→0.96→0.99，把 0.9 之后
        压平到 0.91，首次跨过 0.93 的时刻就从第 3 步挪到"永不"——**压平不可能保持摆幅为 0**。
        断言这条对照本身失败，防止以后又把它当"应给 0"的检验放回去。"""
        c = [0.80, 0.88, 0.91, 0.96, 0.99]
        flat = ledger._flatten_after_cross(c, 0.9)
        self.assertEqual(flat, [0.80, 0.88, 0.91, 0.91, 0.91])
        self.assertEqual(ledger.first_cross(c, 0.93), 3)
        self.assertIsNone(ledger.first_cross(flat, 0.93))


class TestInjectedDefects(unittest.TestCase):
    """把缺陷真的注入归档输入，检查必须随之改变——检查若对输入不敏感，就是装饰。
    这里改的是**被测脚本读取的 per_seed**，而不是被测输出。"""

    def _checks(self, mutate):
        import tempfile

        p4 = json.loads(P4.read_text())
        mutate(p4)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "p4.json"
            path.write_text(json.dumps(p4))
            return ledger.build(
                ledger.load_p3(P3), ledger.load_p4(path),
                {"p3_meta": json.loads(P3.read_text())["meta"], "p4_meta": p4["meta"]},
            )["account_checks"]

    def test_saturated_accuracy_moves_the_ceiling(self):
        def mutate(d):
            for s in d["arms"]["freq"]["per_seed"]:
                s["final_acc"] = 1.0

        ch = self._checks(mutate)
        self.assertEqual(ch["ceiling"]["p4:freq"]["arm_at_ceiling"], 1.0)
        self.assertEqual(ch["signs"]["freq"]["acc_n_nonzero"], 2)  # 只与 base 的 28 个满分打平

    def test_gen_decoupled_from_cost_is_caught(self):
        def mutate(d):
            for s in d["arms"]["cap"]["per_seed"]:
                s["total_gen"] += 100

        ch = self._checks(mutate)
        self.assertFalse(ch["collinear"]["cap"]["gen_equals_cost_total"])
        self.assertFalse(ch["collinear"]["cap"]["diff_equals_cost_diff"]["gen"])

    def test_degenerate_traj_is_caught(self):
        def mutate(d):
            for s in d["arms"]["cap"]["per_seed"]:
                s["total_traj"] = s["total_cost"]

        # 基线：cap 的 traj 0.282 / cost 1.000，是本模型里唯一能分开两个字段的臂。
        base_checks = self._checks(lambda d: None)
        self.assertEqual(base_checks["degeneracy"]["separating_arms"], ["cap"])
        # 逐种子置 traj := cost 时，总量比仍为 1.000（30×268800 被 30 整除），
        # 故总量口径对这一格天生不敏感——这本身就是要记录的局限，不是检查失效。
        self.assertFalse(self._checks(mutate)["degeneracy"]["traj_equals_cost"]["cap"])
        # 换逐种子不同值后敏感性才显现：freq 的 traj 不再是 cost 的等比放大。
        def mutate2(d):
            for i, s in enumerate(d["arms"]["freq"]["per_seed"]):
                s["total_cost"] += i * 1000
                s["total_traj"] = s["total_cost"]

        self.assertFalse(self._checks(mutate2)["degeneracy"]["traj_equals_cost"]["freq"])

    def test_threshold_band_follows_the_curve(self):
        """注入：把 `freq` 每条曲线的 0.90 以上整体拉高一档（到点更早），
        `steps_ratio` 五档必须一起下移、且 `level_matched` 的档间形状跟着变——
        检查若读的是缓存或写死的常数，这条会照样绿。"""

        def mutate(d):
            for s in d["arms"]["freq"]["per_seed"]:
                s["acc_curve"] = [min(1.0, v + 0.05) for v in s["acc_curve"]]

        before = self._checks(lambda d: None)["threshold_band"]["p4:freq"]
        after = self._checks(mutate)["threshold_band"]["p4:freq"]
        for t in ("0.85", "0.90", "0.95"):
            self.assertLess(
                after["by_threshold"][t]["steps_ratio"], before["by_threshold"][t]["steps_ratio"]
            )
        self.assertNotAlmostEqual(
            after["level_matched"]["by_threshold"]["0.95"]["mean_steps_arm_minus_base"],
            before["level_matched"]["by_threshold"]["0.95"]["mean_steps_arm_minus_base"],
            places=1,
        )
        # 天花板钳位后不应出现新的缺省对（每种子都仍能跨过 0.9）
        self.assertEqual(after["by_threshold"]["0.90"]["n_miss"], 0)


if __name__ == "__main__":
    unittest.main()
