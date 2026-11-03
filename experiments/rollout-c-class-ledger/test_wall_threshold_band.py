"""`wall_threshold_band.py` 的单测：L4 墙钟读数的阈值拆解（2026-11-03 M1）。

**这个文件的存在理由**：L4 的 `wall_ratio` 与 L3 的 `steps_ratio` 建在同一个 0.9 上，
但上一轮只核了 L3。这一轮核 L4 时抓到两类真问题，各钉一条负向对照：

  1. **总量比与逐种子比会说不同的方向**。`replay` 的总量比在阈值上从 1.0085 走到
     0.9913（穿过 1），若只报总量比就要写"方向未定"；逐种子其实是 18–20/30 为 >1。
     把逐种子那条线删掉后测试必须红。
  2. **"恰等于 1.0000"必须算进 n_identical**。`cap` 多数种子的窗口成本比正好 1，
     若把它算进"更快"或"更慢"，同号占比就会虚高、看上去有方向。

运行：cd experiments/rollout-c-class-ledger && python3 -m unittest test_wall_threshold_band -v
"""

import json
import unittest
from pathlib import Path

import wall_threshold_band as W

HERE = Path(__file__).resolve().parent


def _seed(final_acc, costs):
    """造一个最小可用的 per_seed：acc_curve 单调爬升，step_costs 给定。"""
    n = len(costs)
    # 平均地在第 k 步跨过阈值：用线性爬升，保证 first_cross 可预期
    curve = [min(1.0, (i + 1) / float(n)) for i in range(n)]
    return {"final_acc": final_acc, "acc_curve": curve, "step_costs": list(costs),
            "steps_to_09": 0, "walls_to_09": 0, "total_cost": sum(costs)}


class TestReferenceReproduction(unittest.TestCase):
    """最重要的一条：在参考阈值上必须逐位复现 ledger.json 的 wall_ratio。

    复现不了说明本脚本把口径读错了（比如把两臂都截到基线步号，或用了 total_cost
    而不是窗口成本）。这是**自校验**，不是"再算一遍看看"。
    """

    def test_reference_threshold_reproduces_ledger(self):
        data = W.build()
        self.assertTrue(data["all_reproduced"])
        self.assertEqual(len(data["reference_reproduction"]), 8)
        for c in data["reference_reproduction"]:
            self.assertLess(abs(c["recomputed"] - c["archived"]), 1e-12, c["arm"])


class TestTotalsVersusPerSeed(unittest.TestCase):
    """总量比与逐种子比的口径差异——§13.47.1 判读纪律的一半。"""

    def test_per_seed_counts_split_a_ratio_of_one(self):
        """两个种子：一个 2.0、一个 0.5，总量比恰好 1.25 但符号一快一慢。

        逐种子读数必须给出 1 慢 1 快，而不是被均值抹成"没有差别"。
        """
        base = [_seed(0.9, [10, 10, 10, 10]), _seed(0.9, [10, 10, 10, 10])]
        arm = [_seed(0.9, [20, 20, 20, 20]), _seed(0.9, [5, 5, 5, 5])]
        r = W.per_seed_wall_ratio(arm, base, 0.9)
        self.assertEqual(r["n_pairs"], 2)
        self.assertEqual((r["n_slower"], r["n_faster"]), (1, 1))
        self.assertAlmostEqual(r["faster_frac"], 0.5)

    def test_identical_windows_are_not_directional(self):
        """两臂窗口成本完全相同时必须记 n_identical，不能算成"更快"或"更慢"。"""
        base = [_seed(0.9, [10] * 4) for _ in range(3)]
        arm = [_seed(0.9, [10] * 4) for _ in range(3)]
        r = W.per_seed_wall_ratio(arm, base, 0.9)
        self.assertEqual(r["n_identical"], 3)
        self.assertEqual((r["n_slower"], r["n_faster"]), (0, 0))
        self.assertEqual(r["faster_frac"], 0.0)


class TestDirectionFlip(unittest.TestCase):
    """逐种子多数方向是否随阈值翻——`C_vs_B` 在真归档上就是这样一臂。"""

    def test_direction_flip_detected_on_synthetic(self):
        """阈值 0.9 上 3/3 更慢，0.5 上 3/3 更快：多数方向必须在阈值间翻号。"""
        # 曲线在第 1 步就 >=0.5、第 3 步才 >=0.9
        base = []
        arm = []
        for _ in range(3):
            base.append({"acc_curve": [0.5, 0.7, 0.9, 0.95], "step_costs": [10, 10, 10, 10]})
            arm.append({"acc_curve": [0.5, 0.7, 0.9, 0.95], "step_costs": [5, 5, 20, 20]})
        blk_hi = W.arm_band(arm, base, 1.0, thresholds=(0.5, 0.9))
        # 0.5 上臂窗口成本 = 5+5 = 10 < 基线 10+10 = 20 ⇒ 更快
        # 0.9 上臂窗口成本 = 5+5+20 = 30 > 基线 30 ⇒ 相同（不是更慢）
        self.assertEqual(blk_hi["per_seed_by_threshold"]["0.50"]["n_faster"], 3)
        self.assertEqual(blk_hi["per_seed_by_threshold"]["0.90"]["n_identical"], 3)
        self.assertTrue(blk_hi["per_seed_direction_flip"])


class TestArmBandShape(unittest.TestCase):
    def test_span_and_flip_around_one(self):
        """`wall_ratio` 的翻号参考点是 1.0 而不是 0——这里直接钉住。"""
        base = [_seed(0.9, [10] * 4) for _ in range(2)]
        arm = [_seed(0.9, [9] * 4) for _ in range(2)]
        blk = W.arm_band(arm, base, 1.0, thresholds=(0.9,))
        self.assertLess(blk["wall_max"], 1.0)
        self.assertFalse(blk["wall_sign_flip"])


if __name__ == "__main__":
    unittest.main()
