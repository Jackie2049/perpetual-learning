#!/usr/bin/env python3
"""real_lengths_policy 的单元测试（§13.30 第四批）。

钉住两件事：
  ① 排程器复现第三批的构造档读数（口径没漂）；
  ② **"更强调度抬税"不是无条件成立的**——组内离散度低于约 0.4 时符号相反。
     这条是本批最重要的发现，必须写成测试：否则下次谁再把 cv=0.6 的构造档结论
     当成通用规律引用，测试不会红。

    python3 -m unittest test_real_lengths_policy -v
"""

import json
import os
import random
import statistics
import unittest

import group_split_tradeoff as m
import real_lengths_policy as r

SEEDS = list(range(30))


def mean_tax(cv, backlog, policy):
    vals = []
    for s in SEEDS:
        rng = random.Random(s)
        lens = [[max(1.0, rng.gauss(50.0, 50.0 * cv)) for _ in range(8)] for _ in range(16)]
        vals.append(m._makespan_by_policy(lens, backlog, policy)['tax'])
    return statistics.mean(vals)


class TestConstructBaselineReproduces(unittest.TestCase):
    """口径对齐：本脚本用的排程器必须给出第三批的 34.0%／36.8%。"""

    def test_fifo_matches_third_batch(self):
        self.assertAlmostEqual(mean_tax(0.6, 4, m.POLICY_FIFO), 0.3402, places=3)

    def test_lof_matches_third_batch(self):
        self.assertAlmostEqual(mean_tax(0.6, 4, m.POLICY_LOF), 0.3677, places=3)


class TestSignFlipsWithWithinGroupDispersion(unittest.TestCase):
    """负向对照：把第三批的结论当通用规律 → 在低离散负载上必须失败。"""

    def test_high_cv_lof_raises_tax(self):
        self.assertGreater(mean_tax(0.6, 4, m.POLICY_LOF), mean_tax(0.6, 4, m.POLICY_FIFO) + 0.02)

    def test_low_cv_lof_lowers_tax(self):
        """真实 rollout 的组内 cv≈0.155 落在这一侧：更强调度**降**税。"""
        self.assertLess(mean_tax(0.155, 4, m.POLICY_LOF), mean_tax(0.155, 4, m.POLICY_FIFO) - 0.02)

    def test_zero_load_still_clean(self):
        """无论离散度多少，组内长度全同时税必须为 0（排程器本身不能引入伪排队）。"""
        for policy in (m.POLICY_FIFO, m.POLICY_LOF):
            lens = [[50.0] * 8 for _ in range(16)]
            self.assertLess(abs(m._makespan_by_policy(lens, 4, policy)['tax']), 1e-12)


class TestIccIsNotTheControl(unittest.TestCase):
    """ICC 单独扫描时符号不变 ⇒ 不能用"组内相关高"解释真实档的符号翻转。"""

    def test_icc_sweep_keeps_positive_sign(self):
        for icc in (0.0, 0.5, 0.75):
            fifo, lof = [], []
            for s in SEEDS:
                lens = r._sample_icc_group_lens(random.Random(s), icc)
                fifo.append(m._makespan_by_policy(lens, 4, m.POLICY_FIFO)['tax'])
                lof.append(m._makespan_by_policy(lens, 4, m.POLICY_LOF)['tax'])
            # 组内 cv 固定 0.6 时，四档 ICC 的位移都应为正
            self.assertGreater(statistics.mean(lof), statistics.mean(fifo))


class TestTruncationUsesArmCap(unittest.TestCase):
    """负向对照：截断率必须按本档的 max_response 算。

    抓的是实际发生的错误——`describe()` 曾把上限写死 256，于是 cap1024 档被报成
    "截断 45%／61%"，而 §13.50 的实测是 0.03%／0.65%，四个臂还被抹成同一对值。
    """

    def test_cap1024_arm_not_read_as_256(self):
        lens = [[300.0] * 8 for _ in range(16)]   # 全部 300：超过 256、不到 1024
        self.assertEqual(r.describe(lens, cap=1024.0)['truncated_fraction'], 0.0)
        self.assertEqual(r.describe(lens, cap=256.0)['truncated_fraction'], 1.0)
        self.assertEqual(r.describe(lens, cap=1024.0)['max_response'], 1024.0)

    def test_arm_cap_routes_by_label(self):
        self.assertEqual(r.arm_cap('qwen15b_cap1024'), 1024.0)
        self.assertEqual(r.arm_cap('qwen05b_cap1024'), 1024.0)
        self.assertEqual(r.arm_cap('qwen15b_cap256'), 256.0)
        self.assertEqual([c for _, f in r.REAL_ARMS for c in [r.arm_cap(_)]],
                         [256.0, 1024.0, 256.0, 1024.0])

    def test_build_summary_passes_arm_cap_to_describe(self):
        """接线检查：`build_summary()` 必须把 cap 传进 `describe()`，否则前两条形同虚设。

        这里替换 `describe()` 截获它收到的 cap，不需要跑排程，成本与臂数同阶。
        """
        seen = {}
        orig = r.describe

        def spy(groups, cap=256.0):
            seen[cap] = seen.get(cap, 0) + 1
            return orig(groups, cap=cap)

        r.describe = spy
        try:
            r.build_real_arm_structure()
        finally:
            r.describe = orig
        self.assertEqual(seen, {256.0: 2, 1024.0: 2})

    def test_archived_truncation_matches_13_50(self):
        """归档值与 §13.50 实测必须同向：cap256 高截断、cap1024 近乎不截断。"""
        with open(r.SUMMARY_PATH) as f:
            real = json.load(f)['real_arms']
        for label, arm in real.items():
            frac = arm['structure']['truncated_fraction']
            self.assertEqual(arm['structure']['max_response'], r.arm_cap(label))
            if label.endswith('cap1024'):
                self.assertLess(frac, 0.01, label)      # 实测 0.03%／0.65%
            else:
                self.assertGreater(frac, 0.4, label)    # 实测 45.3%／60.8%


class TestSignStabilityDiagnostics(unittest.TestCase):
    """§13.30 第五批：均值不够，还要看符号与基线抖动。

    上一批在 cap256 两档上把「−0.95pp（27/30 为负）」写成方向性结论，本批补的
    三个诊断是判它能不能当结论用的门槛。每条都配一个**在把被测代码改坏时必须
    变红**的断言。
    """

    @staticmethod
    def _flat_groups():
        """组内长度全同 → 两臂读数恒等，用于造"无分辨率"的输入。"""
        return [[100.0] * 8 for _ in range(16)]

    def test_identical_arms_report_no_resolution(self):
        """负向对照：两臂完全相同时 `n_identical` 必须吃掉全部种子。

        若实现把"差为 0"错算成正号（例如写成 `d >= 0`），这里会报 30 个阳性、
        `same_sign_fraction` 报 1.0——一个本无分辨率的格子被读成"稳定"。
        """
        st = r.policy_sign_stability(self._flat_groups(), 16, 4, m.POLICY_LOF, seeds=SEEDS)
        self.assertEqual(st['n_identical'], len(SEEDS))
        self.assertEqual(st['n_positive'], 0)
        self.assertEqual(st['n_negative'], 0)

    def test_sign_counts_sum_to_seed_count(self):
        """三态计数必须闭合，且符号混杂的格子不能报满分同号。"""
        groups = r.load_real_groups(os.path.join(r.REAL_DIR, 'qwen15b_gsm8k_n8.jsonl.gz'))
        st = r.policy_sign_stability(groups, 16, 8, m.POLICY_LOF, seeds=SEEDS)
        self.assertEqual(st['n_positive'] + st['n_negative'] + st['n_identical'], len(SEEDS))
        # 8 批这一格在 30 个种子上符号混杂
        self.assertLess(st['same_sign_fraction'], 0.8)

    def test_fifo_band_is_zero_on_identical_groups(self):
        """负向对照：组内长度全同时换序不改变 FIFO 读数，带宽必须是 0。"""
        band = r.fifo_reorder_band(self._flat_groups(), 16, 4, seeds=SEEDS)
        self.assertEqual(band['max_pp'], 0.0)

    def test_fifo_band_actually_shuffles(self):
        """负向对照：若实现忘了 shuffle，带宽恒 0，这条会失败。"""
        groups = r.load_real_groups(os.path.join(r.REAL_DIR, 'qwen15b_gsm8k_n8.jsonl.gz'))
        band = r.fifo_reorder_band(groups, 16, 4, seeds=list(range(6)))
        self.assertGreater(band['max_pp'], 0.5)

    def test_arrival_shift_reports_both_orders(self):
        """到达顺序假设的对照：两种到达序都要给出读数，不能只报一种。"""
        groups = r.load_real_groups(os.path.join(r.REAL_DIR, 'qwen05b_gsm8k_n8_cap1024.jsonl.gz'))
        sh = r.arrival_assumption_shift(groups, 16, 4, m.POLICY_LOF, seeds=list(range(10)))
        self.assertEqual(sh['n_total'], 10)
        # cap1024 4 批是本批符号最稳的一格：换到达序后绝大多数种子仍同号
        self.assertGreaterEqual(sh['n_same_sign'], 8)


if __name__ == '__main__':
    unittest.main()
