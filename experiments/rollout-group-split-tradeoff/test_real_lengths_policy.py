#!/usr/bin/env python3
"""real_lengths_policy 的单元测试（§13.30 第四批）。

钉住两件事：
  ① 排程器复现第三批的构造档读数（口径没漂）；
  ② **"更强调度抬税"不是无条件成立的**——组内离散度低于约 0.4 时符号相反。
     这条是本批最重要的发现，必须写成测试：否则下次谁再把 cv=0.6 的构造档结论
     当成通用规律引用，测试不会红。

    python3 -m unittest test_real_lengths_policy -v
"""

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


if __name__ == '__main__':
    unittest.main()
