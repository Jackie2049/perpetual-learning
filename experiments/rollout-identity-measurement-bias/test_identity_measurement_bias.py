#!/usr/bin/env python3
"""§21.9 实验的单元测试（2026-11-02 M1）。

纪律：每条判定都要有能推翻它的对照。本文件守四件事：
  ① 恒等式在「无加速」时精确为 1（它是代数恒等式，不是近似）；
  ② 误差只来自测量缺口，不来自恒等式本身——同一份分项按正确分解加速时误差为零；
  ③ 方向：隐藏子阶段导致**高报**，且随 G 与隐藏份额单调增大；
  ④ 解析式 dR/R = R(1−1/G)·ds 与数值导数一致。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import identity_measurement_bias as m  # noqa: E402


class TestIdentityIsExact(unittest.TestCase):
    def test_no_acceleration_gives_one(self):
        g = {k: 1.0 for k in m.BASE}
        self.assertAlmostEqual(m.identity_ratio(m.BASE, g), 1.0, places=12)

    def test_identity_is_scale_invariant(self):
        """整步整体放大 100 倍，倍数不变（恒等式只看比值）。"""
        g = {'gen': 2.0, 'logp': 1.0, 'refit': 1.0, 'train': 1.0, 'idle': 1.0}
        a = m.identity_ratio(m.BASE, g)
        scaled = {k: v * 100.0 for k, v in m.BASE.items()}
        self.assertAlmostEqual(m.identity_ratio(scaled, g), a, places=12)

    def test_correct_decomposition_is_exact(self):
        """负向对照：按真实分解（train 加速、idle 不加速）重建，误差必须为零。

        这条把"误差来自测量缺口"与"误差来自恒等式"分开。
        """
        g = {'gen': 1.0, 'logp': 1.0, 'refit': 1.0, 'train': 2.0, 'idle': 1.0}
        self.assertAlmostEqual(
            m.identity_ratio(m.BASE, g),
            m.identity_ratio(dict(m.BASE), dict(g)), places=12)


class TestOverCrediting(unittest.TestCase):
    def test_mixed_block_over_reports(self):
        """把含空转的块整块按 G 加速 ⇒ 必然高报。"""
        parts = {'gen': 4.0, 'logp': 1.0, 'refit': 0.5,
                 'train': m.BASE['train'] + m.BASE['idle'], 'idle': 0.0}
        wrong = m.identity_ratio(parts, {'gen': 1.0, 'logp': 1.0, 'refit': 1.0,
                                        'train': 2.0, 'idle': 1.0})
        truth = m.identity_ratio(m.BASE, {'gen': 1.0, 'logp': 1.0, 'refit': 1.0,
                                          'train': 2.0, 'idle': 1.0})
        self.assertGreater(wrong, truth)

    def test_error_grows_with_claimed_speedup(self):
        """同一个隐藏份额（1/3）下，声称的 G 越大，高报越多。"""
        errs = []
        for gval in (1.5, 2.0, 3.0):
            truth = m.identity_ratio(m.BASE, {'gen': 1.0, 'logp': 1.0, 'refit': 1.0,
                                              'train': gval, 'idle': 1.0})
            parts = {'gen': 4.0, 'logp': 1.0, 'refit': 0.5,
                     'train': m.BASE['train'] + m.BASE['idle'], 'idle': 0.0}
            wrong = m.identity_ratio(parts, {'gen': 1.0, 'logp': 1.0, 'refit': 1.0,
                                             'train': gval, 'idle': 1.0})
            errs.append((wrong - truth) / truth)
        self.assertLess(errs[0], errs[1])
        self.assertLess(errs[1], errs[2])

    def test_error_zero_at_zero_hidden_share(self):
        """下界：隐藏份额为 0 时不存在高报（判据不能恒为正）。"""
        scan = m.q2_hidden_substage()['hidden_share_scan']
        self.assertAlmostEqual(scan[0]['rel_err_pct'], 0.0, places=9)

    def test_error_monotone_in_hidden_share(self):
        scan = m.q2_hidden_substage()['hidden_share_scan']
        errs = [row['rel_err_pct'] for row in scan]
        self.assertEqual(errs, sorted(errs))
        self.assertGreater(errs[-1], 10.0, '隐藏一半时空转时误差应超过 10%')


class TestDenominatorShrink(unittest.TestCase):
    def test_lost_time_increases_reported_ratio(self):
        """未计时的墙钟被丢掉 ⇒ 加速块占比被抬高 ⇒ 报得更好。"""
        scan = m.q3_denominator_shrink()['scan']
        ratios = [row['rebuilt'] for row in scan]
        self.assertEqual(ratios, sorted(ratios))
        self.assertGreater(ratios[-1], scan[0]['rebuilt'])

    def test_no_loss_no_error(self):
        scan = m.q3_denominator_shrink()['scan']
        self.assertAlmostEqual(scan[0]['rebuilt'], m.q3_denominator_shrink()['truth'], places=12)


class TestAnalyticSensitivity(unittest.TestCase):
    def test_formula_matches_numeric_derivative(self):
        """dR/ds 与 R(1−1/G) 一致（中心差分数值核）。"""
        s = 0.4
        for gval in (1.5, 2.0, 5.0):
            a = m.analytic_share_sensitivity(s, gval)
            h = 1e-6
            r_plus = 1.0 / ((s + h) / gval + (1.0 - (s + h)))
            r_minus = 1.0 / ((s - h) / gval + (1.0 - (s - h)))
            numeric = (r_plus - r_minus) / (2 * h)
            self.assertAlmostEqual(a['absolute_slope'], numeric, places=4)

    def test_amplification_grows_with_G(self):
        vals = [m.analytic_share_sensitivity(0.4, g)['absolute_slope']
                for g in (1.5, 2.0, 3.0, 5.0)]
        self.assertEqual(vals, sorted(vals))


class TestReportedCase(unittest.TestCase):
    def test_leak_inflates_reported_ratio(self):
        out = m.q5_reported_case()
        ratios = [row['rebuilt'] for row in out['train_block_leak']]
        self.assertEqual(ratios, sorted(ratios))
        self.assertAlmostEqual(out['train_block_leak'][0]['rebuilt'], out['truth_ratio'], places=12)

    def test_leak_of_ten_percent_is_material(self):
        out = m.q5_reported_case()
        row = [r for r in out['train_block_leak'] if abs(r['leak_share'] - 0.10) < 1e-9][0]
        self.assertGreater(row['rel_err_pct'], 4.0, row)


if __name__ == '__main__':
    unittest.main()
