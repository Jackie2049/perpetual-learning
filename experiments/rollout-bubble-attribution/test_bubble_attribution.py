#!/usr/bin/env python3
"""§21.10 实验的单元测试（2026-11-02 M1）。

纪律：每条判定都要有能推翻它的对照。
  ① 形状统计量在纯机制上必须取端点值（sim∈{0,1}、conc∈{0,1}）；
  ② 三类机制两两之间必须有一个统计量能把它们分开（否则判据不成立）；
  ③ 混淆必须真的让判据退化（碎片化降低同时性、tick 粗化让 B2/B3 并列）；
  ④ 同层可加性在支撑集不相交时必须精确（ratio=1），重叠时必须 delta=0。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bubble_attribution as m  # noqa: E402


class TestShapeEndpointValues(unittest.TestCase):
    def test_simultaneous_kinds_have_sim_one(self):
        for kind in ('B2', 'B3'):
            for seed in range(5):
                c = m.classifiers(m.make_trace(kind, seed))
                self.assertAlmostEqual(c['simultaneity'], 1.0, places=9, msg=(kind, seed))

    def test_staggered_kinds_have_sim_zero(self):
        for kind in ('B5', 'B4'):
            for seed in range(5):
                c = m.classifiers(m.make_trace(kind, seed))
                self.assertAlmostEqual(c['simultaneity'], 0.0, places=9, msg=(kind, seed))

    def test_b4_is_concentrated_b5_is_flat(self):
        for seed in range(5):
            self.assertAlmostEqual(m.classifiers(m.make_trace('B4', seed))['concentration'], 1.0, places=9)
            self.assertAlmostEqual(m.classifiers(m.make_trace('B5', seed))['concentration'], 0.0, places=9)

    def test_busy_has_no_bubble(self):
        c = m.classifiers(m.make_trace('busy', 0))
        self.assertEqual(c['bubble_share'], 0.0)
        self.assertEqual(c['free_ticks'], 0)


class TestSeparability(unittest.TestCase):
    def test_b3_vs_b5_separated_by_simultaneity(self):
        """负向对照：若删掉同时性，这两类就不可分（气泡份额几乎相同）。"""
        b3 = m.classifiers(m.make_trace('B3', 0))
        b5 = m.classifiers(m.make_trace('B5', 0))
        self.assertLess(abs(b3['bubble_share'] - b5['bubble_share']), 0.02,
                        '两类气泡份额本就接近——可分性必须来自形状而非大小')
        self.assertGreater(abs(b3['simultaneity'] - b5['simultaneity']), 0.99)

    def test_b5_vs_b4_separated_by_concentration(self):
        b5 = m.classifiers(m.make_trace('B5', 0))
        b4 = m.classifiers(m.make_trace('B4', 0))
        self.assertLess(abs(b5['bubble_share'] - b4['bubble_share']), 0.02)
        self.assertGreater(abs(b5['concentration'] - b4['concentration']), 0.99)

    def test_b2_vs_b3_not_separated_by_any_shape_stat(self):
        """B2 与 B3 只差窗口长度——同时性与集中度完全相同。"""
        b2 = m.classifiers(m.make_trace('B2', 0))
        b3 = m.classifiers(m.make_trace('B3', 0))
        self.assertAlmostEqual(b2['simultaneity'], b3['simultaneity'], places=9)
        self.assertAlmostEqual(b2['concentration'], b3['concentration'], places=9)
        self.assertGreater(b3['longest_run'], b2['longest_run'],
                           '唯一的区别只有窗口长度——这恰是它脆弱的证据')


class TestConfounders(unittest.TestCase):
    def test_fragmentation_erodes_simultaneity(self):
        vals = []
        for frag in (0.0, 0.02, 0.05, 0.10):
            rows = [m.classifiers(m.make_trace('B3', s, random_share=frag))['simultaneity']
                    for s in range(10)]
            vals.append(sum(rows) / len(rows))
        self.assertEqual(vals, sorted(vals, reverse=True))
        self.assertGreater(vals[0] - vals[-1], 0.3)

    def test_mixed_kind_lands_between_pure_kinds(self):
        pure3 = [m.classifiers(m.make_trace('B3', s))['simultaneity'] for s in range(10)]
        mixed = [m.classifiers(m.make_trace('mixed_B3_B4', s))['simultaneity'] for s in range(10)]
        self.assertLess(sum(mixed) / len(mixed), sum(pure3) / len(pure3))
        self.assertGreater(sum(mixed) / len(mixed), 0.3,
                           '混合体仍有相当的同时性——但已低于纯 B3')

    def test_coarse_ticks_tie_b2_and_b3(self):
        out = m.scenario_tick_granularity()
        self.assertTrue(out['tick1_s']['order_B3_gt_B2'])
        self.assertTrue(out['tick10_s']['tie'], '粗粒度下两类窗口被量化成同一长度')
        self.assertTrue(out['tick20_s']['tie'])


class TestAdditivity(unittest.TestCase):
    def test_disjoint_windows_are_exactly_additive(self):
        out = m.scenario_additivity()['disjoint_windows']
        self.assertAlmostEqual(out['ratio']['mean'], 1.0, places=9)

    def test_overlap_adds_nothing(self):
        out = m.scenario_additivity()['overlapping_idle']
        self.assertAlmostEqual(out['delta']['mean'], 0.0, places=9)


if __name__ == '__main__':
    unittest.main()
