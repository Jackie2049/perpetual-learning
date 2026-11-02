#!/usr/bin/env python3
"""§21.11 实验的单元测试（2026-11-02 M1）。

纪律：每条判决都要有能推翻它的对照，并且期望值全部写成常数（不从被测函数推导）。
  ① 采样二值化的支撑数是**精确枚举**来的，不能用 ceil/round 近似；
     w/k 为整数时支撑集必须唯一（这正是"窗口整体丢失"不可能发生的条件）；
  ② 短窗在粗采样下会整段消失——这是 B2 在 10s／20s 下"读不到"的机制，
     而不是"读到了但相等"；
  ③ 镜像对必须逐位相同：同一实例、不同标签，任何统计量都不许分开；
  ④ 可加性是可加**读数**，不是可加**质量**（份额和相等，真气泡份额不变）。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import resolution_limits as rl  # noqa: E402


class TestSupportSpectrum(unittest.TestCase):
    def test_integer_ratio_gives_single_support(self):
        """w/k ∈ ℤ ⇒ 支撑集只有一个元素、且等于 w/k（位移不变，窗口必然可见）。"""
        for w, k in ((30, 1), (30, 2), (30, 5), (30, 10), (20, 20), (100, 20)):
            self.assertEqual(rl.support_spectrum(w, k), [w // k], msg=(w, k))

    def test_short_window_can_vanish_completely(self):
        """窗口短于采样间隔时，整段可能落在两个采样点之间 ⇒ 支撑 0。"""
        self.assertEqual(rl.support_spectrum(2, 10), [0, 1])
        self.assertEqual(rl.support_spectrum(8, 20), [0, 1])
        self.assertNotIn(0, rl.support_spectrum(12, 10))

    def test_support_is_never_ceil_of_ratio(self):
        """负向对照：ceil(w/k) 会在 w=2, k=10 时报 1，而真实最坏情况是 0。"""
        self.assertEqual(rl.support_spectrum(2, 10), [0, 1])
        self.assertNotEqual(min(rl.support_spectrum(2, 10)), 1)

    def test_support_set_has_at_most_two_values(self):
        for w in (1, 2, 3, 5, 7, 11, 13, 30, 47):
            for k in (1, 2, 5, 10, 20):
                self.assertLessEqual(len(rl.support_spectrum(w, k)), 2, msg=(w, k))


class TestTauThreshold(unittest.TestCase):
    def test_resolution_scales_with_tick(self):
        """最小可分辨窗口随采样间隔线性上升：tick=2→2s、5→5s、10→10s、20→20s。"""
        out = rl.scenario_tau_threshold()
        self.assertEqual(out['tick2_s']['min_tau_b2_full_accuracy'], 2)
        self.assertEqual(out['tick5_s']['min_tau_b2_full_accuracy'], 5)
        self.assertEqual(out['tick10_s']['min_tau_b2_full_accuracy'], 10)
        self.assertEqual(out['tick20_s']['min_tau_b2_full_accuracy'], 20)

    def test_below_one_tick_never_separable(self):
        """低于一个采样间隔的窗口：10s 采样下 1–9 秒全部 0/30——不是并列，是看不见。"""
        out = rl.scenario_tau_threshold()['tick10_s']
        self.assertEqual(out['never_separable'], list(range(1, 10)))

    def test_aliasing_band_exists_at_five_seconds(self):
        """5s 采样存在混叠带：τ_B2 ∈ {6,7,8,9} 时命中率既不是 0 也不是 1。"""
        band = rl.scenario_tau_threshold()['tick5_s']['ambiguous_band']
        self.assertEqual(band[:4], [6, 7, 8, 9])

    def test_min_ratio_at_ten_seconds_is_two(self):
        """10s 采样、τ_B2=10s：窗长比 1.2–1.5 仍不可分，≥2 才 30/30。"""
        sweep = {r['ratio']: r['hit_rate'] for r in rl.scenario_ratio_curve()['sweep']}
        self.assertEqual(sweep[1.0], 0.0)
        self.assertEqual(sweep[1.5], 0.0)
        self.assertEqual(sweep[2.0], 1.0)


class TestMirrorUnidentifiability(unittest.TestCase):
    def test_same_trace_two_labels(self):
        """同一份轨迹：所有统计量逐位相同——标签不可由轨迹恢复。"""
        out = rl.scenario_mirror_context()
        self.assertTrue(out['identical_instance'])
        for key in ('simultaneity', 'concentration', 'bubble_share', 'free_ticks', 'longest_run'):
            self.assertEqual(out['A'][key], out['B'][key], msg=key)

    def test_length_alone_sorts_nothing(self):
        """同时性、集中度相同而份额差 5 倍：形状统计量不含份额，无法据此排序。"""
        out = rl.scenario_mirror_length()
        self.assertTrue(out['same_sim'])
        self.assertTrue(out['same_conc'])
        self.assertAlmostEqual(out['share_ratio'], 5.0, places=9)

    def test_overlap_window_reads_like_b3(self):
        """全体同时的长窗口在三个统计量上与 B3 同形——§21.10 判据会把它归进 B2/B3 族。"""
        out = rl.scenario_component_mass()
        for w in ('window_30', 'window_60', 'window_120'):
            self.assertEqual(out[w]['simultaneity'], 1.0)
            self.assertEqual(out[w]['concentration'], 0.0)

    def test_alignment_changes_reading_not_bucket(self):
        """尾部窗口挪 40 格：读数变 0.1，判据的三个桶一个都没动。"""
        out = rl.scenario_mirror_misaligned()
        self.assertAlmostEqual(out['share_delta'], 0.1, places=9)
        self.assertEqual(out['aligned']['simultaneity'], out['shifted_by_40']['simultaneity'])


class TestAdditivityIsReadsNotMass(unittest.TestCase):
    def test_shares_add_but_true_bubble_does_not(self):
        out = rl.scenario_additivity_mass()
        self.assertAlmostEqual(out['union']['share'], out['union']['sum_of_parts'], places=9)
        self.assertAlmostEqual(out['true_bubble_share_in_union'], 0.375, places=9)
        self.assertNotAlmostEqual(out['union']['share'], out['true_bubble_share_in_union'], places=3)
        self.assertEqual(out['tail_contribution'], 0.0)


class TestRuleCircularity(unittest.TestCase):
    def test_rule_always_returns_a_bucket(self):
        """判据没有"以上都不是"：近乎随机的轨迹照样被分进某一桶。"""
        out = rl.scenario_rule_circularity()
        self.assertIn(out['assigned_bucket'], ('B2/B3', 'B4', 'B5'))


if __name__ == '__main__':
    unittest.main()
