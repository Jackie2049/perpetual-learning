#!/usr/bin/env python3
"""fusion_evidence 的单测。期望值一律写死常数（不与被测输出互推）。"""

import json
import unittest

import fusion_evidence as F


class TestConstruction(unittest.TestCase):

    def test_traces_identical_across_semantics(self):
        """§21.11 的前提不能放松：两个语义的忙/闲轨迹逐位相同。"""
        for level in F.LEVELS:
            a = F.make_replicas(level)
            b = F.make_replicas(level)
            self.assertEqual(a, b)

    def test_signal_streams_differ(self):
        """外部通道在两个语义下必须不同，否则判据无从谈起。"""
        a = F.make_signals(F.RUN_NLPC, 'high', 0)
        b = F.make_signals(F.RUN_LPC, 'high', 0)
        self.assertNotEqual(a['H_hits'], b['H_hits'])
        self.assertNotEqual(a['D_occupancy'], b['D_occupancy'])

    def test_nlpc_has_zero_hits_inside_window_only(self):
        """真空泡只在空窗里没有活；忙段照常有命中和 prefill。"""
        free = set(F.idle_ticks('high'))
        a = F.make_signals(F.RUN_NLPC, 'high', 3)
        self.assertEqual(sum(v for t, v in enumerate(a['H_hits']) if t in free), 0.0)
        self.assertEqual(sum(v for t, v in enumerate(a['I_prefill']) if t in free), 0.0)
        self.assertGreater(sum(v for t, v in enumerate(a['H_hits']) if t not in free), 0.0)


class TestRules(unittest.TestCase):

    def test_fuse_and_requires_both(self):
        self.assertTrue(F.fuse_and(True, True))
        self.assertFalse(F.fuse_and(True, False))
        self.assertFalse(F.fuse_and(False, True))
        self.assertFalse(F.fuse_and(False, False))

    def test_threshold_type_uses_busy_mean(self):
        # 忙段均值 1.0，阈值 0.5：窗口均值 0.4 < 0.5 ⇒ 空闲；0.6 ⇒ 不是
        self.assertTrue(F.threshold_type([0.4], [1.0] * 4))
        self.assertFalse(F.threshold_type([0.6], [1.0] * 4))

    def test_rate_ratio(self):
        self.assertAlmostEqual(F.rate_ratio([2.0, 2.0, 4.0, 4.0], {0, 1}, {2, 3}), 0.5)
        # 忙段为 0 时不能除零
        self.assertEqual(F.rate_ratio([0.0, 0.0], {0}, {1}), 0.0)
        self.assertEqual(F.rate_ratio([3.0, 0.0], {0}, {1}), float('inf'))


class TestDecisionDirection(unittest.TestCase):

    def test_high_load_lpc_misread_as_idle(self):
        """高负载下缓存命中的占用率也过阈值 ⇒ 被反判成真空泡（specificity 0）。"""
        c = F.classify(F.RUN_LPC, 'high', 0)
        self.assertTrue(c['occ_verdict_idle'])
        self.assertTrue(c['fused_verdict_idle'])

    def test_asymmetric_failure(self):
        """背景命中变大：真空泡 recall 掉到 0，缓存命中 specificity 保持 1。"""
        fail = F.scenario_and_rule()['failure_direction']
        self.assertEqual(fail['bg_rel_0.00']['nlpc_recall'], 1.0)
        self.assertEqual(fail['bg_rel_1.00']['nlpc_recall'], 0.0)
        for k, v in fail.items():
            self.assertEqual(v['lpc_false_idle'], v['lpc_n'], k)

    def test_identifiability_flips(self):
        """负向对照：背景命中等于缓存命中量级时，判据必须失效。"""
        ident = F.scenario_identifiability()
        self.assertEqual(ident['bg_hits_0']['levels_separable'], 2)
        self.assertEqual(ident['bg_hits_60']['levels_separable'], 0)


class TestResolutionCoupling(unittest.TestCase):

    def test_sampling_gate(self):
        """w < k 时任何相位都采不到；w < 2k 时采不到两次。"""
        s = F.scenario_min_conditions()['sampling']['sampling_10s']
        by_w = {r['window_s']: r for r in s}
        self.assertEqual(by_w[5]['phase_visible_rate'], 0.5)   # 10/5 相位中一半能看到
        self.assertEqual(by_w[10]['two_sample_phase_rate'], 0.0)
        self.assertEqual(by_w[20]['two_sample_phase_rate'], 1.0)
        s1 = F.scenario_min_conditions()['sampling']['sampling_1s']
        self.assertTrue(all(r['two_sample_phase_rate'] == 1.0 for r in s1))


class TestArchive(unittest.TestCase):

    def test_verify_passes(self):
        self.assertEqual(F.verify(), 0)

    def test_archive_has_all_sections(self):
        with open(F.OUT_PATH) as f:
            d = json.load(f)
        for k in ('config', 'channel_families', 'threshold_sweep',
                  'relative_vs_absolute', 'and_rule', 'min_conditions',
                  'identifiability'):
            self.assertIn(k, d)


if __name__ == '__main__':
    unittest.main()
