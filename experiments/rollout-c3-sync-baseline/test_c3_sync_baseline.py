#!/usr/bin/env python3
"""§21.8 实验的单元测试（2026-11-02 M1）。

纪律：每条判定都要有能推翻它的对照。本文件重点守四件事：
  ① 因果记账（版本只在 update 完成时推进）——否则 η 会被记成 0 而失去意义；
  ② 暂停语义（暂停同步必须 η 恒 0；流水同步必须出现 η=1）；
  ③ 估计量的失效方式（B0 结构性、B2 的条件可辨识、B2' 的退化）；
  ④ 一个"看起来合理但会错"的读法必须真的被判错（负向对照）。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c3_sync_baseline as m  # noqa: E402


class TestCausalAccounting(unittest.TestCase):
    def test_eta_zero_only_at_cold_start(self):
        """本模型的因果约束：η=0 只可能出现在第 0 步。"""
        for k in (1, 2, 4):
            for r in (0.1, 0.7, 2.0):
                tr = m.simulate(m.cfg_of(k, r), 3)
                zeros = [s['idx'] for s in tr['steps'] if s['eta'] == 0]
                self.assertEqual(zeros, [0], (k, r, zeros))

    def test_eta_never_negative(self):
        for k in (1, 4):
            for r in (0.1, 2.0):
                tr = m.simulate(m.cfg_of(k, r), 5)
                self.assertTrue(all(s['eta'] >= 0 for s in tr['steps']))

    def test_version_advances_once_per_update(self):
        """一步一个版本：每步的 update 时长必须等于该步自带的 u。"""
        tr = m.simulate(m.cfg_of(1, 0.3), 0)
        for s in tr['steps']:
            self.assertAlmostEqual(s['end'] - s['start'], s['u'], places=9,
                                   msg='每步的 update 时长必须等于该步自带的 u（版本只推进一次）')


class TestPauseSemantics(unittest.TestCase):
    def test_paused_is_zero_eta_everywhere(self):
        for r in (0.1, 0.5, 2.0):
            tr = m.simulate(m.cfg_of(1, r, paused=True), 0)
            self.assertTrue(all(s['eta'] == 0 for s in tr['steps']),
                            '暂停同步必须 η 恒 0（否则暂停没有把版本钉住）')

    def test_pipelined_never_zero_after_start(self):
        """流水同步（更新期间引擎继续跑）在因果记账下必然产生 η≥1。"""
        for k in (1, 4):
            tr = m.simulate(m.cfg_of(k, 0.5, paused=False), 0)
            tail = [s['eta'] for s in tr['steps'][1:]]
            self.assertTrue(tail and min(tail) >= 1, (k, tail[:5]))

    def test_gap_equals_update_time_when_generation_limited(self):
        """K=1、生成受限（U<T）时 gap = U·(1−1/steps)：每步代价是 U，首段被冷启动稀释。

        首段（第 0 步）两形态墙钟相同（都是 T+U），故均值差恰为 (steps−1)/steps·U。
        这条结构性恒等式同时守住了「版本只推进一次」与「暂停只延迟重启」两件事。
        """
        for r in (0.1, 0.5):
            for steps in (20, 200):
                cfg_p = m.cfg_of(1, r, paused=True)
                cfg_p['steps'] = steps
                cfg_q = m.cfg_of(1, r, paused=False)
                cfg_q['steps'] = steps
                a = m.trace_stats(m.simulate(cfg_p, 0))['step_wall_mean']
                b = m.trace_stats(m.simulate(cfg_q, 0))['step_wall_mean']
                self.assertAlmostEqual(a - b, r * (steps - 1) / float(steps), places=9,
                                       msg=(r, steps, a, b, a - b))

    def test_gap_shrinks_with_replicas(self):
        """K 变大后队列吸收等待：同一 U 下 gap 明显变小（K=4 的 0.0245 对 K=1 的 0.0995）。"""
        g1 = (m.trace_stats(m.simulate(m.cfg_of(1, 0.1, paused=True), 0))['step_wall_mean'] -
              m.trace_stats(m.simulate(m.cfg_of(1, 0.1, paused=False), 0))['step_wall_mean'])
        g4 = (m.trace_stats(m.simulate(m.cfg_of(4, 0.1, paused=True), 0))['step_wall_mean'] -
              m.trace_stats(m.simulate(m.cfg_of(4, 0.1, paused=False), 0))['step_wall_mean'])
        self.assertGreater(g1, 3.0 * g4, (g1, g4))

    def test_gap_vanishes_when_update_limited(self):
        """更新受限（U>T/K）时队列吸收等待：两形态墙钟相同。"""
        a = m.trace_stats(m.simulate(m.cfg_of(4, 2.0, paused=True), 0))['step_wall_mean']
        b = m.trace_stats(m.simulate(m.cfg_of(4, 2.0, paused=False), 0))['step_wall_mean']
        self.assertLess(abs(a - b), 0.02, (a, b))


class TestB0Structural(unittest.TestCase):
    def test_b0_recovers_truth_on_paused_sync(self):
        for r in (0.1, 0.3, 0.5):
            tr = m.simulate(m.cfg_of(1, r, paused=True), 0)
            b0, n0 = m.b0_eta0_step_wall(tr)
            self.assertAlmostEqual(b0, 1.0 + r, places=6)
            self.assertEqual(n0, len(tr['steps']), '暂停同步应当每一步都是 η=0')

    def test_b0_undefined_after_discarding_cold_start(self):
        """丢弃第 0 步后，B0 在 async trace 上结构性无定义（不是数值问题）。"""
        for k in (1, 4):
            tr = m.simulate(m.cfg_of(k, 0.3), 0)
            b0, n0 = m.b0_eta0_step_wall({'steps': tr['steps'][1:]})
            self.assertIsNone(b0)
            self.assertEqual(n0, 0)

    def test_b0_value_insensitive_to_cluster_size(self):
        """B0 恒等于 T+U，与副本数无关——它读不到集群规模，用它当基线会把倍数按 K 压扁。"""
        vals = []
        for k in (1, 2, 4, 8):
            tr = m.simulate(m.cfg_of(k, 0.3), 0)
            b0, _ = m.b0_eta0_step_wall(tr)
            vals.append(round(b0, 9))
        self.assertEqual(len(set(vals)), 1, vals)

    def test_b0_understates_async_speedup(self):
        """负向对照：拿 B0 当基线时，K=8 的 async 收益被读成 1.00×（真实 4.26–8.53×）。"""
        tr = m.simulate(m.cfg_of(8, 0.3), 0)
        b0, _ = m.b0_eta0_step_wall(tr)
        speed_b0 = 1.3 / b0
        self.assertAlmostEqual(speed_b0, 1.0, places=6)
        cycle = m.trace_stats(tr)['step_wall_mean']
        self.assertGreater(1.3 / cycle / speed_b0, 3.0,
                           '真实收益至少是 B0 读数的 3 倍，负向对照才成立')


class TestB2Conditions(unittest.TestCase):
    def test_b2_undefined_without_no_wait_step(self):
        """K=1 确定性从不空等 ⇒ 没有"无等待步" ⇒ Û 无定义 ⇒ B2 结构性不可用。"""
        tr = m.simulate(m.cfg_of(1, 0.1), 0)
        self.assertEqual(m.trace_stats(tr)['no_wait_frac'], 0.0)
        self.assertIsNone(m.b2_count_minus_idle(tr, 1))

    def test_b2_close_in_identifiable_region(self):
        tr = m.simulate(m.cfg_of(4, 0.3), 0)
        e = m.b2_count_minus_idle(tr, 4)
        self.assertIsNotNone(e)
        self.assertLess(abs(e['baseline'] - 1.3) / 1.3, 0.02)

    def test_b2_naive_undervalues_u_under_jitter(self):
        tr = m.simulate(m.cfg_of(4, 0.7, update_cv=0.3), 0)
        e = m.b2_naive_min_wall(tr, 4)
        self.assertLess(e['u_hat'], 0.7, '最小墙钟只能低估 U，不可能等于多档抖动的中位数')

    def test_median_gap_estimator_biased_low(self):
        """B2'（到达间隔中位数）在 K=4 有抖动时仍系统性低估 T。"""
        tr = m.simulate(m.cfg_of(4, 0.3, arrival_cv=0.3), 0)
        e = m.b2_median_gap(tr, 4)
        self.assertLess(e['t_hat'], 1.0, e)

    def test_median_gap_degenerates_without_jitter(self):
        """退化对照：副本同步启动且无抖动时，间隔中位数 → 0，B2' 完全失效。"""
        tr = m.simulate(m.cfg_of(4, 0.3), 0)
        e = m.b2_median_gap(tr, 4)
        self.assertLess(e['t_hat'], 0.2, e)


class TestWindowEscape(unittest.TestCase):
    def test_window_zero_forces_eta_zero_and_kills_throughput(self):
        tr = m.simulate(m.cfg_of(4, 0.3, queue='fifo', window=0), 0)
        st = m.trace_stats(tr)
        self.assertEqual(st['eta0_frac'], 1.0, '窗口收到 0 时每一步都是 η=0')
        self.assertGreater(st['dropped'], 0, 'η=0 的代价是丢弃旧批')
        self.assertLess((1.3) / st['step_wall_mean'], 1.0, '合成吞吐必须低于同步')

    def test_window_zero_b0_no_longer_matches_truth(self):
        """窗口 0 时 B0 虽然处处有定义，仍不等于 T+U（它读的是被窗口改造后的节奏）。"""
        tr = m.simulate(m.cfg_of(4, 0.3, queue='fifo', window=0), 0)
        b0, _ = m.b0_eta0_step_wall(tr)
        self.assertGreater(abs(b0 - 1.3) / 1.3, 0.4, b0)

    def test_window_large_keeps_cold_start_only(self):
        tr = m.simulate(m.cfg_of(4, 0.3, queue='fifo', window=8), 0)
        self.assertLess(m.trace_stats(tr)['eta0_frac'], 0.01)


if __name__ == '__main__':
    unittest.main()
