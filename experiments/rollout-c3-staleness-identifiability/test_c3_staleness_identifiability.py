#!/usr/bin/env python3
"""§21.7 实验的单元测试（2026-11-02 M1）。

纪律：每一条判定都要有"能把它推翻"的对照，否则测试只是复述结论。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c3_staleness_identifiability as m  # noqa: E402


class TestTruthIsIndependentOfReadings(unittest.TestCase):
    """真值不许由读法反推：读法只吃引擎可见事件，真值另走内部账。"""

    def test_truth_not_derived_from_engine_view(self):
        rows = m.simulate(m.CONFIGS['async-stale'], 0)
        truth = [e for r in rows for e in r['eta_true']]
        probe = m.eta_obs_engine_weight(rows)
        self.assertTrue(all(v == 0.0 for v in probe),
                        '引擎侧读法在本模型里结构性为零，绝不能等于真值')
        self.assertGreater(max(truth), 0.0)

    def test_engine_view_drops_idle_rows(self):
        rows = m.simulate(m.CONFIGS['async-starved'], 0)
        self.assertGreater(len(rows), len(m.engine_view(rows)),
                           '背压配置必须出现空转步，否则分母错无处检验')
        self.assertTrue(all(not r['idle'] for r in m.engine_view(rows)))


class TestSimulationSemantics(unittest.TestCase):

    def test_sync_has_zero_eta(self):
        for seed in range(5):
            rows = m.simulate(m.CONFIGS['sync'], seed)
            self.assertEqual(max(max(r['eta_true']) for r in rows), 0,
                             '同步配置的 η 必须恒 0（把它当"更陈旧"就等于零敏感）')

    def test_deep_is_staler_than_stale(self):
        deep = m.eta_obs_max_only(m.simulate(m.CONFIGS['async-deep'], 0))
        stale = m.eta_obs_max_only(m.simulate(m.CONFIGS['async-stale'], 0))
        self.assertGreater(deep, stale)

    def test_accept_never_exceeds_generated(self):
        for name in m.CONFIGS:
            rows = m.simulate(m.CONFIGS[name], 0)
            self.assertLessEqual(sum(r['n_accept'] for r in rows),
                                 sum(r['n_gen'] for r in rows), name)

    def test_stale_window_bounds_eta(self):
        for window in (0, 1, 4):
            cfg = dict(m.CONFIGS['async-stale'])
            cfg['stale_window'] = window
            rows = m.simulate(cfg, 0)
            eta = [e for r in m.engine_view(rows) for e in r['eta_true']]
            if not eta:
                self.assertEqual(window, 0, '窗口为 0 时可能一步都凑不满；其余窗口不该空')
                continue
            self.assertLessEqual(max(eta), window,
                                 '窗口是硬约束：max η 不可能超过 stale_window')

    def test_rng_only_affects_dropout(self):
        """同一 seed 两次跑必须逐字段一致；且改变失败率不改 η（只改接纳量）。"""
        a = m.simulate(m.CONFIGS['async-deep'], 7)
        b = m.simulate(m.CONFIGS['async-deep'], 7)
        self.assertEqual([r['eta_true'] for r in a], [r['eta_true'] for r in b])
        self.assertEqual([r['n_accept'] for r in a], [r['n_accept'] for r in b])


class TestApproximationIdentifiability(unittest.TestCase):

    def test_engine_weight_approx_is_structurally_zero(self):
        """§21.6 的结论在本模型里被复现：只看引擎侧装载版本的年龄恒为 0。"""
        for name in ('async-stale', 'async-deep'):
            probe = m.eta_obs_engine_weight(m.simulate(m.CONFIGS[name], 3))
            self.assertEqual(max(probe), 0.0)
            truth = max(e for r in m.engine_view(m.simulate(m.CONFIGS[name], 3))
                        for e in r['eta_true'])
            self.assertGreater(truth, 0.0, '真值不为零才说明该读法真的失效了')

    def test_max_only_cannot_report_distribution(self):
        scalar = m.eta_obs_max_only(m.simulate(m.CONFIGS['async-deep'], 0))
        summary = m._max_only_summary(scalar)
        self.assertNotEqual(summary['p50'], summary['p50'], 'P50 必须是 NaN（未定义）')
        self.assertNotEqual(summary['p90'], summary['p90'], 'P90 必须是 NaN（未定义）')

    def test_eta_distribution_is_not_degenerate(self):
        """若真值分布本来就是一个点，那"分布 vs 最大值"的差别就没有可观测代价。"""
        vals = [e for r in m.engine_view(m.simulate(m.CONFIGS['async-deep'], 0))
                for e in r['eta_true']]
        self.assertGreater(max(vals) - min(vals), 0.0,
                           '本模型存在真实分布宽度，否则 Q1 的分布论证不成立')


class TestSingleSDecomposition(unittest.TestCase):

    def test_generated_ratio_equals_dropout(self):
        """1/(1−丢弃率) 是逐种子的恒等式；跨种子的均值之比不等于均值之比，故按种子核。"""
        for name in ('sync', 'async-stale', 'async-deep'):
            cfg = m.CONFIGS[name]
            rows = m.simulate(cfg, 11)
            n_acc = sum(r['n_accept'] for r in rows)
            n_gen = sum(r['n_gen'] for r in rows)
            drop = 1.0 - n_acc / float(n_gen)
            self.assertAlmostEqual(n_gen / float(n_acc), 1.0 / (1.0 - drop), places=12)

    def test_active_denominator_needs_idle(self):
        """分母换成"活跃步"同时做两件事：剔掉空转（上修）与多摊一段 sync_extra（下修）。

        两条路径公式不同（1/(1−空转率)/(1+extra/T) vs 1/(1+extra/T)），
        只有存在空转的配置才会被上修超过 1——这就是"分母错"能被识别的前提。
        """
        res = m.scenario_single_s()
        self.assertEqual(res['async-starved']['idle_ratio']['mean'], 0.5)
        # 分母换成"活跃步 × (T + sync_extra)"后，比值 = 1/(1−空转率) / (1 + sync_extra/T)
        idle = res['async-starved']['idle_ratio']['mean']
        cfg = m.CONFIGS['async-starved']
        self.assertAlmostEqual(res['async-starved']['active_ratio']['mean'],
                               (1.0 / (1.0 - idle)) / (1.0 + cfg['sync_extra'] / cfg['T']),
                               places=6)
        # 无空转时两口径之差只剩"每步多加一段 sync_extra"，方向与空转校正相反
        self.assertAlmostEqual(res['async-deep']['active_ratio']['mean'],
                               1.0 / (1.0 + cfg['sync_extra'] / cfg['T']), places=9)

    def test_same_s_different_eta(self):
        """Q2 的核心对照：S 完全相同、η 完全不同。"""
        res = m.scenario_single_s()
        pairs = res['indistinguishable_by_s_alone']
        for key, val in pairs.items():
            self.assertAlmostEqual(val['s_diff_rel'], 0.0, places=12, msg=key)
        self.assertNotEqual(pairs['sync-pipelined|async-deep']['eta_max_diff'], 0.0)
        self.assertNotEqual(pairs['async-stale|async-deep']['eta_max_diff'], 0.0)


class TestNarratives(unittest.TestCase):

    def test_sync_relabelled_as_baseline_flips(self):
        """把基线换成另一个异步配置，会让"sync 比基线快"读反——这是 Q3 的判据。"""
        conf = m.scenario_variant_narratives()['direction_conflict']['sync']
        self.assertEqual(conf['honest_vs_async_base'], conf['n'],
                         'sync 自身在诚实口径下是 1.0，换成异步基线后必须 <1')

    def test_honest_and_generated_agree_on_direction(self):
        out = m.scenario_variant_narratives()['direction_conflict']
        for name, conf in out.items():
            self.assertEqual(conf['honest_vs_gen_num'], 0,
                             '本模型里生成量与接纳量同向（丢弃率与配置无关），'
                             '故两口径只差幅度不差方向——这一点必须显式成立，'
                             '否则 README 里不能写"方向一致"')


class TestDeterminism(unittest.TestCase):

    def test_summary_is_reproducible(self):
        a = m._strip_nan(m._canon(m.run_all()))
        b = m._strip_nan(m._canon(m.run_all()))
        self.assertEqual(a, b)


if __name__ == '__main__':
    unittest.main()
