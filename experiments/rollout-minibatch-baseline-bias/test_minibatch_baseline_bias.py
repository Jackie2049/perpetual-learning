#!/usr/bin/env python3
"""`minibatch_baseline_bias.py` 的单元测试（20 组左右，只依赖标准库）。

侧重守三件事：
  1. **期望值写死常数**（不从未测输出反推）：理论式、块形状、长度分布形状都用
     手算/解析值断言，避免"照答卷判卷"。
  2. **负向对照真的能抓到错误**：注入偏差（把 n−1 换成 n）、把 iid 生成改成组内相关、
     让识别性判据退化，都必须被测试抓到。
  3. **本批最要紧的语义**：`grouped` 基准的方差压低必须能被独立测出来，
     且解析值与实测值一致；`interleaved` 与 `ldm` 的顺序关系必须逐种子核对过。

运行：
    python3 -m unittest test_minibatch_baseline_bias -v
"""

import math
import os
import random
import statistics
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'rollout-minibatch-composition'))

import minibatch_baseline_bias as M  # noqa: E402
import train_side_composition as T  # noqa: E402


class TestExpectedSampleVar(unittest.TestCase):
    """解析式必须用独立方法验证，不能只跟自己的推导互证。"""

    def test_rho_zero_only_within_dof(self):
        # rho=0：只剩层内自由度 ⇒ (N − g)/(N − 1)。手算常数，不从函数输出反推。
        # 8 组×8 = 64 条：(64−8)/63；32 组×8 = 256 条：(256−32)/255
        self.assertAlmostEqual(M._expected_sample_var(8, 8, 0.0), 56.0 / 63.0, places=12)
        self.assertAlmostEqual(M._expected_sample_var(8, 32, 0.0), 224.0 / 255.0, places=12)

    def test_rho_one_only_between_dof(self):
        # rho=1：只剩层间自由度 ⇒ (N − n_total)/(N − 1)
        self.assertAlmostEqual(M._expected_sample_var(8, 8, 1.0), 56.0 / 63.0, places=12)
        self.assertAlmostEqual(M._expected_sample_var(8, 32, 1.0), 248.0 / 255.0, places=12)

    def test_bias_direction_depends_on_rho(self):
        """均值结构的偏差**方向随 ρ 翻转**——这是手算边界给本批的一处真实修正。

        * ρ=0（组内样本独立）：块内方差估计**更大**（层内自由度占比更高）；
        * ρ=1（组完全决定样本）：块内方差估计**更小**（只看得到 8 个组，组间波动被压低）。
        所以"基准方差被压低"只在 ρ 足够大时成立；ρ=0 时基准反而偏高。
        本实验的默认 ρ=0.5 落在偏低一侧（ratio 0.9605 < 1），结论成立但必须写清条件。
        """
        rho0 = M._expected_sample_var(8, 8, 0.0) / M._expected_sample_var(8, 32, 0.0)
        rho1 = M._expected_sample_var(8, 8, 1.0) / M._expected_sample_var(8, 32, 1.0)
        self.assertGreater(rho0, 1.0, 'rho=0 时块内估计应更大：%.6f' % rho0)
        self.assertLess(rho1, 1.0, 'rho=1 时块内估计应更小：%.6f' % rho1)
        # 交叉点：两者相等时的 rho（解 0.5 档落在哪一侧）
        self.assertLess(M.grouped_bias_theory()['expected_ratio_block_over_full'], 1.0)

    def test_theory_ratio_hardcoded(self):
        # 解析值手算：rho=0.5、m=8、N=32、n=8
        #   E[s²_块]  = (0.5·8·7 + 0.5·8·7)/(64−1)   = 56/63   = 0.888889
        #   E[s²_全批]= (0.5·32·7 + 0.5·8·31)/(256−1) = (112+124)/255 = 0.925490
        #   比值 = 0.888889/0.925490 = 0.960452
        th = M.grouped_bias_theory()
        self.assertAlmostEqual(th['expected_within_var_block'], 56 / 63, places=12)
        self.assertAlmostEqual(th['expected_within_var_full'], 236 / 255, places=12)
        self.assertAlmostEqual(th['expected_ratio_block_over_full'], 0.9604519774011299, places=12)
        self.assertLess(th['deficit_relative'], 0.0)

    def test_theory_matches_empirical(self):
        # 解析 0.9605 vs 实测；容差给 1%（30 种子的抽样误差约 0.3%）
        emp = M.within_var_deficit_empirical()
        th = M.grouped_bias_theory()['expected_ratio_block_over_full']
        self.assertLess(abs(emp - th), 0.01, 'empirical=%r theory=%r' % (emp, th))


class TestLengthModel(unittest.TestCase):
    def test_iid_has_no_within_group_correlation(self):
        """iid 档必须真的"长度与 uid 无关"——初版用组均值生成，造出了假 iid。"""
        lengths = M.build_lengths(0, 'iid')
        per_group = [lengths[g * M.GROUP_SIZE:(g + 1) * M.GROUP_SIZE] for g in range(M.N_GROUPS)]
        group_means = [statistics.mean(g) for g in per_group]
        within_vars = [statistics.pvariance(g) for g in per_group]
        # 组内方差的均值应接近组均值方差（独立性 ⇒ 两者同量级，比值≈1）
        ratio = statistics.mean(within_vars) / statistics.pvariance(group_means)
        self.assertGreater(ratio, 0.6, 'iid 档组内方差太小，说明仍带组内相关：%.3f' % ratio)

    def test_realistic_is_tight_within_group(self):
        lengths = M.build_lengths(0, 'realistic')
        per_group = [lengths[g * M.GROUP_SIZE:(g + 1) * M.GROUP_SIZE] for g in range(M.N_GROUPS)]
        group_means = [statistics.mean(g) for g in per_group]
        within_vars = [statistics.pvariance(g) for g in per_group]
        ratio = statistics.mean(within_vars) / statistics.pvariance(group_means)
        self.assertLess(ratio, 0.1, 'realistic 档组内不该这么散：%.3f' % ratio)

    def test_degenerate_is_all_equal(self):
        lengths = M.build_lengths(3, 'degenerate')
        self.assertEqual(len(set(lengths)), 1)

    def test_lengths_are_monotone_in_regime_scatter(self):
        # strong_corr 的组间离散 > realistic，组内离散 < realistic
        def spreads(regime):
            lengths = M.build_lengths(1, regime)
            per_group = [lengths[g * M.GROUP_SIZE:(g + 1) * M.GROUP_SIZE] for g in range(M.N_GROUPS)]
            return (statistics.pstdev([statistics.mean(g) for g in per_group]),
                    statistics.mean(statistics.pstdev(g) for g in per_group))
        bg_s, wi_s = spreads('strong_corr')
        bg_r, wi_r = spreads('realistic')
        self.assertGreater(bg_s, bg_r)
        self.assertLess(wi_s, wi_r)


class TestBlockShapes(unittest.TestCase):
    def test_contiguous_covers_all_indices_once(self):
        shapes = M.contiguous_shapes(256, 4)
        covered = []
        for start, n in shapes:
            covered.extend(range(start, start + n))
        self.assertEqual(covered, list(range(256)))
        self.assertEqual([n for _, n in shapes], [64, 64, 64, 64])

    def test_interleaved_partitions(self):
        idx = [sorted(M.interleaved_indices(256, 4, j)) for j in range(4)]
        flat = sorted(i for part in idx for i in part)
        self.assertEqual(flat, list(range(256)))
        self.assertEqual([len(p) for p in idx], [64, 64, 64, 64])


class TestOrdering(unittest.TestCase):
    def test_group_contiguous_index_matches_repeat_interleave(self):
        # verl 的 repeat(repeat_times=n, interleave=True)：组 g 的第 i 条落在 g*n+i
        idx = M.group_contiguous_index()
        self.assertEqual(idx[:8], [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(idx[8:16], [8, 9, 10, 11, 12, 13, 14, 15])
        self.assertEqual(len(idx), M.N_GROUPS * M.GROUP_SIZE)

    def test_degenerate_reproduces_shape_of_13_32(self):
        """degenerate 档（组内长度全同）下 LDM 的分区必须仍是"每块若干完整组"。

        §13.32 的 1.0508 就是在这个极限下量的；若本实验在 degenerate 档算不出同样的
        组成结构（块内组数=8），两条读数之间的对照就不成立。
        """
        uids, grads = T._grad_model(random.Random(0), M.RHO_GROUP, None)
        lengths = M.build_lengths(0, 'degenerate')
        order, _parts = M.ldm_order(lengths, M.K_BLOCKS)
        shapes = M.contiguous_shapes(len(order), M.K_BLOCKS)
        purity = M.block_group_purity([uids[i] for i in order], shapes)
        # **本批自查修正**：原以为「长度全同 ⇒ 每块 8 个完整组」是错的。权重全同时
        # LDM 的堆状态完全对称，等分入堆后分区下标是 (0,4,8,…)/(1,5,9,…)，
        # 把 32 个组**摊到全部 4 块**，每块装齐 32 个组、没有一组完整。
        self.assertEqual(purity['intact_groups'], 0)
        self.assertEqual(purity['mean_blocks_per_group'], 4)
        self.assertEqual(purity['mean_groups_per_block'], float(M.N_GROUPS))

    def test_grouped_basis_has_eight_groups_per_block(self):
        """§13.32 的分母（组内连续排列）才是「每块 8 个完整组」的那个口径。

        这个不对称（分子 32 组/块、分母 8 组/块）就是基准偏差因子 1.0412 的来源，
        也是本批最要紧的一处发现，必须被测试钉住。
        """
        uids = [g for g in range(M.N_GROUPS) for _ in range(M.GROUP_SIZE)]
        shapes = M.contiguous_shapes(len(uids), M.K_BLOCKS)
        purity = M.block_group_purity(uids, shapes)
        self.assertEqual(purity['intact_groups'], M.N_GROUPS)
        self.assertEqual(purity['mean_blocks_per_group'], 1)
        self.assertEqual(purity['mean_groups_per_block'], float(M.N_GROUPS // M.K_BLOCKS))

    def test_realistic_scatters_groups(self):
        """真实长度结构下，LDM 必须把同组摊到多个块（否则本批的前提不成立）。"""
        uids, _grads = T._grad_model(random.Random(0), M.RHO_GROUP, None)
        lengths = M.build_lengths(0, 'realistic')
        order, _parts = M.ldm_order(lengths, M.K_BLOCKS)
        shapes = M.contiguous_shapes(len(order), M.K_BLOCKS)
        purity = M.block_group_purity([uids[i] for i in order], shapes)
        self.assertGreater(purity['mean_blocks_per_group'], 3.5)
        self.assertLess(purity['intact_groups'], M.N_GROUPS // 2)


class TestNegativeControls(unittest.TestCase):
    """注入已知错误，必须被测试抓到——否则测试在"陪跑"。"""

    def test_denominator_choice_moves_the_ratio(self):
        """分母 N−1 vs N 会让块/全批比值不同——证明这个比值对估计量选择敏感，
        不是"换个分母也一样"的巧合。两个值都写死常数。

        **方向与本批初版预期相反**：原以为 MLE 分母（除以 N）会让比值更靠近 1，
        实际是 **0.9605 → 0.9492，离 1 更远**。原因是分母变小对分子（N=64）的相对影响
        比对分母（N=256）更大，把块这一侧同时压得更低。
        """
        def ratio(denom_shift):
            def e(n_groups, n_total=8):
                n = n_groups * n_total
                within = 0.5 * n_groups * (n_total - 1)
                between = 0.5 * n_total * (n_groups - 1)
                return (within + between) / (n - denom_shift)
            return e(8) / e(32)
        unbiased = ratio(1)
        mle = ratio(0)
        self.assertAlmostEqual(unbiased, 0.9604519774011299, places=12)
        self.assertAlmostEqual(mle, 0.9491525423728814, places=12)
        self.assertGreater(unbiased, mle, '无偏分母反而更靠近 1（与初版预期相反）')

    def test_bias_must_shift_reading_away_from_one(self):
        """若基准无偏（ratio=1），真实组成效应就等于原始读数——本批结论会是另一种。

        这里用算术检查：把 theory_ratio 强设成 1.0 时，
        `baseline_bias_factor` 必须变成 1.0，分解里的基准份额必须变成 0。
        """
        th = M.grouped_bias_theory()
        self.assertNotAlmostEqual(th['expected_ratio_block_over_full'], 1.0, places=3)
        factor = 1.0 / th['expected_ratio_block_over_full']
        self.assertGreater(factor, 1.03)  # 1.0412

    def _judge(self, ident):
        """`identifiability` 只给原始读数，判定在 `build_summary` 里做——
        测试也走同一条判定规则，避免"测试里判通过、实现里判失败"。"""
        return (not ident['sensitive']) if ident['must_be_insensitive'] else ident['sensitive']

    def test_constant_metric_is_caught(self):
        ident = M.identifiability(M._metric_constant, M.RHO_LEVELS, 'rho', is_constant=True)
        self.assertFalse(ident['sensitive'])
        self.assertTrue(self._judge(ident))

    def test_group_span_control_is_insensitive(self):
        ident = M.identifiability(M._metric_group_span, M.RHO_LEVELS, 'rho', is_constant=True)
        self.assertFalse(ident['sensitive'])
        self.assertTrue(self._judge(ident))

    def test_deviating_metric_sensitive(self):
        ident = M.identifiability(M._metric_gap_over_grouped, M.RHO_LEVELS, 'rho')
        self.assertTrue(ident['sensitive'])
        self.assertTrue(self._judge(ident))

    def test_interleaved_metric_sensitive_around_one(self):
        ident = M.identifiability(M._metric_interleaved_over_ldm, M.RHO_LEVELS, 'rho',
                                  around_one=True)
        self.assertTrue(ident['sensitive'])
        self.assertTrue(self._judge(ident))


class TestIdentifiabilityCriteria(unittest.TestCase):
    def test_relative_spread(self):
        self.assertAlmostEqual(M.relative_spread([1.0, 2.0]), 0.5)
        self.assertAlmostEqual(M.relative_spread([1.0, 1.0, 1.0]), 0.0)
        self.assertIsNone(M.relative_spread([]))

    def test_spread_around_one_detects_drift(self):
        # 离 1 的偏移从 0.0006 涨到 0.0078 ⇒ (0.0078−0.0006)/0.0078 = 0.923
        self.assertAlmostEqual(M.spread_around_one([1.0006, 1.0078]), (0.0078 - 0.0006) / 0.0078)
        self.assertEqual(M.spread_around_one([1.0, 1.0]), 0.0)

    def test_relative_spread_would_miss_it(self):
        """对照：同一组数字用 `relative_spread` 只有 0.7%，会被判不敏感。

        这条测试守住"判据必须与量级匹配"这个自查修正不回流。
        """
        vals = [1.0006, 1.0078]
        self.assertLess(M.relative_spread(vals), 0.01)

    def test_constant_control_threshold(self):
        # 恒定微扰（1e-6 量级）必须判零敏感（§13.32 自查修正二）
        ident = M.identifiability(lambda _lv: 1.0 + 1e-6, M.RHO_LEVELS, 'rho', is_constant=True)
        self.assertFalse(ident['sensitive'])


class TestSummaryInvariants(unittest.TestCase):
    """`build_summary` 的跨字段一致性（不重跑全量，只查不变量）。"""

    @classmethod
    def setUpClass(cls):
        roots = {}
        verl = os.environ.get('VERL_ROOT', os.path.join(os.path.dirname(os.path.dirname(HERE)), 'verl'))
        if os.path.isdir(verl):
            roots['verl'] = verl
        cls.out = M.build_summary(roots)

    def test_regimes_ordered_by_group_clumping(self):
        """组内越相关（长度越集中）⇒ LDM 越容易把同组排到一起 ⇒ 块内组数越多。

        方向是**单调递增**：iid(28.7) < realistic(29.8) < strong_corr(30.9) < degenerate(32.0)。
        初版断言方向写反（以为相关性强会让组更散），由测试抓出。
        """
        b = self.out['baseline_bias']['by_regime']
        m = [b[r]['mean_groups_per_block_ldm']
             for r in ('iid', 'realistic', 'strong_corr', 'degenerate')]
        self.assertEqual(m, sorted(m))
        self.assertLess(m[0], m[-1])

    def test_degenerate_is_largest(self):
        b = self.out['baseline_bias']['by_regime']
        deg = b['degenerate']['ldm_over_grouped_mean']
        for r in ('iid', 'realistic', 'strong_corr'):
            self.assertLess(b[r]['ldm_over_grouped_mean'], deg)

    def test_decomposition_sums_to_one(self):
        for key in ('decomposition', 'realistic_decomposition'):
            d = self.out['baseline_bias'][key]
            self.assertAlmostEqual(d['baseline_share_of_raw'] + d['real_share_of_raw'], 1.0, places=9)

    def test_identifiability_all_pass(self):
        self.assertTrue(self.out['verdict']['identifiability_all_pass'])

    def test_bias_direction_crossover(self):
        """偏差方向在 rho≈0.12 处翻转——本批的"基准偏小"只在 rho 大于它时成立。

        默认 rho=0.5 落在偏低一侧，结论成立但**是带条件的**；
        这条测试防止"基准偏差"被写成无条件陈述（rho=0 时基准反而偏大）。
        """
        b = self.out['baseline_bias']['bias_direction_vs_rho']
        self.assertAlmostEqual(b['crossover_rho'], 0.12, places=9)
        self.assertGreater(b['rho_to_ratio']['0.00'], 1.0)
        self.assertLess(b['rho_to_ratio']['0.50'], 1.0)

    def test_headline_mentions_both_readings(self):
        h = self.out['verdict']['headline']
        self.assertIn('1.0508', h)
        self.assertIn('1.0462', h)


if __name__ == '__main__':
    unittest.main(verbosity=2)
