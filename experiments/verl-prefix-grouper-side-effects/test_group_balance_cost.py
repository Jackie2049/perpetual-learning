#!/usr/bin/env python3
"""训练侧副作用实验的单元测试（2026-11-02 M1）。

纪律：每条判定都要有能推翻它的对照。
  ① 跑的必须是**上游源码**，不是我重写的算法——用上游不变量校验；
  ② 组完整性必须真的被保住（同 uid 同分区），否则实验测的不是组级均衡；
  ③ 方向必须站得住：组级均衡的不均衡度 ≥ 逐条均衡；
  ④ 规模效应必须单调：批越大，组级劣势越小。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import group_balance_cost as m  # noqa: E402


class TestUpstreamSourceIsReal(unittest.TestCase):
    def test_functions_extracted_from_upstream_file(self):
        U = m.load_upstream_funcs()
        for name in m.WANT:
            self.assertIn(name, U)
            self.assertTrue(callable(U[name]))

    def test_partitions_cover_all_without_dup(self):
        U = m.load_upstream_funcs()
        lens = m.gen_lengths(0, 200, 5)
        w = [int(U['calculate_workload'](U['_tensor']([L])).item()) for L in lens]
        for parts in (U['get_seqlen_balanced_partitions'](w, k_partitions=8, equal_size=True),
                      U['get_group_balanced_partitions'](w, m.uid_list_for(200, 5), k_partitions=8)):
            flat = [i for p in parts for i in p]
            self.assertEqual(sorted(flat), list(range(len(w))))
            self.assertEqual(len(set(flat)), len(flat))


class TestWorkloadModel(unittest.TestCase):
    def test_workload_is_superlinear(self):
        """上游 calculate_workload = 24576L + L²。

        它在实用长度区间**不是**纯二次：L=400→800 比值约 2.03（线性项仍占 82%），
        要到 L≈24576 后二次项才占优。这条测试固定住"超线性但接近线性"这一事实，
        防止把负载模型当成纯平方去解释结果。
        """
        U = m.load_upstream_funcs()
        ratio = (U['calculate_workload'](U['_tensor']([800])).item() /
                 U['calculate_workload'](U['_tensor']([400])).item())
        self.assertGreater(ratio, 2.0, '必须严格超线性')
        self.assertLess(ratio, 2.2, '但在 L≈800 处仍远小于 4')
        # 长序列下二次项才显出来
        long_ratio = (U['calculate_workload'](U['_tensor']([32768])).item() /
                      U['calculate_workload'](U['_tensor']([16384])).item())
        self.assertGreater(long_ratio, 2.6, '二次项在长序列上应显著')

    def test_workload_matches_formula(self):
        U = m.load_upstream_funcs()
        for L in (128, 512, 2048):
            got = U['calculate_workload'](U['_tensor']([L])).item()
            self.assertEqual(got, 24576 * L + L * L)


class TestGroupIntegrity(unittest.TestCase):
    def test_group_partitions_keep_uid_together(self):
        U = m.load_upstream_funcs()
        n_prompts, n_samples = 200, 5
        uids = m.uid_list_for(n_prompts, n_samples)
        lens = m.gen_lengths(0, n_prompts, n_samples)
        w = [int(U['calculate_workload'](U['_tensor']([L])).item()) for L in lens]
        parts = U['get_group_balanced_partitions'](w, uids, k_partitions=8)
        where = {}
        for pi, p in enumerate(parts):
            for i in p:
                where.setdefault(uids[i], set()).add(pi)
        self.assertTrue(all(len(v) == 1 for v in where.values()))

    def test_flag_off_does_not_keep_uid_together(self):
        """负向对照：逐条均衡**可以**把同 uid 拆到不同分区——这才说明组级均衡有约束。"""
        U = m.load_upstream_funcs()
        n_prompts, n_samples = 200, 5
        uids = m.uid_list_for(n_prompts, n_samples)
        lens = m.gen_lengths(0, n_prompts, n_samples)
        w = [int(U['calculate_workload'](U['_tensor']([L])).item()) for L in lens]
        parts = U['get_seqlen_balanced_partitions'](w, k_partitions=8, equal_size=True)
        where = {}
        for pi, p in enumerate(parts):
            for i in p:
                where.setdefault(uids[i], set()).add(pi)
        split = sum(1 for v in where.values() if len(v) > 1)
        self.assertGreater(split, 0, '若一条都没被拆开，组完整性就没有代价可言')


class TestCostDirection(unittest.TestCase):
    def test_group_is_never_better_balanced(self):
        """组级均衡是逐条均衡的子约束 ⇒ 不均衡度不应更低。"""
        out = m.scenario_group_cost()
        for tag, blk in out.items():
            self.assertGreaterEqual(blk['imbalance_group']['mean'],
                                    blk['imbalance_flat']['mean'] - 1e-9, tag)

    def test_cost_is_small_at_realistic_batch(self):
        """DP8 批 1024 上，代价应在个位数百分比内（否则结论会完全不同）。"""
        out = m.scenario_batch_size()
        self.assertLess(out['batch1024']['imbalance_ratio'], 1.05)
        self.assertGreater(out['batch1024']['imbalance_ratio'], 1.0)

    def test_cost_monotone_decreasing_in_batch(self):
        out = m.scenario_batch_size()
        ratios = [out['batch%d' % b]['imbalance_ratio'] for b in (256, 1024, 4096, 16384)]
        self.assertEqual(ratios, sorted(ratios, reverse=True), ratios)

    def test_small_batch_cost_is_material(self):
        """批 256 时代价放大到 >5%——这是这条副作用的真正风险区。"""
        out = m.scenario_batch_size()
        self.assertGreater(out['batch256']['imbalance_ratio'], 1.05)


if __name__ == '__main__':
    unittest.main()
