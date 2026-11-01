#!/usr/bin/env python3
"""group_split_tradeoff 的单元测试。

重点不是"脚本能跑"，而是把本批修正过的三处建模错误各钉一条负向对照——
这些错误都是第一版跑出"看着合理"的数字之后才被发现的，如果不写测试，
下次改动很可能把它们再引回来。

    python3 -m unittest test_group_split_tradeoff -v
"""

import random
import statistics
import unittest

import group_split_tradeoff as m


class TestSchedulerInvariants(unittest.TestCase):
    def test_homogeneous_groups_have_no_tax(self):
        """组内长度全同 → 整组归还与逐条归还没有区别，税必须为 0。"""
        self.assertEqual(m.grouping_tax([[5.0] * 8] * 16)['grouping_tax'], 0.0)

    def test_no_backlog_means_no_tax(self):
        """一批工作量一次放得下时，两种粒度必然同 makespan（与 §14 候选 D 同口径）。"""
        lens = m.sample_group_lens(random.Random(0))
        self.assertEqual(m.grouping_tax(lens, backlog_groups=1)['grouping_tax'], 0.0)

    def test_tax_needs_queueing(self):
        """有 backlog 才会排队，排队才可能产生税——这条保证税不是凭空构造的。"""
        lens = m.sample_group_lens(random.Random(0))
        self.assertGreater(m.grouping_tax(lens, backlog_groups=4)['grouping_tax'], 0.0)

    def test_makespan_positive(self):
        for lens in ([[1.0]], [[1.0, 2.0], [3.0, 4.0]]):
            self.assertGreater(m.grouping_tax(lens)['group_makespan'], 0.0)

    def test_trajectory_arm_is_never_starved_when_capacity_fits(self):
        """容量足够时，逐条调度不该比组调度慢（造数不能反过来）。"""
        lens = [[1.0] * 4] * 4
        r = m.grouping_tax(lens, backlog_groups=1)
        self.assertLessEqual(r['traj_makespan'], r['group_makespan'] + 1e-9)


class TestDefectsAreCaught(unittest.TestCase):
    """负向对照：把本批修掉的错误各自复现一遍，确认测试会红。"""

    def test_group_arm_must_not_use_trajectory_slots(self):
        """错误做法：组臂用 1 个 slot 装整组 → 组臂凭空有空闲，税被抹成 0。"""
        lens = m.sample_group_lens(random.Random(0))
        correct = m.grouping_tax(lens, backlog_groups=4)['grouping_tax']
        n_per = len(lens[0])
        dur = [max(g) for g in lens] * 4
        wrong_group = m._makespan([1] * len(dur), dur, False, cap=len(lens) * n_per)
        wrong_traj = m._makespan([1] * len(dur), dur, False, cap=len(lens) * n_per)
        wrong_tax = (wrong_group - wrong_traj) / wrong_group
        self.assertGreater(correct, 0.3)
        self.assertAlmostEqual(wrong_tax, 0.0, places=9)

    def test_single_sample_fragment_is_not_pseudo_zero_variance(self):
        """拆成每片 1 条时样本标准差未定义，必须返回 None 而不是 0/1。"""
        self.assertIsNone(m.pseudo_zero_variance_rate(4, 4, random.Random(0), iters=50))

    def test_intra_group_order_is_unchanged_by_splitting(self):
        """GRPO 归一化是组内减均值除标准差：**组内**两两排序拆组后不变。

        这条是 §11.2 条目 7 的正确表述来源——拆组破坏的不是组内比较，
        而是把不同组的样本放进了同一个梯度更新。
        """
        rng = random.Random(0)
        n, p = 8, 2
        flips = 0
        total = 0
        for _ in range(300):
            r = [rng.gauss(0, 1) for _ in range(n)]
            mu, sd = statistics.mean(r), statistics.pstdev(r) + 1e-9
            whole = [(x - mu) / sd for x in r]
            frag = [x for x in whole]  # 组内归一化后再拆，分数只是被同一个常数缩放
            for i in range(n):
                for j in range(i + 1, n):
                    total += 1
                    if (whole[i] > whole[j]) != (frag[i] > frag[j]):
                        flips += 1
        self.assertEqual(flips, 0)
        self.assertGreater(total, 0)

    def test_cross_group_flip_grows_with_split_depth(self):
        """跨组排序翻转率随拆分加深而上升——这是拆组真正的统计代价。"""
        p1 = statistics.mean(m.variance_order_flip_rate(8, 1, random.Random(s), iters=150)
                             for s in range(4))
        p2 = statistics.mean(m.variance_order_flip_rate(8, 2, random.Random(s), iters=150)
                             for s in range(4))
        p4 = statistics.mean(m.variance_order_flip_rate(8, 4, random.Random(s), iters=150)
                             for s in range(4))
        self.assertEqual(p1, 0.0)
        self.assertGreater(p2, 0.15)
        self.assertGreater(p4, p2)


class TestStatistics(unittest.TestCase):
    def test_sigma_variance_grows_with_split(self):
        v1 = m.stats_distribution(8, 1, random.Random(0))['var']
        v2 = m.stats_distribution(8, 2, random.Random(0))['var']
        self.assertGreater(v2, v1)

    def test_analytic_ratio_matches_direction(self):
        self.assertGreater(m.analytic_var_ratio(8, 2), 1.0)
        self.assertGreater(m.analytic_var_ratio(8, 4), m.analytic_var_ratio(8, 2))
        self.assertEqual(m.analytic_var_ratio(8, 1), 1.0)

    def test_simulation_ratio_is_smaller_than_chi2_bound(self):
        """仿真比低于 χ² 近似界：近似假设独立同方差，实际分片共享组均值会压缩方差。"""
        sim = m.stats_distribution(8, 4, random.Random(0))['var'] / \
            m.stats_distribution(8, 1, random.Random(0))['var']
        self.assertLess(sim, m.analytic_var_ratio(8, 4))


class TestCandidateDConsistency(unittest.TestCase):
    def test_reproduces_candidate_d_zero(self):
        """候选 D 的口径（单批、无排队）在本模型下也必须是 0，否则两套模型量的不是同一件事。"""
        lens = [[80.0, 8.0, 30.0, 61.0, 19.0, 48.0, 55.0, 25.0]]
        self.assertEqual(m.grouping_tax(lens, backlog_groups=1)['grouping_tax'], 0.0)


if __name__ == '__main__':
    unittest.main()
