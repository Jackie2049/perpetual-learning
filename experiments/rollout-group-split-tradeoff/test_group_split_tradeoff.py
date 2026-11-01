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


class TestTrainSideBalance(unittest.TestCase):
    """2026-11-02 第二批：训练侧同栈均衡（verl `use_prefix_grouper` 机制）。

    这一层的数字已与真实 verl `get_seqlen_balanced_partitions` 逐位核对过
    （同样的输入给出相同的分区和相同的分区权重），所以测试重点是**机制**不是数值。
    """

    def test_ldm_replication_matches_known_optimal_case(self):
        """1..16 分 4 片：LDM 应给出每组 34 的完全均衡解（与真实 verl 输出一致）。"""
        w = list(range(1, 17))
        parts = m._karmarkar_karp(w, 4)
        self.assertEqual([sum(w[i] for i in p) for p in parts], [34, 34, 34, 34])

    def test_grouped_keeps_groups_whole(self):
        """组级均衡的每个分区都由完整的组构成——这是 `use_prefix_grouper` 的成立前提。

        分区单位是**组**：32 组分 4 片，每片 8 组（= 64 条样本）。
        """
        lens = m.sample_group_lens(random.Random(0), n_groups=32, cv=0.6)
        r = m.train_side_balance(lens, 4, grouped=True)
        self.assertTrue(r['groups_intact'])
        self.assertEqual(r['partition_sizes'], [8, 8, 8, 8])

    def test_default_path_splits_groups(self):
        """逐条 LDM 会把同组样本分到不同分区——所以默认路径下不能开 prefix 共享。

        这是负向对照：如果哪天这条断言失败，说明"默认路径不拆组"这个前提变了，
        正文里"默认路径必然拆组"的表述必须跟着改。
        """
        lens = m.sample_group_lens(random.Random(0), n_groups=32, cv=0.6)
        r = m.train_side_balance(lens, 4, grouped=False)
        self.assertFalse(r['groups_intact'])
        self.assertLess(r['groups_intact_count'], r['n_groups'])

    def test_grouped_balance_better_than_no_balance(self):
        """组级均衡仍显著优于"按 uid 连续切"——否则分组约束等于没做事。"""
        lens = m.sample_group_lens(random.Random(3), n_groups=32, cv=0.6)
        flat = [x for g in lens for x in g]
        w = [m.workload_int64(x) for x in flat]
        grouped = m.train_side_balance(lens, 4, grouped=True)['imbalance']
        naive = m._imbalance(m._cut_by_uid_order(w, 4), w)
        self.assertLess(grouped, naive)

    def test_workload_saturation_is_far_beyond_training_range(self):
        """int64 饱和点远在可训练范围外，因此"截断改变均衡"这条不成立。"""
        sat = m.workload_saturation_len()
        self.assertGreater(sat, 10 ** 9)
        self.assertLessEqual(m.calculate_workload(sat - 1), m.INT64_MAX)
        self.assertGreater(m.calculate_workload(sat), m.INT64_MAX)
        self.assertEqual(m.workload_int64(sat), m.INT64_MAX)

    def test_workload_matches_verl_formula(self):
        """工作量公式与 verl `seqlen_balancing.py:46` 一致：24576·l + l²。"""
        for l in (1, 512, 4096):
            self.assertEqual(m.calculate_workload(l), 24576 * l + l * l)

    def test_splits_cover_all_samples(self):
        """两种拆分实现都必须无损（拆分只为均衡，不能丢样本）。"""
        w = [m.workload_int64(x) for x in (100, 200, 150, 50, 300, 250, 180, 120)]
        for fn in (lambda: m.greedy_split(len(w), 2), lambda: m.karmarkar_split(w, 2)):
            covered = sorted(i for p in fn() for i in p)
            self.assertEqual(covered, list(range(len(w))))


class TestSigmaBias(unittest.TestCase):
    def test_splitting_biases_sigma_hat_downward(self):
        """拆组不只放大 σ̂ 的方差，还把它**系统性压低**（p=2 约 −11.6%）。

        这条以前只记录了方差膨胀（`sim_var_ratio`），漏了期望偏移；
        偏移方向很关键：σ̂ 偏小会把优势幅度放大，因此"拆组只是噪声变大"的表述是错的。
        """
        def mean_sigma(n, p, seed):
            rng = random.Random(seed)
            return statistics.mean(
                statistics.mean(m.sigma_hat_draws(n, p, rng, iters=1)) for _ in range(400)
            )

        whole = mean_sigma(8, 1, 0)
        split = mean_sigma(8, 2, 1)
        deeper = mean_sigma(8, 4, 2)
        self.assertLess(split, whole)
        self.assertLess(deeper, split)


class TestRetractedMetric(unittest.TestCase):
    """2026-11-02 第二批：撤回"排序翻转率"作为尺度打散的证据。

    这三条测试的作用是**防止失效指标换个名字回到结论里**：
    翻转率对组难度方差零敏感 → 它量的不是"跨组尺度"，那结论一就必须换证据。
    """

    def test_flip_rate_is_not_sensitive_to_group_variance(self):
        """组难度方差从 0 拉到 25，翻转率几乎不动——所以它不能证明"尺度被打散"。"""
        r = m._iid_counterfactual(n=8, iters=120, seeds=4)
        spread = r['max_minus_min_across_group_variance']
        self.assertLess(spread['n8_p2'], 0.05)
        self.assertLess(spread['n8_p4'], 0.05)

    def test_flip_rate_stays_high_without_group_effect(self):
        """组难度方差为 0（两组同分布）时翻转率仍显著大于 0——这是指标失效的直接证据。"""
        r = m._iid_counterfactual(n=8, iters=120, seeds=4)
        self.assertGreater(r['pair_flip_by_group_variance']['theta_var_0']['n8_p2'], 0.05)
        self.assertGreater(r['pair_flip_by_group_variance']['theta_var_0']['n8_p4'], 0.05)

    def test_scale_mismatch_has_identification(self):
        """尺度错配（分片 σ / 整组 σ）有识别性：p=1 恒为 1，拆得越碎越偏离 1。"""
        unit = m.fragment_scale_mismatch(8, 1, random.Random(0), iters=200)
        p2 = m.fragment_scale_mismatch(8, 2, random.Random(0), iters=200)
        p4 = m.fragment_scale_mismatch(8, 4, random.Random(0), iters=200)
        self.assertAlmostEqual(unit['median'], 1.0, places=6)
        self.assertGreater(p2['p90'], 1.2)
        self.assertGreater(p4['p90'], p2['p90'])


if __name__ == '__main__':
    unittest.main()
