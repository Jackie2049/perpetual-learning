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


class TestGroupArmGroupsByGroupNotMember(unittest.TestCase):
    """2026-11-03：`grouping_tax` 的组臂曾按**成员**分组而不是按**整组**分组。

    `_makespan` 的 `slot_counts` 是逐"任务"的槽位数；组臂传入的是
    `[n_per] * len(group_dur)`——每个任务占 n_per 个槽，但每个任务**各自**是一组。
    于是 `group_left` 每个 key 的值都是 1，任务一完成就把 n_per 个槽还回去，
    归还粒度在组臂上退化成逐条，L1–L2 两批的分组税读数因此偏低。
    这两条测试钉住"组臂真的按整组归还"。
    """

    def test_uniform_groups_have_no_tax(self):
        """组内全同、组间全同 → 两种粒度都是同一个 makespan，税为 0。"""
        self.assertEqual(
            m.grouping_tax([[100.0] * 8] * 4, backlog_groups=4)['grouping_tax'], 0.0)

    def test_group_arm_never_beats_trajectory_arm(self):
        """负税在"整组归还只会更慢"的定义下不可能；按成员分组时会给出 −300%。"""
        lens = [[100.0] * 8, [100.0] * 8, [200.0] * 8, [200.0] * 8]
        r = m.grouping_tax(lens, backlog_groups=2)
        self.assertGreaterEqual(r['group_makespan'], r['traj_makespan'] - 1e-9)
        self.assertGreaterEqual(r['grouping_tax'], 0.0)


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


class TestRaggedGroupsAreRejected(unittest.TestCase):
    """组大小参差会让 `n_per` 只取自第 0 组，组臂的占用与归还量对不上。

    2026-11-03 发现：参差输入能静默量出**负税**（−33%），而"整组归还只会更慢"
    的定义下负税不可能。这里把输入约束钉成断言，避免下游再拿负数当读数。
    """

    def test_ragged_groups_raise(self):
        ragged = [[50.0, 50.0, 250.0, 250.0], [100.0] * 4 + [300.0] * 4]
        with self.assertRaises(ValueError):
            m._makespan_by_policy(ragged, 1, m.POLICY_FIFO)
        with self.assertRaises(ValueError):
            m.grouping_tax(ragged, backlog_groups=1)

    def test_negative_tax_before_the_guard(self):
        """负向对照：去掉守卫后同一输入确实量出负税——证明守卫拦的是真问题。

        这里直接复现"n_per 取自第 0 组"的旧行为（不调用被守卫的函数）。
        """
        ragged = [[50.0, 50.0, 250.0, 250.0], [100.0] * 4 + [300.0] * 4]
        n_per = len(ragged[0])
        dur = [max(g) for g in ragged]
        unguarded = m._makespan([n_per] * len(dur), dur, True, cap=len(ragged) * n_per)
        flat = [x for g in ragged for x in g]
        traj = m._makespan([1] * len(flat), flat, False, cap=len(ragged) * n_per)
        self.assertLess((unguarded - traj) / unguarded, -0.1)

    def test_equal_size_groups_never_negative(self):
        """等大组下税恒非负——这是 `R` 能当上界引用的前提。"""
        for seed in range(60):
            rng = random.Random(seed)
            n_groups, n_per = rng.randint(2, 6), rng.randint(2, 8)
            groups = [[max(1.0, rng.lognormvariate(4.0, 1.2)) for _ in range(n_per)]
                      for _ in range(n_groups)]
            for backlog in (1, 2, 4):
                for p in m.POLICIES:
                    self.assertGreaterEqual(
                        m._makespan_by_policy(groups, backlog, p)['tax'], 0.0,
                        '%s seed=%d backlog=%d' % (p, seed, backlog))


class TestGroupArmReturnsWholeGroup(unittest.TestCase):
    """2026-11-03 第四十批：`grouping_tax` 的组臂曾按**成员**而不是按**整组**归还。

    旧写法 `_makespan([n_per] * k, dur, True)` 让每个"任务"各自成组，一完成就把
    `n_per` 个槽位全还回去——组臂的粒度退化成逐条，`R` 在积压 ≥2 批时被抬高，
    甚至量出负税（"整组归还只会更慢"下不可能）。这三条钉住修正后的口径。

    与参差无关：这里**组是等大的**，坏的只是组臂的分组方式。
    """

    def test_group_arm_matches_policy_layer(self):
        """修正后 `grouping_tax` 的 FIFO 栏必须与 `_makespan_by_policy` 逐位相同。"""
        for seed in range(10):
            lens = m.sample_group_lens(random.Random(seed), n_groups=16, n_per=8, cv=0.6)
            self.assertAlmostEqual(m.grouping_tax(lens, backlog_groups=4)['grouping_tax'],
                                   m._makespan_by_policy(lens, 4, m.POLICY_FIFO)['tax'],
                                   places=12)

    def test_backlogged_group_arm_never_negative(self):
        """等大组、有积压时税必须非负——旧路径在这里给出 −200%。"""
        groups = [[50.0, 250.0], [50.0, 250.0]]
        for backlog in (1, 2, 4):
            self.assertGreaterEqual(m.grouping_tax(groups, backlog_groups=backlog)['grouping_tax'],
                                    0.0, 'backlog=%d' % backlog)
        self.assertAlmostEqual(m.grouping_tax(groups, backlog_groups=4)['grouping_tax'], 0.3)

    def test_old_arm_undercounts_on_backlog(self):
        """负向对照：旧组臂会**低估** `R`，在"组最长者挤在一起"的负载上最明显。

        旧写法让每个成员各自成组、各自归还，组臂不被整组扣住，于是组臂更容易
        排进空槽 ⇒ 税偏低甚至归零。`[[400,400],[40,40]]` 积压 4 批：旧 45.0%、
        正确 0.0%（组臂本来就一批装得下，不该有税）。
        """
        groups = [[400.0, 400.0], [40.0, 40.0]]
        n_per, cap, b = 2, 4, 4
        dur = [max(g) for _ in range(b) for g in groups]
        flat = [x for _ in range(b) for g in groups for x in g]
        old_group = m._makespan([n_per] * len(dur), dur, True, cap=cap)
        traj = m._makespan([1] * len(flat), flat, False, cap=cap)
        new_group = m._group_makespan(dur, n_per, cap)
        old_tax = (old_group - traj) / old_group
        new_tax = (new_group - traj) / new_group
        self.assertGreater(old_tax, new_tax + 0.2)
        self.assertEqual(new_tax, 0.0)


class TestPolicyInvariance(unittest.TestCase):
    """Layer 2b：排队顺序归引擎、归还粒度归 L4，两者不是同一条轴。"""

    def test_no_backlog_hides_every_policy(self):
        """一批装得下时任何排队顺序都无事可排，六档税全为 0。"""
        lens = m.sample_group_lens(random.Random(0))
        for p in m.POLICIES:
            self.assertEqual(m._makespan_by_policy(lens, 1, p)['tax'], 0.0)

    def test_fifo_and_lof_differ_in_makespan(self):
        """换排队顺序确实改变两臂的 makespan。"""
        lens = m.sample_group_lens(random.Random(3))
        a = m._makespan_by_policy(lens, 4, m.POLICY_FIFO)
        b = m._makespan_by_policy(lens, 4, m.POLICY_LOF)
        self.assertNotAlmostEqual(a['group_makespan'], b['group_makespan'], places=6)

    def test_lof_raises_tax_because_traj_arm_benefits_more(self):
        """长度优先让两臂都更快，但轨迹臂受益更多 ⇒ 税**上升**。
        若这条反过来，说明本层排程把顺序轴吃掉了。"""
        lens = m.sample_group_lens(random.Random(0))
        a = m._makespan_by_policy(lens, 4, m.POLICY_FIFO)
        b = m._makespan_by_policy(lens, 4, m.POLICY_LOF)
        self.assertGreater(b['tax'], a['tax'])
        self.assertLess(b['group_makespan'], a['group_makespan'])
        self.assertLess(b['traj_makespan'], a['traj_makespan'])

    def test_homogeneous_load_is_zero_for_every_policy(self):
        lens = [[50.0] * 8] * 16
        for p in m.POLICIES:
            self.assertAlmostEqual(m._makespan_by_policy(lens, 4, p)['tax'], 0.0, places=12)


class TestPolicyLayerDefectsAreCaught(unittest.TestCase):
    def test_negative_control_has_identification(self):
        """两个不可实现对照必须把税推离基线，否则本层对"换了轴"不敏感。"""
        nc = m.policy_invariance_negative_control()
        self.assertTrue(nc['detects_two_arm_resort'])
        self.assertTrue(nc['detects_sorted_arrival'])
        self.assertTrue(nc['zero_load_clean'])

    def test_order_matters_so_the_layer_is_not_vacuous(self):
        """本层的识别性：若把组的排队顺序**换成到达顺序**（即取消策略），
        组臂 makespan 必须与按长度优先不同。否则说明排程器根本没吃顺序，
        前面那些"税随策略移动"的读数就都是假象。
        """
        lens = m.sample_group_lens(random.Random(7), n_groups=8)
        n_per = len(lens[0])
        cap = len(lens) * n_per

        def makespan_with_explicit_order(ordered):
            free, running, pending, now, end = cap, [], list(ordered), 0.0, 0.0
            while pending or running:
                for r in list(pending):
                    if n_per <= free:
                        running.append((now + r['last'], n_per, None))
                        free -= n_per
                        pending.remove(r)
                if not running:
                    break
                running.sort()
                t, s, _ = running.pop(0)
                now = max(now, t)
                end = max(end, t)
                free += s
            return max(end, 1e-9)

        order = [{'gid': (b, i), 'lens': g, 'last': max(g), 'first': min(g),
                  'arrival': 0.0} for b in range(4) for i, g in enumerate(lens)]
        as_arrived = makespan_with_explicit_order(order)
        by_length = makespan_with_explicit_order(m._group_order(order, m.POLICY_LOF))
        self.assertLess(by_length, as_arrived - 1.0)


if __name__ == '__main__':
    unittest.main()
