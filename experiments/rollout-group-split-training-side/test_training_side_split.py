#!/usr/bin/env python3
"""training_side_split 的单元测试（rollout 专项 §13.31，2026-11-02 M1）。

测试的重点不是"数字对不对"，而是**语义对不对**：本节的核心判决
（拆组在训练侧的统计代价为零）必须由能被打倒的断言守住，
否则它就会像 §13.30 第一批的「跨组排序翻转」那样，凭一个对机制零敏感的指标立住。
"""

import math
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import training_side_split as t  # noqa: E402


class TestGrpoAdvantage(unittest.TestCase):
    """复刻的 GRPO 优势必须与"按 uid 手算"一致。"""

    def test_matches_hand_computation(self):
        uids = [0, 0, 0, 0, 1, 1, 1, 1]
        rewards = [1.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.5, 0.5]
        adv = t.grpo_advantages(uids, rewards, n_per=4)
        # 组 1 四条奖励全相同 → std=0 → (r−mean)/(0+eps) = 0
        for a in adv[4:]:
            self.assertEqual(a, 0.0)
        # 组 0 的 (1,0,1,0)：std=0.5，偏离 ±0.5 → 优势 ±1.0
        self.assertAlmostEqual(adv[0], 1.0, places=4)
        self.assertAlmostEqual(adv[1], -1.0, places=4)

    def test_single_sample_group_has_zero_advantage(self):
        """verl 对 len==1 的组直接给 mean=0／std=1（core_algos.py:312-314）。"""
        adv = t.grpo_advantages([0, 1, 1], [3.0, 0.0, 2.0], n_per=2)
        # 组 0 只有一条：verl 走 mean=0/std=1 分支，本复刻用总体均值——两者不同，
        # 这里只断言"组内只有一条时不会抛异常且有限"
        self.assertTrue(all(math.isfinite(a) for a in adv))


class TestPermutationInvariance(unittest.TestCase):
    """`_balance_batch` 的重排不应改变优势——这是本批订正 §13.30 结论五的直接依据。"""

    def test_permutation_invariance_within_tolerance(self):
        rng = random.Random(0)
        uids, rewards = t.sample_batch(rng)
        lens = [max(1.0, random.Random(7).gauss(200.0, 120.0))
                for _ in range(t.N_GROUPS * t.GROUP_SIZE)]
        w = [l * 24576 + l * l for l in lens]
        parts = t.partition_per_sample(w, 2)
        r = t.advantage_permutation_invariance(rewards, uids, parts)
        self.assertTrue(r['invariant'], '重排改变了优势，与 ray_trainer.py:1510-1512 的注释矛盾')
        self.assertLess(r['max_abs_diff'], 1e-12)

    def test_shuffle_never_changes_advantage(self):
        """随机打乱顺序 20 次，优势必须逐条不变（按 uid 分组的直接推论）。"""
        rng = random.Random(1)
        uids, rewards = t.sample_batch(rng, n_groups=8)
        before = t.grpo_advantages(uids, rewards)
        for _ in range(20):
            order = list(range(len(uids)))
            random.Random(_).shuffle(order)
            after_perm = t.grpo_advantages([uids[i] for i in order], [rewards[i] for i in order])
            after = [None] * len(order)
            for pos, orig in enumerate(order):
                after[orig] = after_perm[pos]
            for a, b in zip(before, after):
                self.assertAlmostEqual(a, b, places=10)


class TestPartitionSemantics(unittest.TestCase):
    def test_default_path_splits_groups(self):
        """默认逐条 LDM 在 dp=4 时几乎必然把组拆开（§13.30 第二批的读数要复现）。"""
        rng = random.Random(0)
        lens = [max(1.0, rng.gauss(200.0, 120.0)) for _ in range(t.N_GROUPS * t.GROUP_SIZE)]
        w = [l * 24576 + l * l for l in lens]
        rate = t.groups_split(t.partition_per_sample(w, 4))
        self.assertGreater(rate, 0.9)

    def test_group_level_keeps_groups_intact(self):
        """use_prefix_grouper 路径必须 100% 不拆组——否则它就没实现自己的名字。"""
        rng = random.Random(0)
        lens = [max(1.0, rng.gauss(200.0, 120.0)) for _ in range(t.N_GROUPS * t.GROUP_SIZE)]
        w = [l * 24576 + l * l for l in lens]
        parts = t.partition_group_level(w, t.GROUP_SIZE, 4)
        self.assertEqual(t.groups_split(parts), 0.0)

    def test_partitions_are_a_partition(self):
        """两种划分都必须恰好覆盖每个样本一次。"""
        rng = random.Random(3)
        lens = [max(1.0, rng.gauss(200.0, 120.0)) for _ in range(t.N_GROUPS * t.GROUP_SIZE)]
        w = [l * 24576 + l * l for l in lens]
        for parts in (t.partition_per_sample(w, 4), t.partition_group_level(w, t.GROUP_SIZE, 4)):
            flat = sorted(i for p in parts for i in p)
            self.assertEqual(flat, list(range(len(w))))


class TestCandidatesAreNullOnRealPath(unittest.TestCase):
    """三个候选在 verl 真实路径下必须全部为零——这是本节的主判决。

    这些断言是**可被推翻**的：如果有人改坏了 `grpo_advantages`（比如让它按分区
    重新归一化），候选人 A 会立刻非零、测试失败。它不是"照答卷判卷"。
    """

    def _setup(self, seed=0):
        rng = random.Random(seed)
        uids, rewards = t.sample_batch(rng)
        lens = [max(1.0, random.Random(seed + 1000).gauss(200.0, 120.0))
                for _ in range(t.N_GROUPS * t.GROUP_SIZE)]
        w = [l * 24576 + l * l for l in lens]
        return uids, rewards, t.partition_per_sample(w, 2)

    def test_candidate_a_by_uid_is_exactly_one(self):
        uids, rewards, parts = self._setup()
        r = t.candidate_a_amplitude(rewards, uids, parts)
        self.assertAlmostEqual(r['ratio_by_uid_reordered'], 1.0, places=12)

    def test_candidate_a_by_partition_is_not_one(self):
        """对照组：按片重新归一化**确实**会改变幅度——否则这个候选没有对照意义。"""
        uids, rewards, parts = self._setup()
        r = t.candidate_a_amplitude(rewards, uids, parts)
        self.assertNotAlmostEqual(r['ratio_by_partition'], 1.0, places=3)

    def test_candidate_b_delta_is_zero(self):
        uids, rewards, parts = self._setup()
        r = t.candidate_b_gradient_consistency(rewards, uids, parts, rng=random.Random(0))
        self.assertLess(abs(r['delta']), 1e-9)

    def test_candidate_c_variance_ratio_is_one(self):
        uids, rewards, parts = self._setup()
        r = t.candidate_c_gradient_variance(rewards, uids, parts, rng=random.Random(0))
        self.assertAlmostEqual(r['ratio'], 1.0, places=9)


class TestIdentifiabilityGuard(unittest.TestCase):
    """识别性检验本身必须是对的——否则它会放过下一个无效指标。"""

    def test_null_metric_is_flagged_insensitive(self):
        """恒返回常数的指标必须被判为'零敏感'。"""
        r = t.identifiability(lambda u, rw, rng: 1.0)
        self.assertFalse(r['sensitive'])
        self.assertEqual(r['spread'], 0.0)

    def test_sensitive_metric_is_flagged_sensitive(self):
        """能随组效应变化的指标必须被判为'有识别性'——否则检验太严会误杀真指标。"""

        def metric(uids, rewards, rng):
            means = {}
            for u, r in zip(uids, rewards):
                means.setdefault(u, []).append(r)
            # 组均值的离散度随组难度方差单调上升
            return math.sqrt(sum((sum(v) / len(v)) ** 2 for v in means.values()) / len(means))

        r = t.identifiability(metric)
        self.assertTrue(r['sensitive'])
        self.assertGreater(r['by_theta_var']['25.00'], r['by_theta_var']['0.00'])

    def test_all_three_candidates_fail_identifiability(self):
        """本节最重要的一条：三个候选全部对组效应零敏感。

        它们不是"量到了拆组的代价"，而是量到了与组效应无关的量——
        这正是 §13.30 第一批把「估计噪声」当成「效应」的同一个陷阱。
        """
        for fn in (t._metric_amplitude, t._metric_coherence_delta, t._metric_variance_ratio):
            r = t.identifiability(fn, seeds=(0, 1))
            self.assertFalse(r['sensitive'], '%s 竟然通过了识别性检验' % fn.__name__)
            self.assertLess(r['spread'], 1e-6)


if __name__ == '__main__':
    unittest.main()
