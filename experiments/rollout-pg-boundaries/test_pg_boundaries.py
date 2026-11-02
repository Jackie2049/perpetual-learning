#!/usr/bin/env python3
"""§13.34 的单元测试：验证机制复刻的形状，而不是验证读数好看。

每条断言都绑一个「如果实现写错就必然失败」的条件；读数本身见 summary.json。
"""

import random
import unittest

import pg_boundaries as pb


def _layout(n_per=8, n_groups=32, seed=0):
    rng = random.Random(seed)
    return [[max(1.0, rng.gauss(pb.SEQ_MEAN, pb.SEQ_MEAN * pb.SEQ_CV)) for _ in range(n_per)]
            for _ in range(n_groups)]


class TestContinousCut(unittest.TestCase):
    def test_chunks_are_contiguous_and_cover(self):
        chunks = pb.contig_chunks(64, 16)
        self.assertEqual([len(c) for c in chunks], [16, 16, 16, 16])
        self.assertEqual(sorted(i for c in chunks for i in c), list(range(64)))

    def test_non_divisible_raises(self):
        with self.assertRaises(AssertionError):
            pb.contig_chunks(65, 16)


class TestGroupLevelBalance(unittest.TestCase):
    def test_groups_stay_together_after_reorder(self):
        gl = _layout()
        reordered, part = pb.balance_batch_pg(pb.batch_layout(gl), 4)
        n = len(gl[0])
        for gi in range(len(gl)):
            positions = [i for i, it in enumerate(reordered) if it[0] == gi]
            self.assertEqual(len(positions), n)
            self.assertEqual(max(positions) - min(positions), n - 1, '组在重排后必须连续')
            self.assertEqual(len({part[p] for p in positions}), 1, '整组必须落在同一分区')

    def test_partition_sizes_match_dispatch_views(self):
        """dispatch 用连续 chunk：只有「分区大小 = B/dp」时两者才逐一相等。"""
        gl = _layout()
        reordered, part = pb.balance_batch_pg(pb.batch_layout(gl), 4)
        for view in pb.dispatch_views(reordered, 4):
            self.assertEqual(len({part[p] for p in view}), 1)

    def test_requires_divisible_group_count(self):
        gl = _layout(n_groups=30)
        with self.assertRaises(AssertionError):
            pb.balance_batch_pg(pb.batch_layout(gl), 4)


class TestDefaultBalance(unittest.TestCase):
    def test_default_balance_splits_groups(self):
        gl = _layout()
        reordered, _ = pb.balance_batch_default(pb.batch_layout(gl), 4)
        n = len(gl[0])
        split = 0
        for gi in range(len(gl)):
            pos = [i for i, it in enumerate(reordered) if it[0] == gi]
            if max(pos) - min(pos) != n - 1:
                split += 1
        self.assertGreater(split, 0, '默认逐条 LDM 应当把组打散（§13.32 已核）')


class TestMiniCut(unittest.TestCase):
    def test_group_balance_has_no_mini_cut_when_divisible(self):
        gl = _layout()
        reordered, _ = pb.balance_batch_pg(pb.batch_layout(gl), 1)
        views = pb.dispatch_views(reordered, 1)
        self.assertEqual(pb.mini_bs_cut(reordered, views, 32), 0.0)

    def test_singleton_groups_never_cut(self):
        gl = [[900.0] for _ in range(32)]
        reordered, _ = pb.balance_batch_pg(pb.batch_layout(gl), 1)
        views = pb.dispatch_views(reordered, 1)
        self.assertEqual(pb.mini_bs_cut(reordered, views, 8), 0.0)


class TestMicroSplit(unittest.TestCase):
    def test_partitions_cover_all_samples(self):
        rng = random.Random(1)
        lens = [float(int(rng.uniform(200, 4000))) for _ in range(32)]
        parts = pb.micro_split(lens, 16384)
        self.assertEqual(sorted(i for p in parts for i in p), list(range(32)))

    def test_loose_budget_gives_single_micro(self):
        lens = [900.0] * 32
        self.assertEqual(pb.micro_split(lens, 10 ** 9), [list(range(32))])

    def test_num_micro_follows_ceildiv(self):
        lens = [1000.0] * 32
        # total = 32000, budget = 16384 -> ceil = 2
        self.assertEqual(len(pb.micro_split(lens, 16384)), 2)

    def test_dynamic_bsz_balance_reorders_by_workload(self):
        """use_dynamic_bsz_balance=True 时，重的 micro-batch 拿走更多样本；
        关掉时顺序保持分区返回顺序。两者都必须在，才能证明这个开关被复刻了。"""
        rng = random.Random(2)
        lens = [float(int(rng.uniform(200, 4000))) for _ in range(32)]
        with_bal = pb.micro_split(lens, 16384, use_dynamic_bsz_balance=True)
        without = pb.micro_split(lens, 16384, use_dynamic_bsz_balance=False)
        self.assertEqual(sorted(map(sorted, with_bal)), sorted(map(sorted, without)))
        self.assertNotEqual([sorted(p) for p in with_bal], [sorted(p) for p in without])


class TestPrefixCost(unittest.TestCase):
    def test_shared_prefix_is_cheaper_than_repeated(self):
        self.assertLess(pb.prefix_rounds(512, [1024.0] * 8, 1),
                        pb.prefix_rounds(512, [1024.0] * 8, 8))

    def test_ratio_scales_with_rounds(self):
        r1 = pb.prefix_rounds(512, [1024.0] * 8, 1)
        r4 = pb.prefix_rounds(512, [1024.0] * 8, 4)
        self.assertAlmostEqual(r4 / r1, 4.0)

    def test_loose_budget_equals_baseline(self):
        r = pb.layer3_case(8, 1, 4, 512, seed=0, budget_tokens=10 ** 9)
        self.assertAlmostEqual(r['real_over_baseline'], 1.0)

    def test_default_budget_is_worse_than_baseline(self):
        r = pb.layer3_case(8, 1, 4, 512, seed=0, budget_tokens=16384)
        self.assertGreater(r['real_over_baseline'], 1.0)

    def test_best_per_minibatch_is_never_better_than_baseline(self):
        for seed in range(5):
            r = pb.layer3_case(8, 1, 4, 512, seed=seed, budget_tokens=16384)
            self.assertGreaterEqual(r['best_per_mb_over_baseline'], 1.0 - 1e-12)


class TestIdentifiability(unittest.TestCase):
    def test_singleton_groups_have_no_cost(self):
        r = pb.layer3_case(1, 1, 4, 512, seed=1)
        self.assertAlmostEqual(r['real_over_baseline'], 1.0)

    def test_loose_budget_never_splits(self):
        r = pb.layer3_case(8, 1, 4, 512, seed=1, budget_tokens=10 ** 9)
        self.assertEqual(r['fragments_per_group'], 1.0)

    def test_shrinking_budget_increases_fragments(self):
        a = pb.layer3_case(8, 1, 4, 512, seed=1, budget_tokens=8192)
        b = pb.layer3_case(8, 1, 4, 512, seed=1, budget_tokens=16384)
        self.assertGreater(a['fragments_per_group'], b['fragments_per_group'])


class TestIteration(unittest.TestCase):
    def test_formula_matches_when_divisible(self):
        r = pb.iterate_minibatches(64, 32, 3)
        self.assertEqual(r['mismatch'], 0)
        self.assertEqual(r['n_batches'], 6)

    def test_tail_batch_counted(self):
        r = pb.iterate_minibatches(70, 32, 2)
        self.assertEqual(r['tail_per_epoch'], 6)
        self.assertEqual(r['mismatch'], 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
