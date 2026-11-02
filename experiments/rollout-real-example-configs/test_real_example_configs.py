#!/usr/bin/env python3
"""real_example_configs 的单元测试。

期望值全部写死为常数，不从被测函数的输出反推（否则是「照答卷判卷」）。
"""
import unittest

import real_example_configs as m


class TestBatchCountFormula(unittest.TestCase):
    """num_micro_batches 必须与 seqlen_balancing.py:402-414 同序同式。"""

    def test_three_assertions_in_real_order(self):
        # 405 行：max_token_len < max_seq_len → shape（先于任何批数计算）
        self.assertEqual(m.num_micro_batches(4, [100, 200], 150), (None, 'shape'))
        # 正常切分：max_seq=200 <= 预算 200，ceil(300/200)=2，组数 2 → 2
        self.assertEqual(m.num_micro_batches(2, [100, 200], 200), (2, 'ok'))

    def test_batch_count_is_ceil_of_total_over_budget(self):
        # 400 token / 每批 150 → ceil = 3；组数 4 不封顶
        self.assertEqual(m.num_micro_batches(4, [100] * 4, 150), (3, 'ok'))

    def test_group_count_caps_batch_count(self):
        # 8 组、total 800、预算 100 → ceil 8，与组数相等
        self.assertEqual(m.num_micro_batches(8, [100] * 8, 100), (8, 'ok'))
        # 预算再小也只能到组数
        self.assertEqual(m.num_micro_batches(8, [100] * 8, 50)[1], 'shape')
        self.assertEqual(m.num_micro_batches(8, [100] * 8, 100)[0], 8)

    def test_vpp_roundup_then_cap(self):
        # 32 样本 / g=8 → 4 组；total 64000、预算 16384 → ceil 4；roundup(4,3)=6 > 4 → cap
        lens = [2000] * 32
        self.assertEqual(m.num_micro_batches(32, lens, 16384), (4, 'ok'))
        self.assertEqual(
            m.num_micro_batches(32, lens, 16384, force_group_size=8,
                                num_batches_divided_by=3), (None, 'cap'))

    def test_vpp_roundup_without_exceeding_is_ok(self):
        # 32 样本 / g=8 → 4 组；roundup(4,2)=4，不顶破
        lens = [2000] * 32
        self.assertEqual(
            m.num_micro_batches(32, lens, 16384, force_group_size=8,
                                num_batches_divided_by=2), (4, 'ok'))


class TestPartition(unittest.TestCase):
    def test_single_partition_when_one_batch(self):
        self.assertEqual(m.micro_split([10, 20, 30], 1), [[0, 1, 2]])

    def test_grouped_never_splits_a_group(self):
        lens = [100, 110, 120, 130, 500, 510, 520, 530]
        # 8 样本 / 每组 4 条 → 2 组，批数不能超过 2（414 行断言）
        for n_micro in (1, 2):
            parts = m.micro_split(lens, n_micro, force_group_size=4)
            for p in parts:
                groups = [i // 4 for i in p]
                self.assertEqual(len(set(groups)), len(groups) // 4,
                                 'a group was split: %s' % p)

    def test_partition_covers_every_sample_once(self):
        lens = [100, 300, 200, 450, 120, 260]
        parts = m.micro_split(lens, 3)
        flat = sorted(i for p in parts for i in p)
        self.assertEqual(flat, [0, 1, 2, 3, 4, 5])

    def test_light_batches_go_to_both_ends(self):
        # use_dynamic_bsz_balance：先按工作量降序，再取奇数位反转放前面
        # 三条路径各 1 条样本时，最轻的那批必须排在第一位
        lens = [10, 5000, 500]
        parts = m.micro_split(lens, 3)
        w = [sum(m.WORKLOAD_LINEAR * lens[i] + m.WORKLOAD_QUAD * lens[i] ** 2 for i in p)
             for p in parts]
        self.assertEqual(min(w), w[0], 'lightest batch must come first: %s' % w)
        self.assertEqual(max(w), w[1], 'heaviest batch must come second: %s' % w)


class TestRankView(unittest.TestCase):
    def test_rank_view_uses_prompt_times_n_over_dp(self):
        ex = dict(mini_bs=256, dp=8, train_bs=1024, n=5)
        # 256/8 = 32 组 × 5 = 160 样本；不整除时返回 None
        self.assertEqual(m.rank_view(ex), (160, 32))
        self.assertIsNone(m.rank_view(dict(mini_bs=256, dp=3, train_bs=1024, n=5)))


class TestExampleTable(unittest.TestCase):
    def test_all_examples_have_required_fields(self):
        for ex in m.EXAMPLES:
            for k in ('name', 'train_bs', 'n', 'mini_bs', 'prompt_len', 'resp_len',
                      'budget', 'dp', 'pp', 'dyn'):
                self.assertIn(k, ex, '%s missing %s' % (ex['name'], k))

    def test_budget_is_not_confused_with_max_seq(self):
        # 回归：这曾是本批最严重的口径错误（把 ppo_max_token_len_per_gpu 当长度上限）
        ex = [e for e in m.EXAMPLES if e['name'] == 'qwen3_30b_a3b_megatron_vpp2'][0]
        self.assertEqual(ex['budget'], 30720)
        self.assertEqual(ex['prompt_len'] + ex['resp_len'], 10240)
        self.assertGreater(ex['budget'], ex['prompt_len'] + ex['resp_len'])

    def test_static_bshd_configs_are_flagged(self):
        names = {e['name'] for e in m.EXAMPLES if not e['dyn']}
        self.assertEqual(names, {'qwen3_5_35b_bshd_static', 'qwen3_5_397b_bshd_static',
                                 'qwen3_5_122b_bshd_static'})


class TestFragmentation(unittest.TestCase):
    def test_identical_lengths_fragment_maximally(self):
        # 长度全相同时 LDM 只能连续切片，每条都可能落在不同批
        ex = dict(m.EXAMPLES[0])
        b, ng = m.rank_view(ex)
        lens = [2000] * b
        group_of = [g for g in range(ng) for _ in range(ex['n'])]
        n, _ = m.num_micro_batches(b, lens, ex['budget'])
        frag = m.fragments(m.micro_split(lens, n), group_of)
        # 每批约 b/n 条，一条组（n=5 条）几乎必然跨 ~4.9 批
        self.assertGreater(sum(frag.values()) / ng, 4.0)

    def test_noop_grouping_gives_identical_fragmentation(self):
        ex = dict(m.EXAMPLES[0])
        b, ng = m.rank_view(ex)
        lens, group_of = m.make_lengths(ex, ng, 0)
        n, _ = m.num_micro_batches(b, lens, ex['budget'])
        a = m.fragments(m.micro_split(lens, n, force_group_size=1), group_of)
        b2 = m.fragments(m.micro_split(lens, n, force_group_size=1), group_of)
        self.assertEqual(a, b2)


class TestPPBubble(unittest.TestCase):
    def test_single_stage_has_zero_bubble(self):
        self.assertEqual(m.bubble_frac(10, 1), 0.0)

    def test_bubble_formula_matches_1f1b(self):
        # 1 - m/(m+p-1)：m=9, p=2 → 1 - 9/10 = 0.1
        self.assertAlmostEqual(m.bubble_frac(9, 2), 0.1, places=12)
        # m=4, p=4 → 1 - 4/7
        self.assertAlmostEqual(m.bubble_frac(4, 4), 1 - 4 / 7, places=12)

    def test_fewer_batches_means_more_bubble(self):
        self.assertGreater(m.bubble_frac(8, 2), m.bubble_frac(18, 2))

    def test_degenerate_inputs_return_none(self):
        self.assertIsNone(m.bubble_frac(0, 4))
        self.assertIsNone(m.bubble_frac(4, 0))


class TestDeadPathRecord(unittest.TestCase):
    """前缀共享线在 HEAD 上不可达，这条记录必须带可复核锚点。"""

    def test_record_has_anchors(self):
        d = m.PREFIX_GROUPER_DEAD_PATH
        for k in ('head', 'knob_never_forwarded', 'prefix_grouper_never_constructed',
                  'dropped_by', 'proof_wiring_existed_before'):
            self.assertIn(k, d)
        self.assertEqual(d['head'], '1a9e711e')
        self.assertIn('044bbba2', d['dropped_by'])

    def test_verdict_says_unreachable(self):
        self.assertFalse(m._verdict_line()['prefix_grouper_reachable_at_head'])


class TestIdentifiability(unittest.TestCase):
    """结论必须对应当敏感的变量敏感、对不该敏感的变量不敏感。"""

    def test_n_equals_one_makes_grouping_a_noop(self):
        rows = m.identifiability()['by_group_size']
        r1 = [r for r in rows if r['n'] == 1][0]
        self.assertEqual(r1['n_default_median'], r1['n_grouped_median'])

    def test_huge_budget_collapses_to_one_batch(self):
        rows = m.identifiability()['by_budget']
        last = rows[-1]
        self.assertEqual(last['n_default_median'], 1)
        self.assertEqual(last['n_grouped_median'], 1)

    def test_pp_one_has_no_bubble_ratio(self):
        rows = m.identifiability()['bubble_by_pp']
        r1 = [r for r in rows if r['pp'] == 1][0]
        self.assertIsNone(r1['ratio'])

    def test_bubble_ratio_is_above_one_for_pp_ge_2(self):
        for r in m.identifiability()['bubble_by_pp']:
            if r['pp'] >= 2:
                self.assertGreater(r['ratio'], 1.0, 'pp=%s ratio=%s' % (r['pp'], r['ratio']))


if __name__ == '__main__':
    unittest.main()
