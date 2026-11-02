"""§13.35 的单元测试：钉住机制复刻的形状与三条会反转结论的陷阱。

每个测试的期望值都是**写死的常数或结构性恒等式**，不从被测函数的输出反推
（记忆库 test-expected-values-must-be-constants）。
"""

import unittest

import microbatch_group_integrity as m


class TestMicroSplitShape(unittest.TestCase):
    """micro_split 必须复刻 verl seqlen_balancing.py:348-466 的三件事。"""

    def test_budget_fits_whole_batch_gives_one_partition(self):
        parts, info = m.micro_split([100] * 8, 10 ** 9)
        self.assertEqual(len(parts), 1)
        self.assertEqual(sorted(parts[0]), list(range(8)))
        self.assertFalse(info['infeasible'])

    def test_num_micro_is_ceildiv_total_over_budget(self):
        # 8 条 × 1000 = 8000，预算 3000 → ceil = 3
        parts, _ = m.micro_split([1000] * 8, 3000, use_dynamic_bsz_balance=False)
        self.assertEqual(len(parts), 3)

    def test_force_group_size_never_splits_a_group(self):
        lens = [100, 500, 900, 1300, 200, 600, 1000, 1400]
        parts, _ = m.micro_split(lens, 2000, force_group_size=4)
        where = {}
        for pid, p in enumerate(parts):
            for i in p:
                where.setdefault(i // 4, set()).add(pid)
        self.assertEqual(sorted(where), [0, 1])
        self.assertTrue(all(len(v) == 1 for v in where.values()))

    def test_force_group_size_equals_one_is_the_default(self):
        lens = [300, 700, 1100, 1500, 400, 800, 1200, 1600]
        a, _ = m.micro_split(lens, 2000)
        b, _ = m.micro_split(lens, 2000, force_group_size=1)
        self.assertEqual(a, b)

    def test_vpp_roundup(self):
        # ceil(8000/3000)=3，vPP=2 → roundup 到 4
        parts, info = m.micro_split([1000] * 8, 3000, num_batches_divided_by=2,
                                    use_dynamic_bsz_balance=False)
        self.assertEqual(info['n_micro_before_cap'], 4)
        self.assertEqual(len(parts), 4)

    def test_group_cap_makes_some_vpp_cases_infeasible(self):
        """组完整把上限从 B 压到 B/n：8 组、ceil=8、vPP=3 → roundup 9 > 8 → 真实实现断言失败。"""
        parts, info = m.micro_split([2000] * 64, 16384, force_group_size=8, num_batches_divided_by=3)
        self.assertIsNone(parts)
        self.assertTrue(info['infeasible'])
        self.assertEqual(info['n_micro_before_cap'], 9)
        # 同一条 mini-batch 走逐条切时上限是 64，不触发
        ps, ps_info = m.micro_split([2000] * 64, 16384, force_group_size=1, num_batches_divided_by=3)
        self.assertIsNotNone(ps)
        self.assertFalse(ps_info['infeasible'])


class TestFragments(unittest.TestCase):
    def test_singleton_groups_make_wiring_a_noop(self):
        r = m.case(1, 1, 8, seed=3)
        self.assertEqual(r['ideal_over_current'], 1.0)

    def test_larger_budget_never_increases_fragments(self):
        tight = m.case(8, 1, 4, seed=1, budget=8192)
        loose = m.case(8, 1, 4, seed=1, budget=10 ** 9)
        self.assertGreaterEqual(tight['fragments_per_group_current'], loose['fragments_per_group_current'])
        self.assertEqual(loose['fragments_per_group_current'], 1)

    def test_ideal_is_never_worse_than_current(self):
        for seed in range(5):
            r = m.case(8, 2, 4, seed)
            self.assertLessEqual(r['ideal_over_current'], 1.0 + 1e-12)
            self.assertEqual(r['fragments_per_group_ideal_minibatch'], 1.0)


class TestReferenceFrame(unittest.TestCase):
    """「整批共享 1.00×」不是可达点；可达上限是 mini-batch 参照系。"""

    def test_wholebatch_baseline_is_invariant_to_budget(self):
        """整批参照点与预算无关：改预算只改现状，不改 whole-batch 分母。"""
        loose = m.case(8, 1, 4, seed=2, budget=10 ** 9)
        tight = m.case(8, 1, 4, seed=2, budget=16384)
        self.assertEqual(loose['current_over_wholebatch'], loose['ideal_minibatch_over_wholebatch'])
        self.assertGreater(tight['current_over_wholebatch'], 1.0)

    def test_one_wholebatch_prefix_per_group_regardless_of_dp(self):
        """外参照是「每个组一份前缀」：它的绝对 FLOP 与 dp、mini_groups 都无关。"""
        a = m.case(8, 1, 4, seed=4)['ideal_wholebatch_flop']
        b = m.case(8, 2, 8, seed=4)['ideal_wholebatch_flop']
        c = m.case(8, 4, 1, seed=4)['ideal_wholebatch_flop']
        self.assertEqual(a, b)
        self.assertEqual(a, c)


class TestPrefixLengthInvariance(unittest.TestCase):
    def test_ratios_identical_across_prefix_lengths(self):
        out = m.prefix_length_invariance()
        vals = {k: round(v['current_over_wholebatch'], 12) for k, v in out.items()}
        self.assertEqual(len(set(vals.values())), 1)


class TestVppLayer(unittest.TestCase):
    def test_grouped_path_never_splits_groups(self):
        r = m.vpp_effect(mini_groups=8, budget=8192)
        for row in r['rows']:
            if not row['grouped_infeasible']:
                self.assertTrue(row['group_integrity_grouped'])
                self.assertFalse(row['group_integrity_per_sample'])

    def test_grouped_far_fewer_micro_batches(self):
        row = next(x for x in m.vpp_effect(mini_groups=8, budget=4096)['rows'] if x['vpp'] == 2)
        self.assertEqual(row['grouped_n_micro'], 8)      # 上限 = 组数
        self.assertGreater(row['per_sample_n_micro'], 8)

    def test_vpp3_all_infeasible_when_budget_starved(self):
        for budget in (4096, 8192, 16384):
            row = next(x for x in m.vpp_effect(mini_groups=8, budget=budget)['rows'] if x['vpp'] == 3)
            self.assertTrue(row['grouped_infeasible'])


if __name__ == '__main__':
    unittest.main()
