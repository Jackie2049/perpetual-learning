#!/usr/bin/env python3
"""pp_microbatch_boundary 的单元测试（2026-11-02 M1）。

覆盖三类：① 本批自查修正过的地方（口径不可比、方向猜反）必须被钉住；
② 骨架正确性（切片、排法、模型）；③ 识别性检验自身的松紧。
"""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import pp_microbatch_boundary as P  # noqa: E402


class TestShapes(unittest.TestCase):
    def test_shapes_cover_all_items(self):
        for n in (8, 32, 64):
            for m in (1, 2, 4, 8):
                shapes = P.micro_batch_shapes(n, m)
                covered = [i for a, b in shapes for i in range(a, b)]
                self.assertEqual(sorted(covered), list(range(n)), (n, m))

    def test_shapes_count_equals_m_when_divisible(self):
        for m in (1, 2, 4, 8):
            self.assertEqual(len(P.micro_batch_shapes(32, m)), m)

    def test_shapes_allow_remainder_but_less_than_m(self):
        shapes = P.micro_batch_shapes(10, 4)
        self.assertEqual(sum(b - a for a, b in shapes), 10)
        self.assertLessEqual(len(shapes), 4)
        self.assertGreaterEqual(min(b - a for a, b in shapes), 1)


class TestArms(unittest.TestCase):
    def test_arms_preserve_membership(self):
        """所有排法成员集合必须完全相同——这是同口径比较的前提。"""
        vals = [3, 1, 9, 5, 7, 2]
        for arm in P.ARMS:
            self.assertEqual(sorted(P.rank_order(arm, vals)), sorted(vals), arm)

    def test_cur_is_sorted_then_interleaved(self):
        vals = [5, 1, 4, 2]
        # 升序 [1,2,4,5] ⇒ [::2]=[1,4]，[1::2][::-1]=[5,2] ⇒ [1,4,5,2]
        self.assertEqual(P.rank_order('cur', vals), [1, 4, 5, 2])

    def test_plain_interleaves_without_sorting(self):
        vals = [5, 1, 4, 2]
        # [::2]=[5,4]，[1::2][::-1]=[2,1] ⇒ [5,4,2,1]
        self.assertEqual(P.rank_order('plain', vals), [5, 4, 2, 1])

    def test_desc_is_reverse_of_sorted(self):
        vals = [5, 1, 4, 2]
        self.assertEqual(P.rank_order('desc', vals), [5, 4, 2, 1])
        self.assertEqual(P.rank_order('sorted', vals), [1, 2, 4, 5])

    def test_rejects_unknown_arm(self):
        with self.assertRaises(ValueError):
            P.rank_order('nope', [1])


class TestCostModels(unittest.TestCase):
    def test_sum_model_is_membership_invariant(self):
        """载荷可加时，排法完全不影响总代价——这是"排法只影响 PP 边界"的结构性判据。"""
        parts = P.seed_parts(0, 0.6, 8)
        base = P.batch_cost(parts, 'sorted', 4, 'sum')
        for arm in P.ARMS:
            self.assertAlmostEqual(P.batch_cost(parts, arm, 4, 'sum') / base, 1.0, places=12, msg=arm)

    def test_max_model_matches_manual(self):
        self.assertEqual(P.rank_cost([1, 5, 2, 3], 2, 'max'), max(1, 5) + max(2, 3))

    def test_sum_plus_max_is_between(self):
        seq = [1, 5, 2, 3]
        self.assertLessEqual(P.rank_cost(seq, 2, 'sum'),
                             P.rank_cost(seq, 2, 'sum_plus_max'))
        self.assertLessEqual(P.rank_cost(seq, 2, 'sum_plus_max'),
                             P.rank_cost(seq, 2, 'max') + P.rank_cost(seq, 2, 'sum'))


class TestSingleMicroBatch(unittest.TestCase):
    def test_m1_all_arms_identical(self):
        """m=1 时排法不应产生任何差异——排法只在 PP 微批边界存在时才有意义。"""
        self.assertTrue(P.single_micro_batch_identity()['all_identical'])

    def test_m1_identity_holds_for_any_seed(self):
        for seed in (0, 3, 7):
            parts = P.seed_parts(seed, 0.6, 4)
            costs = {P.rank_cost(P.rank_order(a, parts[0]), 1, 'max') for a in P.ARMS}
            self.assertEqual(len(costs), 1)


class TestSelfCorrections(unittest.TestCase):
    """本批自查推翻的两处结论，各留一条测试防回流。"""

    def test_block_level_ordering_is_not_comparable(self):
        """错误 1 防回流：把 partition 当不可分块排序会给每 rank 1 个元素，
        与逐样本排法不是同一个数量——任何比较都必须先核 rank 内元素个数一致。"""
        parts = P.seed_parts(0, 0.6, 8)
        n_per = len(parts[0])
        block_level = sorted(sum(p) for p in parts)          # 错误写法会得到的形态
        self.assertEqual(len(block_level), 8)
        self.assertNotEqual(len(block_level), n_per)
        for arm in P.ARMS:
            self.assertEqual(len(P.rank_order(arm, parts[0])), n_per)

    def test_cur_is_worse_than_sorted_not_better(self):
        """错误 2 防回流：本批起初假设"去掉重排会变差"，实测方向相反。
        在 model=max、cv=0.6、dp=8、m=4 下 cur 必须比 sorted 更差。"""
        row = P.order_effect(0.6, 8, 4, 'max')
        self.assertGreater(row['cur'], 1.0)

    def test_sorted_and_desc_are_identical_under_max(self):
        """max 模型只看每片最大值：升序与降序切出来的最大值多重集相同。"""
        row = P.order_effect(0.6, 8, 4, 'max')
        self.assertAlmostEqual(row['sorted'], row['desc'], places=12)


class TestMechanism(unittest.TestCase):
    def test_stage_spread_orders_arms_as_expected(self):
        """mechanism 的判据：cur 的 stage_max_spread 必须小于 plain（挤堆更轻），
        且两者都大于 sorted——三者方向与 model=max 的代价方向一致。"""
        mech = P.mechanism_table()
        arms = mech['arms']
        self.assertLess(arms['cur']['stage_max_spread'], arms['plain']['stage_max_spread'])
        self.assertGreater(arms['cur']['stage_max_spread'], arms['sorted']['stage_max_spread'])
        self.assertGreater(arms['cur']['cost_over_sorted'], 1.0)

    def test_piece_cv_is_flatter_for_sorted_than_cur(self):
        """sorted/cur 的 piece_max_cv 高于 orig/plain——说明"压扁每片最大载荷"不是这段重排在做的事。"""
        mech = P.mechanism_table()['arms']
        self.assertGreater(mech['sorted']['piece_max_cv'], mech['orig']['piece_max_cv'])
        self.assertGreater(mech['cur']['piece_max_cv'], mech['plain']['piece_max_cv'])


class TestIdentifiability(unittest.TestCase):
    def test_constant_metric_is_not_sensitive(self):
        ctrl = P.identifiability()['control_constant']
        self.assertFalse(ctrl['sensitive'])
        self.assertEqual(ctrl['relative_spread'], 0.0)

    def test_group_dispersion_control_is_sensitive(self):
        ctrl = P.identifiability()['control_group_dispersion']
        self.assertTrue(ctrl['sensitive'])

    def test_cv_and_m_drivers_are_sensitive(self):
        ident = P.identifiability()
        self.assertTrue(ident['cv_driver']['sensitive'])
        self.assertTrue(ident['m_driver']['sensitive'])

    def test_relative_spread_threshold(self):
        self.assertEqual(P.relative_spread([1.0, 1.0]), 0.0)
        self.assertAlmostEqual(P.relative_spread([1.0, 1.02]), 0.02 / 1.01, places=12)


class TestDirection(unittest.TestCase):
    def test_cur_worse_holds_across_cv_levels(self):
        for cv in P.CV_LEVELS:
            row = P.order_effect(cv, 8, 4, 'max')
            self.assertGreater(row['cur'], 1.0, cv)

    def test_effect_grows_with_cv(self):
        vals = [P.order_effect(cv, 8, 4, 'max')['cur'] for cv in P.CV_LEVELS]
        self.assertTrue(all(b > a for a, b in zip(vals, vals[1:])), vals)

    def test_effect_vanishes_at_m1(self):
        for cv in P.CV_LEVELS:
            self.assertEqual(P.order_effect(cv, 8, 1, 'max')['cur'], 1.0)


if __name__ == '__main__':
    unittest.main()
