#!/usr/bin/env python3
"""`prediction_floor.py` 的守护测试（第十七批，2026-11-04 M1）。

三条判据各对应本批踩过或差点踩过的一个坑：

  1. **恒等式是本批的承重读数**——`max(floor_predict(g)) == max(g)`，不许写成"经验上
     大多成立"。任何一组反例都要让测试红。
  2. **接线自证**——`real` 预测下 `R` 必须与 `_makespan_by_policy` 逐位相同。这条在
     本批红了四次（两臂槽位数不同 / 轨迹臂多排了一次序），全部是接线错、不是读数错。
  3. **判据敏感性**——把预测打坏，`R` 必须动。不动说明 `R` 里根本没有预测这一项，
     "预测对 `R` 无空间"就成了"度量对预测无反应"。

跑法：`python3 test_prediction_floor.py`（约 3 秒，只读 8 组真值，不跑全量）。
"""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)),
                                'tools', 'paired-diff'))

import prediction_floor as pf  # noqa: E402
from group_split_tradeoff import POLICY_LOF, _makespan_by_policy  # noqa: E402

GROUPS = None


def setUpModule():
    global GROUPS
    path = os.path.join(pf.REAL_DIR, pf.REAL_ARMS[0][1])
    GROUPS = pf.load_real_groups(path)[:pf.N_ID_GROUPS]


class TestGroupMaxIdentity(unittest.TestCase):
    """判据 1：截断上界对"组最大"是**恒等**，不是近似。"""

    def test_identity_holds_on_all_groups(self):
        for label, fname in pf.REAL_ARMS:
            groups = pf.load_real_groups(os.path.join(pf.REAL_DIR, fname))
            got = pf.group_max_identity(groups, pf.arm_cap(label))
            self.assertEqual(got['n_floor_max_exact'], got['n_groups'], label)
            self.assertTrue(got['all_exact'], label)

    def test_branch_coverage_is_complete(self):
        """三条分支（全撞顶／部分／无）之和必须等于组数——否则恒等式的覆盖是虚的。"""
        for label, fname in pf.REAL_ARMS:
            groups = pf.load_real_groups(os.path.join(pf.REAL_DIR, fname))
            got = pf.group_max_identity(groups, pf.arm_cap(label))
            self.assertEqual(got['n_branch_all_at_cap'] + got['n_branch_some_at_cap']
                             + got['n_branch_none_at_cap'], got['n_groups'], label)

    def test_cap256_arm_actually_covers_the_some_branch(self):
        """cap256 档必须真有"部分撞顶"的组，否则恒等式只测了平凡分支。"""
        groups = pf.load_real_groups(
            os.path.join(pf.REAL_DIR, 'qwen15b_gsm8k_n8.jsonl.gz'))
        got = pf.group_max_identity(groups, 256.0)
        self.assertGreater(got['n_branch_some_at_cap'], 100)
        self.assertGreater(got['n_branch_all_at_cap'], 10)

    def test_identity_counterexample_is_detected(self):
        """负向对照：撞顶样本的**值不等于 c** 时恒等式必须报出反例。

        这是恒等式唯一的失效方式。构造：`c=256`，但撞顶那条被引擎记成 260（比如按
        别的口径计数）⇒ 预测给 256、真值给 260 ⇒ 组最大预测偏低。
        """
        group = [260.0, 100.0, 90.0]           # 撞顶了，但记下来的值不是 c
        self.assertNotEqual(max(pf.floor_predict(group, 256.0)), max(group))
        got = pf.group_max_identity([group], 256.0)
        self.assertFalse(got['all_exact'])
        self.assertEqual(got['n_floor_max_exact'], 0)

    def test_identity_holds_when_at_cap_value_is_exactly_c(self):
        """同一输入把值改成恰好 c，恒等式就成立——差别只在"撞顶样本记成什么"。"""
        got = pf.group_max_identity([[256.0, 100.0, 90.0]], 256.0)
        self.assertTrue(got['all_exact'])


class TestWiring(unittest.TestCase):
    """判据 2：`real` 预测下必须与既有实现逐位相同。"""

    def test_tax_matches_existing_implementation(self):
        for backlog in (4, 16):
            got = pf.tax_canonical_check(GROUPS, backlog)
            self.assertTrue(got['ok'], (backlog, got))

    def test_traj_arm_matches_when_both_share_the_order(self):
        """两臂同序档：轨迹臂展开后不再重排，`R` 与 `_makespan_by_policy` 相同。"""
        ref = _makespan_by_policy(GROUPS, 16, POLICY_LOF)['tax']
        mine = pf.shared_tax(GROUPS, 16, None, 256.0)
        self.assertAlmostEqual(ref, mine, places=12)

    def test_separate_arm_differs_from_shared_because_traj_reorders(self):
        """分序档必须**不等于**同序档（轨迹臂自己按真值排），否则两档是一回事。"""
        shared = pf.shared_tax(GROUPS, 16, None, 256.0)
        separate = pf.separate_tax(GROUPS, 16, None, 256.0)
        self.assertGreater(abs(shared - separate), 1e-6)


class TestSensitivity(unittest.TestCase):
    """判据 3：打坏预测，`R` 必须动。"""

    def test_degraded_predictor_moves_r(self):
        """打坏预测后 `R` 必须动——但**三档负载的强度差很多**，不能一律要求大。

        16 组下实测：cap1024 两档 30/30 个种子都动、1.5B cap256 21/30、0.5B cap256 只
        5/30（那档 61% 轨迹撞顶，组序几乎只由"撞没撞顶"决定 ⇒ 预测的可争空间本来就小）。
        所以判据写成"两条 cap1024 与 1.5B cap256 明显在动"，弱档的存在写进断言消息。
        """
        moved = {}
        for label, fname in pf.REAL_ARMS:
            groups = pf.load_real_groups(os.path.join(pf.REAL_DIR, fname))
            got = pf.sensitivity_check(groups, pf.arm_cap(label))
            moved[label] = got['backlog16']['separate_moves']
        self.assertGreaterEqual(moved['qwen15b_cap1024'], 30, moved)
        self.assertGreaterEqual(moved['qwen05b_cap1024'], 30, moved)
        self.assertGreaterEqual(moved['qwen15b_cap256'], 10, moved)

    def test_floor_predictor_equals_truth_predictor(self):
        """反过来：截断上界与全真值的 `R` 必须**逐位**相同（恒等式的推论）。"""
        a = pf.separate_tax(GROUPS, 16, pf.floor_predict, 256.0)
        b = pf.separate_tax(GROUPS, 16, pf.full_predict, 256.0)
        self.assertEqual(a, b)

    def test_order_agreement_is_complete(self):
        got = pf.group_order_agreement(GROUPS, 256.0, n_groups=pf.N_ID_GROUPS)
        self.assertEqual(got['n_same_order'], got['n_seeds'])


class TestTautologicalErrorIsMarked(unittest.TestCase):
    """逐条侧误差在本数据上结构性为 0，归档必须写明它是同义反复。"""

    def test_flag_present(self):
        got = pf.traj_level_error(GROUPS, 256.0)
        self.assertTrue(got['error_is_tautological'])
        self.assertEqual(got['n_with_error'], 0)
        self.assertIn('unknown', got)


if __name__ == '__main__':
    unittest.main(verbosity=2)
