#!/usr/bin/env python3
"""train_side_composition 的单元测试（rollout 专项 §13.32，2026-11-02 M1）。

测试重点不是"数字对不对"，而是**每个结论能不能被可执行的断言打死**：
  * 组成效应必须随 rho 变（否则量的是别的东西）；
  * 噪声比必须随块数 K 变（否则指标退化）；
  * 识别性检验本身必须能抓到恒常数（太松）与能放过真信号（太严）；
  * 估计量必须用 n-1（用 n 会让 K>1 的读数凭空消失——本批自查真实踩过）；
  * 交错排列必须真的把每组摊到所有块（纯度 0），否则"交错"这个名字就是假的。
"""

import math
import os
import random
import statistics
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import train_side_composition as t  # noqa: E402


class TestGradientModel(unittest.TestCase):
    """梯度模型必须真的把 rho 当成"组信号强度"。"""

    def test_rho_one_makes_group_members_identical(self):
        rng = random.Random(0)
        uids, grads = t._grad_model(rng, 1.0, None)
        for i in range(1, t.GROUP_SIZE):
            diff = t._norm([a - b for a, b in zip(grads[0], grads[i])])
            self.assertEqual(diff, 0.0)

    def test_rho_zero_removes_group_signal(self):
        rng = random.Random(0)
        uids, grads = t._grad_model(rng, 0.0, None)
        # 同组与异组的样本差应同量级（没有系统性组方向）
        same = statistics.mean(
            t._norm([a - b for a, b in zip(grads[i], grads[i + 1])]) for i in range(0, 64, 8))
        cross = statistics.mean(
            t._norm([a - b for a, b in zip(grads[i], grads[8 + i])]) for i in range(0, 64, 8))
        self.assertLess(abs(same / cross - 1.0), 0.35)

    def test_group_size_and_count(self):
        rng = random.Random(3)
        uids, grads = t._grad_model(rng, 0.5, None)
        self.assertEqual(len(grads), t.N_GROUPS * t.GROUP_SIZE)
        self.assertEqual(len(set(uids)), t.N_GROUPS)


class TestBlockShapes(unittest.TestCase):
    def test_contiguous_covers_everything_without_overlap(self):
        shapes = t._block_shapes_contiguous(256, 4)
        self.assertEqual(sum(n for _, n in shapes), 256)
        pos = 0
        for start, n in shapes:
            self.assertEqual(start, pos)
            pos += n

    def test_contiguous_handles_non_divisible(self):
        shapes = t._block_shapes_contiguous(10, 4)
        self.assertEqual(sum(n for _, n in shapes), 10)
        self.assertEqual([n for _, n in shapes], [3, 3, 2, 2])

    def test_interleaved_partitions_modulo_k(self):
        n, k = 256, 4
        union = set()
        for j in range(k):
            idx = t._block_indices_interleaved(n, k, j)
            self.assertTrue(all(i % k == j for i in idx))
            self.assertFalse(union & set(idx))
            union |= set(idx)
        self.assertEqual(union, set(range(n)))


class TestEstimator(unittest.TestCase):
    def test_homogeneous_batch_has_no_defined_ratio(self):
        """所有梯度完全相同时，全批方差为 0，比值无定义（返回 None，不是伪造一个数）。"""
        g = [0.3, -0.1, 0.2]
        grads = [list(g) for _ in range(8)]     # 必须是独立副本，共用一个 list 会被原地改坏
        self.assertIsNone(t._variance_ratio_from_indices(grads, [[i] for i in range(8)]))

    def test_must_use_n_minus_one(self):
        """用 1/n 估计时，"同组完全不相交"的切法会把方差估成 0。

        构造：每块恰好装一个完整组（组内 8 条样本各不相同、组均值不同）。
        无偏估计（n−1）下块均值的方差 > 0；分母用 n 会得到 0。
        """
        rng = random.Random(0)
        uids, grads = t._grad_model(rng, 0.0, None)
        # 每块 = 一个完整组（8 条）：块内异质、块均值 = 该组均值
        blocks = [list(range(g * t.GROUP_SIZE, (g + 1) * t.GROUP_SIZE)) for g in range(4)]
        unbiased = t._asymptotic_var(uids, grads, 4, [(b[0], len(b)) for b in blocks])
        dim = len(grads[0])
        biased = 0.0
        for b in blocks:
            m = [0.0] * dim
            for i in b:
                m = t._add(m, grads[i])
            m = [x / len(b) for x in m]
            s2 = sum(t._norm([grads[i][d] - m[d] for d in range(dim)]) ** 2 for i in b) / len(b)
            biased += s2 / len(b)
        self.assertGreater(unbiased, 0.0)
        self.assertGreater(unbiased, biased)   # n-1 版严格更大

    def test_ratio_grows_with_k(self):
        rng = random.Random(0)
        uids, grads = t._grad_model(rng, 0.5, None)
        n = len(grads)
        rs = [t._asymptotic_gradient_variance_ratio(
            uids, grads, k, t._block_shapes_contiguous(n, k)) for k in (1, 2, 4, 8)]
        for a, b in zip(rs, rs[1:]):
            self.assertGreater(b, a)

    def test_k1_k2_relation(self):
        """K=1 的读数 = 块方差估计 / 全批方差估计，比值有限且为正；K 增大单调上升。"""
        rng = random.Random(0)
        uids, grads = t._grad_model(rng, 0.5, None)
        r1 = t._asymptotic_gradient_variance_ratio(
            uids, grads, 1, t._block_shapes_contiguous(len(grads), 1))
        r2 = t._asymptotic_gradient_variance_ratio(
            uids, grads, 2, t._block_shapes_contiguous(len(grads), 2))
        self.assertGreater(r1, 0.0)
        self.assertGreater(r2, r1)


class TestCompositionEffect(unittest.TestCase):
    def test_interleaved_beats_contiguous_when_groups_intact(self):
        """组内连续排列时，把组摊开（交错）一定提高噪声；rho=0 时两者相等。"""
        n, k = t.N_GROUPS * t.GROUP_SIZE, 4
        sizes = [n // k] * k

        def ratio(grouped_grads, blocks):
            return t._variance_ratio_from_indices(grouped_grads, blocks)

        rng = random.Random(1)
        uids, grads = t._grad_model(rng, 0.9, None)
        contig = [list(range(j * sizes[0], (j + 1) * sizes[0])) for j in range(k)]
        inter = [t._block_indices_interleaved(n, k, j) for j in range(k)]
        self.assertGreater(ratio(grads, inter), ratio(grads, contig))

        # rho=0 时样本梯度都是独立随机方向，块怎么切都不改变组成分布
        rr = []
        for s_ in range(8):
            _, g0 = t._grad_model(random.Random(s_), 0.0, None)
            rr.append(ratio(g0, inter) / ratio(g0, contig))
        self.assertLess(abs(statistics.mean(rr) - 1.0), 0.02)

    def test_purity_detects_interleaving(self):
        uids = [g for g in range(8) for _ in range(8)]
        n = len(uids)
        inter_shapes = [(0, n)]
        # 用交错切分后 uid 的顺序来判断纯度
        order = [i for j in range(4) for i in t._block_indices_interleaved(n, 4, j)]
        pure_inter = t.group_purity([uids[i] for i in order], t._block_shapes_contiguous(n, 4))
        self.assertEqual(pure_inter['intact_ratio'], 0.0)
        self.assertEqual(pure_inter['mean_blocks_per_group'], 4)
        pure_grouped = t.group_purity(uids, t._block_shapes_contiguous(n, 4))
        self.assertEqual(pure_grouped['intact_ratio'], 1.0)


class TestIdentifiabilityGuard(unittest.TestCase):
    def test_constant_metric_is_flagged_insensitive(self):
        r = t.identifiability(t._metric_constant, t.RHO_LEVELS, 'constant')
        self.assertFalse(r['sensitive'])

    def test_real_signal_is_flagged_sensitive(self):
        r = t.identifiability(t._metric_composition_ratio, t.RHO_LEVELS, 'compos')
        self.assertTrue(r['sensitive'])

    def test_relative_not_absolute_threshold(self):
        """指标量级只有 1e-2 时，绝对阈值会把噪声判成"动了"。"""
        small = lambda rng, lv: 0.05 + 1e-6 * lv     # 变化只有 1e-6，绝对上"动了"
        r = t.identifiability(small, t.RHO_LEVELS, 'tiny')
        self.assertFalse(r['sensitive'])


class TestFrameworkSemantics(unittest.TestCase):
    def test_every_framework_declares_grouping(self):
        for name, ev in t.FRAMEWORK_EVIDENCE.items():
            self.assertIn(ev['advantage_grouping'], ('uid', 'prompt'), name)
            self.assertGreaterEqual(len(ev['anchors']), 2, name)

    def test_local_snapshots_verify_all_anchors(self):
        """本目录的 source-snapshots 必须能让四个框架的每条 anchor 都命中。"""
        for name, ev in t.FRAMEWORK_EVIDENCE.items():
            root = t.snapshot_root(name)
            self.assertIsNotNone(root, '缺少 %s 快照' % name)
            for rel, snippet in ev['anchors']:
                ok, detail = t.check_anchor(root, rel, snippet)
                self.assertTrue(ok, '%s %s: %s' % (name, rel, detail))

    def test_anchor_check_reports_missing_snapshot(self):
        ok, detail = t.check_anchor(None, 'a/b.py', 'x = 1')
        self.assertFalse(ok)
        self.assertIn('no local snapshot', detail)

    def test_anchor_check_detects_present_snippet(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, 'pkg'))
            with open(os.path.join(d, 'pkg', 'm.py'), 'w') as f:
                f.write('def f():\n    return 1\n')
            ok, detail = t.check_anchor(d, 'pkg/m.py', 'def f():\n    return 1')
            self.assertTrue(ok)
            self.assertEqual(detail, 'hit')
            ok2, _ = t.check_anchor(d, 'pkg/m.py', 'def unrelated():\n    pass')
            self.assertFalse(ok2)


class TestSummaryShape(unittest.TestCase):
    def test_build_summary_is_self_consistent(self):
        out = t.build_summary({})
        self.assertTrue(out['verdict']['no_framework_normalizes_per_partition'])
        k = out['composition']['noise_ratio_by_K']
        self.assertLess(k['K1']['contiguous_mean'], k['K8']['contiguous_mean'])
        d4 = out['composition']['composition_effect_at_K4']['dp4']
        self.assertGreater(d4['ldm_over_grouped_mean'], 1.0)
        self.assertLess(d4['interleaved_over_ldm_mean'], 1.05)
        # 负向对照必须明显偏离 1
        self.assertGreater(
            out['composition']['negative_control_block_correlated']['interleaved_over_contiguous'],
            1.2)


if __name__ == '__main__':
    unittest.main()
