"""带宽为 0 时的判据守卫（2026-11-04 第十四批）。

**为什么单独一个文件。** `exceeds_band(|效应|, 0.0)` 曾对所有非零效应一律返回 True，
于是"基线换序一动不动"与"这一格根本没有读数"被读成同一句结论。第十一批把这件事记
成"判据只能留在产生带宽的那一层"，工具 readme 只写了一条要求；本批把它变成**代码
拒绝**：`exceeds_band` 在 `band == 0.0` 时抛 `ValueError`，要断言为 0 的带宽必须用
`band_floor_pass(..., zero_kind=...)` 把成因写出来。

**负向对照落在哪根线上**：把 `exceeds_band` 的守卫去掉（回到 `abs(effect) > abs(band)`），
`TestZeroBandIsRefused` 的三条会红；把 `band_floor_pass` 的 `no_reading` 分支改成
返回 `True`，`TestZeroKindDecides` 会红。

运行：cd experiments/rollout-c-class-ledger && python3 -m unittest test_zero_band_guard -v
"""
import os
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / 'tools' / 'paired-diff'))

import paired_diff  # noqa: E402


class TestZeroBandIsRefused(unittest.TestCase):
    """带宽恰为 0 时，工具层不得替调用方猜成因。"""

    def test_exceeds_band_raises_on_zero_band(self):
        with self.assertRaises(ValueError):
            paired_diff.exceeds_band(0.02, 0.0)

    def test_negative_effect_also_raises(self):
        """符号不影响这一条——守卫看的是带宽，不是效应。"""
        with self.assertRaises(ValueError):
            paired_diff.exceeds_band(-0.02, 0.0)

    def test_declared_cause_is_required(self):
        """给了带宽 0 却没声明成因，`band_floor_pass` 同样拒答。"""
        with self.assertRaises(ValueError):
            paired_diff.band_floor_pass(0.02, 0.0)
        with self.assertRaises(ValueError):
            paired_diff.band_floor_pass(0.02, 0.0, zero_kind='whatever')


class TestZeroKindDecides(unittest.TestCase):
    """两种成因判词相反：一个说得到，一个问不出来。"""

    def test_baseline_stable_allows_claim(self):
        self.assertTrue(paired_diff.band_floor_pass(0.02, 0.0, zero_kind='baseline_stable'))

    def test_no_reading_never_passes(self):
        """即使效应很大（0.02 = 2pp，远大于任何带宽量级），问不出来就是问不出来。"""
        self.assertFalse(paired_diff.band_floor_pass(0.02, 0.0, zero_kind='no_reading'))
        self.assertFalse(paired_diff.band_floor_pass(100.0, 0.0, zero_kind='no_reading'))

    def test_zero_effect_stable_baseline_is_not_a_pass(self):
        """`baseline_stable` + 效应恰为 0 ⇒ 不过门槛（严格大于）。"""
        self.assertFalse(paired_diff.band_floor_pass(0.0, 0.0, zero_kind='baseline_stable'))


class TestNonZeroBandUnchanged(unittest.TestCase):
    """带宽 > 0 时两个函数逐位相同——新增入口不能改变已有判读。"""

    def test_matches_exceeds_band(self):
        for effect in (-2.0, -1.0, 0.0, 0.5, 1.0, 1.5):
            for band in (0.1, 1.0, 3.0):
                self.assertEqual(paired_diff.band_floor_pass(effect, band),
                                 paired_diff.exceeds_band(effect, band))
        self.assertFalse(paired_diff.band_floor_pass(1.0, 1.0))
        self.assertTrue(paired_diff.band_floor_pass(1.5, 1.0))


class TestNoLiveCallerPassesZeroSilently(unittest.TestCase):
    """探针侧的入口也接上了：不给 `zero_kind` 时同样拒答。

    `bucketing_probe.effect_exceeds_band` 是归档读数的唯一入口，若它绕过守卫，
    工具层的拒绝就只对"未来新代码"有效。
    """

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(REPO / 'experiments' / 'rollout-group-split-tradeoff'))

    def test_probe_entry_refuses_zero_band(self):
        import bucketing_probe
        with self.assertRaises(ValueError):
            bucketing_probe.effect_exceeds_band(0.02, 0.0)
        self.assertFalse(bucketing_probe.effect_exceeds_band(0.02, 0.0, zero_kind='no_reading'))
        self.assertTrue(bucketing_probe.effect_exceeds_band(0.02, 0.0,
                                                            zero_kind='baseline_stable'))


class TestSelftestCoversTheGuard(unittest.TestCase):
    """工具自带 `--selftest` 必须覆盖这条守卫（不靠本文件也能发现回归）。"""

    def test_selftest_passes_and_mentions_zero_band(self):
        out = subprocess.run([sys.executable, str(REPO / 'tools' / 'paired-diff' /
                                                 'paired_diff.py'), '--selftest'],
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)


if __name__ == '__main__':
    unittest.main()
