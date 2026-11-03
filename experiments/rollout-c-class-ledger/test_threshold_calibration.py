"""门槛校准的守护测试（2026-11-04 第十一批）。

**为什么单独一个文件。** `calibrate_thresholds.py` 报的是"判据在已知真相上误判
多少"。这类读数最容易在下一次改判据时静默变坏——没人会重跑一遍校准再看数字
变没变。这里把**实测到的边界**钉成断言；判据（`sign_regime` 的门槛、`band_floor`
的"过半"）哪天被改动，这里会先红。

**与"注入"的分工。** 注入检验的是"改了测试能不能抓到"；这里检验的是"抓到的
那条边界是不是就是当初量出来的那条"。两者都要，不能互相代替。

运行：cd experiments/rollout-c-class-ledger && python3 -m unittest test_threshold_calibration -v
"""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / 'tools' / 'paired-diff'))

import paired_diff  # noqa: E402
import calibrate_thresholds as C  # noqa: E402


class TestSignRegimeCalibration(unittest.TestCase):
    """`sign_regime` 的四档门槛在已知真相上的误判率。"""

    @classmethod
    def setUpClass(cls):
        cls.out = C.calibrate_regime(trials=2000)

    def test_direction_stable_is_never_called_mixed(self):
        """真值同向（每个种子都明显为正）时，报 `mixed` 就是误报"方向混杂"。"""
        self.assertEqual(self.out['stable']['mixed'], 0.0)

    def test_pure_coin_is_almost_never_called_stable(self):
        """真值是无信息（每个种子独立掷硬币）时，报 `stable` 就是凭空给出方向。

        实测 1.1%（n=20 时约 1/2^19 组合恰好同号）——这个数不是门槛能调掉的，
        它是"20 个种子全同号"本身的概率。
        """
        self.assertLessEqual(self.out['coin']['stable'], 0.02)

    def test_thin_but_consistent_is_never_called_mixed(self):
        """表态少但同向 ⇒ `consistent_but_thin`。这正是第十批拆分母要解决的问题。"""
        self.assertEqual(self.out['thin_same']['mixed'], 0.0)
        self.assertEqual(self.out['thin_same']['consistent_but_thin'], 1.0)

    def test_all_zero_is_no_resolution(self):
        self.assertEqual(self.out['all_zero']['no_resolution'], 1.0)

    def test_thin_and_mixed_lands_in_one_of_the_two_thin_buckets(self):
        """表态少且表态的自己混杂 ⇒ 不能落到 `stable`／`mixed`（后者会暗示"够量的混杂"）。"""
        row = self.out['thin_zero_mix']
        self.assertEqual(row['stable'], 0.0)
        self.assertEqual(row['mixed'], 0.0)
        self.assertAlmostEqual(row['consistent_but_thin'] + row['no_resolution'], 1.0, places=6)


class TestBandFloorCalibration(unittest.TestCase):
    """`band_floor` 的 `no_reading` 是**全零**判据，不是"接近零"判据。"""

    @classmethod
    def setUpClass(cls):
        cls.out = C.calibrate_band_floor(trials=600)

    def test_all_zero_is_no_reading(self):
        self.assertEqual(self.out['truth_all_zero']['p_nonzero=0.00']['dist']['no_reading'], 1.0)

    def test_one_nonzero_seed_breaks_no_reading(self):
        """期望非零种子 1.5 个（p=0.05）时，`no_reading` 只剩约四分之一。

        这是**结构性**的：判据要求 30 个种子的 max R 全为 0，任何一个种子抽到
        非零就整格翻档。所以引用 `band_floor` 时必须同时报 `n_r_identically_zero`
        的计数，不能只看档名——真实 `qwen15b_cap256` 积压 4 就是 26/30 非零却被
        判 `usable`，而其中位带宽只有 2.832pp。
        """
        row = self.out['truth_all_zero']['p_nonzero=0.05']['dist']
        self.assertLess(row['no_reading'], 0.5)
        self.assertGreater(row['degenerate'], 0.5)

    def test_half_nonzero_seeds_can_be_called_usable(self):
        """一半种子非零时，过半种子带宽非零 ⇒ 档名给 `usable`，与"恒零"真相无关。

        断言这条是为了钉住：`usable` **不代表基线稳**，只代表"过半种子看不出来
        是常量"。效应量是否超出带宽仍须各自过 `effect_exceeds_band`。
        """
        row = self.out['truth_all_zero']['p_nonzero=0.50']['dist']
        self.assertGreater(row['usable'], 0.3)


class TestRealCellsAreReportedWithCounts(unittest.TestCase):
    """真实 16 格里，**同一个档名下零种子计数差一个数量级**——档名不能单独引用。

    实测（`real-lengths-summary.json`，30 种子）：
    `qwen15b_cap256` 积压 4 判 `usable`，但 26/30 个种子的 `R` 非零、中位带宽
    2.83pp；`qwen05b_cap256` 积压 8 也判 `usable`，30/30 非零、中位带宽 2.15pp。
    两者档名相同，一个是"多数种子撑起带宽"、一个是"全部种子都有带宽"。
    """

    @classmethod
    def setUpClass(cls):
        cls.cells = C.real_cells()

    def test_no_reading_only_where_every_seed_is_zero(self):
        for label, cells in self.cells.items():
            for name, c in cells.items():
                if c['band_floor'] == 'no_reading':
                    self.assertEqual(c['n_r_identically_zero'], 30,
                                     '%s %s 判 no_reading 但零种子只有 %d/30'
                                     % (label, name, c['n_r_identically_zero']))

    def test_usable_cells_span_a_wide_range_of_zero_counts(self):
        """`usable` 的格子零种子计数从 0/30 到 4/30 ⇒ 档名单独不足以判读。"""
        zero_counts = sorted(c['n_r_identically_zero']
                             for cells in self.cells.values()
                             for c in cells.values() if c['band_floor'] == 'usable')
        self.assertGreater(len(set(zero_counts)), 1)
        self.assertLessEqual(max(zero_counts), 4)

    def test_degenerate_cell_exists_and_is_stricter_than_usable(self):
        """0.5B cap256 积压 4：22/30 种子带宽为 0，档名 `degenerate`。"""
        c = self.cells['qwen05b_cap256']['backlog4']
        self.assertEqual(c['band_floor'], 'degenerate')
        self.assertEqual(c['n_band_zero'], 22)


class TestCalibrationIsNotTheInjectionTest(unittest.TestCase):
    """校准量的是判据误判率，不能拿注入测试代替；反之亦然。"""

    def test_calibration_archive_matches_script(self):
        """归档必须能由脚本逐字段重算——否则引用的数字与代码脱钩。"""
        import json
        built = C.build()
        text = json.dumps(built, ensure_ascii=False, indent=1, sort_keys=True)
        with open(C.OUT_PATH) as f:
            ref = json.load(f)
        self.assertEqual(json.dumps(ref, ensure_ascii=False, indent=1, sort_keys=True), text)

    def test_known_answer_input_still_classifies_as_before(self):
        """第十批用过的已知答案输入，档名不能漂（`[0,1,-1,0,2]` → mixed）。"""
        self.assertEqual(paired_diff.sign_regime([0, 1, -1, 0, 2])['regime'], 'mixed')


if __name__ == '__main__':
    unittest.main(verbosity=2)
