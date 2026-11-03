#!/usr/bin/env python3
"""real_lengths_policy 的单元测试（§13.30 第四批）。

钉住两件事：
  ① 排程器复现第三批的构造档读数（口径没漂）；
  ② **"更强调度抬税"不是无条件成立的**——组内离散度低于约 0.4 时符号相反。
     这条是本批最重要的发现，必须写成测试：否则下次谁再把 cv=0.6 的构造档结论
     当成通用规律引用，测试不会红。

    python3 -m unittest test_real_lengths_policy -v
"""

import json
import os
import random
import statistics
import unittest

import group_split_tradeoff as m
import real_lengths_policy as r

SEEDS = list(range(30))


def mean_tax(cv, backlog, policy):
    vals = []
    for s in SEEDS:
        rng = random.Random(s)
        lens = [[max(1.0, rng.gauss(50.0, 50.0 * cv)) for _ in range(8)] for _ in range(16)]
        vals.append(m._makespan_by_policy(lens, backlog, policy)['tax'])
    return statistics.mean(vals)


class TestConstructBaselineReproduces(unittest.TestCase):
    """口径对齐：本脚本用的排程器必须给出第三批的 34.0%／36.8%。"""

    def test_fifo_matches_third_batch(self):
        self.assertAlmostEqual(mean_tax(0.6, 4, m.POLICY_FIFO), 0.3402, places=3)

    def test_lof_matches_third_batch(self):
        self.assertAlmostEqual(mean_tax(0.6, 4, m.POLICY_LOF), 0.3677, places=3)


class TestSignFlipsWithWithinGroupDispersion(unittest.TestCase):
    """负向对照：把第三批的结论当通用规律 → 在低离散负载上必须失败。"""

    def test_high_cv_lof_raises_tax(self):
        self.assertGreater(mean_tax(0.6, 4, m.POLICY_LOF), mean_tax(0.6, 4, m.POLICY_FIFO) + 0.02)

    def test_low_cv_lof_lowers_tax(self):
        """真实 rollout 的组内 cv≈0.155 落在这一侧：更强调度**降**税。"""
        self.assertLess(mean_tax(0.155, 4, m.POLICY_LOF), mean_tax(0.155, 4, m.POLICY_FIFO) - 0.02)

    def test_zero_load_still_clean(self):
        """无论离散度多少，组内长度全同时税必须为 0（排程器本身不能引入伪排队）。"""
        for policy in (m.POLICY_FIFO, m.POLICY_LOF):
            lens = [[50.0] * 8 for _ in range(16)]
            self.assertLess(abs(m._makespan_by_policy(lens, 4, policy)['tax']), 1e-12)


class TestIccIsNotTheControl(unittest.TestCase):
    """ICC 单独扫描时符号不变 ⇒ 不能用"组内相关高"解释真实档的符号翻转。"""

    def test_icc_sweep_keeps_positive_sign(self):
        for icc in (0.0, 0.5, 0.75):
            fifo, lof = [], []
            for s in SEEDS:
                lens = r._sample_icc_group_lens(random.Random(s), icc)
                fifo.append(m._makespan_by_policy(lens, 4, m.POLICY_FIFO)['tax'])
                lof.append(m._makespan_by_policy(lens, 4, m.POLICY_LOF)['tax'])
            # 组内 cv 固定 0.6 时，四档 ICC 的位移都应为正
            self.assertGreater(statistics.mean(lof), statistics.mean(fifo))


class TestTruncationUsesArmCap(unittest.TestCase):
    """负向对照：截断率必须按本档的 max_response 算。

    抓的是实际发生的错误——`describe()` 曾把上限写死 256，于是 cap1024 档被报成
    "截断 45%／61%"，而 §13.50 的实测是 0.03%／0.65%，四个臂还被抹成同一对值。
    """

    def test_cap1024_arm_not_read_as_256(self):
        lens = [[300.0] * 8 for _ in range(16)]   # 全部 300：超过 256、不到 1024
        self.assertEqual(r.describe(lens, cap=1024.0)['truncated_fraction'], 0.0)
        self.assertEqual(r.describe(lens, cap=256.0)['truncated_fraction'], 1.0)
        self.assertEqual(r.describe(lens, cap=1024.0)['max_response'], 1024.0)

    def test_arm_cap_routes_by_label(self):
        self.assertEqual(r.arm_cap('qwen15b_cap1024'), 1024.0)
        self.assertEqual(r.arm_cap('qwen05b_cap1024'), 1024.0)
        self.assertEqual(r.arm_cap('qwen15b_cap256'), 256.0)
        self.assertEqual([c for _, f in r.REAL_ARMS for c in [r.arm_cap(_)]],
                         [256.0, 1024.0, 256.0, 1024.0])

    def test_build_summary_passes_arm_cap_to_describe(self):
        """接线检查：`build_summary()` 必须把 cap 传进 `describe()`，否则前两条形同虚设。

        这里替换 `describe()` 截获它收到的 cap，不需要跑排程，成本与臂数同阶。
        """
        seen = {}
        orig = r.describe

        def spy(groups, cap=256.0):
            seen[cap] = seen.get(cap, 0) + 1
            return orig(groups, cap=cap)

        r.describe = spy
        try:
            r.build_real_arm_structure()
        finally:
            r.describe = orig
        self.assertEqual(seen, {256.0: 2, 1024.0: 2})

    def test_archived_truncation_matches_13_50(self):
        """归档值与 §13.50 实测必须同向：cap256 高截断、cap1024 近乎不截断。"""
        with open(r.SUMMARY_PATH) as f:
            real = json.load(f)['real_arms']
        for label, arm in real.items():
            frac = arm['structure']['truncated_fraction']
            self.assertEqual(arm['structure']['max_response'], r.arm_cap(label))
            if label.endswith('cap1024'):
                self.assertLess(frac, 0.01, label)      # 实测 0.03%／0.65%
            else:
                self.assertGreater(frac, 0.4, label)    # 实测 45.3%／60.8%


class TestSignStabilityDiagnostics(unittest.TestCase):
    """§13.30 第五批：均值不够，还要看符号与基线抖动。

    上一批在 cap256 两档上把「−0.95pp（27/30 为负）」写成方向性结论，本批补的
    三个诊断是判它能不能当结论用的门槛。每条都配一个**在把被测代码改坏时必须
    变红**的断言。
    """

    @staticmethod
    def _flat_groups():
        """组内长度全同 → 两臂读数恒等，用于造"无分辨率"的输入。"""
        return [[100.0] * 8 for _ in range(16)]

    def test_identical_arms_report_no_resolution(self):
        """负向对照：两臂完全相同时 `n_identical` 必须吃掉全部种子。

        若实现把"差为 0"错算成正号（例如写成 `d >= 0`），这里会报 30 个阳性、
        `same_sign_fraction` 报 1.0——一个本无分辨率的格子被读成"稳定"。
        """
        st = r.policy_sign_stability(self._flat_groups(), 16, 4, m.POLICY_LOF, seeds=SEEDS)
        self.assertEqual(st['n_identical'], len(SEEDS))
        self.assertEqual(st['n_positive'], 0)
        self.assertEqual(st['n_negative'], 0)

    def test_sign_counts_sum_to_seed_count(self):
        """三态计数必须闭合，且符号混杂的格子不能报满分同号。"""
        groups = r.load_real_groups(os.path.join(r.REAL_DIR, 'qwen15b_gsm8k_n8.jsonl.gz'))
        st = r.policy_sign_stability(groups, 16, 8, m.POLICY_LOF, seeds=SEEDS)
        self.assertEqual(st['n_positive'] + st['n_negative'] + st['n_identical'], len(SEEDS))
        # 8 批这一格在 30 个种子上符号混杂
        self.assertLess(st['same_sign_fraction'], 0.8)

    def test_fifo_band_is_zero_on_identical_groups(self):
        """负向对照：组内长度全同时换序不改变 FIFO 读数，带宽必须是 0。"""
        band = r.fifo_reorder_band(self._flat_groups(), 16, 4, seeds=SEEDS)
        self.assertEqual(band['max_pp'], 0.0)

    def test_fifo_band_actually_shuffles(self):
        """负向对照：若实现忘了 shuffle，带宽恒 0，这条会失败。"""
        groups = r.load_real_groups(os.path.join(r.REAL_DIR, 'qwen15b_gsm8k_n8.jsonl.gz'))
        band = r.fifo_reorder_band(groups, 16, 4, seeds=list(range(6)))
        self.assertGreater(band['max_pp'], 0.5)

    def test_arrival_shift_reports_both_orders(self):
        """到达顺序假设的对照：两种到达序都要给出读数，不能只报一种。"""
        groups = r.load_real_groups(os.path.join(r.REAL_DIR, 'qwen05b_gsm8k_n8_cap1024.jsonl.gz'))
        sh = r.arrival_assumption_shift(groups, 16, 4, m.POLICY_LOF, seeds=list(range(10)))
        self.assertEqual(sh['n_total'], 10)
        # cap1024 4 批是本批符号最稳的一格：换到达序后绝大多数种子仍同号
        self.assertGreaterEqual(sh['n_same_sign'], 8)

    def test_no_resolution_cell_reports_zero_active(self):
        """两臂恒等时：没有种子在表态，ssf 与 active 档都必须反映这一点。

        第十批：`same_sign_fraction` 保留 identical 在分母（既有字段不动），
        但配套的 `resolve_active_fraction` 必须为 0、`same_sign_fraction_active`
        必须为 None——不能因为"全部为 0"就报一个像模像样的同号占比。
        """
        st = r.policy_sign_stability(self._flat_groups(), 16, 4, m.POLICY_LOF, seeds=SEEDS)
        self.assertEqual(st['resolve_active_fraction'], 0.0)
        self.assertIsNone(st['same_sign_fraction_active'])
        # 负向对照：若把 active 档写成复用全列 ssf，下面这条会变成 0.0 而失败
        self.assertNotEqual(st['same_sign_fraction_active'], st['same_sign_fraction'])

    def test_low_backlog_cell_active_fraction_exceeds_ssf(self):
        """低积压格：identical 吃掉大半种子，active 档必须高于全列 ssf。

        1.5B cap256 积压 2 是 18/20 个种子两臂相等的一格；全列 ssf 只有 0.1，
        看上去像"方向混杂"，而表态的种子里其实是同向的。两档必须分开。
        """
        groups = r.load_real_groups(os.path.join(r.REAL_DIR, 'qwen15b_gsm8k_n8.jsonl.gz'))
        st = r.policy_sign_stability(groups, 16, 2, m.POLICY_LOF, seeds=SEEDS)
        self.assertGreaterEqual(st['n_identical'], 15)
        self.assertLessEqual(st['resolve_active_fraction'], 0.25)
        self.assertGreater(st['same_sign_fraction_active'],
                           st['same_sign_fraction'])
        # 第十批：这一格的档名必须是 `consistent_but_thin` 而不是 `no_resolution`。
        # 两者的判词不同——前者说"方向一致但没分辨率"，后者说"什么也不能说"；
        # 旧的三步把两者并成一句"符号混杂"。把分档写成按全列 ssf 判断时这条会红
        # （全列 ssf=0.1 且表态占比 0.25，两种错误实现都会给 no_resolution 或 mixed）。
        self.assertEqual(st['regime'], 'consistent_but_thin')

    def test_regime_is_not_derived_from_full_denominator_ssf(self):
        """负向对照：档名不能由图省事的全列 ssf 决定。

        这一格表态的种子 3 个全同向、27 个重复 ⇒ 全列 ssf=0.1（看着像混杂），
        `same_sign_fraction_active`=1.0。若 `sign_regime` 退化成只看全列 ssf，
        档名会落到 `mixed`/`no_resolution`，本断言即失败。
        """
        st = r.policy_sign_stability(self._flat_groups(), 16, 4, m.POLICY_LOF, seeds=SEEDS)
        self.assertEqual(st['regime'], 'no_resolution')


class TestBandFloor(unittest.TestCase):
    """带宽为 0 是不是"基线很稳"（第十一批，2026-11-04）。

    `exceeds_band(|效应|, 0)` 对所有非零效应一律通过，所以带宽为 0 的格子最容易被
    读成"效应显著超出基线抖动"。而这里的 0 有两种成因，判词完全不同：

      - 组最大全部贴 `max_response` ⇒ `R` 恒等 0，**打乱顺序也不变** ⇒ `no_reading`；
      - 只有部分种子为 0 ⇒ 带宽由少数种子撑起 ⇒ `degenerate`。

    两条负向对照都落在"把分档实现改坏"的那根线上：把 `no_reading` 并进 `usable`、
    或把 `degenerate` 并进 `usable`，对应断言即红。
    """

    @staticmethod
    def _pinned_groups(n_groups=16, n_per=8):
        """16 组的组最大全部相等 ⇒ 任何顺序下组臂与轨迹臂都只受同一条约束。

        这就是 cap256 档的形态：86%／94% 的组最大恰好等于 256，于是抽 16 组时
        组最大全等的概率约 0.09／0.38——整格 `R ≡ 0` 不是"很稳"，是问不出来。
        """
        lens = [100.0, 103.0, 107.0, 111.0, 115.0, 120.0, 126.0, 133.0]
        return [[x for x in lens[:n_per]] for _ in range(n_groups)]

    def test_pinned_groups_are_no_reading(self):
        """负向对照：撤掉 `no_reading` 档（并进 `usable`），这条即红。"""
        floor = r.fifo_band_floor(self._pinned_groups(), 16, 4, seeds=SEEDS)
        self.assertEqual(floor['n_r_identically_zero'], len(SEEDS))
        self.assertEqual(floor['band_floor'], 'no_reading')

    def test_mixed_pinned_and_ladder_scores_degenerate(self):
        """14 组贴顶 + 2 组组最大略低 ⇒ 只有抽到那一组的种子带宽为 0。

        这就是 `qwen05b_cap256` 积压 4 的形态：22/30 个种子带宽为 0、8 个不为 0。
        负向对照：把 `degenerate` 并进 `usable`，这条即红。
        """
        pinned = [[250.0, 252.0, 254.0, 256.0, 256.0, 256.0, 256.0, 256.0]] * 14
        lower = [[150.0, 170.0, 190.0, 210.0, 230.0, 236.0, 243.0, 251.0]] * 2
        floor = r.fifo_band_floor(pinned + lower, 16, 4, seeds=SEEDS)
        self.assertGreater(floor['n_band_zero'], 0)
        self.assertLess(floor['n_r_identically_zero'], len(SEEDS))
        self.assertEqual(floor['band_floor'], 'degenerate')

    def test_uniform_groups_inside_are_not_enough(self):
        """组内每条都相同也不产生税：`R` 需要组**内部**有极差，不是组间有差。

        `[[100]*8, [200]*8, ...]` 四个组两两不同，税却是 0——组臂按整组放时
        每条同时完成、轨迹臂也一完成就还格子，两者 makespan 相同。所以构造
        "带宽退化"的输入必须让组内长短不一，否则连 `usable` 都谈不上。
        """
        uniform = [[100.0 * (i + 1)] * 8 for i in range(16)]
        floor = r.fifo_band_floor(uniform, 16, 4, seeds=SEEDS)
        self.assertEqual(floor['n_r_identically_zero'], len(SEEDS))
        self.assertEqual(floor['band_floor'], 'no_reading')

    def test_constructed_load_is_usable(self):
        """构造档（组最大 100–400 均匀）带宽不为 0：分档不能把所有格子都报成退化。"""
        groups = m.sample_group_lens(random.Random(0), n_groups=16, n_per=8, cv=0.6)
        floor = r.fifo_band_floor(groups, 16, 4, seeds=SEEDS)
        self.assertEqual(floor['n_band_zero'], 0)
        self.assertEqual(floor['band_floor'], 'usable')

    def test_real_cap256_low_backlog_is_no_reading(self):
        """实测格（归档里的形态）：cap256 积压 2 → `no_reading`；cap256 积压 4 → `degenerate`。

        这两格正是先前"带宽为 0／过半为 0"的那两个，判词不同：
        积压 2 整格问不出来，积压 4 仍有 8 个种子给出非零带宽。
        """
        groups = r.load_real_groups(os.path.join(r.REAL_DIR, 'qwen05b_gsm8k_n8.jsonl.gz'))
        low = r.fifo_band_floor(groups, 16, 2, seeds=SEEDS)
        self.assertEqual(low['band_floor'], 'no_reading')
        mid = r.fifo_band_floor(groups, 16, 4, seeds=SEEDS)
        self.assertEqual(mid['band_floor'], 'degenerate')

    def test_exactly_half_zero_band_is_still_usable(self):
        """**边界本身**：恰好一半种子带宽为 0 时判 `usable`，不是 `degenerate`。

        第十一批 readme 自记的缺口——此前的构造全是"严格过半"或"全部"，把边界
        从 `>` 改成 `>=` 时 5 条测试全绿。这里用实测搜到的输入钉住边界：40 组里
        33 组贴顶，30 个抽样种子中恰好 15 个抽到全贴顶的 16 组（另 15 个抽到梯子组）。

        「过半」在这里是**严格多于一半**的意思：一半为 0 说明还有一半种子在表态，
        带宽仍有含量。改判据（`>` → `>=`）时这条会红。
        """
        pinned = [[250.0, 252.0, 254.0, 256.0, 256.0, 256.0, 256.0, 256.0]]
        ladder = [[50.0, 100.0, 150.0, 200.0, 230.0, 240.0, 250.0, 255.0]]
        groups = pinned * 33 + ladder * 7
        floor = r.fifo_band_floor(groups, 16, 4, seeds=SEEDS)
        self.assertEqual(floor['n_band_zero'], len(SEEDS) // 2)
        self.assertLess(floor['n_r_identically_zero'], len(SEEDS))
        self.assertEqual(floor['band_floor'], 'usable')

    def test_one_more_zero_band_flips_to_degenerate(self):
        """边界另一侧：把同一个输入多推一组贴顶，带宽为 0 的种子过半 ⇒ `degenerate`。

        与上一条成对——只测一侧的话，"`>` 恒真"或"`>` 恒假"都只会有一种失败。
        """
        pinned = [[250.0, 252.0, 254.0, 256.0, 256.0, 256.0, 256.0, 256.0]]
        ladder = [[50.0, 100.0, 150.0, 200.0, 230.0, 240.0, 250.0, 255.0]]
        groups = pinned * 34 + ladder * 6
        floor = r.fifo_band_floor(groups, 16, 4, seeds=SEEDS)
        self.assertGreater(floor['n_band_zero'], len(SEEDS) // 2)
        self.assertLess(floor['n_r_identically_zero'], len(SEEDS))
        self.assertEqual(floor['band_floor'], 'degenerate')


class TestSamplingReach(unittest.TestCase):
    """`no_reading` 是"这个负载问不出来"还是"这一次抽样问不出来"（第十四批，2026-11-04）。

    第十一批把 `no_reading` 写成"这一格问不出来"，但没分清两件事：负载上**每条路**
    都排不出先后（换抽样也没用），还是这次 `n_groups` 太小、每次都抽不到能排的组
    （加大就问得出来）。这里是纯解析与纯函数的守护，不跑真实负载——真实读数在
    [sampling-reach.json](sampling-reach.json) 与 readme 第十四批。

    负向对照落在哪根线：把 `free_share_min` 的 `n_free<=0` 分支改成返回 1（"总能问
    出来"），`test_no_free_group_means_never` 即红；把 `sampling_floor` 的
    `n_r_zero == SEED_COUNT` 判据改成 `>= 1`，`test_n16_matches_archive_verdict` 即红。
    """

    def test_no_free_group_means_never(self):
        """没有一组的组最大低于上限 ⇒ 无论 n_groups 多大都问不出来（返回 None）。"""
        import sampling_reach as s
        self.assertIsNone(s.free_share_min(n_total=480, n_free=0, share=0.5))

    def test_min_n_is_monotone_and_bounded(self):
        """n_free 越大、要求越松，所需 n_groups 越小；且不超过总量。"""
        import sampling_reach as s
        self.assertLess(s.free_share_min(480, 67, 0.5), s.free_share_min(480, 67, 0.9))
        self.assertLess(s.free_share_min(480, 67, 0.5), s.free_share_min(480, 28, 0.5))
        self.assertLessEqual(s.free_share_min(480, 400, 0.9), 480)

    def test_pinned_groups_never_reach_any_n(self):
        """构造输入：16 组全贴顶时任何 n_groups 都 `no_reading`——与真实 cap256 同形。"""
        import sampling_reach as s
        lens = [250.0, 252.0, 254.0, 256.0, 256.0, 256.0, 256.0, 256.0]
        groups = [list(lens) for _ in range(16)]
        for n in (4, 8, 16):
            floor = s.sampling_floor(groups, 256.0, n, 2)
            self.assertEqual(floor['band_floor'], 'no_reading', 'n=%d' % n)

    def test_n16_matches_archive_verdict(self):
        """自检：本脚本 n_groups=16 的三档与归档用的 `fifo_band_floor` 一致。

        **必须逐格一致**，否则"扫描结论"与"归档结论"会各说各话。只核 cap256 积压 2
        这一格（最快一格）；其余 16 格在 `sampling_reach.py` 的 `at_n16` 里逐格列出。
        """
        import sampling_reach as s
        groups = r.load_real_groups(os.path.join(r.REAL_DIR, 'qwen15b_gsm8k_n8.jsonl.gz'))
        mine = s.sampling_floor(groups, 256.0, 16, 2)
        official = r.fifo_band_floor(groups, 16, 2, seeds=SEEDS)
        self.assertEqual(mine['band_floor'], official['band_floor'])
        self.assertEqual(mine['n_r_identically_zero'], official['n_r_identically_zero'])
        self.assertEqual(mine['n_band_zero'], official['n_band_zero'])


if __name__ == '__main__':
    unittest.main()
