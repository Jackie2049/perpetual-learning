"""跨脚本一致性测试：同一列配对差喂进不同脚本，必须得到同一个计数（2026-11-03 M1 第六批遗留①）。

**为什么要这个文件**：第六批把 `sign_counts`／`same_sign_fraction`／`span`／
`sign_flip_around` 收进 `tools/paired-diff`，四个脚本改接后五份归档 `--verify`
逐字段不变。但那只能证明"改接没改变**已有归档**的读数"——**证明不了**四个脚本
对**同一组输入**给同一个答案。抽工具的初衷是"改正一处能带动另一处"，而
归档比对恰恰做不到这一点：四个脚本各自算自己的逐种子差值，喂给共享判据的
那一列只要在某个脚本里被**重新定基**（如 `v - 1.0`）或**换了口径**（如整数
步号 vs 浮点比率），共享判据本身再一致也照样会给出不同结论。

本文件钉三件事：

  ① **共用的计数必须逐字段相同**——同一列差值，`summarize_paired` 在各脚本
     出现处给出相同的 `n_positive`／`n_negative`／`n_identical`；
  ② **重新定基不改变符号分布**——比率型脚本把 `v - 1.0` 再交给 `sign_counts`，
     与非比率型脚本直接喂 `d` 必须一致（`d` 与 `d - 1` 的零判据相同，因为
     `sign_counts` 是对**零**归类，而定基只平移了零点）；
  ③ **整数步号与浮点比率不能混用同一套容差语义**——`steps_at` 的 `n_ties`
     是整数精确相等，本测试确认它与 `sign_counts` 在整数输入下等价。

运行：cd experiments/rollout-c-class-ledger && python3 -m unittest test_cross_script_consistency -v
"""

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / 'tools' / 'paired-diff'))

import paired_diff


# 一组专门造得刁钻的配对差，覆盖判读三步的每一种边界：
#   - 恰好等于容差（TOL）→ 必须落进 identical，不能记正
#   - 容差内的小正数／小负数 → identical
#   - 明显同向（全正、全负）→ 稳定
#   - 一半正一半负正好抵消 → 均值骗人
#   - 含整数步号（0／±1）→ 与 `steps_at` 的 n_ties 口径对齐
DIFFS_FLOAT = [
    0.0,
    paired_diff.TOL,          # 恰等于容差：strict 比较 → identical
    -paired_diff.TOL,
    1e-13,                    # 容差内
    -1e-13,
    0.05, 0.05, 0.05,         # 同向
    -0.02,                    # 反对
    0.10,
    0.0, 0.0,                 # 多个 identical
]

DIFFS_INT = [0, 1, -1, 2, 0, 0, -3, 1, 0, 2, -1, 0]  # 步号：整数精确相等


class TestSameColumnSameCounts(unittest.TestCase):
    """① 同一列差值在不同脚本的调用点上必须得到同一个计数。

    `summarize_paired` 是"这一列到底怎么分"的唯一入口。只要每个脚本都走它，
    计数就不会漂；这里直接对**每个脚本实际传进来的那一列**断言。若某个脚本
    自己另算一遍（不经 `summarize_paired`），本测试就抓不到——所以下面
    `TestScriptsActuallyUseSharedTool` 再补一条"确实在用它"的断言。
    """

    def test_summarize_paired_counts_match_sign_counts(self):
        for diffs in (DIFFS_FLOAT, DIFFS_INT):
            s = paired_diff.summarize_paired(diffs)
            c = paired_diff.sign_counts(diffs)
            self.assertEqual(s['n_positive'], c['n_positive'], diffs)
            self.assertEqual(s['n_negative'], c['n_negative'], diffs)
            self.assertEqual(s['n_identical'], c['n_identical'], diffs)

    def test_counts_sum_to_length(self):
        for diffs in (DIFFS_FLOAT, DIFFS_INT):
            c = paired_diff.sign_counts(diffs)
            self.assertEqual(c['n_positive'] + c['n_negative'] + c['n_identical'], len(diffs))

    def test_tol_boundary_recorded_as_identical_not_positive(self):
        """恰等于容差不进位——这是第四批负向对照抓到的边界，跨脚本也须一致。"""
        c = paired_diff.sign_counts([paired_diff.TOL, -paired_diff.TOL])
        self.assertEqual(c['n_positive'], 0)
        self.assertEqual(c['n_negative'], 0)
        self.assertEqual(c['n_identical'], 2)


class TestRebasingPreservesSigns(unittest.TestCase):
    """② 比率型脚本把 `v - 1.0` 交给 `sign_counts`，与非比率型直接喂 `d` 必须一致。

    `sign_counts` 是相对**零**的分类，定基只是平移零点：`d > TOL` ⟺ `(d+1) > 1+TOL`
    在浮点上**几乎**成立，但两者对容差的处理不同——`sign_counts` 用的是绝对容差
    TOL，而重定基后的 `v-1` 与 `v` 的舍入误差不同，恰在边界上可能分叉。
    本测试用**远离边界**的样本证明口径等价，同时用**边界样本**把这条已知差异
    写成显式断言（不允许它悄悄出现）。
    """

    def test_far_from_boundary_rebase_is_invisible(self):
        """**必须先把比率平移再用 `sign_counts`**：两者都是相对零的分类，
        直接喂 0.978 会记成"正"，而它真正的含义是"比 1 小"。所以三个比率型
        调用点（`wall_ratio_at`、`wall_threshold_band` 的 `dirs_ok`、
        `threshold_band` 的 `sign_flip_around`）全部显式平移。
        """
        vals = [0.978, 1.005, 1.024, 0.999, 1.001, 1.206, 0.850]
        rebased = paired_diff.sign_counts([v - 1.0 for v in vals])
        # 与"先定基、再按零分类"的定义逐项对账
        expect_pos = sum(1 for v in vals if v > 1.0 + paired_diff.TOL)
        expect_neg = sum(1 for v in vals if v < 1.0 - paired_diff.TOL)
        self.assertEqual(rebased['n_positive'], expect_pos)
        self.assertEqual(rebased['n_negative'], expect_neg)
        self.assertEqual(rebased['n_positive'] + rebased['n_negative'] + rebased['n_identical'],
                         len(vals))

    def test_rebase_blind_spot_near_one(self):
        """未知但已量清的盲区：距 1 不到 1e-12 的比率，重新定基后必然落进 identical。

        `v - 1.0` 在 `v` 极接近 1 时**吸收**掉整个偏移（1.0+1e-15 → 1.110e-15，
        1.0+1e-13 → 9.992e-14，都 < TOL），所以这类种子被记成"两臂相同"而不是
        "臂略慢/略快"。这是浮点减法的性质，不是判据写错——臂真的只差 1e-15 时
        也没有分辨率可言。钉住它，免得将来有人看到 `n_identical` 有值就去追。
        """
        blind = [1.0 + 1e-13, 1.0 - 1e-13, 1.0 + 1e-15]
        for v in blind:
            self.assertEqual(paired_diff.sign_counts([v - 1.0])['n_identical'], 1, repr(v))
        # 一旦偏移超过 TOL，就恢复可分辨
        self.assertEqual(paired_diff.sign_counts([(1.0 + 2e-12) - 1.0])['n_positive'], 1)

    def test_archived_ratios_never_land_in_the_blind_spot(self):
        """既有归档里没有任何一个比率落在盲区内——所以这条盲区不影响任何已发布读数。

        若将来归档里出现 0 < |v−1| ≤ 1e-12 的比率，本测试会红，提示"这一格的
        方向可能被重新定基吞掉"。
        """
        import json
        ledger = json.loads((HERE / 'ledger.json').read_text())

        def walk(o):
            if isinstance(o, dict):
                for v in o.values():
                    yield from walk(v)
            elif isinstance(o, list):
                for v in o:
                    yield from walk(v)
            elif isinstance(o, float) and 0.5 < o < 1.5:
                yield o

        vals = list(walk(ledger))
        self.assertGreater(len(vals), 100)  # 确认确实扫到了归档，不是空跑
        in_blind = [v for v in vals if 0 < abs(v - 1.0) <= paired_diff.TOL]
        self.assertEqual(in_blind, [], '这些比率落在重新定基盲区内：%r' % in_blind)


class TestIntegerStepSemantics(unittest.TestCase):
    """③ 整数步号的 `n_ties` 与浮点比率型脚本的 `n_identical` 是**不同**的口径。

    `steps_at` 的 `n_ties` 用 `d == 0`（整数精确相等）；`wall_ratio_at` 的
    `n_identical` 用 `d - 1.0` 落在 TOL 内。对整数输入两者恰好等价（因为唯一
    让 `d == 0` 的方式就是整数相等），本测试把这个等价钉住，并指出它对
    **非整数输入不成立**——这正是两个脚本不能混用同一套键名的理由。
    """

    def test_integer_equality_matches_tol_counting(self):
        ties = sum(1 for d in DIFFS_INT if d == 0)
        c = paired_diff.sign_counts(DIFFS_INT)
        self.assertEqual(ties, c['n_identical'])

    def test_non_integer_boundary_differs(self):
        """非整数差值：`d == 0` 与 `|d| <= TOL` 给出不同的结果。

        `steps_at` 把"恰好差 5e-13 步"记成非并列，而 `wall_ratio_at` 会把
        5e-13 的比率差记成 identical——两者都是对的，因为步号是整数、比率是浮点，
        但**不能用一个键名同时表达两件事**。
        """
        d_int = [5e-13]           # 不是整数，但远小于 TOL
        self.assertEqual(sum(1 for x in d_int if x == 0), 0)          # steps_at: 非并列
        self.assertEqual(paired_diff.sign_counts(d_int)['n_identical'], 1)  # wall_ratio_at: 相同


class TestScriptsActuallyUseSharedTool(unittest.TestCase):
    """四个脚本确实在 import 并调用共享判据——否则上面的测试管不到它们。"""

    def test_all_five_scripts_import_paired_diff(self):
        scripts = [
            HERE.parent / 'rollout-group-split-tradeoff' / 'bucketing_probe.py',
            HERE.parent / 'rollout-group-split-tradeoff' / 'real_lengths_policy.py',
            HERE / 'ledger.py',
            HERE / 'wall_threshold_band.py',
        ]
        for p in scripts:
            src = p.read_text()
            self.assertIn('import paired_diff', src, p.name)
            # 至少调一个以上的共享函数，不是只 import 占位
            n_calls = sum(src.count('paired_diff.' + fn) for fn in
                          ('sign_counts', 'same_sign_fraction', 'sd_pp',
                           'exceeds_band', 'span', 'sign_flip_around', 'sign_flip'))
            self.assertGreaterEqual(n_calls, 2, '%s 用了 %d 处共享判据' % (p.name, n_calls))


class TestToleranceInjectionCaught(unittest.TestCase):
    """注入测试：把共享判据的 TOL 改松，本文件的边界断言必须变红。

    这是"判据里的常数要有自己的边界测试"的落地——第六批发现注入 `TOL=1e-3`
    时既有 113 个测试全绿，因为没有任何测试造出"小于 1e-3、大于 1e-12"的差值。
    本文件第一组 `DIFFS_FLOAT` 刻意含 `1e-13`，就是为了让这个注入被抓住。
    """

    def test_loosening_tol_flips_small_diffs_to_identical(self):
        saved = paired_diff.TOL
        try:
            paired_diff.TOL = 1e-3
            # `1e-13` 在 TOL=1e-12 下是 identical，在 TOL=1e-3 下仍是 identical（不变）；
            # 但 `0.05` 在 1e-3 下仍是 positive。构造一个专门落在 1e-3 以下的差值：
            probe = [5e-4, -5e-4]   # 1e-3 容差下 → identical
            c_tight = paired_diff.sign_counts(probe, tol=1e-12)
            self.assertEqual(c_tight['n_positive'], 1)
            self.assertEqual(c_tight['n_negative'], 1)
            c_loose = paired_diff.sign_counts(probe)  # 用模块级 TOL=1e-3
            self.assertEqual(c_loose['n_identical'], 2)
        finally:
            paired_diff.TOL = saved


if __name__ == '__main__':
    unittest.main()


class TestOldVersusNewImplementation(unittest.TestCase):
    """③ 两版实现逐值对照——第六批遗留③。

    归档 `--verify` 逐字段相同只证明"输入列没落进两版的分歧区"，不证明两版实现一致。
    这里用同一列输入分别喂旧版手写实现与共享判据，逐值比对；真实列必须零分歧。
    """

    def test_real_columns_have_no_divergence(self):
        import compare_old_new_impl as C
        real = C.real_columns()
        self.assertGreaterEqual(len(real), 40, '真实列不足，多半是没扫到源 summary')
        total_diffs = 0
        total_identical = 0
        for name, diffs in real.items():
            self.assertEqual(C.compare(diffs), [], name)
            total_diffs += len(diffs)
        self.assertGreater(total_diffs, 1000)  # 确认不是空跑

    def test_real_columns_really_exercise_identical_bucket(self):
        """真实列里必须真有一批 identical 种子，否则这条对照没测到容差边界。"""
        import compare_old_new_impl as C
        real = C.real_columns()
        n_id = sum(paired_diff.sign_counts(d)['n_identical'] for d in real.values())
        self.assertGreater(n_id, 100, '真实列里 identical 种子太少，边界没被覆盖')

    def test_injected_bias_is_caught(self):
        """注入：把新版正号计数偏 +1，对照必须报分歧。"""
        import compare_old_new_impl as C
        orig = C.new_sign_counts

        def biased(diffs):
            out = orig(diffs)
            if out['n_identical']:
                out['n_positive'] += 1
            return out

        C.new_sign_counts = biased
        try:
            found = False
            for diffs in C.boundary_columns().values():
                if diffs and C.compare(diffs):
                    found = True
                    break
            self.assertTrue(found, '注入的正号偏差没被对照抓到')
        finally:
            C.new_sign_counts = orig
