#!/usr/bin/env python3
"""负向对照：把本批三处"先说错、后来改对"的地方做成可执行的反证。

三个对照，每个都模拟一个"说错了会怎样"的世界，并断言**错误版本给出不同答案**：

  N1  **假设 LDM 在长度全同时是「每块 8 个完整组」**（本批初版错误）
      → 用真实 LDM 复刻件跑同一个长度分布，必须看到"每组被摊到 4 块、每块 32 个组全在"。
      若某个世界的 LDM 真的给出 8 组/块，说明复刻件或对真实实现的理解错了。

  N2  **忽略"偏差方向随 rho 翻转"**（本批第二处修正）
      → 在 rho=0 时"基准方差被压低"这句话必须**不成立**（ratio > 1）。
      若在 rho=0 也说"基准偏小"，那本批的机制解释就是编的。

  N3  **把基准偏差因子按错的分母算**（分子 32 组/块 vs 全批 32 组/块）
      → 若错误地按"分子也是 8 组/块"算，偏差因子会变成 1.0（无偏），
      1.0508 就会被读成 100% 真实组成效应。两种算法必须给出不同答案。

用法：
    python3 negative_controls.py     # 全部通过打印 OK，退出码 0
"""

import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'rollout-minibatch-composition'))

import minibatch_baseline_bias as M  # noqa: E402
import train_side_composition as T  # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    if cond:
        print('  [ok]   %s %s' % (name, detail))
    else:
        print('  [FAIL] %s %s' % (name, detail))
        FAILS.append(name)


def n1_degenerate_is_not_eight_groups_per_block():
    print('N1 真实 LDM 在「长度全同」下是否给出 8 组/块？')
    uids = [g for g in range(M.N_GROUPS) for _ in range(M.GROUP_SIZE)]
    lengths = M.build_lengths(0, 'degenerate')
    order, parts = M.ldm_order(lengths, M.K_BLOCKS)
    purity = M.block_group_purity([uids[i] for i in order], M.contiguous_shapes(len(order), M.K_BLOCKS))
    check('每块装齐全部 32 个组', purity['mean_groups_per_block'] == float(M.N_GROUPS),
          '=> %.1f' % purity['mean_groups_per_block'])
    check('没有一组完整', purity['intact_groups'] == 0, '=> %d' % purity['intact_groups'])
    check('每组跨 4 块', purity['mean_blocks_per_group'] == M.K_BLOCKS,
          '=> %d' % purity['mean_blocks_per_group'])
    # 而 grouped（组内连续）才是 8 组/块
    gp = M.block_group_purity(uids, M.contiguous_shapes(len(uids), M.K_BLOCKS))
    check('对照组：grouped 是 8 组/块', gp['mean_groups_per_block'] == 8.0,
          '=> %.1f' % gp['mean_groups_per_block'])
    check('两者不同', purity['mean_groups_per_block'] != gp['mean_groups_per_block'])


def n2_crossover_exists():
    print('N2 偏差方向是否随 rho 翻转？')
    b = M._expected_sample_var(8, 8, 0.0) / M._expected_sample_var(8, 32, 0.0)
    check('rho=0 时"块内 8 组"反而更大', b > 1.0, '=> ratio=%.5f' % b)
    b2 = M._expected_sample_var(8, 8, 1.0) / M._expected_sample_var(8, 32, 1.0)
    check('rho=1 时"块内 8 组"更小', b2 < 1.0, '=> ratio=%.5f' % b2)
    check('默认 rho=0.5 落在偏小一侧',
          M.grouped_bias_theory()['expected_ratio_block_over_full'] < 1.0,
          '=> ratio=%.5f' % M.grouped_bias_theory()['expected_ratio_block_over_full'])


def n3_wrong_denominator_gives_different_answer():
    print('N3 按错的分母算偏差因子会不会得到不同结论？')
    right = (M._expected_sample_var(M.GROUP_SIZE, M.N_GROUPS, M.RHO_GROUP)
             / M._expected_sample_var(M.GROUP_SIZE, M.N_GROUPS // M.K_BLOCKS, M.RHO_GROUP))
    # 错误版：以为分子也只装 8 个组 ⇒ 因子变成 1
    wrong = (M._expected_sample_var(M.GROUP_SIZE, M.N_GROUPS // M.K_BLOCKS, M.RHO_GROUP)
             / M._expected_sample_var(M.GROUP_SIZE, M.N_GROUPS // M.K_BLOCKS, M.RHO_GROUP))
    check('正确因子 > 1', right > 1.0, '=> %.5f' % right)
    check('错误因子 == 1', abs(wrong - 1.0) < 1e-12, '=> %.5f' % wrong)
    # 精确值：正确因子 = E[32]/E[8] = 0.925490/0.888889 = 1.041176…
    check('正确因子等于解析常数', abs(right - (0.9254901960784314 / 0.8888888888888888)) < 1e-12,
          '=> %.6f' % right)
    # (1.041176 − 1)/1.041176 = 3.9548% —— 这是"错误版会把偏差整个抹掉"的量级
    check('两者相差约 3.95%', abs((right - wrong) / right - 0.039548022598870136) < 1e-9,
          '=> %.4f%%' % ((right - wrong) / right * 100))


def n4_identifiability_rejects_constant():
    print('N4 识别性检验能否打死恒常数指标？')
    ident = M.identifiability(M._metric_constant, M.RHO_LEVELS, 'rho', is_constant=True)
    check('恒常数判不敏感', ident['sensitive'] is False)
    ident2 = M.identifiability(M._metric_gap_over_grouped, M.RHO_LEVELS, 'rho')
    check('有信号的指标判敏感', ident2['sensitive'] is True,
          '=> spread=%.4f' % (ident2['relative_spread'] or 0))


def main():
    print('负向对照（本批三处自查修正的反证）\n')
    n1_degenerate_is_not_eight_groups_per_block()
    print()
    n2_crossover_exists()
    print()
    n3_wrong_denominator_gives_different_answer()
    print()
    n4_identifiability_rejects_constant()
    print()
    if FAILS:
        print('FAILED: %d 项' % len(FAILS))
        return 1
    print('ALL PASSED: 4 组负向对照全部成立')
    return 0


if __name__ == '__main__':
    sys.exit(main())
