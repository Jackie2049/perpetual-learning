#!/usr/bin/env python3
"""与真实 verl 的 `karmarkar_karp` 逐位核对分区（本实验的 LDM 口径复核）。

`experiments/rollout-group-split-tradeoff/group_split_tradeoff.py` 里的 `_karmarkar_karp`
是复刻件，§13.30／§13.31 已用它做过 180 组输入的交叉验证。本文件做**同一件事的复验**，
但只用 `ast` 从 `verl/utils/seqlen_balancing.py` 抽出 `karmarkar_karp` 一个函数、
在沙箱里执行（不 import torch，因为 verl 模块顶部会拉 torch），
然后把本实验真实用到的**长度分布**（`build_lengths` 的四档 × 30 种子）逐位比对。

**为什么要单独做一个**：本实验的结论取决于"LDM 把同组样本摊开到什么程度"，
而这个量完全由分区结果决定。若复刻件与真实实现有一个分区的差异，
`interleaved_over_ldm` 那一列就不可信。

用法：
    python3 xcheck_verl.py            # 默认读 ../verl；可用 VERL_ROOT 覆盖
退出码 0 = 全部一致；1 = 有差异（会打印首个反例）。
"""

import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'rollout-group-split-tradeoff'))

import minibatch_baseline_bias as M  # noqa: E402

EXPECTED_SOURCE = 'verl/utils/seqlen_balancing.py'


def load_real_karmarkar(verl_root):
    """用 ast 抽出 `karmarkar_karp` 源码并在沙箱里执行（不触 torch）。"""
    path = os.path.join(verl_root, EXPECTED_SOURCE)
    if not os.path.exists(path):
        raise SystemExit('找不到 %s（用 VERL_ROOT 指向 verl checkout）' % path)
    src = open(path, encoding='utf-8').read()
    tree = ast.parse(src)
    fn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'karmarkar_karp':
            fn = node
            break
    if fn is None:
        raise SystemExit('该 checkout 的 seqlen_balancing.py 里没有 karmarkar_karp')
    mod = ast.Module(body=[fn], type_ignores=[])
    ns = {'heapq': __import__('heapq')}
    exec(compile(ast.fix_missing_locations(mod), path, 'exec'), ns)
    return ns['karmarkar_karp'], path


def normalized(parts):
    """分区内排序、分区之间按首元素排序——只比"谁和谁在一块"，不比顺序。"""
    return sorted(tuple(sorted(p)) for p in parts)


def main():
    verl_root = os.environ.get('VERL_ROOT', os.path.join(os.path.dirname(os.path.dirname(HERE)), 'verl'))
    real_kk, path = load_real_karmarkar(verl_root)
    from group_split_tradeoff import _karmarkar_karp as replica_kk

    checked = 0
    mismatches = []
    for regime in M.LENGTH_REGIMES:
        for s in M.SEEDS:
            lengths = M.build_lengths(s, regime)
            weights = [l * M.LEN_LINEAR_COEF + l * l for l in lengths]
            a = normalized(real_kk(seqlen_list=weights, k_partitions=M.K_BLOCKS, equal_size=True))
            b = normalized(replica_kk(weights, M.K_BLOCKS, equal_size=True))
            checked += 1
            if a != b:
                mismatches.append({'regime': regime, 'seed': s,
                                   'real': a, 'replica': b})
    print('source:', path)
    print('cases checked: %d（四档长度 × %d 种子，k=%d）' % (checked, len(M.SEEDS), M.K_BLOCKS))
    if mismatches:
        print('MISMATCH %d 例，首例：' % len(mismatches))
        print(mismatches[0])
        return 1
    print('XCHECK OK: 分区逐位一致')
    return 0


if __name__ == '__main__':
    sys.exit(main())
