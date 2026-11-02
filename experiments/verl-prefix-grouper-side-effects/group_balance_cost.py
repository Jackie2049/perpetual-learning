#!/usr/bin/env python3
"""§21 之外、掘金线：verl `use_prefix_grouper` 的**训练侧副作用**值多少

支撑 [open-source/verl/evidence/prefix-grouper-inert](../../open-source/verl/evidence/prefix-grouper-inert/readme.md)。

上游事实（main `6879e1ab07`，已由取证脚本确认）：
  · 引擎侧从不传 `use_prefix_grouper` ⇒ **前缀共享前向不生效**；
  · 但 `ray_trainer._balance_batch` 仍按该开关改行为，且**它是生效的**。

`_balance_batch` 在开关为真时做两件事：
  (a) 用 `get_group_balanced_partitions` 做**组级**均衡（同 uid 必须在同一 rank）；
  (b) **跳过** rank 内重排（"Place smaller micro-batches at both ends"）。

本实验只回答一件事：**这两件事加起来，相对"逐条均衡 + 重排"是赚还是亏？**

关键前提（来自本仓库既有结论，不是本实验的假设）：
  · §13.39：跳过那段重排在 `max` 代价模型下是**负优化**（跳过反而不亏）；
  · §13.32：优势归一化按 uid 分组，**与分区方式无关** ⇒ 组完整不改变统计口径。

因此本实验的全部变量集中在 (a)：**组完整把 DP 负载均衡弄差了多少**。

方法：直接抽取上游 `seqlen_balancing.py` 的三个函数源码（AST 取源码，
注入轻量 torch 替身）运行，**跑的是上游代码本身**，不是重写版。

用法：
    python3 group_balance_cost.py
    python3 group_balance_cost.py --verify
    python3 -m unittest test_group_balance_cost -v
"""

import ast
import builtins
import copy
import heapq
import itertools
import json
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent
SUMMARY = HERE / 'summary.json'
UPSTREAM = HERE / 'upstream-seqlen_balancing.py'
SEEDS = list(range(30))

builtins_zip = builtins.zip

WANT = ['calculate_workload', 'karmarkar_karp',
        'get_seqlen_balanced_partitions', 'get_group_balanced_partitions']


# ---------------------------------------------------------------- 上游源码注入


def load_upstream_funcs(path=UPSTREAM):
    """从上游文件里原样抽出函数源码并执行，返回可调用对象。

    上游模块 import 了 torch / verl.protocol / tensordict，本实验只需要分区算法；
    这里给 `calculate_workload` 一个纯 Python 的 `torch` 替身
    （它只调 torch.tensor([x]) 与 .item()，语义上是恒等映射）。
    """
    src = pathlib.Path(path).read_text(encoding='utf-8')
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    chunks = []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in WANT:
            chunks.append(''.join(lines[n.lineno - 1:n.end_lineno]))
    if len(chunks) != len(WANT):
        raise RuntimeError('上游函数抽取失败：找到 %d / %d' % (len(chunks), len(WANT)))

    class _T:
        """torch.tensor 的最小替身：只支持本实验用到的逐元素算术。

        上游 `calculate_workload` 计算 24576*L + L**2（FLOPs 近似），
        因此替身必须支持 + * ** 与 .item()。
        """
        def __init__(self, v):
            self.v = list(v)

        def __add__(self, o):
            return _T([a + (o.v[i] if isinstance(o, _T) else o) for i, a in enumerate(self.v)])

        __radd__ = __add__

        def __mul__(self, o):
            return _T([a * (o.v[i] if isinstance(o, _T) else o) for i, a in enumerate(self.v)])

        __rmul__ = __mul__

        def __pow__(self, p):
            return _T([a ** p for a in self.v])

        def item(self):
            return self.v[0]

    class _Torch:
        Tensor = _T

        @staticmethod
        def tensor(v):
            # 上游既传 list 也传张量；本实验只取标量，统一包成 _T
            if isinstance(v, _T):
                return v
            return _T(v if isinstance(v, (list, tuple)) else [v])

    def _zip(*a, strict=False):
        # 上游用了 3.10+ 的 zip(strict=)，本机 3.9 无此参数；
        # 长度不等时按 strict=True 的语义抛错，保持行为一致。
        if strict:
            n = {len(x) for x in a}
            if len(n) > 1:
                raise ValueError('zip() arguments have different lengths')
        return builtins_zip(*a)

    ns = {'torch': _Torch(), 'copy': copy, 'heapq': heapq,
          'chain': itertools.chain, 'zip': _zip}
    for c in chunks:
        exec(compile(c, str(path), 'exec'), ns)   # noqa: S102 - 只跑上游纯算法源码
    out = {k: ns[k] for k in WANT}
    out['_tensor'] = _Torch.tensor
    return out


# ---------------------------------------------------------------- 负载生成


def gen_lengths(seed, n_prompts, n_samples, mean=400.0, cv=0.8, group_rho=0.5):
    """生成一个训练批的样本长度。

    group_rho 控制**同 prompt 各条回答长度相关性的强度**——真实模型里同 prompt
    的 n 条回答共享难度，长度既不独立也不完全相同。用共享分量实现：
        len = exp(mu + rho_share + (1-rho_share) * indiv)
    """
    rng = random.Random(seed)
    out = []
    for _ in range(n_prompts):
        shared = rng.gauss(0.0, 1.0)
        for _ in range(n_samples):
            indiv = rng.gauss(0.0, 1.0)
            z = (group_rho ** 0.5) * shared + ((1.0 - group_rho) ** 0.5) * indiv
            out.append(max(32, int(mean * (2.0 ** z))))
    return out


def uid_list_for(n_prompts, n_samples):
    return [('uid%d' % i) for i in range(n_prompts) for _ in range(n_samples)]


# ---------------------------------------------------------------- 不均衡度量


def imbalance(partitions, workloads):
    """最重分区 ÷ 平均分区负载。1.0 = 完美均衡。"""
    sums = [sum(workloads[i] for i in p) for p in partitions]
    mean = sum(sums) / float(len(sums))
    return max(sums) / mean if mean else float('nan')


def spread(partitions, workloads):
    sums = [sum(workloads[i] for i in p) for p in partitions]
    mean = sum(sums) / float(len(sums))
    return (max(sums) - min(sums)) / mean if mean else float('nan')


# ---------------------------------------------------------------- 场景


def scenario_group_cost():
    """(a) 组级均衡 vs 逐条均衡：不均衡差多少（同长度分布、同 dp_size）。"""
    U = load_upstream_funcs()
    out = {}
    for dp in (4, 8):
        for n_samples in (4, 5):
            tag = 'dp%d_n%d' % (dp, n_samples)
            rows = []
            for seed in SEEDS:
                # 条数须能被 dp 整除（equal_size 要求），组数亦然
                n_prompts = (2048 // n_samples) // dp * dp
                lens = gen_lengths(seed, n_prompts, n_samples)
                w = [int(U['calculate_workload'](U['_tensor']([L])).item()) for L in lens]
                uids = uid_list_for(n_prompts, n_samples)

                flat = U['get_seqlen_balanced_partitions'](w, k_partitions=dp, equal_size=True)
                grp = U['get_group_balanced_partitions'](w, uids, k_partitions=dp)
                rows.append({'imbalance_flat': imbalance(flat, w),
                             'imbalance_group': imbalance(grp, w),
                             'spread_flat': spread(flat, w),
                             'spread_group': spread(grp, w),
                             'n_items': len(w)})
            out[tag] = summarize(rows, ['imbalance_flat', 'imbalance_group',
                                        'spread_flat', 'spread_group', 'n_items'])
            out[tag]['imbalance_ratio'] = (out[tag]['imbalance_group']['mean'] /
                                           out[tag]['imbalance_flat']['mean'])
    return out


def scenario_group_rho():
    """组内长度相关性（rho）如何影响组级均衡的代价。

    rho 越大，同组内各条长度越接近 ⇒ 组是不可分的"齐整块" 或 "齐整大块"，
    组级均衡对逐条均衡的劣势应当先增后减；rho=1 时组内完全同长。
    """
    U = load_upstream_funcs()
    out = {}
    for rho in (0.0, 0.25, 0.5, 0.75, 1.0):
        rows = []
        for seed in SEEDS:
            n_samples = 5
            n_prompts = (2048 // n_samples) // 8 * 8
            lens = gen_lengths(seed, n_prompts, n_samples, group_rho=rho)
            w = [int(U['calculate_workload'](U['_tensor']([L])).item()) for L in lens]
            uids = uid_list_for(n_prompts, n_samples)
            flat = U['get_seqlen_balanced_partitions'](w, k_partitions=8, equal_size=True)
            grp = U['get_group_balanced_partitions'](w, uids, k_partitions=8)
            rows.append({'imbalance_flat': imbalance(flat, w),
                         'imbalance_group': imbalance(grp, w)})
        blk = summarize(rows, ['imbalance_flat', 'imbalance_group'])
        blk['imbalance_ratio'] = blk['imbalance_group']['mean'] / blk['imbalance_flat']['mean']
        out['rho%.2f' % rho] = blk
    return out


def scenario_batch_size():
    """批越大，组级均衡的粒度劣势应当越小（每组固定 n 条，组数随批线性增长）。"""
    U = load_upstream_funcs()
    out = {}
    for batch in (256, 1024, 4096, 16384):
        rows = []
        for seed in SEEDS:
            n_samples = 5
            n_prompts = (batch // n_samples) // 8 * 8
            lens = gen_lengths(seed, n_prompts, n_samples)
            w = [int(U['calculate_workload'](U['_tensor']([L])).item()) for L in lens]
            uids = uid_list_for(n_prompts, n_samples)
            flat = U['get_seqlen_balanced_partitions'](lens, k_partitions=8, equal_size=True)
            grp = U['get_group_balanced_partitions'](lens, uids, k_partitions=8)
            rows.append({'imbalance_flat': imbalance(flat, w),
                         'imbalance_group': imbalance(grp, w)})
        blk = summarize(rows, ['imbalance_flat', 'imbalance_group'])
        blk['imbalance_ratio'] = blk['imbalance_group']['mean'] / blk['imbalance_flat']['mean']
        out['batch%d' % batch] = blk
    return out


def scenario_upstream_identity():
    """把上游源码抽出来跑，必须与上游行为一致：equal_size 时各分区等长、且不重不漏。"""
    U = load_upstream_funcs()
    lens = gen_lengths(0, 200, 5)
    w = [int(U['calculate_workload'](U['_tensor']([L])).item()) for L in lens]
    uids = uid_list_for(200, 5)
    flat = U['get_seqlen_balanced_partitions'](w, k_partitions=8, equal_size=True)
    grp = U['get_group_balanced_partitions'](w, uids, k_partitions=8)
    out = {}
    for name, parts in (('flat', flat), ('group', grp)):
        flat_idx = [i for p in parts for i in p]
        out[name] = {'k': len(parts),
                     'sizes': sorted(len(p) for p in parts),
                     'covers_all': sorted(flat_idx) == list(range(len(lens))),
                     'no_dup': len(set(flat_idx)) == len(flat_idx)}
    # 组完整：同 uid 是否真的落在同一分区
    uid_of = {}
    for p_idx, part in enumerate(grp):
        for i in part:
            uid_of.setdefault(uids[i], set()).add(p_idx)
    out['group_keeps_uid_together'] = all(len(v) == 1 for v in uid_of.values())
    out['n_uids'] = len(uid_of)
    return out


def summarize(items, keys):
    res = {}
    for k in keys:
        vals = [it[k] for it in items if it[k] == it[k]]
        if not vals:
            res[k] = {'mean': float('nan'), 'std': float('nan'),
                      'min': float('nan'), 'max': float('nan'), 'n': 0}
            continue
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1)
        res[k] = {'mean': m, 'std': var ** 0.5, 'min': min(vals),
                  'max': max(vals), 'n': len(vals)}
    return res


def run_all():
    return {
        'config': {'seeds': len(SEEDS)},
        'upstream_identity': scenario_upstream_identity(),
        'group_cost': scenario_group_cost(),
        'group_rho': scenario_group_rho(),
        'batch_size': scenario_batch_size(),
    }


def _canon(x):
    if isinstance(x, dict):
        return {k: _canon(v) for k, v in sorted(x.items())}
    if isinstance(x, list):
        return [_canon(v) for v in x]
    if isinstance(x, float):
        return round(x, 9)
    return x


def _strip_nan(x):
    if isinstance(x, dict):
        return {k: _strip_nan(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_strip_nan(v) for v in x]
    if isinstance(x, float) and x != x:
        return 'NaN'
    return x


def verify():
    if not SUMMARY.exists():
        print('summary.json 不存在，先运行一次全量')
        return 1
    archived = json.loads(SUMMARY.read_text())
    fresh = _strip_nan(_canon(run_all()))
    if _strip_nan(_canon(archived)) != fresh:
        print('差异字段：', [k for k in fresh if _strip_nan(_canon(archived.get(k))) != fresh[k]])
        return 1
    print('--verify 通过：与归档 summary.json 逐字段一致')
    return 0


def main():
    if '--verify' in sys.argv:
        return verify()
    SUMMARY.write_text(json.dumps(_canon(run_all()), ensure_ascii=False,
                                  indent=1, sort_keys=True))
    print('写出', SUMMARY)
    return 0


if __name__ == '__main__':
    sys.exit(main())
