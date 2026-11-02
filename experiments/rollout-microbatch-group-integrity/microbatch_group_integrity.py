#!/usr/bin/env python3
"""micro-batch 层的组完整性代价：可接线的那一条（rollout 专项，2026-11-02 M1）。

支撑 notebook/rollout-performance-research.md §13.35；接 §13.34 登记的两条残留——
「micro-batch 层 81.3% 的组被切开」是否有解，「被跳过的 PP 气泡」能不能补上。

**本脚本只做一件事**：把 §13.34 的代价归因拆干净。§13.34 报了「micro-batch 层每组被切成
3.62 段、前缀块开销是整组共享的 3.62 倍」，读起来像是「micro-batch 层有个 bug 可以修」。
核完源码路径后，原因是三段叠加的，只有第一段是可接线的：

  A. 代码路径（本批新核）  verl 1a9e711e 里能让 micro-batch 看组的开关 `force_group_size`
                           没有任何 TRAIN 调用方写入；`rearrange_micro_batches` 用它的第三个参数
                           默认值 =1。因此 §13.34 的条件句「若 force_group_size > 1」在本快照
                           的**所有**训练路径上都不可达。（残留：ray trainer 不在本 checkout 的
                           `enable` 流程里——`engine_workers.update_actor` 是 `@register` 方法，
                           由上游调用点触发，只读核到此为止，不作结构保证。）
  B. mini-batch 结构         §13.34 混用了两个参照系（见「模型修正 #1」）：组完整最好的可达点是
                           mini-batch 层（割点 0%），不是整批（§13.34 报的 1.00×）。
  C. vPP 整数倍约束          megatron 路径强制 `micro-batch 数 % num_batches_divided_by == 0`，
                           组完整会让可分的批数骤减（min(num_groups, ceildiv(...)) 被下限顶住），
                           这个代价与 A 无关、与代码是否有 bug 无关。

FLOP 口径与 §13.34 一致（前缀 O(L²) + 跨块 O(p·s) + 前缀线性项），只算前缀块；
**不折算墙钟、不与任何报值相除**（§19.1 规则一）。前缀长度无关的恒等式 `1/n` 在本口径成立
（见 counterfactuals），所以「前缀越长收益越大」同样不能从本实验推出。

用法：
    python3 microbatch_group_integrity.py            # 全量重跑并覆盖 summary.json
    python3 microbatch_group_integrity.py --verify   # 重跑并与归档 summary.json 逐字段比对
    python3 -m unittest test_microbatch_group_integrity -v
"""

import json
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY_PATH = os.path.join(HERE, 'summary.json')
sys.path.insert(0, os.path.join(HERE, '..', 'rollout-pg-boundaries'))

# LDM / 切分复刻直接复用 §13.34 的实现与同一个真实 verl 交叉验证过的版本，避免第二份漂移。
from pg_boundaries import (  # noqa: E402
    WORKLOAD_LINEAR, WORKLOAD_QUAD, _karmarkar_karp, balance_batch_default, balance_batch_pg,
    batch_layout, contig_chunks, dispatch_views, prefix_rounds,
)

SEEDS = list(range(30))

N_GROUPS = 32
N_PER = 8                 # rollout.n
SEQ_MEAN, SEQ_CV = 2200.0, 0.45
RESPONSE_LEN = 1024
PREFIX_LEN = 512          # 只影响绝对 FLOP；所有比值与它无关（恒等式守住）
DEFAULT_BUDGET = 16384    # ppo_max_token_len_per_gpu 默认
DP_SIZES = (1, 2, 4)
MINI_GROUP_CHOICES = (1, 2, 4, 8)
VPP_SIZES = (2, 3, 4)


# ---------------------------------------------------------------------------
# 与 verl 快照对齐的 micro-batch 切分（seqlen_balancing.py:348-466）
# ---------------------------------------------------------------------------
def roundup_divisible(x, n):
    """seqlen_balancing.roundup_divisible：向上取到 n 的整数倍。"""
    return ((x + n - 1) // n) * n


def micro_split(seg_lens, budget, force_group_size=1, num_batches_divided_by=None,
                use_dynamic_bsz_balance=True):
    """rearrange_micro_batches 的完整形状（比 §13.34 的版本多两件事，都是本批要量的一侧）：

      * `force_group_size > 1` 时（441-452 行）：先按长度把**组**聚成一个权重做 LDM，
        再把组下标展开回样本下标；`use_dynamic_bsz_balance` 的排序键也改用组工作量（444-448）。
      * `num_batches_divided_by` 时（412 行）：`num_micro_batches = roundup_divisible(n, vpp)`，
        并且 414 行的断言 `num_micro_batches <= num_groups` 会对组数硬上限——
        组完整时上限是「组数」，逐条时上限是「样本数」。

    返回 (parts, info)。parts 是按真实顺序排好的分区（每组为样本下标列表）。
    """
    lens = [int(l) for l in seg_lens]
    b = len(lens)
    assert force_group_size >= 1
    # 395-401 行：原实现按 num_groups 切；本批整条 mini-batch 都由整组构成时两者等价，
    # 不等时才是「组被前置切分」的情形，这里直接拒绝建模（见 readme「边界」）。
    assert b % force_group_size == 0, 'batch size must be divisible by force_group_size'
    num_groups = b // force_group_size
    total = sum(lens)
    n_micro = min(num_groups, -(-total // int(budget)))
    if num_batches_divided_by:
        n_micro = roundup_divisible(n_micro, num_batches_divided_by)
    info = {'n_micro_raw': min(num_groups, -(-total // int(budget))),
            'n_micro_before_cap': n_micro,
            'cap_violated': n_micro > num_groups,
            'infeasible': n_micro > num_groups}
    if n_micro > num_groups:
        # 412 行 roundup 之后 414 行 assert n_micro <= num_groups——组完整时这个上限从「样本数」
        # 掉到「组数」，被 roundup 顶破后直接 AssertionError。不 rescale，原样报出（见 verdict）。
        info['infeasible'] = True
        return None, info
    if n_micro <= 1:
        return [list(range(b))], info

    w = [WORKLOAD_LINEAR * l + WORKLOAD_QUAD * l * l for l in lens]
    if force_group_size > 1:
        group_w = [sum(w[g * force_group_size:(g + 1) * force_group_size]) for g in range(num_groups)]
        gparts = _karmarkar_karp(group_w, n_micro, equal_size=False)
        parts = []
        for gp in sorted(gparts):
            idx = []
            for g in gp:
                idx.extend(range(g * force_group_size, (g + 1) * force_group_size))
            parts.append(idx)
        sort_w = group_w
    else:
        parts = [sorted(p) for p in _karmarkar_karp(w, n_micro, equal_size=False)]
        sort_w = w
    if use_dynamic_bsz_balance:
        parts.sort(key=lambda p: (sum(sort_w[i // force_group_size] for i in p[::force_group_size]),
                                  p[0] if p else 0), reverse=True)
        parts = parts[::2][::-1] + parts[1::2]
    return parts, info


# ---------------------------------------------------------------------------
# 三个参照点（**必须用同一套 mini-batch 参照系**）
# ---------------------------------------------------------------------------
def group_fragments(reordered, views, mini_groups, budget, per_sample, n_per=N_PER):
    """在一个 rank 的每个 mini-batch 内部切 micro-batch，统计每组的片段数。

    per_sample=True  时 force_group_size=1（真实默认）
    per_sample=False 时 force_group_size=n（组完整；等价于在 mini-batch 里对组做 LDM）
    """
    per_group = {}
    for view in views:
        for mb in contig_chunks(len(view), mini_groups * n_per):
            seg = [reordered[view[i]] for i in mb]
            parts, _ = micro_split([x[2] for x in seg], budget,
                                   force_group_size=1 if per_sample else n_per)
            for pid, part in enumerate(parts):
                for i in part:
                    per_group.setdefault(seg[i][0], set()).add(pid)
    return per_group


def prefix_cost(per_group, n_per=N_PER, prefix_len=PREFIX_LEN, suffix_len=RESPONSE_LEN):
    """前缀块在这一次遍历里算了几遍（每组的片段数之和）。"""
    return sum(prefix_rounds(prefix_len, [suffix_len] * n_per, len(pids)) for pids in per_group.values())


def case(n_per, dp, mini_groups, seed, budget=DEFAULT_BUDGET, with_whole_batch=True):
    """一个 (n, dp, mini_groups, seed) 案例的三个参照点与其比值。

    ideal_minibatch  组完整、micro-batch 层对组做 LDM（改一行参数即可达）；**可达上限**
    current          现状：force_group_size=1
    ideal_wholebatch  §13.34 的 baseline——每个组只算一次前缀，**要求一个 micro-batch
                      装下整批**，与 mini-batch 迭代互斥，故不可达；只作外参照

    注意两组比值用的分母不同、不可互相换算：
      ideal_over_current        = 组完整可达上限 ÷ 现状（两个都按 mini-batch 切）
      ideal_minibatch_over_wholebatch = 可达上限 ÷ 外参照（这个**不是**差距）
    """
    rng = random.Random(4242 + seed * 1000 + n_per * 100 + dp * 10 + mini_groups)
    gl = [[float(max(1, int(round(rng.gauss(SEQ_MEAN, SEQ_MEAN * SEQ_CV))))) for _ in range(n_per)]
          for _ in range(N_GROUPS)]
    reordered, _ = balance_batch_pg(batch_layout(gl), dp)
    views = dispatch_views(reordered, dp)
    mini_groups = min(mini_groups, len(views[0]) // n_per)
    if mini_groups <= 0:
        return None
    cur = group_fragments(reordered, views, mini_groups, budget, per_sample=True, n_per=n_per)
    idm = group_fragments(reordered, views, mini_groups, budget, per_sample=False, n_per=n_per)
    out = {
        'n_cases': len(cur),
        'fragments_per_group_current': statistics.mean(len(v) for v in cur.values()),
        'fragments_per_group_ideal_minibatch': statistics.mean(len(v) for v in idm.values()),
        'frags_current': sum(len(v) for v in cur.values()),
        'frags_ideal_minibatch': sum(len(v) for v in idm.values()),
    }
    out['ideal_over_current'] = out['frags_ideal_minibatch'] / out['frags_current']
    if with_whole_batch:
        units = len({it[0] for it in reordered})
        whole = sum(prefix_rounds(PREFIX_LEN, [RESPONSE_LEN] * n_per, 1) for _ in range(units))
        out['ideal_wholebatch_flop'] = whole
        out['current_over_wholebatch'] = prefix_cost(cur) / whole
        out['ideal_minibatch_over_wholebatch'] = prefix_cost(idm) / whole
    return out


def layer0_structure():
    """可接线性：floor（§13.34 报的 3.62×）里有多少属于「mini-batch 参照系选错」。"""
    rows = []
    for n_per in (4, 8):
        for dp in DP_SIZES:
            for mg in MINI_GROUP_CHOICES:
                for budget, tag in ((DEFAULT_BUDGET, 'default_16384'), (10 ** 9, 'loose')):
                    for seed in SEEDS[:10]:
                        r = case(n_per, dp, mg, seed, budget=budget)
                        if r:
                            rows.append(dict(r, n_per=n_per, dp=dp, mini_groups=mg, budget=tag))
    by = {}
    for key in ('default_16384', 'loose'):
        sel = [r for r in rows if r['budget'] == key]
        by[key] = {
            'current_over_wholebatch': statistics.mean(r['current_over_wholebatch'] for r in sel),
            'ideal_minibatch_over_wholebatch': statistics.mean(r['ideal_minibatch_over_wholebatch'] for r in sel),
            'ideal_over_current': statistics.mean(r['ideal_over_current'] for r in sel),
            'fragments_per_group_current': statistics.mean(r['fragments_per_group_current'] for r in sel),
            'fragments_per_group_ideal_minibatch': statistics.mean(r['fragments_per_group_ideal_minibatch'] for r in sel),
            'n_cases': len(sel),
        }
    return {'by_budget': by, 'rows': rows,
            'reading': ('两个比值用同一个「整批共享」分母：current_over_wholebatch 是 §13.34 报的 3.62×，'
                        'ideal_minibatch_over_wholebatch 是「组完整」在 mini-batch 参照系下的可达上限；'
                        'ideal_over_current 是两者之比，也就是「接线」那一行能拿回多少。'
                        '**「整批共享 1.00×」不是可达点**——它要求一个 mini-batch 装下整批，'
                        '而训练是按 mini-batch 迭代的（ray_trainer._update_actor 的 ppo_mini_batch_size）。')}


def layer1_ideal_minibatch():
    """可接线收益：把 force_group_size 接成 n 之后，micro-batch 层还剩多少不均衡。"""
    rows = []
    for n_per in (4, 8):
        for dp in DP_SIZES:
            for mg in MINI_GROUP_CHOICES:
                for seed in SEEDS:
                    r = case(n_per, dp, mg, seed)
                    if r:
                        rows.append(dict(r, n_per=n_per, dp=dp, mini_groups=mg))
    return {
        'n_cases': len(rows),
        'fragments_per_group_current': statistics.mean(r['fragments_per_group_current'] for r in rows),
        'fragments_per_group_ideal_minibatch': statistics.mean(r['fragments_per_group_ideal_minibatch'] for r in rows),
        'ideal_over_current_mean': statistics.mean(r['ideal_over_current'] for r in rows),
        'ideal_over_current_min': min(r['ideal_over_current'] for r in rows),
        'ideal_over_current_max': max(r['ideal_over_current'] for r in rows),
        'ideal_minibatch_over_wholebatch': statistics.mean(r['ideal_minibatch_over_wholebatch'] for r in rows),
    }


# ---------------------------------------------------------------------------
# vPP 整数倍约束：组完整给不出的「批数粒度」
# ---------------------------------------------------------------------------
def vpp_effect(dp=1, n_per=N_PER, mini_groups=8, seed=0, budget=DEFAULT_BUDGET, vpp_list=VPP_SIZES):
    """同一个 mini-batch，组完整与逐条切两种做法可分的 micro-batch 数。

    逐条切的 n_micro = min(B, ceildiv(total, budget))，真实代码不要求整除 vPP；
    组完整时 n_micro = min(B/n, ceildiv(total, budget))，随后被 roundup 到 vPP 的倍数——
    组完整把上限从 B 压到 B/n。若 roundup 顶破这个上限，真实代码在 414 行直接断言失败，
    本函数把它作为 `grouped_infeasible` 报出，不替它 rescale。
    """
    rng = random.Random(4242 + seed * 1000 + n_per * 100 + dp * 10 + mini_groups)
    gl = [[float(max(1, int(round(rng.gauss(SEQ_MEAN, SEQ_MEAN * SEQ_CV))))) for _ in range(n_per)]
          for _ in range(N_GROUPS)]
    reordered, _ = balance_batch_pg(batch_layout(gl), dp)
    views = dispatch_views(reordered, dp)
    mini_groups = min(mini_groups, len(views[0]) // n_per)
    mb = [reordered[views[0][i]] for i in range(mini_groups * n_per)]
    lens = [x[2] for x in mb]
    total = sum(lens)
    per_sample_ceil = min(len(lens), -(-int(total) // int(budget)))
    grouped_ceil = min(len(lens) // n_per, -(-int(total) // int(budget)))
    rows = []
    for vpp in vpp_list:
        ps_parts, ps_info = micro_split(lens, budget, force_group_size=1, num_batches_divided_by=vpp)
        gp_parts, gp_info = micro_split(lens, budget, force_group_size=n_per, num_batches_divided_by=vpp)

        def imbalance(parts):
            loads = [sum(lens[i] for i in p) for p in parts]
            mean = sum(loads) / len(loads)
            return (max(loads) - mean) / mean

        def group_integrity_ok(parts):
            where = {}
            for pid, p in enumerate(parts):
                for i in p:
                    where.setdefault(i // n_per, set()).add(pid)
            return all(len(v) == 1 for v in where.values())

        rows.append({
            'vpp': vpp,
            'per_sample_n_micro': len(ps_parts),
            'grouped_n_micro': None if gp_parts is None else len(gp_parts),
            'grouped_infeasible': gp_parts is None,
            'per_sample_ceil': per_sample_ceil, 'grouped_ceil': grouped_ceil,
            'load_imbalance_per_sample': imbalance(ps_parts),
            'load_imbalance_grouped': None if gp_parts is None else imbalance(gp_parts),
            'group_integrity_per_sample': group_integrity_ok(ps_parts),
            'group_integrity_grouped': None if gp_parts is None else group_integrity_ok(gp_parts),
        })
    return {'mini_batch_tokens': round(total), 'n_samples': len(lens), 'rows': rows}


def layer2_vpp():
    out = {}
    for vpp in VPP_SIZES:
        rows = []
        for mg in (2, 4, 8):
            for budget in (4096, 8192, 16384):
                rows.append(vpp_effect(mini_groups=mg, budget=budget))
        sel = [r_ for r in rows for r_ in r['rows'] if r_['vpp'] == vpp]
        ok = [r_ for r_ in sel if not r_['grouped_infeasible']]
        agg = {
            'n_cases': len(sel),
            'n_grouped_infeasible': len(sel) - len(ok),
            'mean_n_micro_ratio_grouped_over_per_sample':
                statistics.mean(r_['grouped_n_micro'] / r_['per_sample_n_micro'] for r_ in ok) if ok else None,
            'mean_imbalance_per_sample':
                statistics.mean(r_['load_imbalance_per_sample'] for r_ in ok) if ok else None,
            'mean_imbalance_grouped':
                statistics.mean(r_['load_imbalance_grouped'] for r_ in ok) if ok else None,
            'grouped_never_splits_groups': all(r_['group_integrity_grouped'] for r_ in ok),
        }
        out[str(vpp)] = {'summary': agg, 'details': rows}
    return out


def layer2_budget_sweep(vpp=2, mini_groups=8):
    """预算从「刚好装下整批」往下扫：逐条切能分几批、组完整能分几批。"""
    rows = []
    base = vpp_effect(mini_groups=mini_groups, budget=10 ** 9)
    full = base['mini_batch_tokens']
    for frac in (1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05):
        r = vpp_effect(mini_groups=mini_groups, budget=max(1, int(full * frac)))
        row = next(x for x in r['rows'] if x['vpp'] == vpp)
        rows.append(dict(row, budget_fraction=frac, budget_tokens=int(full * frac)))
    return {'full_mini_batch_tokens': full, 'rows': rows}


# ---------------------------------------------------------------------------
# 识别性检验与负向对照
# ---------------------------------------------------------------------------
def identifiability():
    """先验后量：这几个量在「该为零 / 该恒等」的条件下是否真的归零 / 恒等。"""
    # 1) n=1 时「组完整」与「逐条」是同一个切法，比值必须恒为 1
    r1 = case(1, 1, 8, seed=3)
    # 2) 预算装得下整批时，组完整与逐条都只有一个 micro-batch 装着这个 mini-batch，
    #    但逐条切也可能因为预算不足而多切——这里把预算放到极大，两者都必须 1 段/组
    r2 = case(8, 1, 8, seed=3, budget=10 ** 9)
    # 3) 组完整后每个组的片段数不可能多于组数
    r3 = case(8, 1, 4, seed=3)
    return {
        'n1_ideal_over_current': r1['ideal_over_current'],
        'loose_budget_fragments_current': r2['fragments_per_group_current'],
        'loose_budget_fragments_ideal': r2['fragments_per_group_ideal_minibatch'],
        'ideal_fragments_per_group_never_exceeds_groups_per_minibatch':
            r3['fragments_per_group_ideal_minibatch'] <= 4,
        'vpp2_grouped_always_keeps_groups': all(
            d['group_integrity_grouped'] for v in ('2', '4') for blk in layer2_vpp()[v]['details']
            for d in blk['rows'] if not d['grouped_infeasible']),
        'reading': ('n=1 时两者切法等价，比值必须是 1；预算装得下时两者都只能切 1 段；'
                    '组完整下每组片段数不超过一个 mini-batch 的组数；'
                    'vPP=2 下组完整路径必须从不拆组（否则是模型错，不是结论）。'),
    }


def counterfactuals():
    """负向对照：三条在正确实现下必须成立的恒等式 / 单调性。"""
    rows = []
    for seed in SEEDS[:5]:
        rows.append(case(8, 1, 4, seed))
    # 1) 预算越小，现状的碎片数不减
    # 2) 组完整时碎片数不随 dp 变化（每个 rank 上是同样的组级 LDM）
    # 3) 可达上限在每个案例上都不劣于现状
    tight = case(8, 1, 4, seed=1, budget=8192)
    loose = case(8, 1, 4, seed=1, budget=10 ** 9)
    d1 = case(8, 1, 4, seed=1)['fragments_per_group_ideal_minibatch']
    d2 = case(8, 2, 4, seed=1)['fragments_per_group_ideal_minibatch']
    d4 = case(8, 4, 4, seed=1)['fragments_per_group_ideal_minibatch']
    return {
        'current_fragments_shrink_with_bigger_budget':
            tight['fragments_per_group_current'] >= loose['fragments_per_group_current'],
        'ideal_fragments_same_across_dp': max(d1, d2, d4) - min(d1, d2, d4) < 1e-9,
        'ideal_over_current_le_one_everywhere':
            all(r['ideal_over_current'] <= 1.0 + 1e-12 for r in rows),
        'values': {'tight': tight['fragments_per_group_current'],
                   'loose': loose['fragments_per_group_current'],
                   'd1': d1, 'd2': d2, 'd4': d4,
                   'ideal_over_current': [r['ideal_over_current'] for r in rows]},
    }


def prefix_length_invariance():
    """前缀长度只改绝对 FLOP、不改两个比值：在 3 档前缀长度上逐位核对。

    这条同时是 §13.34「共享增益恒等于 1/n、与前缀长度无关」的回归——前缀块口径下
    分子分母同乘一个只与前缀长度有关的常数，任何比值都不该动。
    """
    out = {}
    for p in (128, 512, 2048):
        rng = random.Random(4242 + 17 * 1000 + 8 * 100 + 1 * 10 + 4)
        gl = [[float(max(1, int(round(rng.gauss(SEQ_MEAN, SEQ_MEAN * SEQ_CV))))) for _ in range(8)]
              for _ in range(N_GROUPS)]
        reordered, _ = balance_batch_pg(batch_layout(gl), 1)
        views = dispatch_views(reordered, 1)
        cur = group_fragments(reordered, views, 4, DEFAULT_BUDGET, per_sample=True)
        idm = group_fragments(reordered, views, 4, DEFAULT_BUDGET, per_sample=False)
        whole = sum(prefix_rounds(p, [RESPONSE_LEN] * 8, 1) for _ in range(len(cur)))
        out[str(p)] = {
            'current_over_wholebatch': prefix_cost(cur, prefix_len=p) / whole,
            'ideal_minibatch_over_wholebatch': prefix_cost(idm, prefix_len=p) / whole,
        }
    return out


def build_summary():
    out = {
        'seeds': SEEDS,
        'constants': {
            'n_groups': N_GROUPS, 'n_per': N_PER, 'seq_gauss': [SEQ_MEAN, SEQ_CV],
            'response_len': RESPONSE_LEN, 'prefix_len': PREFIX_LEN,
            'default_budget': DEFAULT_BUDGET,
            'workload': 'calculate_workload(l) = 24576*l + l**2',
            'flop_model': 'prefix O(L^2) + cross O(p*s) + prefix linear；只算前缀块',
        },
    }
    out['layer0_structure'] = layer0_structure()
    out['layer1_ideal_minibatch'] = layer1_ideal_minibatch()
    out['layer2_vpp'] = layer2_vpp()
    out['layer2_budget_sweep'] = layer2_budget_sweep()
    out['identifiability'] = identifiability()
    out['counterfactuals'] = counterfactuals()
    out['prefix_length_invariance'] = prefix_length_invariance()
    out['verdict'] = _verdict(out)
    return out


def _verdict(out):
    l0 = out['layer0_structure']['by_budget']['default_16384']
    l0tight = out['layer0_structure']['by_budget']['loose']
    l1 = out['layer1_ideal_minibatch']
    vpp = out['layer2_vpp']
    b2 = out['layer2_budget_sweep']['rows']
    by_frac = {r['budget_fraction']: r for r in b2}
    return {
        'ceiling_is_minibatch_not_wholebatch': {
            'floor_current_over_wholebatch_default_budget': l0['current_over_wholebatch'],
            'reachable_ceiling_ideal_minibatch_over_wholebatch': l0['ideal_minibatch_over_wholebatch'],
            'floor_current_over_wholebatch_loose_budget': l0tight['current_over_wholebatch'],
            'note': ('外参照（每个组只算一次前缀）要求一个 micro-batch 装下整批；'
                     '可达上限是「每个 mini-batch 里每个组算一次」，二者之比恒等于每 mini-batch 的组数。'
                     '§13.34 报的 3.62× 是前者，不是落差。'),
        },
        'wiring_gain': {
            'fragments_per_group_current': l1['fragments_per_group_current'],
            'fragments_per_group_ideal_minibatch': l1['fragments_per_group_ideal_minibatch'],
            'ideal_over_current_mean': l1['ideal_over_current_mean'],
            'ideal_over_current_range': [l1['ideal_over_current_min'], l1['ideal_over_current_max']],
        },
        'vpp_constraint': {
            'budget_is_soft_vpp_is_hard': {
                'per_sample_n_micro_by_budget': [(r['budget_fraction'], r['per_sample_n_micro'])
                                                 for r in b2],
                'grouped_n_micro_by_budget': [(r['budget_fraction'], r['grouped_n_micro'])
                                              for r in b2],
                'note': ('逐条切的 n_micro = min(样本数, ceil) 随预算增长，组完整的 n_micro '
                         '= roundup(min(组数, ceil), vPP) 在组数处封顶；预算越紧，组完整越只有 '
                         'vPP 的倍数可选——预算是软约束，vPP 是硬约束。'),
            },
            'vpp2_imbalance_flips_at_mini_groups': [
                {'mini_groups': r['mini_groups'],
                 'per_sample_n_micro': r['per_sample_n_micro'],
                 'grouped_n_micro': r['grouped_n_micro'],
                 'load_imbalance_per_sample': r['load_imbalance_per_sample'],
                 'load_imbalance_grouped': r['load_imbalance_grouped']}
                for r in _vpp2_at_default_budget()],
            'vpp_groups': {v: {k: vpp[v]['summary'][k] for k in
                               ('mean_n_micro_ratio_grouped_over_per_sample',
                                'mean_imbalance_per_sample', 'mean_imbalance_grouped',
                                'n_grouped_infeasible') if k in vpp[v]['summary']}
                           for v in vpp},
        },
    }


def _vpp2_at_default_budget():
    out = []
    for mg in (2, 4, 8):
        row = next(x for x in vpp_effect(mini_groups=mg, budget=DEFAULT_BUDGET)['rows'] if x['vpp'] == 2)
        out.append(dict(row, mini_groups=mg))
    return out


def verify():
    with open(SUMMARY_PATH) as f:
        archived = json.load(f)
    fresh = json.loads(json.dumps(build_summary()))
    if fresh == archived:
        print('verify: OK — 重跑结果与归档 summary.json 逐字段相同')
        return 0
    print('verify: MISMATCH')
    _diff(archived, fresh, '')
    return 1


def _diff(a, b, path):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                print('  only in one side:', path + '/' + k)
            else:
                _diff(a[k], b[k], path + '/' + k)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            print('  list length differs:', path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(x, y, path + '[%d]' % i)
    elif a != b:
        print('  differs:', path, a, b)


def main():
    if '--verify' in sys.argv:
        sys.exit(verify())
    summary = build_summary()
    with open(SUMMARY_PATH, 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write('\n')
    print(json.dumps(summary['verdict'], indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
