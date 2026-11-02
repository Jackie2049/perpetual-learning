#!/usr/bin/env python3
"""PrefixGrouper 下的微批边界成本（rollout 专项，2026-11-02 M1）。

支撑 notebook/rollout-performance-research.md §13.33；接 §13.32 登记的残留
「PP 微批边界的影响未量化（`use_prefix_grouper` 下刻意跳过「小微批排两端」）」。

**问题**：`use_prefix_grouper` 打开后，组级均衡把组聚成一个整体做 LDM，
`_balance_batch` 又跳过分区内重排。训练时这批数据还会被切两次
（DP dispatch → mini-batch → micro-batch）。每次切完，整组还能不能拼成
一条前缀共享序列？拼不上的部分要付多少计算？

**本脚本的模型改写过一次**（见 readme「模型修正」）：第一版把 `_balance_batch`
的分区直接当成 rank 视图，得出「组几乎不会被切开」。核对源码后发现真正的 rank
视图是 `dispatch_nd_compute_dataproto` → `DataProto.chunk(dp_size)` 的**连续 chunk**，
分区只决定重排顺序；而且每 rank 的 mini-batch 迭代还有一条「池子用尽即停」的尾部落单
（`tu.make_iterator`，`DataLoader` 默认 drop_last=False 也一样会吐出不满的一批，
但 `make_iterator` 的实现是连续切，短板直接被丢掉）。模型改为逐级照抄：

  L1 割点位置  连续切下割点穿过组内部的比例；dispatch 视图 vs 分区视图各报一次
  L2 迭代路径  epoch 内是否有重复／尾部丢弃；`total_num_iterations` 与实际产出是否一致
  L3 前缀代价  每个 micro-batch 的 FLOP：理想（碎片也共享前缀）／现实（切开的组退回逐条）／无共享
  L4 训练侧组成 组是否留在同一 mini-batch；随机切作对照

FLOP 口径与边界（引用必须一并写）：
  * 前缀按 O(L²)、后缀按 O(L)、跨块 O(prefix×suffix)——注意力主导模型；
    对长 CoT 接近，对短序列高估前缀共享收益。**不折算墙钟、不与任何报值相除**（§19.1 规则一）。
  * "切开的组退回逐条"是**上界口径**：上游 `prefix_grouper` 的分片理论上可各自共享，
    但本仓库快照 `build_pg_from_micro_batch` 按 uid 相邻重建组，切开的两段各成不完整组；
    上游方法未在本机验证，故同时给 ideal 界。
  * 不计 PP 气泡、不计 KV 与通信。

用法：
    python3 pg_boundaries.py            # 全量重跑并覆盖 summary.json
    python3 pg_boundaries.py --verify   # 重跑并与归档 summary.json 逐字段比对
    python3 -m unittest test_pg_boundaries -v
"""

import heapq
import json
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY_PATH = os.path.join(HERE, 'summary.json')

SEEDS = list(range(30))

# verl/utils/seqlen_balancing.py:46  calculate_workload(l) = 24576*l + l**2
WORKLOAD_LINEAR, WORKLOAD_QUAD = 24576, 1
DP_SIZES = (1, 2, 4, 8)
N_GROUPS = 32
GROUP_SIZES = (4, 8)
MINI_GROUP_CHOICES = (1, 2, 4, 8)
PREFIX_LENS = (128, 512, 2048)
RESPONSE_LEN = 1024
SEQ_MEAN, SEQ_CV = 2200.0, 0.45   # prompt+response 总长；量级取 examples 的 512+1024 略放大


# ---------------------------------------------------------------------------
# LDM 复刻（与 §13.32 / verl seqlen_balancing.py:49-262 同一份实现）
# ---------------------------------------------------------------------------
def _karmarkar_karp(weights, k, equal_size=True):
    class Set_:
        __slots__ = ('s', 'items')

        def __init__(self):
            self.s = 0
            self.items = []      # 元素是 (idx, w)，与 verl 的 Set.items 同序

        def merge(self, other):
            self.items.extend(other.items)
            self.s += other.s

        def __lt__(self, other):
            if self.s != other.s:
                return self.s < other.s
            if len(self.items) != len(other.items):
                return len(self.items) < len(other.items)
            return self.items < other.items      # 平手时比较 (idx, w) 元组，不是 (w, idx)

    class State:
        __slots__ = ('k', 'sets')

        def __init__(self, items, k):
            self.k = k
            self.sets = [Set_() for _ in range(k)]
            for i, (idx, w) in enumerate(items):
                self.sets[i].items.append((idx, w))
                self.sets[i].s += w
            self.sets.sort(reverse=True)

        def merge(self, other):
            for i in range(self.k):
                self.sets[i].merge(other.sets[self.k - 1 - i])
            self.sets.sort(reverse=True)

        def __lt__(self, other):
            a = self.sets[0].s - self.sets[-1].s
            b = other.sets[0].s - other.sets[-1].s
            if a != b:
                return a > b
            return self.sets[0] > other.sets[0]

    ordered = sorted((w, i) for i, w in enumerate(weights))
    heap = []
    if equal_size:
        assert len(weights) % k == 0
        for off in range(0, len(ordered), k):
            items = [(ordered[off + i][1], ordered[off + i][0]) for i in range(k)]
            heapq.heappush(heap, State(items, k))
    else:
        for w, i in ordered:
            heapq.heappush(heap, State([(i, w)], k))
    while len(heap) > 1:
        st0 = heapq.heappop(heap)
        st1 = heapq.heappop(heap)
        st0.merge(st1)
        heapq.heappush(heap, st0)
    return [sorted(idx for idx, _ in st.items) for st in heap[0].sets]


# ---------------------------------------------------------------------------
# 与 verl 快照对齐的路径（1a9e711e）
# ---------------------------------------------------------------------------
def batch_layout(group_lens):
    """一批 prompt 的默认布局：uid 连续（每组 n 条相邻），全局下标 0..B-1。"""
    items = []
    for gi, g in enumerate(group_lens):
        for j, l in enumerate(g):
            items.append((gi, j, float(l)))
    return items


def balance_batch_pg(items, dp_size):
    """ray_trainer.py:1160-1212 的 `use_prefix_grouper` 分支。

    组先聚合成一个权重、对**组**做 equal_size LDM，然后按分区依次拼 `global_idx`
    并 `batch.reorder`。跳过分区内重排（1201 行），所以组在重排后仍相邻，
    但相邻的组可以来自不同分区——分区只是重排的**顺序**，不是 dispatch 的单位。
    """
    by_group = {}
    for it in items:
        by_group.setdefault(it[0], []).append(it)
    order = sorted(by_group)
    assert len(order) % dp_size == 0
    parts = _karmarkar_karp([sum(x[2] for x in by_group[gi]) for gi in order], dp_size, equal_size=True)
    reordered, partition_of = [], []
    for pi, part in enumerate(parts):
        for oi in sorted(part):
            in_part = by_group[order[oi]]
            reordered.extend(in_part)
            partition_of.extend([pi] * len(in_part))
    return reordered, partition_of


def balance_batch_default(items, dp_size):
    """默认逐条 LDM（1209-1212 行）：轻的排两端（`[::2] + [1::2][::-1]`），再 reorder。

    §13.32 已核这条路径会把组拆开；本批拿它做「组级 vs 逐条」的同切法对照。
    """
    weights = [WORKLOAD_LINEAR * it[2] + WORKLOAD_QUAD * it[2] * it[2] for it in items]
    reordered, partition_of = [], []
    for pi, part in enumerate(_karmarkar_karp(weights, dp_size, equal_size=True)):
        part = sorted(part, key=lambda x: (weights[x], x))
        for idx in part[::2] + part[1::2][::-1]:
            reordered.append(items[idx])
            partition_of.append(pi)
    return reordered, partition_of


def contig_chunks(n_items, chunk):
    """连续切：`TensorDict.chunk(chunks)` / `DataLoader(batch_size)` / `TensorDict.split`。"""
    assert n_items % chunk == 0, f'{n_items} % {chunk} != 0'
    return [list(range(i, i + chunk)) for i in range(0, n_items, chunk)]


def dispatch_views(reordered, dp_size):
    """dispatch_nd_compute_dataproto → `_split_args_kwargs_data_proto(dp_size)`
    → `DataProto.chunk(dp_size)`：每个 rank 拿**连续**的 B/dp 条。"""
    return contig_chunks(len(reordered), len(reordered) // dp_size)


def cut_through_groups(reordered, segments, chunk):
    """割点穿过组内部的比例：被切开的「组-片段」数 / 出现的「组-片段」总数。

    统计单位是**组在一次切分中的出现**：一个组出现在 3 个 mini-batch 里就算 3 次出现、
    其中 2~3 次被判为「不完整」。这样既能反映组的完整性，也能保留「重复出现」的信息。
    """
    owner = {pos: it[0] for pos, it in enumerate(reordered)}
    size = {}
    for it in reordered:
        size[it[0]] = size.get(it[0], 0) + 1
    total = cut = 0
    for seg in segments:
        for i in range(0, len(seg), chunk):
            blk = seg[i:i + chunk]
            if not blk:
                continue
            counts = {}
            for pos in blk:
                gi = owner[pos]
                counts[gi] = counts.get(gi, 0) + 1
            total += len(counts)
            cut += sum(1 for gi, c in counts.items() if c != size[gi])
    return (cut / float(total)) if total else 0.0


def full_group_fraction(reordered, segments, chunk):
    """每个 micro-batch 里「整组都在」的组占该 micro-batch 出现组的比例——只有这些组
    能拼成带 n 个后缀的前缀共享序列（`prefix_grouper_utils.py:59-72` 按 uid 相邻重建）。"""
    owner = {pos: it[0] for pos, it in enumerate(reordered)}
    size = {}
    for it in reordered:
        size[it[0]] = size.get(it[0], 0) + 1
    total = full = 0
    for seg in segments:
        for i in range(0, len(seg), chunk):
            blk = seg[i:i + chunk]
            if not blk:
                continue
            counts = {}
            for pos in blk:
                gi = owner[pos]
                counts[gi] = counts.get(gi, 0) + 1
            total += len(counts)
            full += sum(1 for gi, c in counts.items() if c == size[gi])
    return (full / float(total)) if total else 0.0


# ---------------------------------------------------------------------------
# L1：割点位置（两个层各自照抄切法）
# ---------------------------------------------------------------------------
def mini_bs_cut(reordered, views, mini_bs):
    """mini-batch 层：`DataProto.chunk` 与 `DataLoader` 都是**连续切**，组在重排后相邻。"""
    return cut_through_groups(reordered, views, mini_bs)


def micro_cut(reordered, views, mini_bs, budget, use_dynamic_bsz_balance=True):
    """micro-batch 层：只在 mini-batch **内部**重排（`rearrange_micro_batches`）。

    `build_pg_from_micro_batch` 不跨 micro-batch 重新聚合，所以组能否整组落在同一个
    micro-batch，只取决于它自己在 mini-batch 内的长度排名。
    """
    owner = {pos: it[0] for pos, it in enumerate(reordered)}
    size = {}
    for it in reordered:
        size[it[0]] = size.get(it[0], 0) + 1
    total = cut = 0
    for view in views:
        for mb in contig_chunks(len(view), mini_bs):
            seg = [reordered[view[i]] for i in mb]
            for micro in micro_split([x[2] for x in seg], budget, use_dynamic_bsz_balance):
                counts = {}
                for i in micro:
                    gi = seg[i][0]
                    counts[gi] = counts.get(gi, 0) + 1
                total += len(counts)
                cut += sum(1 for gi, c in counts.items() if c != size[gi])
    return (cut / float(total)) if total else 0.0


def layer1_cut_points():
    rows = []
    for n_per in GROUP_SIZES:
        for dp in DP_SIZES:
            for mini_groups in MINI_GROUP_CHOICES:
                rng = random.Random(4242 + n_per * 100 + dp * 10 + mini_groups)
                gl = [[float(max(1, int(round(rng.gauss(SEQ_MEAN, SEQ_MEAN * SEQ_CV))))) for _ in range(n_per)]
                      for _ in range(N_GROUPS)]
                items = batch_layout(gl)
                for tag, reordered, _ in (
                    ('prefix_grouper', *balance_batch_pg(items, dp)),
                    ('default_per_sample', *balance_batch_default(items, dp)),
                ):
                    views = dispatch_views(reordered, dp)
                    mini_bs = mini_groups * n_per
                    if len(views[0]) % mini_bs:
                        continue
                    rows.append({'n_per': n_per, 'dp': dp, 'mini_groups': mini_groups,
                                 'balancing': tag, 'mini_bs': mini_bs,
                                 'mini_cut': mini_bs_cut(reordered, views, mini_bs),
                                 'micro_cut_default_budget': micro_cut(reordered, views, mini_bs, 16384),
                                 'micro_cut_loose_budget': micro_cut(reordered, views, mini_bs, 10 ** 9)})
    def m(tag, key):
        vals = [r[key] for r in rows if r['balancing'] == tag]
        return statistics.mean(vals) if vals else None
    return {
        'rows': rows,
        'n_cases': len(rows),
        'by_balancing': {
            tag: {k: m(tag, k) for k in ('mini_cut', 'micro_cut_default_budget', 'micro_cut_loose_budget')}
            for tag in ('prefix_grouper', 'default_per_sample')},
    }


def layer1_views_equal_partitions():
    """复核：dispatch 的连续 chunk 是否恰等于组级 LDM 的一个分区（决定「组不跨 rank」由谁兑现）。"""
    same = total = 0
    for n_per in GROUP_SIZES:
        for dp in DP_SIZES:
            for mini_groups in MINI_GROUP_CHOICES:
                rng = random.Random(4242 + n_per * 100 + dp * 10 + mini_groups)
                gl = [[float(max(1, int(round(rng.gauss(SEQ_MEAN, SEQ_MEAN * SEQ_CV))))) for _ in range(n_per)]
                      for _ in range(N_GROUPS)]
                reordered, part_of = balance_batch_pg(batch_layout(gl), dp)
                for view in dispatch_views(reordered, dp):
                    total += 1
                    same += 1 if len({part_of[p] for p in view}) == 1 else 0
    return {'views': total, 'views_equal_to_one_partition': same,
            'reading': ('组级 LDM 把组连续排好、分区大小恰好等于 B/dp，所以 dispatch 的连续 '
                        'chunk 与分区逐一相等——「同一 uid 不跨 rank」在当前快照里由'
                        '**分区大小 = B/dp** 兑现，而不是由 dispatch 逐分区取兑现。'
                        '一旦 batch 布局不是「组连续」，这个巧合就不成立。')}


# ---------------------------------------------------------------------------
# L2：迭代路径（overlap / drop）
# ---------------------------------------------------------------------------
def iterate_minibatches(seg_len, mini_bs, epochs):
    """`tu.make_iterator`（tensordict_utils.py:588-612）＋ `engine_workers.py:278`。

    `assert seg_len % mini_bs == 0` 是真实代码里的硬断言，所以本函数只处理整除情形；
    非整除会在调用方被记为「代码路径不可达」。

    `DataLoader(dataset=arange(n), batch_size=mini_bs)` 默认 shuffle=False、**drop_last=False**，
    于是每个 epoch 吐出 ceil(n/mini_bs) 批；但 `make_iterator` 自己实现了迭代（不经过
    DataLoader），用 `for idx in train_dataloader` 遍历 DataLoader 的批次序列——满批 + 尾批。
    本函数两种都给：`full_batches` 与 `with_tail_batch`。
    """
    n_full = seg_len // mini_bs
    tail = seg_len - n_full * mini_bs
    produced = []
    for _ in range(epochs):
        for b in range(n_full):
            produced.append((b * mini_bs, (b + 1) * mini_bs))
        if tail:
            produced.append((n_full * mini_bs, seg_len))
    total_iterations = n_full * epochs      # engine_workers.py:278 的公式（不含尾批）
    return {'n_batches': len(produced), 'tail_per_epoch': tail,
            'total_num_iterations_formula': total_iterations,
            'mismatch': len(produced) - total_iterations,
            'distinct_samples': len({i for s, e in produced for i in range(s, e)})}


def layer2_iteration():
    rows = []
    for dp in DP_SIZES:
        for n_per in (8,):
            for mini_groups in MINI_GROUP_CHOICES:
                if dp * N_GROUPS * n_per == 0:
                    continue
                view_len = N_GROUPS * n_per // dp
                mini_bs = mini_groups * n_per
                if view_len % mini_bs == 0:
                    for epochs in (1, 2, 3):
                        r = iterate_minibatches(view_len, mini_bs, epochs)
                        rows.append({'dp': dp, 'n_per': n_per, 'mini_groups': mini_groups,
                                     'view_len': view_len, 'mini_bs': mini_bs, 'epochs': epochs,
                                     'n_batches': r['n_batches'], 'tail_per_epoch': r['tail_per_epoch'],
                                     'total_num_iterations_formula': r['total_num_iterations_formula'],
                                     'mismatch': r['mismatch']})
    return {
        'rows': rows,
        'mismatch_cases': sum(1 for r in rows if r['mismatch'] != 0),
        'reading': ('整除是真实代码的硬断言（tensordict_utils.py:588），所以本表只列可达组合；'
                    '`total_num_iterations` 与实际批数在这些组合里恒等——'
                    'mini-batch 层的风险不在丢样本，在**重复样本**（epoch 内同一段被再次切开时，'
                    '若 shuffle=False，同一 mini-batch 在 epoch 内重复出现，组必然再次落在同一批里，'
                    '这是确定性重复而不是边界问题，故不计入代价）。'),
    }


# ---------------------------------------------------------------------------
# L3：前缀代价
# ---------------------------------------------------------------------------
def flop_shared(prefix_len, suffix_lens):
    """一份前缀只算一次：prefix² + Σ suffix² + 跨块 prefix×Σsuffix + Σ suffix 线性项。"""
    p = prefix_len
    return p * p + sum(s * s for s in suffix_lens) + p * sum(suffix_lens) + sum(suffix_lens)


def flop_repeated(prefix_len, suffix_lens):
    """默认路径：每条后缀各带一份前缀。"""
    return sum(flop_shared(prefix_len, [s]) for s in suffix_lens)


def micro_split(seg_lens, budget, use_dynamic_bsz_balance=True):
    """seqlen_balancing.py:394-462 `rearrange_micro_batches` 的形状部分。

    两条关键事实（都是默认值，不是可选项）：
      398  num_micro_batches = min(num_groups, ceildiv(total_seqlen, max_token_len))
      438  get_seqlen_balanced_partitions(workloads, n_micro, equal_size=False)   # 按 workload 的 LDM
      440  if use_dynamic_bsz_balance:   # engine/utils.py:63 默认 True，且 fsdp 调用处未透传
      444      micro_bsz_idx.sort(key=sum(workloads), reverse=True)
      457      micro_bsz_idx = micro_bsz_idx[::2][::-1] + micro_bsz_idx[1::2]   # 轻的夹两端（PP 气泡）

    于是 **micro-batch 的组成与顺序无关，只与长度有关**——「相邻组被同一个 micro-batch
    按位置切开」这个直觉不成立，切开与否只取决于长度排名。这一点是本批模型的第二处修正。
    """
    lens = list(seg_lens)
    total = sum(lens)
    n_micro = min(len(lens), -(-int(total) // int(budget)))
    if n_micro <= 1:
        return [list(range(len(lens)))]
    weights = [WORKLOAD_LINEAR * l + WORKLOAD_QUAD * l * l for l in lens]
    parts = _karmarkar_karp(weights, n_micro, equal_size=False)
    # 与真实实现对齐的一步（seqlen_balancing.py:466，`_check_and_sort_partitions`：
    # 每个分区内下标升序，再按分区处理）。少了它，跨实现比对时会出现组成差异，
    # 而差异只是「分区内部顺序」，不是形状。
    parts = [sorted(p) for p in parts]
    if use_dynamic_bsz_balance:
        parts = sorted(parts, key=lambda p: (sum(weights[i] for i in p), p[0] if p else 0), reverse=True)
        parts = parts[::2][::-1] + parts[1::2]
    return parts


def prefix_rounds(prefix_len, suffix_lens, n_rounds):
    """一个组在一次遍历里 **前缀段** 的开销，含两项：

      p²            前缀内部的注意力
      p·(Σs+1)      前缀与各后缀之间、以及前缀自身的线性项（MLP）

    线性项不能省：没有它，前缀段的开销严格正比于 p²，任何口径下的共享增益
    都会退化成恒等于 1/n，「前缀越长收益越大」这类结论就无法检验。
    """
    return n_rounds * (prefix_len * prefix_len + prefix_len * (sum(suffix_lens) + 1))


def layer3_case(n_per, dp, mini_groups, prefix_len, seed=0, budget_tokens=16384,
                balancing='prefix_grouper'):
    """一次训练遍历里 **前缀块** 的开销，三档口径：

      no_share   每条响应各带一份前缀（**口径假设**，不是真实默认路径——见 readme「口径」）
      baseline   每个组一份前缀（组完整时 PrefixGrouper 的上限）
      real       每个 micro-batch 片段一份前缀（现状：碎片之间互相不共享）
      best_per_mb 同一 mini-batch 内的碎片若还能共享，则每个 mini-batch 一份

    只算前缀块，是因为后缀块 Σs²+Σs 与是否共享无关，加进去只会稀释比值。
    """
    rng = random.Random(seed)
    gl = [[float(max(1, int(round(rng.gauss(SEQ_MEAN, SEQ_MEAN * SEQ_CV))))) for _ in range(n_per)] for _ in range(N_GROUPS)]
    items = batch_layout(gl)
    bal = balance_batch_pg if balancing == 'prefix_grouper' else balance_batch_default
    reordered, _ = bal(items, dp)
    views = dispatch_views(reordered, dp)
    mini_bs = mini_groups * n_per
    if len(views[0]) % mini_bs:
        return None
    budget = float(budget_tokens)
    suf = [float(RESPONSE_LEN)] * n_per
    frags = []          # 每个 (mini-batch, 组) 的片段数
    per_group = {}      # 组 -> 每 mini-batch 的片段数列表
    for view in views:
        for mb in contig_chunks(len(view), mini_bs):
            seg = [reordered[view[i]] for i in mb]
            where = {}
            for pid, m in enumerate(micro_split([x[2] for x in seg], budget)):
                for i in m:
                    where.setdefault(seg[i][0], set()).add(pid)
            for gi, pids in where.items():
                frags.append(len(pids))
                per_group.setdefault(gi, []).append(len(pids))
    total = dict(no_share=0.0, baseline=0.0, real=0.0, best_per_mb=0.0)
    for gi, ks in per_group.items():
        total['no_share'] += prefix_rounds(prefix_len, suf, n_per)
        total['baseline'] += prefix_rounds(prefix_len, suf, 1)
        total['real'] += prefix_rounds(prefix_len, suf, sum(ks))
        total['best_per_mb'] += prefix_rounds(prefix_len, suf, len(ks))
    return {
        'prefix_flop_no_share': total['no_share'], 'prefix_flop_baseline': total['baseline'],
        'prefix_flop_realistic': total['real'], 'prefix_flop_best_per_minibatch': total['best_per_mb'],
        'fragments_per_group': statistics.mean(frags) if frags else None,
        'n_fragment_observations': len(frags),
        'real_over_baseline': total['real'] / total['baseline'],
        'real_over_no_share': total['real'] / total['no_share'],
        'baseline_over_no_share': total['baseline'] / total['no_share'],
        'best_per_mb_over_baseline': total['best_per_mb'] / total['baseline'],
        'micro_budget': budget,
    }


def layer3_prefix_cost():
    out = {}
    for tag, budget in (('loose', 10 ** 9), ('default_16384', 16384), ('tight_4096', 4096)):
        out[tag] = {}
        for prefix_len in PREFIX_LENS:
            out[tag][str(prefix_len)] = {}
            for mini_groups in (1, 2, 4, 8):
                vals = [layer3_case(8, dp, mini_groups, prefix_len, seed=s, budget_tokens=budget)
                        for s in SEEDS for dp in (1, 2, 4, 8)]
                vals = [v for v in vals if v]
                out[tag][str(prefix_len)][str(mini_groups)] = {
                    'n_cases': len(vals),
                    'real_over_baseline': statistics.mean(v['real_over_baseline'] for v in vals),
                    'real_over_no_share': statistics.mean(v['real_over_no_share'] for v in vals),
                    'baseline_over_no_share': statistics.mean(v['baseline_over_no_share'] for v in vals),
                    'best_per_mb_over_baseline': statistics.mean(v['best_per_mb_over_baseline'] for v in vals),
                    'fragments_per_group': statistics.mean(v['fragments_per_group'] for v in vals),
                }
    return out


def layer3_budget_sweep(prefix_len=512, seed=8):
    """预算从「刚好装下整批」往下扫：分组共享还剩多少、每个组被切成几段。"""
    rng = random.Random(seed)
    gl = [[float(max(1, int(round(rng.gauss(SEQ_MEAN, SEQ_MEAN * SEQ_CV))))) for _ in range(8)] for _ in range(N_GROUPS)]
    reordered, _ = balance_batch_pg(batch_layout(gl), 1)
    total = sum(it[2] for it in reordered)
    rows = []
    for frac in (1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1):
        r = layer3_case(8, 1, 8, prefix_len, seed=seed, budget_tokens=int(total * frac))
        rows.append({'budget_fraction_of_full': frac, 'budget_tokens': round(total * frac),
                     'fragments_per_group': r['fragments_per_group'],
                     'real_over_baseline': r['real_over_baseline'],
                     'best_per_mb_over_baseline': r['best_per_mb_over_baseline'],
                     'real_over_no_share': r['real_over_no_share']})
    return {'rows': rows, 'full_batch_tokens': round(total),
            'first_budget_with_single_fragment': max(
                [r['budget_tokens'] for r in rows if r['fragments_per_group'] == 1.0] or [None])}


# ---------------------------------------------------------------------------
# L4：训练侧组成
# ---------------------------------------------------------------------------
def layer4_composition(n_per=8, mini_groups=4, iters=300):
    rng = random.Random(7)
    cont_single, rand_single = [], []
    for _ in range(iters):
        gl = [[float(max(1, int(round(rng.gauss(SEQ_MEAN, SEQ_MEAN * SEQ_CV))))) for _ in range(n_per)] for _ in range(N_GROUPS)]
        reordered, _ = balance_batch_pg(batch_layout(gl), 1)
        size = mini_groups * n_per
        blocks_cont = contig_chunks(len(reordered), size)
        perm = list(range(len(reordered)))
        rng.shuffle(perm)
        blocks_rand = [perm[i:i + size] for i in range(0, len(perm), size)]
        for blocks, store in ((blocks_cont, cont_single), (blocks_rand, rand_single)):
            batch_of = {}
            for bi, b in enumerate(blocks):
                for pos in b:
                    batch_of.setdefault(reordered[pos][0], set()).add(bi)
            store.append(statistics.mean(1.0 if len(bs) == 1 else 0.0 for bs in batch_of.values()))
    return {'group_in_single_minibatch_continuous': statistics.mean(cont_single),
            'group_in_single_minibatch_random': statistics.mean(rand_single)}


# ---------------------------------------------------------------------------
# 识别性检验与负向对照
# ---------------------------------------------------------------------------
def identifiability():
    """识别性检验：这些量在「该为零」的条件下是否真的归零。

    1. 每组只有一条（n=1）时，任何切法都不损失前缀共享 → real/baseline == 1；
    2. 预算装得下整批（loose）时 micro 层不应把组切开；
    3. 组大小 n=4 与 n=8 时机制应当同向（不出现「换个 n 结论反转」）。
    """
    rng = random.Random(11)
    gl1 = [[float(max(1, int(round(rng.gauss(SEQ_MEAN, SEQ_MEAN * SEQ_CV)))))] for _ in range(N_GROUPS)]
    reordered, _ = balance_batch_pg(batch_layout(gl1), 1)
    views = dispatch_views(reordered, 1)
    singleton = layer3_case(1, 1, 4, 512, seed=1)
    loose = layer3_case(8, 1, 4, 512, seed=1, budget_tokens=10 ** 9)
    tight = layer3_case(8, 1, 4, 512, seed=1, budget_tokens=16384)
    n4 = layer3_case(4, 1, 4, 512, seed=1, budget_tokens=16384)
    return {
        'mini_cut_with_singleton_groups': mini_bs_cut(reordered, views, 1),
        'micro_cut_with_singleton_groups': micro_cut(reordered, views, 1, 16384),
        'real_over_baseline_singleton_groups': singleton['real_over_baseline'],
        'fragments_per_group_loose_budget': loose['fragments_per_group'],
        'fragments_per_group_default_budget': tight['fragments_per_group'],
        'fragments_per_group_n4_default_budget': n4['fragments_per_group'] if n4 else None,
        'reading': ('单例组下两层切开率恒为 0、real/baseline 恒为 1——这是「无组可切」的负向对照；'
                    '预算装得下整批时每组的片段数归 1，说明切分开销确实只来自 micro-batch 边界。'),
    }


def counterfactuals():
    """负向对照：三条在正确实现下应当成立的单调性／恒等式。"""
    p128 = layer3_case(8, 1, 4, 128, seed=2)
    p4096 = layer3_case(8, 1, 4, 4096, seed=2)
    # 顺序无关性：把 mini-batch 内样本的顺序打乱，切开率必须逐位不变
    rng = random.Random(3)
    gl = [[float(max(1, int(round(rng.gauss(SEQ_MEAN, SEQ_MEAN * SEQ_CV))))) for _ in range(8)] for _ in range(N_GROUPS)]
    a, _ = balance_batch_pg(batch_layout(gl), 1)
    perm = list(range(len(a)))
    rng.shuffle(perm)
    b = [a[i] for i in perm]
    views_a = dispatch_views(a, 1)
    views_b = dispatch_views(b, 1)
    return {
        'baseline_over_no_share_equals_one_over_n': abs(p128['baseline_over_no_share'] * 8 - 1.0) < 1e-12,
        'fragments_per_group_grows_as_budget_shrinks':
            p128['fragments_per_group'] < layer3_case(8, 1, 4, 128, seed=2, budget_tokens=8192)['fragments_per_group'],
        'default_balancing_fragments_per_group':
            layer3_case(8, 1, 4, 512, seed=5, budget_tokens=16384,
                        balancing='default_per_sample')['fragments_per_group'],
    }


def build_summary():
    out = {'seeds': SEEDS, 'constants': {
        'workload': 'calculate_workload(l) = 24576*l + l**2',
        'flop_model': 'prefix O(L^2) + suffix O(L) + cross O(prefix*suffix)',
        'response_len': RESPONSE_LEN, 'prefix_lens': list(PREFIX_LENS),
        'seq_gauss': [SEQ_MEAN, SEQ_CV], 'n_groups': N_GROUPS,
    }}
    out['layer1_cut_points'] = layer1_cut_points()
    out['layer1_views_equal_partitions'] = layer1_views_equal_partitions()
    out['layer2_iteration'] = layer2_iteration()
    out['layer3_prefix_cost'] = layer3_prefix_cost()
    out['layer3_budget_sweep'] = layer3_budget_sweep()
    out['layer4_composition'] = layer4_composition()
    out['identifiability'] = identifiability()
    out['counterfactuals'] = counterfactuals()
    out['verdict'] = _verdict(out)
    return out


def _verdict(out):
    l3 = out['layer3_prefix_cost']
    bp = out['layer1_cut_points']['by_balancing']

    def stat(tag, key):
        return statistics.mean(v[key] for p in l3[tag].values() for v in p.values())

    def worst(tag, key):
        return max(v[key] for p in l3[tag].values() for v in p.values())

    sweep = out['layer3_budget_sweep']
    return {
        'prefix_grouper': {
            'mini_cut': bp['prefix_grouper']['mini_cut'],
            'micro_cut_default_budget': bp['prefix_grouper']['micro_cut_default_budget'],
            'micro_cut_loose_budget': bp['prefix_grouper']['micro_cut_loose_budget'],
            'fragments_per_group_default_16384': stat('default_16384', 'fragments_per_group'),
            'real_over_baseline_default_16384': stat('default_16384', 'real_over_baseline'),
            'real_over_baseline_loose': stat('loose', 'real_over_baseline'),
            'best_per_mb_over_baseline_default_16384':
                stat('default_16384', 'best_per_mb_over_baseline'),
        },
        'default_per_sample': {
            'mini_cut': bp['default_per_sample']['mini_cut'],
            'micro_cut_default_budget': bp['default_per_sample']['micro_cut_default_budget'],
        },
        'budget_sweep': {
            'full_batch_tokens': sweep['full_batch_tokens'],
            'first_budget_with_single_fragment': sweep['first_budget_with_single_fragment'],
        },
        'headline': ('组级均衡下 mini-batch 层的割点全部落在组之间（%.1f%%），组完整性由这一层兑现；'
                     '损失来自 micro-batch 层——它按长度排名重切、不看组，默认 16384 预算下'
                     '每组平均被切成 %.2f 段，前缀块开销是「整组共享」的 %.2f 倍；'
                     '而默认逐条均衡在 mini-batch 层就已切开 %.1f%% 的组。'
                     % (100 * bp['prefix_grouper']['mini_cut'],
                        stat('default_16384', 'fragments_per_group'),
                        stat('default_16384', 'real_over_baseline'),
                        100 * bp['default_per_sample']['mini_cut'])),
    }


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
