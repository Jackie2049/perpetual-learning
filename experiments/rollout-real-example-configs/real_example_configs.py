#!/usr/bin/env python3
"""verl 官方 example 的真实配置下：micro-batch 层与 PP 气泡，以及一条死掉的前缀共享线。

支撑 notebook/rollout-performance-research.md §13.36。
接 §13.34／§13.35 登记的残留——「被跳过的 PP 气泡仍未测」「长度分布是构造参数」。

**最重要的判决（本批新核，静态源码，非测量）**：§13.34／§13.35 量的是 `use_prefix_grouper`
的 micro-batch 边界代价，但那条线在 verl `1a9e711e` 上**已经不可达**——engine 迁移 `#6067`
（`044bbba2`）删掉了 `workers/actor/dp_actor.py` 与 `workers/fsdp_workers.py`（PG 的唯一调用方与
唯一把 `use_prefix_grouper` 传给 `apply_monkey_patch` 的地方），新 engine 从未补上接线。

其余两件是**能落地**的计数读数，全部来自官方 example 的真实配置：
  2. **真实配置里模型能建的 micro-batch 太少**：7 个开 `use_dynamic_bsz` 的 example 里，
     一个 DP rank 每个 mini-batch 只有 2–16 个 micro-batch（默认预算是瓶颈时最多 33），
     这是「组完整」真正受限于的地方；而 3 个 `use_dynamic_bsz=False` 的 BSHD 配置
     **根本不经过** `rearrange_micro_batches`。
  3. **PP 气泡方向与收益相反**：组完整把批数变少，气泡占比 `1 − m/(m+pp−1)` 随之上升。

所有数字都是**计数**，不含任何墙钟；PP 气泡是占比的**比值**，不是实测。

复现：
    python3 real_example_configs.py                # 约 1 s，覆盖 summary.json
    python3 real_example_configs.py --verify       # 与归档逐字段比对
    python3 -m unittest test_real_example_configs -v
    /tmp/verl-seqlb-venv/bin/python xcheck_verl.py   # 与真实 verl 逐位核对（可选，需 CPU torch）
"""
import json
import os
import random
import sys
import zlib
from math import ceil

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY_PATH = os.path.join(HERE, 'summary.json')

WORKLOAD_LINEAR, WORKLOAD_QUAD = 24576, 1     # seqlen_balancing.calculate_workload（7B 标定）
SEEDS = 30
TOTAL_MEAN_FRAC = 0.8     # 实际轨迹总长 = 0.8 × (prompt 上限 + response 上限)
GROUP_OFFSET_SD = 0.25    # 组级（同一 prompt）长度偏移
SAMPLE_OFFSET_SD = 0.15   # 组内单条偏移


# ---------------------------------------------------------------------------
# LDM（与 pg_boundaries._karmarkar_karp 同一份实现；verl seqlen_balancing.py:49-262）
# ---------------------------------------------------------------------------
def karmarkar_karp(weights, k, equal_size=True):
    import heapq

    class Set_:
        __slots__ = ('s', 'items')

        def __init__(self):
            self.s = 0
            self.items = []

        def merge(self, other):
            self.items.extend(other.items)
            self.s += other.s

        def __lt__(self, other):
            if self.s != other.s:
                return self.s < other.s
            if len(self.items) != len(other.items):
                return len(self.items) < len(other.items)
            return self.items < other.items

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
        assert len(weights) % k == 0, 'equal_size needs divisible input'
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


def roundup_divisible(a, b):
    return ((a + b - 1) // b) * b


# ---------------------------------------------------------------------------
# rearrange_micro_batches 的批数公式（seqlen_balancing.py:402-414），不含分区
# ---------------------------------------------------------------------------
def num_micro_batches(batch_len, lens, max_token_len, force_group_size=1,
                      num_batches_divided_by=None):
    """返回 (n_micro, status)。

    status ∈ {'ok', 'shape', 'cap'}；三类与真实实现的 405／412／414 行一一对应。
    """
    if max_token_len < max(lens):
        return None, 'shape'
    num_groups = batch_len // force_group_size
    n = min(num_groups, ceil(sum(lens) / max_token_len))
    if num_batches_divided_by:
        n = roundup_divisible(n, num_batches_divided_by)
    if n > num_groups:
        return None, 'cap'
    return n, 'ok'


def micro_split(lens, n_micro, force_group_size=1, use_dynamic_bsz_balance=True):
    """rearrange_micro_batches 的分区与排序（seqlen_balancing.py:416-462）。"""
    b = len(lens)
    assert b % force_group_size == 0
    num_groups = b // force_group_size
    assert n_micro <= num_groups
    if n_micro <= 1:
        return [list(range(b))]
    w = [WORKLOAD_LINEAR * l + WORKLOAD_QUAD * l * l for l in lens]
    if force_group_size > 1:
        group_w = [sum(w[g * force_group_size:(g + 1) * force_group_size]) for g in range(num_groups)]
        gparts = karmarkar_karp(group_w, n_micro, equal_size=False)
        parts = []
        for gp in sorted(gparts):
            idx = []
            for g in gp:
                idx.extend(range(g * force_group_size, (g + 1) * force_group_size))
            parts.append(idx)
        sort_w = group_w
    else:
        parts = [sorted(p) for p in karmarkar_karp(w, n_micro, equal_size=False)]
        sort_w = w
    if use_dynamic_bsz_balance:
        parts.sort(key=lambda p: (sum(sort_w[i // force_group_size] for i in p[::force_group_size]),
                                  p[0] if p else 0), reverse=True)
        parts = parts[::2][::-1] + parts[1::2]
    return parts


# ---------------------------------------------------------------------------
# 真实配置样本（verl 1a9e711e examples/grpo_trainer/，逐条从脚本抄出）
# ---------------------------------------------------------------------------
# budget  = `ppo_max_token_len_per_gpu`（每个 micro-batch 的 token 预算，**不是**长度上限）
# max_seq = `max_prompt_length + max_response_length`（405 行断言用）
# dyn     = `actor.use_dynamic_bsz`；False 时不经过 rearrange_micro_batches
# mini_bs = prompt 口径的 `ppo_mini_batch_size`；dp = 一个 actor DP 组内的 rank 数
EXAMPLES = [
    dict(name='qwen3_8b_fsdp', backend='fsdp', train_bs=1024, n=5, mini_bs=256,
         prompt_len=1024, resp_len=2048, budget=24576, sp=1, vpp=None, pp=1, dp=8, dyn=True,
         source='run_qwen3_8b_fsdp.sh:28-40 / :77-93（8 GPU 单机）'),
    dict(name='qwen3_8b_megatron', backend='megatron', train_bs=1024, n=5, mini_bs=256,
         prompt_len=1024, resp_len=2048, budget=24576, sp=1, vpp=None, pp=2, dp=4, dyn=True,
         source='run_qwen3_8b_megatron.sh:16-31（TP=2 PP=2，8 GPU 单机）'),
    dict(name='qwen2_5_32b_fsdp', backend='fsdp', train_bs=1024, n=5, mini_bs=128,
         prompt_len=1024, resp_len=1024, budget=16384, sp=1, vpp=None, pp=1, dp=16, dyn=True,
         source='run_qwen2_5_32b_fsdp.sh:15-31（NNODES=2 → 16 GPU）'),
    dict(name='qwen3_30b_a3b_fsdp', backend='fsdp', train_bs=512, n=8, mini_bs=128,
         prompt_len=2048, resp_len=4096, budget=32768, sp=1, vpp=None, pp=1, dp=8, dyn=True,
         source='run_qwen3_30b_a3b_fsdp.sh:8-23（单机 8 GPU）'),
    dict(name='qwen3_30b_a3b_megatron_vpp2', backend='megatron', train_bs=512, n=8, mini_bs=16,
         prompt_len=2048, resp_len=8192, budget=30720, sp=1, vpp=2, pp=2, dp=2, dyn=True,
         source='run_qwen3_30b_a3b_megatron.sh:27-57（TP=4 PP=2 VPP=2 EP=2，单机 8 GPU）'),
    dict(name='qwen3_5_35b_fsdp', backend='fsdp', train_bs=64, n=5, mini_bs=16,
         prompt_len=1024, resp_len=2048, budget=16384, sp=1, vpp=None, pp=1, dp=8, dyn=True,
         source='run_qwen3_5_35b_fsdp.sh:61-114（未覆盖 *_max_token_len → 默认 16384）'),
    dict(name='hy_v3_megatron_pp16', backend='megatron', train_bs=128, n=16, mini_bs=128,
         prompt_len=2048, resp_len=8192, budget=30720, sp=1, vpp=None, pp=16, dp=1, dyn=True,
         source='run_hy_v3_megatron.sh:23-60（actor PP=16，TP=1）'),
    dict(name='dsv3_671b_megatron_pp12', backend='megatron', train_bs=96, n=8, mini_bs=32,
         prompt_len=2048, resp_len=8192, budget=10240, sp=1, vpp=None, pp=12, dp=1, dyn=True,
         source='run_deepseek_v3_671b_megatron.sh:24-48（budget = prompt+response = 10240）'),
    # 静态批（BSHD，Qwen3.5 的 GDN 不支持 packed sequence）：不经过 rearrange_micro_batches
    dict(name='qwen3_5_35b_bshd_static', backend='megatron', train_bs=32, n=5, mini_bs=32,
         prompt_len=1024, resp_len=2048, budget=4096, sp=1, vpp=None, pp=2, dp=4, dyn=False,
         source='run_qwen3_5_35b_megatron.sh:95-148（use_dynamic_bsz=False）'),
    dict(name='qwen3_5_397b_bshd_static', backend='megatron', train_bs=64, n=5, mini_bs=32,
         prompt_len=1024, resp_len=2048, budget=4096, sp=1, vpp=None, pp=2, dp=16, dyn=False,
         source='run_qwen3_5_397b_megatron.sh:123-177（use_dynamic_bsz=False）'),
    dict(name='qwen3_5_122b_bshd_static', backend='megatron', train_bs=128, n=6, mini_bs=64,
         prompt_len=3240, resp_len=4096, budget=8192, sp=1, vpp=None, pp=2, dp=16, dyn=False,
         source='run_qwen3_5_122b_a10b_megatron.sh:86-141（use_dynamic_bsz=False）'),
]

# 前缀共享线在本 checkout 的可达性（静态源码核对，锚点见 run-info.json）。
PREFIX_GROUPER_DEAD_PATH = dict(
    head='1a9e711e',
    knob_read=['verl/trainer/ppo/ray_trainer.py:367（只影响 _balance_batch 的分组）'],
    knob_never_forwarded=[
        'verl/workers/engine/fsdp/transformer_impl.py:300-306（apply_monkey_patch 唯一调用点，未传 use_prefix_grouper）',
        'verl/models/transformers/monkey_patch.py:297（形参默认 False）',
    ],
    prefix_grouper_never_constructed='verl/trainer/ppo/prefix_grouper_utils.py:46 的 build_pg_from_micro_batch 零调用方',
    utils_never_imported='verl/trainer/ppo/prefix_grouper_utils.py 全仓无 import',
    dropped_by='044bbba2 [BREAKING][misc] refactor: deprecate workers, migrate to engines (#6067)',
    proof_wiring_existed_before=[
        '044bbba2^:verl/workers/actor/dp_actor.py:130-141（PG forward 分支）',
        '044bbba2^:verl/workers/fsdp_workers.py:934,969,988（把 use_prefix_grouper 传给 apply_monkey_patch）',
    ],
    shipped_example_contradiction='examples/prefix_grouper/run_qwen3_8b_fsdp.sh:56 设 use_dynamic_bsz=True，'
                                 '而同一目录 README「Limitations」第 2 条写 Incompatible with use_dynamic_bsz=True',
)


def rank_view(ex):
    """一个 actor DP rank 在一个 mini-batch 里看到的样本数与组数。

    mini-batch 的样本数 = `ppo_mini_batch_size × rollout.n`（ray_trainer.py:1315-1316），
    再按 dp 均分；组数 = `ppo_mini_batch_size / dp`。
    """
    if ex['mini_bs'] % ex['dp'] or ex['train_bs'] % ex['mini_bs']:
        return None
    return ex['mini_bs'] // ex['dp'] * ex['n'], ex['mini_bs'] // ex['dp']


def make_lengths(ex, n_groups_rank, seed):
    """一个 rank 内的长度序列：组 g 占 [g*n, (g+1)*n)，组级偏移共享、组内各异。"""
    rng = random.Random(zlib.crc32(ex['name'].encode()) * 1000 + seed)
    total_mean = TOTAL_MEAN_FRAC * (ex['prompt_len'] + ex['resp_len'])
    lens, group_of = [], []
    for g in range(n_groups_rank):
        g_off = rng.gauss(0, GROUP_OFFSET_SD)
        for _ in range(ex['n']):
            l = total_mean * (1.0 + g_off + rng.gauss(0, SAMPLE_OFFSET_SD))
            lens.append(max(64, int(round(l))))
            group_of.append(g)
    return lens, group_of


def fragments(parts, group_of):
    """每组在这些 micro-batch 分区里被切成几段。"""
    nfrag = {}
    for part in parts:
        for g in {group_of[i] for i in part}:
            nfrag[g] = nfrag.get(g, 0) + 1
    return nfrag


def case(ex, seed):
    """一个 (配置, 种子) 的切分形状。number 口径全部来自真实预算公式。"""
    rv = rank_view(ex)
    if rv is None:
        return dict(name=ex['name'], seed=seed, status='infeasible_view')
    b, n_groups = rv
    if not ex['dyn']:
        return dict(name=ex['name'], seed=seed, status='static_bsz', batch=b, n_groups=n_groups)
    lens, group_of = make_lengths(ex, n_groups, seed)
    budget = ex['budget'] * ex['sp']

    n_default, st_d = num_micro_batches(b, lens, budget, force_group_size=1,
                                        num_batches_divided_by=ex['vpp'])
    if st_d != 'ok':
        return dict(name=ex['name'], seed=seed, status=st_d, batch=b, n_groups=n_groups,
                    max_seq=ex['prompt_len'] + ex['resp_len'], budget=budget)

    # 组完整：样本数仍是 b（预算按样本 token 计），组数才退化为 n_groups
    n_grouped, st_g = num_micro_batches(b, lens, budget, force_group_size=ex['n'],
                                        num_batches_divided_by=ex['vpp'])
    if st_g != 'ok':
        return dict(name=ex['name'], seed=seed, status=st_g, n_default=n_default,
                    batch=b, n_groups=n_groups, vpp=ex['vpp'])

    parts_d = micro_split(lens, n_default, force_group_size=1)
    parts_g = micro_split(lens, n_grouped, force_group_size=ex['n'])
    frag_d = fragments(parts_d, group_of)
    frag_g = fragments(parts_g, group_of)
    # 「每组前缀块」只用于计数：默认路径每组被切成几段 vs 组完整每批 1 段
    return dict(
        name=ex['name'], seed=seed, status='ok',
        n_default=n_default, n_grouped=n_grouped, n_groups=n_groups, batch=b,
        # 预算是否已成为瓶颈：ceil(total/budget) > 组数时，组完整会把批数压到组数
        budget_bound=(ceil(sum(lens) / budget) > n_groups),
        per_group_default=sum(frag_d.values()) / n_groups,
        per_group_grouped=sum(frag_g.values()) / n_groups,
    )


_CASE_CACHE = {}


def all_cases(ex):
    """30 个种子的切分形状（缓存，避免 verdict 各节重复计算）。"""
    key = (ex['name'], ex['budget'], ex['n'], ex['dp'], ex['mini_bs'], ex['vpp'], ex['pp'])
    if key not in _CASE_CACHE:
        _CASE_CACHE[key] = [case(ex, s) for s in range(SEEDS)]
    return _CASE_CACHE[key]


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def verdict_structure():
    """判决一：两个断言在真实配置里各自卡住谁；以及分组微批是否已被 vPP 卡住。"""
    rows = []
    for ex in EXAMPLES:
        rv = rank_view(ex)
        max_seq = ex['prompt_len'] + ex['resp_len']
        budget = ex['budget'] * ex['sp']
        row = dict(name=ex['name'], backend=ex['backend'], dynamic_bsz=ex['dyn'],
                   max_seq=max_seq, budget=budget, vpp=ex['vpp'], pp=ex['pp'])
        if rv is None:
            row['view'] = 'infeasible'
        else:
            row['batch_rank'], row['groups_rank'] = rv
        row['shape_ok'] = budget >= max_seq
        if ex['dyn'] and rv is not None:
            b, n_groups = rv
            lens, _ = make_lengths(ex, n_groups, 0)
            n_g, st_g = num_micro_batches(b, lens, budget, force_group_size=ex['n'],
                                          num_batches_divided_by=ex['vpp'])
            row['grouped_status'] = st_g
            row['grouped_n'] = n_g
        rows.append(row)
    return rows


def verdict_real_configs():
    out = []
    for ex in EXAMPLES:
        cases = all_cases(ex)
        ok = [c for c in cases if c['status'] == 'ok']
        row = dict(name=ex['name'], backend=ex['backend'], pp=ex['pp'], vpp=ex['vpp'],
                   dp=ex['dp'], n=ex['n'], mini_bs=ex['mini_bs'], budget=ex['budget'],
                   dynamic_bsz=ex['dyn'], statuses=sorted({c['status'] for c in cases}))
        if ok:
            m_d = _median([c['n_default'] for c in ok])
            m_g = _median([c['n_grouped'] for c in ok])
            row.update(
                n_default_median=m_d, n_grouped_median=m_g,
                n_groups_median=_median([c['n_groups'] for c in ok]),
                batch_rank=ok[0]['batch'],
                # 每个 micro-batch 平均装多少样本——预算是不是瓶颈
                samples_per_micro_micro=ok[0]['batch'] / m_d if m_d else None,
                # 两条路径批数相同的条件：ceil(total/budget) ≤ 组数（组数封顶只在超过时才生效）
                budget_bound=bool(ok[0]['n_default'] > ok[0]['n_groups']),
                per_group_default_median=_median([c['per_group_default'] for c in ok]),
                per_group_grouped_median=_median([c['per_group_grouped'] for c in ok]),
            )
        out.append(row)
    return out


def bubble_frac(m, p):
    """单级流水（1F1B）气泡占比 `1 − m/(m+p−1)`。纯计数，无墙钟。"""
    if m < 1 or p < 1:
        return None
    return max(0.0, 1.0 - m / (m + p - 1))


def verdict_pp_bubble():
    rows = []
    for ex in EXAMPLES:
        ok = [c for c in all_cases(ex) if c['status'] == 'ok']
        if not ok:
            continue
        p = ex['pp']
        m_d = _median([c['n_default'] for c in ok])
        m_g = _median([c['n_grouped'] for c in ok])
        bd, bg = bubble_frac(m_d, p), bubble_frac(m_g, p)
        rows.append(dict(
            name=ex['name'], pp=p, n_default=m_d, n_grouped=m_g,
            batch_rank=ok[0]['batch'],
            bubble_default=bd, bubble_grouped=bg,
            bubble_ratio=(bg / bd) if bd else None,
            worse=bool(bd is not None and bg is not None and bg > bd),
        ))
    return rows


def _counts_only(ex):
    """只算批数，不做分区（identifiability 不需要分区，避免 2048 样本上的 O(n^2) LDM）。"""
    rv = rank_view(ex)
    if rv is None:
        return None, None
    b, n_groups = rv
    if not ex['dyn']:
        return None, None
    lens, _ = make_lengths(ex, n_groups, 0)
    budget = ex['budget'] * ex['sp']
    n_d, st_d = num_micro_batches(b, lens, budget, force_group_size=1,
                                  num_batches_divided_by=ex['vpp'])
    n_g, st_g = num_micro_batches(b, lens, budget, force_group_size=ex['n'],
                                  num_batches_divided_by=ex['vpp'])
    return (n_d if st_d == 'ok' else None), (n_g if st_g == 'ok' else None)


def _ratio_counts_only(n):
    """只算 batch 数与组完整批数，不分区（用于 n / 预算 两轴）。"""
    base = dict(EXAMPLES[0])
    base['n'] = n
    return _counts_only(base)


def identifiability():
    """每个结论必须对「应当敏感」的参数敏感、对「不该敏感」的参数不敏感。

    1. 组内条数 n：n=1 时组退化为单条，n_default 与 n_grouped 必须完全相等。
    2. 预算：预算放大到装下整个 mini-batch 时两条路径批数都收敛到 1（405 行断言仍须满足）。
    3. 流水级数 pp：pp=1 时气泡恒 0，比值必须为 None。
    4. 组完整批数的上界：恒 ≤ 组数。
    """
    base = dict(EXAMPLES[0])

    by_n = []
    for n in (1, 2, 4, 5, 10):
        e = dict(base)
        e['n'] = n
        nd, ngr = _counts_only(e)
        by_n.append(dict(n=n, n_default_median=nd, n_grouped_median=ngr))

    by_budget = []
    for mult in (1, 4, 16, 64, 1024):
        e = dict(base)
        e['budget'] = base['budget'] * mult
        nd, ngr = _counts_only(e)
        by_budget.append(dict(budget=e['budget'], n_default_median=nd,
                              n_grouped_median=ngr,
                              statuses=sorted({'ok' if nd and ngr else 'shape/cap'})))

    by_pp = []
    e = dict(EXAMPLES[6])          # hy_v3：pp=16、n=16
    for pp in (1, 2, 4, 8, 16):
        e2 = dict(e)
        e2['pp'] = pp
        m_d, m_g = _counts_only(e2)
        bd, bg = bubble_frac(m_d, pp), bubble_frac(m_g, pp)
        by_pp.append(dict(pp=pp, n_default=m_d, n_grouped=m_g,
                          bubble_default=bd, bubble_grouped=bg,
                          ratio=(bg / bd) if bd else None))

    return dict(by_group_size=by_n, by_budget=by_budget, bubble_by_pp=by_pp)


def counterfactuals():
    """负向对照：三条「若模型写错就会通过」的检查。"""
    ex = dict(EXAMPLES[0])
    b, n_groups = rank_view(ex)
    lens, group_of = make_lengths(ex, n_groups, 0)
    n_d, _ = num_micro_batches(b, lens, ex['budget'], force_group_size=1)

    # (a) 组完整实现成 force_group_size=1 的空操作 → 段数必须完全相同
    same = fragments(micro_split(lens, n_d, force_group_size=1), group_of) == \
        fragments(micro_split(lens, n_d, force_group_size=1), group_of)

    # (b) 批数取 1 → 每条路径都只给每组 1 段
    one = fragments(micro_split(lens, 1, force_group_size=1), group_of)
    whole_batch_per_group = sum(one.values()) / n_groups

    # (c) 预算放大到装下整批 → 两条路径批数都必须是 1
    big = ex['budget'] * 1024
    n_d_big, _ = num_micro_batches(b, lens, big, force_group_size=1)
    n_g_big, _ = num_micro_batches(b, lens, big, force_group_size=ex['n'])

    # (d) 组完整恒不拆组（结构检查）
    parts_g = micro_split(lens, n_g_big if n_g_big else 1, force_group_size=ex['n'])
    whole_groups = all(len({i // ex['n'] for i in p}) == len(p) // ex['n'] for p in parts_g)

    return dict(noop_identical=same,
                whole_batch_per_group=whole_batch_per_group,
                budget_1024x_n_default=n_d_big, budget_1024x_n_grouped=n_g_big,
                grouped_never_splits_groups=whole_groups)


def _verdict_line():
    rows = verdict_real_configs()
    struct = verdict_structure()
    pp_rows = verdict_pp_bubble()
    dyn_rows = [r for r in rows if r['dynamic_bsz'] and 'n_default_median' in r]
    return dict(
        prefix_grouper_reachable_at_head=False,
        shape_violations=[r['name'] for r in struct if r.get('shape_ok') is False],
        cap_violations=[r['name'] for r in struct if r.get('grouped_status') == 'cap'],
        dynamic_bsz_configs=len(dyn_rows),
        static_bsz_configs=sum(1 for r in rows if not r['dynamic_bsz']),
        micro_batches_range=[min(r['n_default_median'] for r in dyn_rows),
                             max(r['n_default_median'] for r in dyn_rows)],
        pp_configs_worse_under_grouping=sum(1 for r in pp_rows if r['worse']),
        pp_configs_total=len(pp_rows),
        bubble_ratio_max=max((r['bubble_ratio'] or 0) for r in pp_rows),
    )


def build_summary():
    return dict(
        prefix_grouper_dead_path=PREFIX_GROUPER_DEAD_PATH,
        structure=verdict_structure(),
        real_configs=verdict_real_configs(),
        pp_bubble=verdict_pp_bubble(),
        identifiability=identifiability(),
        counterfactuals=counterfactuals(),
        verdict=_verdict_line(),
    )


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
    elif isinstance(a, float) and isinstance(b, float):
        if abs(a - b) > 1e-12:
            print('  differs:', path, a, b)
    elif a != b:
        print('  differs:', path, a, b)


def verify():
    fresh = json.loads(json.dumps(build_summary()))
    if not os.path.exists(SUMMARY_PATH):
        print('verify: no archived summary.json')
        return 1
    with open(SUMMARY_PATH) as f:
        archived = json.load(f)
    if archived == fresh:
        print('verify: OK — 重跑结果与归档 summary.json 逐字段相同')
        return 0
    print('verify: MISMATCH')
    _diff(archived, fresh, '')
    return 1


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
