#!/usr/bin/env python3
"""PP 微批边界成本：`use_prefix_grouper` 下跳过"小块排两端"到底损失多少（rollout 专项，2026-11-02 M1）。

支撑 notebook/rollout-performance-research.md §13.33，接续 §13.32 登记的残留
「PP 微批边界的影响未量化（`use_prefix_grouper` 下刻意跳过『小微批排两端』）」。

**问题从哪来**：verl `_balance_batch` 末尾有一段
「Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel」，
它把每个 partition 内部按载荷升序，再重排成 `[::2] + [1::2][::-1]`——小块排两端、大块居中。
但这段被 `if not getattr(self, "use_prefix_grouper", False)` 包住：**打开 prefix grouper 就不排了**。
上一批把这个"不排"登记为未量化的成本；本批把它量出来。

------------------------------------------------------------------
第 0 步：语义核清（只读源码，决定后面算得对不对）
------------------------------------------------------------------
四个事实串起来才成立：

  ① `_balance_batch` 产出 dp_size 个 partition，每个 partition 是一组**下标**。
  ② `global_idx = [j for partition in lst for j in partition]` —— partition 的顺序就是 rank 内顺序。
  ③ dispatch 用 `chunk(dp_size)` **连续等分** `global_idx`（`decorator.py:71-88`），
     第 r 个 rank 拿到的样本序列**恰好**是 `lst[r]`。
  ④ rank 内再被 `train_mini_batch` 按 `mini_batch_size_per_gpu` 连续切片成 PP micro-batch
     （`engine_workers.py:295-309`；`make_iterator` 的 DataLoader 未设 `shuffle` ⇒ 顺序切）。

于是"跳过重排"= 每个 partition 内部不再经历 ③ 之后的那次重排。
**两个排法的成员集合完全相同**，只差 rank 内顺序——这是严格配对比较的前提。

▶ 两处本批自查推翻的结论，先写下来，因为它们决定了下面所有读数的含义：

  错误 1（口径不可比）：初版把"把 partition 当不可分块做 `[::2]+[1::2][::-1]`"当成恢复旧语义，
     得到"新版是旧版的 0.27–0.39 倍"这种三倍级差异。逐个 rank 打印元素个数后才发现，
     旧写法下**每个 rank 只有 1 个元素**（rank 内部成了一条权重），与新版每 rank `n_per` 个
     根本不是同一个数量，相除没有意义。三倍级差异本身就是该停下来核对的警号。

  错误 2（动力学方向猜反）：本批起初假设"小块排两端"是为了让每个 micro-batch 的最大载荷变小，
     于是预期去掉它会变差。实测方向相反——**去掉重排反而更好**（见 PART2）。
     原因在 PART3：这段重排优化的不是"每片最大载荷"，而是"把大块排到中段"，
     而大块被排进中段后**会挤进同一个 stage**，反而放大了 stage 间的最大载荷差。

------------------------------------------------------------------
PART1：PP 代价的三种载荷模型（先把模型说清楚，再谈差异）
------------------------------------------------------------------
每个 rank 的样本按 micro-batch 连续切片，一个 micro-batch 一个 PP 槽位。三种模型：

  max          槽位时间 = 该片最大单条载荷。模拟"每片都要等最重的那条"的上界。
  sum          槽位时间 = 该片载荷之和。模拟"载荷可叠加、不按 max 计"。
  sum_plus_max 两者相加。模拟"计算量（∝ sum）+ 少量等待（∝ max）"的中性组合。

**关键事实：三种模型下"去掉重排"的方向一致（都变好），但量级差两个数量级。**
所以正文只能报"方向 + 模型区间"，不能只挑一个模型报数——挑 max 会把 1.01 倍的效应
说成 1.65 倍。

------------------------------------------------------------------
PART2：主读数（同成员、只换 rank 内顺序）
------------------------------------------------------------------
五种排法，成员集合完全相同：
  cur           当前实装：partition 内**先按载荷升序**再 `[::2]+[1::2][::-1]`
  sorted        对照基准：升序但不交错（即"不重排"的最近替身）
  desc          负向对照：降序（最重的排最前）
  plain         旧语义：不排序、直接在原顺序上 `[::2]+[1::2][::-1]`
  orig          完全不排（保留 LDM 分区后的原始顺序）

------------------------------------------------------------------
PART3：机制（为什么去掉重排更好）
------------------------------------------------------------------
不靠"看起来合理"的叙述，用两个可观测量钉住：
  stage_max_spread   每个 stage 内最大单条载荷 / 全 rank 平均单条载荷 —— 越大说明"某级 stage
                     全是重活"越严重。这是"大块居中会挤进同一 stage"的直接度量。
  piece_max_cv       各 micro-batch 的最大载荷的离散度 —— 这是那段重排**想**优化的量。

若"去掉重排更好"成立，必须满足：stage_max_spread 下降（挤堆缓解），而 piece_max_cv 上升或持平
（说明它确实没在优化这个量）。两个量同时给，读者才能判断该信哪一侧。

------------------------------------------------------------------
PART4：识别性检验（§13.31／§13.32 同一条检验，先验后量）
------------------------------------------------------------------
  cv（组间／组内长度离散度）0.2 → 1.2：效应应随 cv 增大（排序在齐整负载上没价值）
  m（每 rank micro-batch 数）1 → 8：效应应随 m 变化（m=1 时任何排法都无差别）
  对照：恒定指标必须判零敏感（防检验太松）；组均值离散度过 cv 必须判敏感（防太严）
  **m=1 必须逐位相等**——这是排法"只在 PP 微批边界存在时有意义"的硬判据。

墙钟一律不报。虚拟单位，不可与任何论文报值相除（§19.1 规则一）。

用法：
    python3 pp_microbatch_boundary.py             # 全量重跑并覆盖 summary.json
    python3 pp_microbatch_boundary.py --verify    # 重跑并与归档逐字段比对
    python3 -m unittest test_pp_microbatch_boundary -v
"""

import json
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY_PATH = os.path.join(HERE, 'summary.json')

# LDM 复刻复用 §13.30／§13.31／§13.32 的实现（已与真实 verl 在 180 组输入上交叉验证分区一致），
# 不在本仓库出现第四份会分叉的副本。
_SIBLING_TRADEOFF = os.path.join(os.path.dirname(HERE), 'rollout-group-split-tradeoff')
if _SIBLING_TRADEOFF not in sys.path:
    sys.path.insert(0, _SIBLING_TRADEOFF)

from group_split_tradeoff import _karmarkar_karp, workload_int64  # noqa: E402

SEEDS = list(range(30))
GROUP_SIZE = 8                 # rollout.n（与 §13.30 L4／§13.31／§13.32 一致）
N_GROUPS = 32                  # 一批 prompt 数
MEAN_LEN = 1000.0              # 单条序列长度均值（构造参数）
CV_LEVELS = (0.2, 0.6, 1.2)    # 组间／组内长度离散度（与 §13.30 L4 同档）
DP_SIZES = (2, 4, 8)
MB_PER_RANK = (1, 2, 4, 8)     # 每 rank 的 PP micro-batch 数 m
ARMS = ('cur', 'sorted', 'desc', 'plain', 'orig')
COST_MODELS = ('max', 'sum_plus_max', 'sum')

IDENTIFIABILITY_REL_EPS = 0.01  # 相对极差 > 1% 才算"有识别性"（§13.32 同一条阈值）


# ---------------------------------------------------------------------------
# PART1：排法、切片与三种载荷模型
# ---------------------------------------------------------------------------
def interleave(vals):
    """verl 的 `partition[::2] + partition[1::2][::-1]`。"""
    return vals[::2] + vals[1::2][::-1]


def rank_order(arm, weights_in_partition):
    """给定一个 rank 的成员载荷（LDM 分区后的原始顺序），返回该 rank 的样本序列。"""
    w = list(weights_in_partition)
    if arm == 'cur':
        return interleave(sorted(w))
    if arm == 'sorted':
        return sorted(w)
    if arm == 'desc':
        return sorted(w)[::-1]
    if arm == 'plain':
        return interleave(w)
    if arm == 'orig':
        return w
    raise ValueError('unknown arm: %r' % (arm,))


def micro_batch_shapes(n_items, m):
    """把一个 rank 的 n_items 条按 m 个 micro-batch 连续切（verl 的 dataloader 语义）。"""
    base, rem = n_items // m, n_items % m
    shapes, pos = [], 0
    for j in range(m):
        k = base + (1 if j < rem else 0)
        if k <= 0:
            continue
        shapes.append((pos, pos + k))
        pos += k
    return shapes


def piece_cost(chunk, model):
    if model == 'max':
        return max(chunk)
    if model == 'sum':
        return sum(chunk)
    if model == 'sum_plus_max':
        return sum(chunk) + max(chunk)
    raise ValueError('unknown model: %r' % (model,))


def rank_cost(rank_seq, m, model):
    return sum(piece_cost(rank_seq[a:b], model) for a, b in micro_batch_shapes(len(rank_seq), m))


def batch_cost(parts, arm, m, model):
    return sum(rank_cost(rank_order(arm, p), m, model) for p in parts)


# ---------------------------------------------------------------------------
# 数据构造（与 §13.30 L4／§13.31／§13.32 同一档）
# ---------------------------------------------------------------------------
def group_lens(rng, n_groups=N_GROUPS, n_per=GROUP_SIZE, mean_len=MEAN_LEN, cv=0.6):
    """一批组：先抽组均值，再抽组内每条的载荷；组均值与组内都随 cv 缩放。"""
    out = []
    for _ in range(n_groups):
        mu = max(1.0, rng.gauss(mean_len, mean_len * cv))
        out.extend(max(1.0, rng.gauss(mu, mu * cv * 0.5)) for _ in range(n_per))
    return out


def partition_batch(lens, dp_size):
    """verl 默认路径（逐条 LDM）把一批划到 dp_size 个 rank；返回每个 rank 的成员载荷（原顺序）。"""
    weights = [workload_int64(int(x)) for x in lens]
    parts = _karmarkar_karp(weights, dp_size, equal_size=True)
    return [[weights[i] for i in p] for p in parts]


def seed_parts(seed, cv, dp_size):
    return partition_batch(group_lens(random.Random(seed * 97 + int(cv * 10) + dp_size), cv=cv),
                           dp_size)


# ---------------------------------------------------------------------------
# PART2：主读数
# ---------------------------------------------------------------------------
def order_effect(cv, dp_size, m, model='max', seeds=SEEDS):
    """各排法相对 `sorted`（不重排的最近替身）的整批代价比。"""
    per_arm = {arm: [] for arm in ARMS}
    for seed in seeds:
        parts = seed_parts(seed, cv, dp_size)
        for arm in ARMS:
            per_arm[arm].append(batch_cost(parts, arm, m, model))
    base = per_arm['sorted']
    out = {
        'cv': cv, 'dp_size': dp_size, 'micro_batches_per_rank': m, 'model': model,
        'rank_items': len(parts[0]),
        'divisible': len(parts[0]) % m == 0,
    }
    for arm in ARMS:
        ratios = [a / b for a, b in zip(per_arm[arm], base) if b]
        out[arm] = statistics.mean(ratios)
        out[arm + '_se'] = statistics.stdev(ratios) / len(ratios) ** 0.5 if len(ratios) > 1 else 0.0
    return out


def order_effect_table(model='max'):
    return [order_effect(cv, dp, m, model)
            for cv in CV_LEVELS for dp in DP_SIZES for m in MB_PER_RANK]


# ---------------------------------------------------------------------------
# PART3：机制
# ---------------------------------------------------------------------------
def stage_max_spread(parts, arm, m):
    """各 stage（第 j 个 micro-batch）的最大载荷，相对全 rank 平均单条载荷的比值均值。

    > 1 且越大 ⇒ "某级 stage 集中装重活"越严重。这是"大块挤进同一 stage"的直接度量。
    """
    rank_means = [statistics.mean(p) for p in parts]
    ref = statistics.mean(rank_means)
    spreads = []
    for p in parts:
        seq = rank_order(arm, p)
        shapes = micro_batch_shapes(len(seq), m)
        for a, b in shapes:
            spreads.append(max(seq[a:b]) / ref)
    return statistics.mean(spreads)


def piece_max_cv(parts, arm, m):
    """各 micro-batch 最大载荷之间的离散度（这段重排**想**优化的量）。"""
    cvs = []
    for p in parts:
        seq = rank_order(arm, p)
        maxes = [max(seq[a:b]) for a, b in micro_batch_shapes(len(seq), m)]
        if len(maxes) > 1 and statistics.mean(maxes) > 0:
            cvs.append(statistics.pstdev(maxes) / statistics.mean(maxes))
    return statistics.mean(cvs) if cvs else 0.0


def mechanism_table(cv=0.6, dp_size=8, m=4, seeds=SEEDS):
    rows = {arm: {'stage_max_spread': [], 'piece_max_cv': [], 'cost': []} for arm in ARMS}
    for seed in seeds:
        parts = seed_parts(seed, cv, dp_size)
        for arm in ARMS:
            rows[arm]['stage_max_spread'].append(stage_max_spread(parts, arm, m))
            rows[arm]['piece_max_cv'].append(piece_max_cv(parts, arm, m))
            rows[arm]['cost'].append(batch_cost(parts, arm, m, 'max'))
    out = {}
    for arm, d in rows.items():
        out[arm] = {
            'stage_max_spread': statistics.mean(d['stage_max_spread']),
            'piece_max_cv': statistics.mean(d['piece_max_cv']),
            'cost_max': statistics.mean(d['cost']),
        }
    base = out['sorted']
    for arm in out:
        out[arm]['cost_over_sorted'] = out[arm]['cost_max'] / base['cost_max']
    return {'cv': cv, 'dp_size': dp_size, 'micro_batches_per_rank': m, 'arms': out}


# ---------------------------------------------------------------------------
# PART4：识别性检验
# ---------------------------------------------------------------------------
def relative_spread(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    spread = max(vals) - min(vals)
    ref = abs(statistics.mean(vals))
    return (spread / ref) if ref > 0 else float('inf')


def _metric_cur_ratio(rng, cv):
    """在 (dp=8, m=4, 该 cv) 下 cur 相对 sorted 的代价比（model=max）。"""
    parts = partition_batch(group_lens(rng, cv=cv), 8)
    base = batch_cost(parts, 'sorted', 4, 'max')
    return batch_cost(parts, 'cur', 4, 'max') / base if base else None


def _metric_cur_ratio_at_m(rng, m):
    parts = partition_batch(group_lens(rng, cv=0.6), 8)
    base = batch_cost(parts, 'sorted', m, 'max')
    return batch_cost(parts, 'cur', m, 'max') / base if base else None


def _metric_constant(_rng, _level):
    return 1.0


def _metric_group_dispersion(rng, cv):
    """对照组（必须判敏感）：cv 越大，组均值的离散度越大。"""
    lens = group_lens(rng, n_groups=8, n_per=GROUP_SIZE, cv=cv)
    means = [statistics.mean(lens[i * GROUP_SIZE:(i + 1) * GROUP_SIZE]) for i in range(8)]
    return statistics.pstdev(means) / statistics.mean(means)


def _scan(metric_fn, levels, seed_tag):
    vals = []
    for lv in levels:
        for s in (0, 1, 2, 3):
            v = metric_fn(random.Random(s * 131 + seed_tag), lv)
            if v is not None:
                vals.append(v)
    rel = relative_spread(vals)
    return {
        'levels': list(levels),
        'values': vals,
        'min': min(vals) if vals else None,
        'max': max(vals) if vals else None,
        'relative_spread': rel,
        'sensitive': bool(rel is not None and rel > IDENTIFIABILITY_REL_EPS),
        'threshold': IDENTIFIABILITY_REL_EPS,
    }


def identifiability():
    return {
        'cv_driver': dict(_scan(_metric_cur_ratio, CV_LEVELS, 7), param='cv'),
        'm_driver': dict(_scan(_metric_cur_ratio_at_m, MB_PER_RANK, 11), param='micro_batches_per_rank'),
        'control_group_dispersion': dict(_scan(_metric_group_dispersion, CV_LEVELS, 7), param='cv',
                                         note='对照：真信号必须能被读到（防检验太严）'),
        'control_constant': dict(_scan(_metric_constant, CV_LEVELS, 7), param='cv',
                                 note='对照：恒定指标必须判零敏感（防检验太松）'),
    }


def single_micro_batch_identity():
    """m=1 ⇒ 每 rank 只有 1 个 micro-batch ⇒ 排法不应产生任何差异（逐位相等）。"""
    rows = []
    for cv in CV_LEVELS:
        for dp in DP_SIZES:
            parts = seed_parts(0, cv, dp)
            costs = {arm: rank_cost(rank_order(arm, parts[0]), 1, 'max') for arm in ARMS}
            rows.append({'cv': cv, 'dp_size': dp, 'distinct_costs': len(set(costs.values()))})
    return {'rows': rows, 'all_identical': all(r['distinct_costs'] == 1 for r in rows)}


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def _round(obj, nd=10):
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, nd) for v in obj]
    return obj


def _summarize(model, table):
    cur = [r['cur'] for r in table]
    desc = [r['desc'] for r in table]
    plain = [r['plain'] for r in table]
    by_cv, by_m = {}, {}
    for cv in CV_LEVELS:
        vals = [r['cur'] for r in table if r['cv'] == cv]
        by_cv[str(cv)] = {'min': min(vals), 'max': max(vals), 'mean': statistics.mean(vals)}
    for m in MB_PER_RANK:
        vals = [r['cur'] for r in table if r['micro_batches_per_rank'] == m]
        by_m[str(m)] = {'min': min(vals), 'max': max(vals), 'mean': statistics.mean(vals)}
    return {
        'model': model,
        'cur_min': min(cur), 'cur_max': max(cur), 'cur_mean': statistics.mean(cur),
        'desc_min': min(desc), 'desc_max': max(desc), 'desc_mean': statistics.mean(desc),
        'plain_min': min(plain), 'plain_max': max(plain), 'plain_mean': statistics.mean(plain),
        'all_cells_cur_worse': all(v > 1.0 for v in cur),
        'all_cells_desc_better_or_equal': all(v <= 1.0 + 1e-12 for v in desc),
        'by_cv': by_cv, 'by_m': by_m,
    }


def build_summary():
    per_model = {}
    for model in COST_MODELS:
        table = order_effect_table(model)
        per_model[model] = {
            'table': table,
            'summary': _summarize(model, table),
        }
    return _round({
        'experiment': 'PP 微批边界成本：use_prefix_grouper 下跳过“小块排两端”的代价',
        'doc': 'notebook/rollout-performance-research.md §13.33（接续 §13.32 残留）',
        'config': {
            'seeds': len(SEEDS),
            'group_size': GROUP_SIZE,
            'n_groups': N_GROUPS,
            'cv_levels': list(CV_LEVELS),
            'dp_sizes': list(DP_SIZES),
            'micro_batches_per_rank': list(MB_PER_RANK),
            'arms': list(ARMS),
            'cost_models': list(COST_MODELS),
        },
        'per_model': per_model,
        'mechanism': mechanism_table(),
        'identifiability': identifiability(),
        'single_micro_batch_identity': single_micro_batch_identity(),
        'anchors': check_anchors(),
    })


# ---------------------------------------------------------------------------
# 源码锚点：结论依赖的几行代码还在不在（上游改了必须明确报失效，不能静默沿用）
# ---------------------------------------------------------------------------
SNAPSHOT_ROOT = os.path.join(HERE, 'source-snapshots', 'verl', 'main')

ANCHORS = (
    ('verl/trainer/ppo/ray_trainer.py',
     '# Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.\n'
     '        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping\n'
     '        if not getattr(self, "use_prefix_grouper", False):\n'
     '            for idx, partition in enumerate(global_partition_lst):\n'
     '                partition.sort(key=lambda x: (workload_lst[x], x))\n'
     '                ordered_partition = partition[::2] + partition[1::2][::-1]',
     'PP 气泡重排被 use_prefix_grouper 跳过'),
    ('verl/trainer/ppo/ray_trainer.py',
     'global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])',
     'partition 的顺序 = rank 内顺序'),
    ('verl/single_controller/base/decorator.py',
     'chunked_arg = BatchData(arg).chunk(chunks=chunks)',
     'dispatch 按 chunk 连续等分'),
    ('verl/workers/engine_workers.py',
     'mini_batch_size_per_gpu = mini_batch_size // self.engine.get_data_parallel_size()',
     'rank 内 micro-batch 大小 = ppo_mini_batch_size / dp'),
    ('verl/utils/seqlen_balancing.py',
     'def get_seqlen_balanced_partitions(seqlen_list: list[int], k_partitions: int, equal_size: bool):',
     'LDM 划分入口（本实验复刻已与它逐位对齐）'),
)


def check_anchor(rel_path, snippet):
    path = os.path.join(SNAPSHOT_ROOT, rel_path)
    if not os.path.exists(path):
        return False, 'no snapshot at %s' % rel_path
    with open(path, encoding='utf-8', errors='replace') as f:
        text = f.read()
    norm = lambda s: '\n'.join(line.rstrip() for line in s.split('\n'))
    if norm(snippet) in norm(text):
        return True, 'hit'
    first = snippet.split('\n')[0].strip()
    if first and first in text:
        return True, 'hit-first-line-only (行号可能漂移，需人工确认语义)'
    return False, 'MISSING'


def check_anchors():
    out = []
    for rel_path, snippet, what in ANCHORS:
        ok, detail = check_anchor(rel_path, snippet)
        out.append({'file': rel_path, 'what': what, 'found': ok, 'detail': detail})
    return {'all_found': all(a['found'] for a in out), 'items': out}


def main(argv):
    summary = build_summary()
    if '--verify' in argv:
        if not os.path.exists(SUMMARY_PATH):
            print('no archived summary.json to verify against')
            return 2
        with open(SUMMARY_PATH, encoding='utf-8') as f:
            archived = json.load(f)
        if _round(archived) == summary:
            print('verify OK: 逐字段与归档相同')
            return 0
        print('verify FAILED: 与归档不一致')
        for key in sorted(set(archived) | set(summary)):
            if archived.get(key) != summary.get(key):
                print('  differs:', key)
        return 1
    with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, sort_keys=True)
        f.write('\n')
    print('写入 %s' % SUMMARY_PATH)
    for model in COST_MODELS:
        s = summary['per_model'][model]['summary']
        print('%-13s cur/sorted %.4f–%.4f（均值 %.4f）| desc %.3f–%.3f | plain %.3f–%.3f'
              % (model, s['cur_min'], s['cur_max'], s['cur_mean'],
                 s['desc_min'], s['desc_max'], s['plain_min'], s['plain_max']))
    print('m=1 逐位相等: %s' % summary['single_micro_batch_identity']['all_identical'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
