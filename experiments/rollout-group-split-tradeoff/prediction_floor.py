#!/usr/bin/env python3
"""截断本身就是组的长度预测：它把"组最大"预测得**精确**，因此对 `R` 零空间（第十七批，2026-11-04 M1）。

支撑 notebook/rollout-performance-research.md §13.30 第十七批、TS-001 语义面缺口
「长度预测误差」。

**问题。** §11.5 把语义面剩余缺口写成「长度预测误差／分桶／迁移／抢占／KV 搬运」，
几轮批次的结论都是"要另建模型"。其中**长度预测**这一项在真实负载上已经有一条免费
预测器：**响应撞到 `max_response` 的截断本身就说明"这条至少这么长"**。§13.50 量过
cap256 档 45%–61% 的响应卡在 256，从没被换算成"多少条路可排"。

**本批的答案（先说结论，再看证据）。** 这条预测器对**组臂**要的那个量——
`max(g)`，组臂的物理耗时——是**精确**的，不是下界：

    撞顶记 c、其余记真值 ⇒ max(预测) == max(g)  **恒等**（四条负载 480/480 组，0 误差）

原因是结构性的：组内任一条撞顶 ⇒ `max(g) = c` ⇒ 预测也给 `c`；没有一条撞顶 ⇒ 所有
值都是真值 ⇒ 预测仍是 `max(g)`。**这条恒等式与撞顶比例无关、与模型无关、与 cap 无关**
（cap1024 只 0.03%–0.65% 撞顶，同样 480/480 精确）。

⇒ 对 `R`（组粒度代价，定义为"整组原子 vs 逐条连续"的 makespan 比）而言，长度预测
**没有剩余空间**：`R` 里唯一依赖预测的量就是组臂的排队序，而那个序用截断上界就排得
和用真值一模一样（四条负载 × 30 个种子，组序 30/30 全同）。所以 §11.5 的"长度预测
误差"这一项，在**组粒度**这个口径上可以划掉；真正还依赖逐条预测精度的是**逐条路径**，
而它在 `R` 的分子分母里都不出现。

**必须并列的三条边界（否则这个结论会被读大）：**

  ① `grouping_tax` 原定义里两臂吃**同一份顺序**，于是 `R` 里根本没有"预测"这一项，
     预测器换哪个都一样——这本身就是"预测对 `R` 无杠杆"的一半证据。但真实系统的
     两臂各按自己看得见的信息排（组臂只看组级预测、轨迹臂看得见逐条真值），所以本批
     **两个口径都报**，并用 `separate` 那一档证明组臂排错时 `R` 确实会动（见
     `order_sensitivity`：故意把组序打乱 12 次，`R` 的极差就是这笔杠杆的上界）。
  ② **`floor == full` 是恒等式，不是经验结论**——读数上两者永远相等，所以"给读数
     做统计检验"没有意义；有信息量的是**恒等式成立的条件**：撞顶的那条要求组内最长
     那条确实报了满 c（本批是 `response_tokens >= c` 的截断，`finish_reason=length`）。
     若某个引擎把撞顶样本截在 c−1 或按字节计，恒等式就不再精确。
  ③ 本批只覆盖 **cap256 / cap1024 两档 gsm8k 负载**（两模型）；`R` 之外的口径
     （跨副本负载均衡、KV 搬运）不在范围内。

**用法：**
    python3 prediction_floor.py            # 重算并写 prediction-floor.json（约 2 分钟）
    python3 prediction_floor.py --verify   # 与归档逐字段比对
"""

import gzip
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'tools', 'paired-diff'))
import paired_diff  # noqa: E402

from group_split_tradeoff import (POLICY_FIFO, POLICY_LOF, _makespan,
                                  _makespan_by_policy)

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_DIR = os.path.join(HERE, '..', 'rollout-length-structure-real', 'raw')
OUT_PATH = os.path.join(HERE, 'prediction-floor.json')

SEEDS = list(range(30))
N_GROUPS = 16
N_PER = 8
TRIALS = 12
N_ID_GROUPS = 8          # 自证档：容量恰好是 64 槽，与参照实现同口径
N_ID_BACKLOG = 16
BACKLOGS = (1, 4, 16)

REAL_ARMS = (
    ('qwen15b_cap256', 'qwen15b_gsm8k_n8.jsonl.gz'),
    ('qwen15b_cap1024', 'qwen15b_gsm8k_n8_cap1024.jsonl.gz'),
    ('qwen05b_cap256', 'qwen05b_gsm8k_n8.jsonl.gz'),
    ('qwen05b_cap1024', 'qwen05b_gsm8k_n8_cap1024.jsonl.gz'),
)


def arm_cap(label):
    """该档的 max_response：截断必须按它判定，不能一律用 256。"""
    return 1024.0 if 'cap1024' in label else 256.0


def load_real_groups(path):
    """按 prompt 聚组，返回 [[len, ...], ...]（与 `real_lengths_policy` 同口径）。"""
    groups = {}
    with gzip.open(path, 'rt') as f:
        for line in f:
            d = json.loads(line)
            groups.setdefault(d['prompt_idx'], []).append(float(d['response_tokens']))
    return [groups[k] for k in sorted(groups)]


# ---------------------------------------------------------------- 三档预测列

def floor_predict(g, c):
    """撞顶记 c、其余记真值——只使用"这条撞顶了"这一个比特，不需要任何模型。"""
    return [c if x > c else x for x in g]


def full_predict(g, _c=None):
    """全部真值（不可实现的天花板）。"""
    return list(g)


PREDICTORS = (
    ('real', None),
    ('floor', floor_predict),
    ('full', full_predict),
)


def degraded_predict(g, c):
    """**负向对照**：把预测故意打坏，看 `R` 会不会动。

    只保留"撞顶"这一个比特，其余一律记 **1**——相当于一个只会说"这条短"的预测器。
    若这一档的 `R` 与 `floor` 相同，就说明本批的 `R` 对预测完全无反应（度量失效），
    "预测对 `R` 零空间"也就不能成立。**它必须动**，这是判据的敏感性检验。
    """
    return [c if x >= c else 1.0 for x in g]



# ---------------------------------------------------------------- 恒等式（本批的核心读数）

def group_max_identity(groups, cap):
    """`max(floor_predict(g)) == max(g)` 在多少组上成立——本批的承重读数。

    不抽样、**全部组**都算：这条恒等式若只在抽样上成立，就不能当结构结论。
    """
    n_exact = sum(1 for g in groups
                  if max(floor_predict(g, cap)) == max(g))
    never = sum(1 for g in groups if max(g) < cap)
    all_cap = sum(1 for g in groups if min(g) >= cap)
    return {
        'n_groups': len(groups),
        'n_floor_max_exact': n_exact,
        'all_exact': n_exact == len(groups),
        'n_groups_never_at_cap': never,
        'n_groups_all_at_cap': all_cap,
        # 三个互斥的来源各占多少：全撞顶（预测与真值都是 c）、部分撞顶（走同一分支）、
        # 一个都没撞顶（预测 ≡ 真值）。这是恒等式成立的**机器可核的分支覆盖**。
        'n_branch_all_at_cap': all_cap,
        'n_branch_some_at_cap': len(groups) - never - all_cap,
        'n_branch_none_at_cap': never,
    }


def traj_level_error(groups, cap):
    """逐条这一侧预测器**会不会**有误差——与组级恒等式并列，防止把结论读成"预测完美"。

    这里必须说清一件事：**原始数据里撞顶样本的 `response_tokens` 就记成 c**
    （`finish_reason=length`，生成被截在 c）。所以"预测 = c"与"记下来的值 = c"逐位相同，
    误差结构性为 0——**这条读数在本数据上不可能非零，因此没有信息量**，本函数把它
    明写成 `error_is_tautological: true`，不许当"预测很准"的证据引用。

    真正有信息量的是另一件事：撞顶样本的**真实**生成长度是多少，本语料不知道
    （只是被截断在 c）。⇒ 逐条侧的预测误差 = "撞顶样本真实值超过 c 的那一截"，
    **本批测不出来**，只能是已知未做项。
    """
    err = []
    for g in groups:
        for x in g:
            p = cap if x >= cap else x
            err.append(abs(x - p))
    n_at = sum(1 for g in groups for x in g if x >= cap)
    return {
        'n_trajectories': len(err),
        'n_at_cap': n_at,
        'frac_at_cap': round(n_at / float(len(err)), 4),
        'n_with_error': sum(1 for e in err if e > 0),
        'mean_abs_error_tokens': round(statistics.mean(err), 3),
        'max_abs_error_tokens': round(max(err), 3),
        'error_is_tautological': True,
        'unknown': '撞顶样本的真实生成长度本语料没有记录（生成被截断在 c），'
                   '故"逐条侧预测误差"本批测不出来',
    }


def group_order_agreement(groups, cap, seeds=SEEDS, n_groups=N_GROUPS):
    """用 floor 预测排出的组序，与用真值排出的组序是否相同（逐种子）。"""
    rng = random.Random(0)
    same = 0
    for _ in range(len(seeds)):
        pick = rng.sample(groups, n_groups)
        true_ord = sorted(range(n_groups), key=lambda i: (-max(pick[i]), i))
        pred_ord = sorted(range(n_groups),
                          key=lambda i: (-max(floor_predict(pick[i], cap)), i))
        if true_ord == pred_ord:
            same += 1
    return {'n_seeds': len(seeds), 'n_same_order': same}


# ---------------------------------------------------------------- 排程：两臂各按自己的信息排

def _group_arm_tax(group_lens, backlog, predict, cap, n_slots=None):
    """组臂 makespan：**按预测的组最大排序**，物理耗时用真值 `max(g)`，整组归还。"""
    n_groups = len(group_lens)
    n_per = len(group_lens[0])
    if any(len(g) != n_per for g in group_lens):
        raise ValueError('group_lens 必须等大')
    total = n_groups * n_per
    batch_span = float(max(max(g) for g in group_lens))
    entries = []
    for b in range(backlog):
        for g in group_lens:
            p = predict(g, cap) if predict is not None else list(g)
            entries.append((float(max(p)), b * batch_span, float(max(g))))
    entries.sort(key=lambda r: (-r[0], r[1]))

    free = total
    running = []
    pending = list(entries)
    now = end = 0.0
    while pending or running:
        placed = True
        while placed:
            placed = False
            for r in list(pending):
                if n_per <= free:
                    running.append((now + r[2], n_per))
                    free -= n_per
                    pending.remove(r)
                    placed = True
        if not running:
            break
        running.sort()
        t, s_i = running.pop(0)
        now = max(now, t)
        end = max(end, t)
        free += s_i
    return max(end, 1e-9)


def _traj_arm_makespan(group_lens, backlog, n_slots=None, durations=None):
    """轨迹臂 makespan：**逐条真值自排**（它看得见每条长度，不需要预测），逐条归还。

    `durations` 给了就用它（＝组臂那份排序展开后的轨迹序列，保持 `grouping_tax` 的
    "两臂吃同一批轨迹、同一顺序"），不给就按真值降序自排（＝本批的"两臂各按自己的
    信息排"）。两种口径分开，是因为参照实现 `_makespan_by_policy` 用的是前者。
    """
    total = len(group_lens) * len(group_lens[0])
    n_slots = total if n_slots is None else int(n_slots)
    if durations is None:
        durations = [float(x) for b in range(backlog) for g in group_lens for x in g]
        durations.sort(key=lambda v: -v)
    return _makespan([1] * len(durations), durations, False, cap=n_slots)


def separate_tax(group_lens, backlog, predict, cap, n_slots=None):
    """本批主口径：组臂按 `predict` 排、轨迹臂按真值自排，返回 `R`。

    **为什么不沿用 `labelled_tax` 那种"两臂同序"**：那种接线里预测对两臂影响相同，
    `R` 在结构上不含"预测"这一项，换哪个预测器读数都不动（本批第一次跑就撞上，
    四档标签全同）。真实系统的两臂确实各按自己看得见的信息排——组臂只看得到
    组级预测，轨迹臂一条跑完就归还、本来就看得见逐条真值。
    """
    tg = _group_arm_tax(group_lens, backlog, predict, cap)
    tt = _traj_arm_makespan(group_lens, backlog, n_slots)
    return (tg - tt) / tg if tg else 0.0


# ---------------------------------------------------------------- 自证

def tax_canonical_check(groups, backlog):
    """接线自证：`real` 预测下本文件的 `R` 必须与既有 `_makespan_by_policy` 逐位相同。

    两处都是"组臂按真值排、轨迹臂按真值排、容量 = 单批 slot 数"，所以应当相等；
    不等就说明本文件接错了列或取错了容量。
    """
    ref = _makespan_by_policy(groups, backlog, POLICY_LOF)['tax']
    mine = shared_tax(groups, backlog, None, arm_cap('cap256'))
    return {'reference': ref, 'mine': mine, 'abs_diff': abs(mine - ref),
            'ok': abs(mine - ref) <= 1e-12}


def order_sensitivity(groups, cap, backlog=N_ID_BACKLOG, seeds=SEEDS, trials=TRIALS):
    """组臂的**顺序**这笔杠杆有多大：固定组集合，只换到达序，`R` 的极差。

    注意这里的判词要小心：`separate_tax` 里组臂**先排序再插入**，所以打乱到达序对
    「按预测排」的臂是**空操作**（`R` 极差结构性为 0）。那一档只用来证明"排序真的
    生效了、不是到达序在起作用"。真正有区分度的是**不排序**的那一臂（FIFO）：它按
    到达序放，打乱就动。两档并列报，否则"顺序带宽为 0"会被读成"顺序不重要"。
    """
    rng = random.Random(0)
    lof_bands = []
    fifo_bands = []
    for _ in range(len(seeds)):
        pick = rng.sample(groups, N_GROUPS)
        tt = _traj_arm_makespan(pick, backlog)
        lof_vals, fifo_vals = [], []
        for t in range(trials):
            sh = list(pick)
            random.Random(t * 7919 + len(lof_vals)).shuffle(sh)
            tg = _group_arm_tax(sh, backlog, None, cap)
            lof_vals.append((tg - tt) / tg if tg else 0.0)
            fg = _group_arm_tax_fifo(sh, backlog, tt)
            fifo_vals.append(fg)
        lof_bands.append(max(lof_vals) - min(lof_vals))
        fifo_bands.append(max(fifo_vals) - min(fifo_vals))
    return {
        'lof_median_pp': round(100 * statistics.median(lof_bands), 3),
        'lof_max_pp': round(100 * max(lof_bands), 3),
        'fifo_median_pp': round(100 * statistics.median(fifo_bands), 3),
        'fifo_max_pp': round(100 * max(fifo_bands), 3),
        'note': 'lof 档按预测排序、到达序无用 ⇒ 极差结构性为 0；fifo 档才量得到顺序的价格',
    }


def _group_arm_tax_fifo(group_lens, backlog, tt):
    """组臂按**到达序**放（不排序），返回 `R`——顺序敏感性的载体。"""
    n_per = len(group_lens[0])
    total = len(group_lens) * n_per
    durs = [float(max(g)) for b in range(backlog) for g in group_lens]
    free = total
    running = []
    pending = list(range(len(durs)))
    now = end = 0.0
    while pending or running:
        placed = True
        while placed:
            placed = False
            for i in list(pending):
                if n_per <= free:
                    running.append((now + durs[i], n_per))
                    free -= n_per
                    pending.remove(i)
                    placed = True
        if not running:
            break
        running.sort()
        t, s_i = running.pop(0)
        now = max(now, t)
        end = max(end, t)
        free += s_i
    tg = max(end, 1e-9)
    return (tg - tt) / tg if tg else 0.0


def sensitivity_check(groups, cap, seeds=SEEDS, n_groups=N_GROUPS):
    """判据敏感性：把预测打坏（`degraded_predict`），`R` 必须动。

    这不是新读数，是**判据能不能用**的检验。若 degraded 与 floor 相同，则"预测对 `R`
    无空间"只是"本度量对预测无反应"，两者要分开——本批踩过这个坑：最初两臂同序接线，
    四档标签全同，看起来像"预测没用"，其实是 `R` 里根本没有预测这一项。
    """
    picks = _seed_picks(groups, n_groups=n_groups)
    out = {}
    for b in BACKLOGS:
        flo = [separate_tax(pick, b, floor_predict, cap) for pick in picks]
        deg = [separate_tax(pick, b, degraded_predict, cap) for pick in picks]
        shared_flo = [shared_tax(pick, b, floor_predict, cap) for pick in picks]
        shared_deg = [shared_tax(pick, b, degraded_predict, cap) for pick in picks]
        out['backlog%d' % b] = {
            'separate_floor_pp': round(100 * statistics.mean(flo), 3),
            'separate_degraded_pp': round(100 * statistics.mean(deg), 3),
            'separate_moves': sum(1 for a, c in zip(flo, deg)
                                  if abs(a - c) > paired_diff.TOL),
            'shared_floor_pp': round(100 * statistics.mean(shared_flo), 3),
            'shared_degraded_pp': round(100 * statistics.mean(shared_deg), 3),
            'shared_moves': sum(1 for a, c in zip(shared_flo, shared_deg)
                                if abs(a - c) > paired_diff.TOL),
            'n_seeds': len(picks),
        }
    return out


# ---------------------------------------------------------------- 汇总

def _seed_picks(groups, seeds=SEEDS, n_groups=N_GROUPS):
    """30 个种子各自抽到的组（与 `real_lengths_policy` 同口径：`Random(0)` 续抽）。"""
    rng = random.Random(0)
    return [rng.sample(groups, n_groups) for _ in range(len(seeds))]


def measure_structure(groups, label):
    """负载结构：撞顶比例与组级撞顶形态（后者决定恒等式走哪条分支）。"""
    cap = arm_cap(label)
    flat = [v for g in groups for v in g]
    at_cap = sum(1 for v in flat if v >= cap)
    return {
        'max_response': cap,
        'n_trajectories': len(flat),
        'n_at_cap': at_cap,
        'trajectory_floor_frac': round(at_cap / float(len(flat)), 4),
        'n_groups': len(groups),
    }


def measure_arm(groups, label, backlogs=BACKLOGS):
    """三档标签下的 `R` 均值（只给水平；两档预测的差是恒等式，不是新读数）。"""
    cap = arm_cap(label)
    picks = _seed_picks(groups)
    out = {}
    for b in backlogs:
        cells = {}
        for pol in ('separate', 'shared'):
            for name, pred in PREDICTORS:
                if pol == 'separate':
                    vals = [separate_tax(pick, b, pred, cap) for pick in picks]
                else:
                    vals = [shared_tax(pick, b, pred, cap) for pick in picks]
                cells['%s.%s' % (pol, name)] = round(statistics.mean(vals), 6)
        out['backlog%d' % b] = cells
    return out


def shared_tax(group_lens, backlog, predict, cap, n_slots=None):
    """对照口径：两臂**同序**（`grouping_tax` 原定义）。预测对两臂影响相同 ⇒ `R` 不动。

    保留它是因为"预测对 `R` 无杠杆"这件事在两个口径下都成立，但**成立的理由不同**：
    同序档是"预测被约掉了"，分序档是"预测已经精确"。只报一档会把两件事混成一件。
    """
    n_groups = len(group_lens)
    n_per = len(group_lens[0])
    total = n_groups * n_per
    n_slots = total if n_slots is None else int(n_slots)
    batch_span = float(max(max(g) for g in group_lens))
    entries = []
    for b in range(backlog):
        for g in group_lens:
            p = predict(g, cap) if predict is not None else list(g)
            entries.append((float(max(p)), b * batch_span, g))
    entries.sort(key=lambda r: (-r[0], r[1]))

    free = total
    running = []
    pending = list(entries)
    now = end = 0.0
    while pending or running:
        placed = True
        while placed:
            placed = False
            for r in list(pending):
                if n_per <= free:
                    running.append((now + float(max(r[2])), n_per))
                    free -= n_per
                    pending.remove(r)
                    placed = True
        if not running:
            break
        running.sort()
        t, s_i = running.pop(0)
        now = max(now, t)
        end = max(end, t)
        free += s_i
    tg = max(end, 1e-9)

    # 两臂同序：轨迹序列沿组序展开、不再重排（与 `_makespan_by_policy` 一致）
    flat = [float(x) for _p, _b, g in entries for x in g]
    tt = _traj_arm_makespan(group_lens, backlog, n_slots, durations=flat)
    return (tg - tt) / tg if tg else 0.0


def measure_gap(groups, label, backlogs=BACKLOGS):
    """`floor − real`（本批主判据）与同口径带宽；`full − floor` 只作恒等式核对。

    真值列两臂相同（同一批组、同一负载），只有预测列不同 ⇒ 配对差的随机性只来自
    "抽到哪些组"。带宽取同一预测列、**只换传入顺序**的极差（§13.30 第五批同口径）。
    """
    cap = arm_cap(label)
    picks = _seed_picks(groups)
    out = {}
    for b in backlogs:
        rows = []
        for s, pick in enumerate(picks):
            row = {name: separate_tax(pick, b, pred, cap) for name, pred in PREDICTORS}
            # 带宽取 FIFO 臂（按到达序放、不排序）——排序臂打乱到达序是空操作，
            # 用它算带宽会恒得 0，把"顺序的价格"读没了（本批踩过）。
            tt = _traj_arm_makespan(pick, b)
            bands = []
            for t in range(TRIALS):
                sh = list(pick)
                random.Random(s * 7919 + t).shuffle(sh)
                bands.append(_group_arm_tax_fifo(sh, b, tt))
            row['_band'] = max(bands) - min(bands)
            rows.append(row)

        def paired(a, c):
            d = [r[a] - r[c] for r in rows]
            reg = paired_diff.sign_regime(d)
            return {
                'mean_pp': round(100 * statistics.mean(d), 3),
                'n_positive': sum(1 for x in d if x > paired_diff.TOL),
                'n_negative': sum(1 for x in d if x < -paired_diff.TOL),
                'n_identical': sum(1 for x in d if abs(x) <= paired_diff.TOL),
                'regime': reg['regime'],
            }

        out['backlog%d' % b] = {
            'mean_pp': {name: round(100 * statistics.mean([r[name] for r in rows]), 3)
                        for name, _ in PREDICTORS},
            'floor_minus_real': paired('floor', 'real'),
            'full_minus_floor': paired('full', 'floor'),
            'band_pp': {
                'median': round(100 * statistics.median([r['_band'] for r in rows]), 3),
                'max': round(100 * max(r['_band'] for r in rows), 3),
            },
            'n_r_identically_zero': sum(1 for r in rows if r['real'] == 0.0),
        }
    return out


def identity_check(groups, label):
    """对照：8 组、积压 16 批，c 取本档上限与一个极大值。

    c 取极大时一组也不会撞顶 ⇒ floor 预测 == 全真值 ⇒ 两个预测的 `R` **必须逐位相同**。
    这条是"差距确实来自预测误差"的自证，不是新读数。
    """
    cap = arm_cap(label)
    pick = groups[:N_ID_GROUPS]
    out = {}
    for tag, c in (('c_cap', cap), ('c_huge', 1e9)):
        real = separate_tax(pick, N_ID_BACKLOG, None, cap)
        flo = separate_tax(pick, N_ID_BACKLOG, lambda g, _c: floor_predict(g, c), cap)
        full = separate_tax(pick, N_ID_BACKLOG, full_predict, cap)
        out[tag] = {'real': round(real, 6), 'floor': round(flo, 6),
                    'full': round(full, 6),
                    'abs_diff_floor_full': abs(flo - full)}
    return out


def build():
    t0 = time.time()
    out = {'experiment': 'prediction-floor', 'batch': '第十七批', 'date': '2026-11-04',
           'seeds': len(SEEDS), 'n_groups': N_GROUPS, 'n_per_group': N_PER,
           'trials': TRIALS,
           'note': '截断上界对"组最大"是精确预测 ⇒ 对 R 零空间；逐条侧仍有误差，'
                   '但它在 R 的分子分母里都不出现',
           'arms': {}, 'check': {}}
    for label, fname in REAL_ARMS:
        groups = load_real_groups(os.path.join(REAL_DIR, fname))
        cap = arm_cap(label)
        out['arms'][label] = {
            'structure': measure_structure(groups, label),
            'group_max_identity': group_max_identity(groups, cap),
            'traj_level_error': traj_level_error(groups, cap),
            'group_order_agreement': group_order_agreement(groups, cap),
            'tax_by_label': measure_arm(groups, label),
            'gap': measure_gap(groups, label),
            'order_sensitivity': order_sensitivity(groups, cap),
            'sensitivity': sensitivity_check(groups, cap),
            'identity': identity_check(groups, label),
        }
        out['check'][label] = {
            'tax_canonical': tax_canonical_check(groups[:N_ID_GROUPS], N_ID_BACKLOG),
        }
    out['seconds'] = round(time.time() - t0, 1)
    return out


def _flat(obj, prefix=''):
    if isinstance(obj, dict):
        for k, v in obj.items():
            for x in _flat(v, prefix + '/' + str(k)):
                yield x
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            for x in _flat(v, prefix + '/%d' % i):
                yield x
    else:
        yield prefix, obj


def _flat_diff(a, b, skip=('seconds',)):
    da = dict(_flat(a))
    db = dict(_flat(b))
    out = []
    for k in sorted(set(da) | set(db)):
        if any(s in k for s in skip):
            continue
        if k not in da or k not in db or da[k] != db[k]:
            out.append(k)
    return out


def main(argv):
    out = build()
    if '--verify' in argv:
        with open(OUT_PATH) as f:
            old = json.load(f)
        diffs = _flat_diff(old, out)
        for label, chk in out['check'].items():
            for name, v in chk.items():
                if not v['ok']:
                    diffs.append('%s/%s not ok' % (label, name))
        print('verify: %d differences' % len(diffs))
        for k in diffs[:20]:
            print('  ', k)
        return 0 if not diffs else 1
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write('\n')
    for label, arm in out['arms'].items():
        gm = arm['group_max_identity']
        te = arm['traj_level_error']
        os_ = arm['order_sensitivity']
        print('%-18s floor_traj=%.4f group_max_exact=%d/%d all=%s traj_err(n=%d,max=%s) '
              'fifo_order_band_med=%.2fpp' % (
                  label, arm['structure']['trajectory_floor_frac'],
                  gm['n_floor_max_exact'], gm['n_groups'], gm['all_exact'],
                  te['n_with_error'], te['max_abs_error_tokens'], os_['fifo_median_pp']))
        for b, row in arm['gap'].items():
            print('    %-10s real=%6.2f floor=%6.2f full=%6.2f  floor-real=%5.2f (%s) '
                  'band=%.2f zeros=%d' % (
                      b, row['mean_pp']['real'], row['mean_pp']['floor'],
                      row['mean_pp']['full'], row['floor_minus_real']['mean_pp'],
                      row['floor_minus_real']['regime'], row['band_pp']['median'],
                      row['n_r_identically_zero']))
    bad = [k for label, chk in out['check'].items() for k, v in chk.items() if not v['ok']]
    print('checks failing:', bad)
    print('seconds', out['seconds'])
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
