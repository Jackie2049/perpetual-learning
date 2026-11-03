#!/usr/bin/env python3
"""分桶探针：组的排队顺序能改变 R 多少（接 TM-011）

复用 `_makespan_by_policy`，只换组的入队顺序。**不是分桶模型**——它只量"给定组的
集合改顺序"的效应，不建模"哪些组被放进同一批"。

## 2026-11-03 订正（第二版）

旧版把「长度优先（LOF）」与「10 次随机打乱的中位数」比，读出均值 5.95pp。
那个比法有两个坑，都是本版实测出来的：

  1. **基线泄漏**：`_makespan_by_policy` 的轨迹臂同样吃组的顺序，所以随机打乱
     **自己也会改 R**（组间 100–400 均匀时，随机序之间的极差就有 2.5–3.5pp）。
     5.95pp 里有一块是"随机序碰巧更亏"。
  2. **配对错位**：若让两臂共用同一份乱序，乱序就被当成两臂共有的派发次序，
     而 FIFO 的定义是"按到达顺序"。这样配出来的差在真实负载上是 −6.3pp，
     换成"各臂按自己的顺序"后同样的负载给出反向的读数。

本版：FIFO 臂吃**原顺序**、LOF 臂吃**按组最大降序**，两者吃同一批轨迹；
同时给出两个口径：

  * 绝对差  `R_fifo − R_lof`（pp）——直接可加；
  * 相对比  `R_lof / R_fifo`——与 §13.30 已有的臂间读数同口径
    （退化种子两臂完全相同，比值恒 1，会稀释均值，故另报 `*_live`）。

两个负载来源：

  * `synthetic`：旧版的构造负载（组间 100–400 均匀）；
  * `real`：已入库的 `rollout-length-structure-real/raw` 的 `response_tokens`，
    按 `prompt_idx` 聚组（480 组 × n=8），四档 = {0.5B, 1.5B} × {cap256, cap1024}。

用法：python3 bucketing_probe.py
"""
import collections
import gzip
import importlib.util
import json
import pathlib
import random
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RAW = ROOT / 'experiments' / 'rollout-length-structure-real' / 'raw'

sys.path.insert(0, str(ROOT / 'tools' / 'paired-diff'))
import paired_diff  # noqa: E402  配对差判读的共享判据（符号分布／带宽／门槛）

spec = importlib.util.spec_from_file_location('g', HERE / 'group_split_tradeoff.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

SEEDS = 20
N_GROUPS = 16
N_PER = 8
CV = 0.07          # 真实 1.5B cap256 的组最大 cv（§13.30 第四批）
BACKLOGS = (2, 4, 8, 16)
REAL_FILES = (
    'qwen05b_gsm8k_n8',
    'qwen05b_gsm8k_n8_cap1024',
    'qwen15b_gsm8k_n8',
    'qwen15b_gsm8k_n8_cap1024',
)


def make_load(seed):
    """构造负载：16 组 × 8 条，组间 100–400 均匀，组内 cv 0.07。"""
    rng = random.Random(seed)
    out = []
    for _ in range(N_GROUPS):
        base = rng.uniform(100, 400)
        out.append([base * (1 + rng.gauss(0, CV)) for _ in range(N_PER)])
    return out


def load_real(name):
    """真实负载：按 prompt_idx 聚组，返回 [[len, ...], ...]。"""
    groups = collections.defaultdict(list)
    with gzip.open(RAW / (name + '.jsonl.gz'), 'rt') as fh:
        for line in fh:
            r = json.loads(line)
            groups[r['prompt_idx']].append(float(r['response_tokens']))
    return [groups[k] for k in sorted(groups)]


def tax(groups, policy, backlog):
    return M._makespan_by_policy(groups, backlog, policy)['tax']


def arms(groups, backlog, rng):
    """两臂四量：FIFO 吃原顺序、LOF 吃按组最大降序，两者吃同一批轨迹。

    返回 dict，含两条 makespan、R 与两臂各自的相对变化。R 的差与两臂之比
    是恒等关系（`R1 − R0 = (T0/G0) − (T1/G1)`），这里一并给出，免得读者再去
    猜"是不是某一臂被选得更好"。
    """
    order = list(groups)
    rng.shuffle(order)
    # LOF 是确定性的（按组最大降序），`order` 只用来给种子加独立复本
    fifo = M._makespan_by_policy(groups, backlog, M.POLICY_FIFO)
    lof = M._makespan_by_policy(order, backlog, M.POLICY_LOF)
    g0, t0, g1, t1 = (fifo['group_makespan'], fifo['traj_makespan'],
                      lof['group_makespan'], lof['traj_makespan'])
    return {
        'R_fifo': fifo['tax'],
        'R_lof': lof['tax'],
        'abs': fifo['tax'] - lof['tax'],
        'rel': lof['tax'] / fifo['tax'] if fifo['tax'] else 1.0,
        'group_gain': 1.0 - g1 / g0,
        'traj_gain': 1.0 - t1 / t0,
    }


def seed_sign_stability(rows):
    """配对差在**种子间**是否同号。

    均值不看符号分布：20 个种子若一半 +、一半 −，均值可以很小也可以很大，
    但结论"LOF 让 R 变小"只有在**每个种子都同向**时才算稳定。这里报同号
    比例与符号区间，让读者能判断均值是不是靠少数种子撑起来的。
    """
    diffs = [r['abs'] for r in rows]
    counts = paired_diff.sign_counts(diffs)
    return {
        'n_positive': counts['n_positive'],
        'n_negative': counts['n_negative'],
        'same_sign_fraction': paired_diff.same_sign_fraction(diffs),
        # 差值的种子间标准差：与 |均值| 相当就说明符号在种子间不稳
        'sd_pp': paired_diff.sd_pp(diffs),
        'min_pp': round(min(diffs) * 100, 3),
        'max_pp': round(max(diffs) * 100, 3),
    }


def fifo_order_sensitivity(groups, backlog, trials=30, seed=0):
    """同一批组、只换传入顺序，FIFO 臂的 R 极差（pp）。

    FIFO 的定义是"按到达顺序"，但"到达顺序"在本层就是传入列表的顺序。
    真实负载里每个种子抽到的是**不同的 16 组**，抽出来再按 prompt_idx 排序，
    所以种子间的 FIFO 基线同时混了"抽到哪些组"和"它们谁先到"两个来源。
    这里把第二个来源单独量出来：组集合固定、只打乱顺序。
    """
    vals = []
    for s in range(trials):
        shuffled = list(groups)
        random.Random(seed * 100003 + s).shuffle(shuffled)
        vals.append(M._makespan_by_policy(shuffled, backlog, M.POLICY_FIFO)['tax'])
    return {
        'min': round(min(vals), 4),
        'max': round(max(vals), 4),
        'spread_pp': round((max(vals) - min(vals)) * 100, 3),
    }


def effect_exceeds_band(effect_pp, band_pp):
    """配对差的绝对值是否超出 FIFO 自身的顺序敏感度。

    判据实现已收到 `tools/paired-diff`（`exceeds_band`），这里只保留原来的入口名，
    避免调用方与既有归档键漂移；口径（严格大于、判读门槛非显著性）见那里。
    """
    return paired_diff.exceeds_band(effect_pp, band_pp)


def summarize(rows):
    diffs = [r['abs'] for r in rows]
    rels = [r['rel'] for r in rows]
    same = sum(1 for r in rows if abs(r['abs']) < 1e-12)
    live = [r for r in rows if abs(r['abs']) >= 1e-12]
    return {
        'n': len(rows),
        'identical': same,
        'abs_pp_mean': round(statistics.mean(diffs) * 100, 3),
        'abs_pp_median': round(statistics.median(diffs) * 100, 3),
        # 正号 = LOF 让 R 变小（变好），负号 = 变大（变差）
        'abs_wins': sum(1 for r in rows if r['abs'] > 0),
        'sign_stability': seed_sign_stability(rows),
        'rel_mean': round(statistics.mean(rels), 4),
        'rel_median': round(statistics.median(rels), 4),
        # 退化种子的比值恒 1.0，混进均值会把**变差**的臂的平均比拉到 1 以下，
        # 看上去像变好了。相对比只在"两臂确实不同"的种子上读。
        'rel_mean_live': round(statistics.mean([r['rel'] for r in live]), 4)
        if live else None,
        'rel_median_live': round(statistics.median([r['rel'] for r in live]), 4)
        if live else None,
        # R 的变化由两臂组臂/轨迹臂 makespan 之比**代数决定**（无自由参数）：
        #   R = 1 − T/G ⇒ R1 − R0 = (T0/G0) − (T1/G1)
        # 这里给出改序后组臂与轨迹臂各自的相对变化，用来判断"哪一臂更受益"。
        'group_arm_gain_mean': round(statistics.mean(r['group_gain'] for r in rows), 4),
        'traj_arm_gain_mean': round(statistics.mean(r['traj_gain'] for r in rows), 4),
    }


def synthetic_block():
    """构造负载：FIFO 基线、配对绝对差、配对相对比、随机序自身的离散度。"""
    out = []
    for backlog in BACKLOGS:
        rows, spreads = [], []
        for s in range(SEEDS):
            groups = make_load(s)
            r = arms(groups, backlog, random.Random(s + 5000))
            r['R_fifo_arrival'] = r['R_fifo']
            rng = random.Random(s + 1000)
            shuf = []
            for _ in range(10):
                g = groups[:]
                rng.shuffle(g)
                shuf.append(tax(g, M.POLICY_FIFO, backlog))
            spreads.append(max(shuf) - min(shuf))
            rows.append(r)
        stat = summarize(rows)
        stat.update({
            'backlog': backlog,
            'R_fifo_mean': round(statistics.mean(r['R_fifo'] for r in rows), 4),
            'shuffle_spread_pp_mean': round(statistics.mean(spreads) * 100, 3),
        })
        out.append(stat)
    return out


def real_block():
    """真实负载：每个文件抽 16 组、四档积压，配对读顺序效应。"""
    out = []
    for name in REAL_FILES:
        corpus = load_real(name)
        cv = statistics.pstdev([max(g) for g in corpus]) / statistics.mean(
            [max(g) for g in corpus])
        for backlog in BACKLOGS:
            rows, fifo_spreads = [], []
            for s in range(SEEDS):
                rng = random.Random(s)
                picks = sorted(rng.sample(range(len(corpus)), N_GROUPS))
                groups = [corpus[i] for i in picks]
                rows.append(arms(groups, backlog, random.Random(s + 5000)))
                fifo_spreads.append(
                    fifo_order_sensitivity(groups, backlog, seed=s)['spread_pp'])
            stat = summarize(rows)
            stat.update({
                'file': name,
                'backlog': backlog,
                'group_max_cv': round(cv, 3),
                'n_pool_groups': len(corpus),
                'R_fifo_mean': round(statistics.mean(r['R_fifo'] for r in rows), 4),
                'R_lof_mean': round(statistics.mean(r['R_lof'] for r in rows), 4),
                # 同组集合下 FIFO 自身的顺序敏感度均值：|配对差| 若落在这个带里，
                # 说明读到的"排序效应"与基线自身的顺序抖动不可区分。
                'fifo_order_spread_pp_mean': round(statistics.mean(fifo_spreads), 3),
            })
            out.append(stat)
    return out


def main():
    result = {
        'seeds': SEEDS, 'n_groups': N_GROUPS, 'n_per': N_PER,
        'cv_synthetic': CV, 'backlogs': list(BACKLOGS),
        'synthetic': synthetic_block(),
        'real': real_block(),
    }
    text = json.dumps(result, indent=1, ensure_ascii=False)
    print(text)
    (HERE / 'bucketing-probe.json').write_text(text + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
