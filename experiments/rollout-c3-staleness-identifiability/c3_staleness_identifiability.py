#!/usr/bin/env python3
"""§21.7 C3 成对报值约定的可执行性实证（rollout 专项，2026-11-02 M1）。

支撑 notebook/rollout-performance-research.md §21.7。

§21.3 的 C3 规定 async 下必须**成对**报 η（staleness 版本年龄的分布）与 S
（单位墙钟内被**接纳进训练**的轨迹数），并规定"同步化基准取同栈 η=0 整步墙钟"。
§21.6 已核出四框架都不能按字面算 η，只能给近似。本实验问四件事：

  Q1 三种近似 η（训练步版本差 / 引擎侧已装载权重差 / 只报最大值）能否支撑
     C3 自己规定的两个判断：本配置比 sync（η=0）陈旧、以及两个 async 配置谁更陈旧。
  Q2 单独报 S 会偏多少；分子错（生成侧计数）与分母错（剔掉训练侧空转）分别偏多少。
  Q3 "async 比 sync 快 N 倍"的倍数如何随基线选择变化（基线也是异步、且只报生成侧）。
  Q4 弹性窗口 + FIFO 重放时报出的 η=0 是否真为 0（对照组：同一旋钮改 LIFO 必须变正）。

墙钟是任意虚拟秒，不可与任何论文报值相除（§19.1 规则一）。
只用标准库；随机性全部来自 random.Random(seed)。

用法：
    python3 c3_staleness_identifiability.py             # 全量重跑并覆盖 summary.json
    python3 c3_staleness_identifiability.py --verify     # 重跑并与归档 summary.json 逐字段比对
    python3 -m unittest test_c3_staleness_identifiability -v
"""

import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY_PATH = os.path.join(HERE, 'summary.json')

SEEDS = list(range(30))
STEPS = 200
N_GROUPS = 8            # 每步生成的组数
N_SAMPLES = 4           # 每组轨迹数
GROUP_MEAN_FAIL = 0.25  # 组整体失败概率（失败组被丢弃）

CONFIGS = {
    # 同步：生成完一批才更新，η 恒 0。整步墙钟 = T + sync_extra（见 s_wall）
    'sync': {'T': 1.00, 'sync_extra': 0.20, 'mode': 'sync'},
    # "引擎持续在生成"的同步变体：trainer 每轮仍等本轮生成完，
    # 但引擎不等训练结束就开下一批，流水起来之后整步只受 T 约束（s_wall 按 T 计）。
    # 它消费的仍是本轮最新批，η 恒 0——与三个 async 配置在 Q2 里构成"同 S 不同 η"的对照组。
    'sync-pipelined': {'T': 1.00, 'sync_extra': 0.20, 'pipeline': True, 'mode': 'sync'},
    # 异步、轻度陈旧：引擎比训练快 1.5 倍、策略滞后 1 代
    # 注意：async 与 sync-pipelined 的引擎/训练时间轴相同，差别只在
    # 被消费的批有多少已经陈旧——这正是 η 的唯一差别来源。
    'async-stale': {'T': 1.00, 'sync_extra': 0.20, 'mode': 'async',
                    'engine_lag': 1, 'stale_window': 8, 'gen_rate': 1.5},
    # 异步、重度陈旧：引擎快 2.5 倍、策略滞后 3 代，队列积得更深
    'async-deep': {'T': 1.00, 'sync_extra': 0.20, 'mode': 'async',
                   'engine_lag': 3, 'stale_window': 20, 'gen_rate': 2.5},
    # 背压：引擎慢、训练侧常饿着（用于 S 的分母错与弹性窗口两个场景）
    'async-starved': {'T': 1.00, 'sync_extra': 0.20, 'mode': 'async',
                      'engine_lag': 1, 'stale_window': 8, 'gen_rate': 0.5},
}


# ---------------------------------------------------------------- 模型

def simulate(cfg, seed, queue_mode='fifo'):
    """跑一条 trace。

    引擎落后 engine_lag 代：在训练侧产出第 k 代权重之后，引擎才开始用第 k 代生成；
    因此"生成用的权重代"永远落后于"消费时的最新代"。同步模式下 lag=0 且每步
    生成完立刻消费，故 η 恒 0；异步模式下 lag>0，旧组在队列里等，η 大于 0。
    一批的组成按 queue_mode 从待消费队列里取：fifo 取最早、lifo 取最新。
    """
    rng = random.Random(seed)
    lag = cfg.get('engine_lag', 0)
    pipelined = cfg.get('pipeline', False)
    window = cfg.get('stale_window', 0)
    rows = []
    pending = []
    train_ver = 0          # 训练侧当前产出的最新权重代
    rounds = 0
    units = 0.0            # 累计生成额度（gen_rate 可以是小数，表示引擎更慢）
    rate = float(cfg.get('gen_rate', 1))
    while len(rows) < STEPS:
        # 引擎每轮按 gen_rate 产出若干批：>1 积压成队列，<1 让训练侧饿着
        units += rate
        while units >= 1.0:
            if cfg['mode'] == 'sync':
                gen_ver = train_ver
                train_ver += 1
            else:
                gen_ver = max(0, train_ver - lag)
                train_ver += 1
            for g in range(N_GROUPS):
                pending.append((rounds, g, gen_ver))
            rounds += 1
            units -= 1.0
        if cfg['mode'] == 'sync' and not pipelined:
            batch, pending = pending, []
        elif pipelined:
            # 引擎持续生产、训练侧紧随其后消费最新批：η 恒 0，但没有"等一批生成完"的空档
            batch, pending = pending[:N_GROUPS], pending[N_GROUPS:]
        else:
            pending = [x for x in pending if train_ver - x[2] <= window]
            if len(pending) < N_GROUPS:
                # 引擎还没产出可消费的批：训练侧这一步在等待（同步模式下不会出现）
                waiting = True
            else:
                waiting = False
            if waiting:
                rows.append({'idle': True, 'n_gen': 0, 'n_accept': 0, 'eta_true': [],
                             '_gen_ver': []})
                continue
            if queue_mode == 'fifo':
                batch, pending = pending[:N_GROUPS], pending[N_GROUPS:]
            else:
                batch = sorted(pending, key=lambda x: -x[2])[:N_GROUPS]
                gone = set(batch)
                pending = [x for x in pending if x not in gone]
        consume_ver = train_ver - 1
        n_gen = len(batch) * N_SAMPLES
        failed = sum(1 for _ in batch if rng.random() < GROUP_MEAN_FAIL)
        n_accept = (len(batch) - failed) * N_SAMPLES
        rows.append({
            'idle': False,
            'n_gen': n_gen,
            'n_accept': n_accept,
            # 真值：消费时训练侧最新代 − 生成时所用代
            'eta_true': [consume_ver - v for (_, _, v) in batch],
            '_gen_ver': [v for (_, _, v) in batch],
        })
    return rows


def engine_view(rows):
    """引擎只能看见它真正产出并消费的批：空转步不产生任何可观测记录。"""
    return [r for r in rows if not r.get('idle')]


# ---------------------------------------------------------------- 近似 η（§21.6 已核的四种）


def eta_obs_trainer_step(rows):
    """近似 A：verl 式"按步标签"。消费侧与生成侧都用训练步版本号（即本模型真值）。"""
    return [e for r in engine_view(rows) for e in r['eta_true']]


def eta_obs_engine_weight(rows):
    """近似 B：SkyRL 式，只看引擎侧"这批用哪代权重生成"。

    引擎不知道训练侧已推进到哪一代，只能报出本批内部的权重代差；
    于是年龄被系统性压到 0（同批同代时恒 0）。
    """
    out = []
    for r in engine_view(rows):
        cur = max(r['_gen_ver'])
        out.extend([cur - v for v in r['_gen_ver']])
    return out


def eta_obs_max_only(rows):
    """近似 C：NeMo 式按版本淘汰，只产出"最大年龄"一个数，没有分布。

    返回单个标量（跨整段 trace 的最大值），而不是逐步序列——
    这样它在"比谁更陈旧"上能分辨，却结构性无法给出 P50/P90。
    """
    return float(max((max(r['eta_true']) for r in engine_view(rows)), default=0.0))


def pp_get(items, key, q):
    return [it[key][q] for it in items]


def _isnan(vals):
    return any(v != v for v in vals)


def _max_only_summary(scalar):
    """只报最大值的那一档：只有 max 有定义，分位数结构性缺失。"""
    return {'max': float(scalar), 'p50': float('nan'), 'p90': float('nan'),
            'mean': float(scalar)}


def quantile(xs, q):
    ys = sorted(xs)
    if not ys:
        return float('nan')
    k = (len(ys) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(ys) - 1)
    return ys[lo] + (ys[hi] - ys[lo]) * (k - lo)


def eta_summary(xs):
    return {'max': float(max(xs)), 'p50': quantile(xs, 0.5), 'p90': quantile(xs, 0.9),
            'mean': sum(xs) / len(xs)}


# ---------------------------------------------------------------- S 的三种读法


def s_accept(rows, wall):
    """C3 口径：接纳进训练的轨迹数 / 整段墙钟。"""
    return sum(r['n_accept'] for r in rows) / wall


def s_generated(rows, wall):
    """分子错：生成侧计数（被丢弃的组照样算吞吐）。"""
    return sum(r['n_gen'] for r in rows) / wall


def s_active_step_den(rows, cfg):
    """分母错：把训练侧空转/等待的墙钟剔出去，只按"真正训练过的那几步"折算吞吐。

    每步活跃墙钟按 T + U 计（本模型里生成与更新串行）。
    """
    active = engine_view(rows)
    return sum(r['n_accept'] for r in active) / (len(active) * (cfg['T'] + cfg['sync_extra']))


def s_generated_ratio_exact(n_gen, n_acc, T, U, wall):
    """生成侧计数的相对高报：分子换成生成本量，分母仍是整段墙钟。"""
    return (n_gen / wall) / (n_acc / wall)


# ---------------------------------------------------------------- 统计小工具


def _ms(vals):
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1)
    return {'mean': m, 'std': var ** 0.5, 'min': min(vals), 'max': max(vals)}


def _mean_std(items, keys):
    return {k: _ms([it[k] for it in items]) for k in keys}


# ---------------------------------------------------------------- 场景


def scenario_identifiability():
    """Q1：三种近似能否支撑 C3 的两个判断。"""
    per = {}
    for name, cfg in CONFIGS.items():
        seeds = []
        for seed in SEEDS:
            rows = simulate(cfg, seed)
            seeds.append({
                'true': eta_summary([e for r in engine_view(rows) for e in r['eta_true']]),
                'trainer_step': eta_summary(eta_obs_trainer_step(rows)),
                'engine_weight': eta_summary(eta_obs_engine_weight(rows)),
                'max_only': _max_only_summary(eta_obs_max_only(rows)),
            })
        per[name] = seeds

    # 判据一：async 的 η 是否大于 sync（真值恒 0）
    gt_sync = {}
    for name in ('async-stale', 'async-deep'):
        gt_sync[name] = {}
        for key in ('true', 'trainer_step', 'engine_weight', 'max_only'):
            gt_sync[name][key] = {
                'hit': sum(1 for a, b in zip(per[name], per['sync'])
                           if a[key]['max'] > b[key]['max']),
                'n': len(SEEDS),
            }
    # 判据二：deep 是否比 stale 更陈旧
    order = {}
    for key in ('true', 'trainer_step', 'engine_weight', 'max_only'):
        order[key] = {
            'hit': sum(1 for d, s in zip(per['async-deep'], per['async-stale'])
                       if d[key]['max'] > s[key]['max']),
            'n': len(SEEDS),
        }
    # 分布层面：P50／P90 的偏差（近似 − 真值）
    bias = {}
    for name in CONFIGS:
        bias[name] = {}
        for key in ('trainer_step', 'engine_weight', 'max_only'):
            bias[name][key] = {
                q: _ms([pp[key][q] - pp['true'][q] for pp in per[name]])
                for q in ('max', 'p50', 'p90', 'mean')
                if not _isnan(pp_get(per[name], key, q))}
    summary = {}
    for name in CONFIGS:
        summary[name] = {
            key: {q: _ms([pp[key][q] for pp in per[name]])
                  for q in ('max', 'p50', 'p90', 'mean')}
            for key in ('true', 'trainer_step', 'engine_weight', 'max_only')}
    # 判据三：只报最大值能否同时给出分布（结构性，不依赖种子）
    breadth = {}
    for name in ('async-stale', 'async-deep'):
        vals = [e for r in engine_view(simulate(CONFIGS[name], SEEDS[0])) for e in r['eta_true']]
        breadth[name] = {'true_max_minus_p50': max(vals) - quantile(vals, 0.5),
                         'max_only_reports_p50': False}

    # 判据四：真实分布本身是否"单点"——若每个配置的 η 分布都是常数，
    # 则"只报 max 丢掉分布信息"这件事在本模型里没有可观测代价，必须写明
    spread = {}
    for name in CONFIGS:
        vals = [e for r in engine_view(simulate(CONFIGS[name], SEEDS[0])) for e in r['eta_true']]
        spread[name] = {'min': float(min(vals)) if vals else None,
                        'max': float(max(vals)) if vals else None,
                        'std': _ms([float(v) for v in vals])['std'] if vals else None}
    return {'per_config': summary, 'verdict_stale_gt_sync': gt_sync,
            'verdict_order': order, 'bias': bias, 'max_only_breadth': breadth,
            'eta_distribution_spread': spread}


def s_wall(cfg, n_train_steps):
    """整段墙钟：本模型里每个训练步都要跨过生成周期 T；同步（非流水）再多一段更新时间 sync_extra。"""
    if cfg['mode'] == 'sync' and not cfg.get('pipeline', False):
        return n_train_steps * (cfg['T'] + cfg['sync_extra'])
    return n_train_steps * cfg['T']


def scenario_single_s():
    """Q2：单独报 S 的两类偏差与"同一 S 对应不同 η"。

    读法：
      gennum_ratio   分子错——用生成本量代替接纳量（被丢弃/被淘汰的组照样算吞吐）
      active_ratio   分母错——把空转/等待的墙钟剔出去
      eta_max        同一配置的 η 上限（与 S 同时报，见 C3）
    关键对照：`sync-pipelined` 与三个 async 配置的 S 完全相同（引擎时间轴一样），
    但 η 完全不同。只报 S 时它们不可区分。
    """
    per_seed = []
    for seed in SEEDS:
        item = {'seed': seed}
        for name in ('sync', 'sync-pipelined', 'async-stale', 'async-deep', 'async-starved'):
            cfg = CONFIGS[name]
            rows = simulate(cfg, seed)
            active = engine_view(rows)
            n_acc = sum(r['n_accept'] for r in active)
            n_gen = sum(r['n_gen'] for r in active)
            wall = s_wall(cfg, STEPS)
            base = n_acc / wall
            eta = [e for r in active for e in r['eta_true']]
            item[name] = {
                's_c3': base,
                'gennum_ratio': s_generated_ratio_exact(n_gen, n_acc, cfg['T'],
                                                        cfg['sync_extra'], wall),
                'active_ratio': s_active_step_den(rows, cfg) / base,
                'speedup_vs_sync': base / s_accept(simulate(CONFIGS['sync'], seed), s_wall(CONFIGS['sync'], STEPS)),
                'drop_rate': 1.0 - n_acc / float(n_gen),
                'idle_ratio': 1.0 - len(active) / float(len(rows)),
                'eta_max': float(max(eta)) if eta else 0.0,
            }
        per_seed.append(item)
    res = {}
    for name in ('sync', 'sync-pipelined', 'async-stale', 'async-deep', 'async-starved'):
        res[name] = _mean_std([p[name] for p in per_seed],
                              ['s_c3', 'gennum_ratio', 'active_ratio', 'speedup_vs_sync',
                               'drop_rate', 'idle_ratio', 'eta_max'])
    # 可区分性判据：S 相同而 η 不同的配置对
    pairs = [('sync-pipelined', 'async-stale'), ('sync-pipelined', 'async-deep'),
             ('async-stale', 'async-deep')]
    res['indistinguishable_by_s_alone'] = {
        '%s|%s' % (a, b): {
            's_diff_rel': abs(res[a]['s_c3']['mean'] - res[b]['s_c3']['mean']) /
                          max(res[a]['s_c3']['mean'], 1e-12),
            'eta_max_diff': res[a]['eta_max']['mean'] - res[b]['eta_max']['mean'],
        } for a, b in pairs}
    return res


def scenario_variant_narratives():
    """Q3：同一底层真值、三种"看起来合理"的报值叙事，会导向不同结论。

    三条叙事都只改**报值口径**，不改任何一行仿真：
      honest      S 用接纳量、基线是 η=0 的同步配置（T + sync_extra 每步）
      gen_num     分子用生成本量（丢弃的组照样算）
      async_base  基线换成另一个异步配置（仍只报生成侧吞吐）
    判据：同一配置下，"async 是否比基线快"的方向是否一致。
    """
    collected = []
    for seed in SEEDS:
        sync_cfg = CONFIGS['sync']
        base_ok = s_accept(simulate(sync_cfg, seed), s_wall(sync_cfg, STEPS))
        base_gen = s_generated(simulate(sync_cfg, seed), s_wall(sync_cfg, STEPS))
        async_base_gen = s_generated(simulate(CONFIGS['async-stale'], seed),
                                     s_wall(CONFIGS['async-stale'], STEPS))
        item = {'seed': seed}
        for name in ('sync', 'sync-pipelined', 'async-stale', 'async-deep'):
            cfg = CONFIGS[name]
            rows = simulate(cfg, seed)
            wall = s_wall(cfg, STEPS)
            s_ok = s_accept(rows, wall)
            item[name] = {'honest': s_ok / base_ok,
                          'gen_num': s_generated(rows, wall) / base_gen,
                          'async_base': s_generated(rows, wall) / async_base_gen}
        collected.append(item)
    out = {'per_config': {}, 'direction_conflict': {}}
    for name in ('sync', 'sync-pipelined', 'async-stale', 'async-deep'):
        block = {}
        for key in ('honest', 'gen_num', 'async_base'):
            vals = [p[name][key] for p in collected]
            block[key] = _ms(vals)
            block[key]['n_ge_1'] = sum(1 for v in vals if v >= 1.0 - 1e-12)
        out['per_config'][name] = block
        out['direction_conflict'][name] = {
            'honest_vs_gen_num': sum(
                1 for p in collected
                if (p[name]['honest'] >= 1.0) != (p[name]['gen_num'] >= 1.0)),
            'honest_vs_async_base': sum(
                1 for p in collected
                if (p[name]['honest'] >= 1.0) != (p[name]['async_base'] >= 1.0)),
            'n': len(collected),
        }
    return out


ELASTIC_VARIANTS = {
    # 引擎与训练同速、窗口宽松：FIFO 下队列里压着的旧组仍被消费，η>0
    'rate1_window8': {'gen_rate': 1.0, 'stale_window': 8},
    # 引擎比训练快 2 倍、窗口深：FIFO 吃满可见窗口（η=31），LIFO 只吃最新一步（η=1）
    'rate2_window32': {'gen_rate': 2.0, 'stale_window': 32},
    # 引擎比训练快 2 倍、窗口只到 8：FIFO 吃满窗口（η=7），LIFO 仍是 1
    'rate2_window8': {'gen_rate': 2.0, 'stale_window': 8},
    # 引擎比训练快 5 倍、窗口 32：FIFO 的 max 仍顶在 31——与上一条的 FIFO 读数相同，
    # 但两臂的均值差（积压深度）更大：只报 max 时这两种情形难以区分
    'rate5_window32': {'gen_rate': 5.0, 'stale_window': 32},
    # 窗口收到 0：旧组全被淘汰，η 被窗口而不是重放顺序决定（两臂都为 0）
    'rate2_window0': {'gen_rate': 2.0, 'stale_window': 0},
    # 引擎比训练慢：训练侧饿着，队列里没有旧组
    'rate0.5_window8': {'gen_rate': 0.5, 'stale_window': 8},
}


def scenario_elastic_window():
    """Q4：弹性窗口下"报 η=0"是否真为 0；什么时候为真、什么时候为假。

    每一档都给 FIFO 与 LIFO 两臂作对照：两臂答案相同说明该档下
    重放顺序不起作用（读数只反映积压深度与窗口），必须写明而不是当成"零陈旧"。
    """
    out = {}
    for variant, knobs in ELASTIC_VARIANTS.items():
        cfg = {'T': 1.00, 'U': 0.10, 'mode': 'async', 'engine_lag': 1}
        cfg.update(knobs)
        block = {}
        for qmode in ('fifo', 'lifo'):
            per_seed = []
            for seed in SEEDS:
                rows = simulate(cfg, seed, queue_mode=qmode)
                eta = [e for r in engine_view(rows) for e in r['eta_true']]
                per_seed.append({'eta_max': float(max(eta)) if eta else 0.0,
                                 'eta_mean': (sum(eta) / len(eta)) if eta else 0.0,
                                 'idle_ratio': 1.0 - len(engine_view(rows)) / float(len(rows))})
            block[qmode] = _mean_std(per_seed, ['eta_max', 'eta_mean', 'idle_ratio'])
        block['fifo_is_zero'] = block['fifo']['eta_max']['mean'] == 0.0
        block['lifo_is_zero'] = block['lifo']['eta_max']['mean'] == 0.0
        block['order_matters'] = block['fifo']['eta_max']['mean'] != block['lifo']['eta_max']['mean']
        out[variant] = block
    return out


def run_all():
    return {
        'configs': {k: dict(v) for k, v in CONFIGS.items()},
        'seeds': len(SEEDS), 'steps': STEPS,
        'identifiability': scenario_identifiability(),
        'single_s': scenario_single_s(),
        'variant_narratives': scenario_variant_narratives(),
        'elastic_window': scenario_elastic_window(),
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
    """NaN 与自身不相等，直接比较会永远失败；把它们替换成固定标记再比。"""
    if isinstance(x, dict):
        return {k: _strip_nan(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_strip_nan(v) for v in x]
    if isinstance(x, float) and x != x:
        return 'NaN'
    return x


def verify():
    if not os.path.exists(SUMMARY_PATH):
        print('summary.json 不存在，先运行一次全量')
        return 1
    with open(SUMMARY_PATH) as f:
        archived = json.load(f)
    fresh = _strip_nan(_canon(run_all()))
    if _strip_nan(_canon(archived)) != fresh:
        print('差异字段：', [k for k in fresh if _strip_nan(_canon(archived.get(k))) != fresh[k]])
        return 1
    print('--verify 通过：与归档 summary.json 逐字段一致')
    return 0


def main():
    if '--verify' in sys.argv:
        return verify()
    with open(SUMMARY_PATH, 'w') as f:
        json.dump(_canon(run_all()), f, ensure_ascii=False, indent=1, sort_keys=True)
    print('写出', SUMMARY_PATH)
    return 0


if __name__ == '__main__':
    sys.exit(main())
