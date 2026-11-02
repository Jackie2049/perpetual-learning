#!/usr/bin/env python3
"""§21.8 C3「同步化基准」的可辨识性（rollout 专项，2026-11-02 M1）。

支撑 notebook/rollout-performance-research.md §21.8。

§21.3 的 C3 规定 async 收益必须相对"同栈 η=0 整步墙钟"报；§21.7 已实证
单独报 S 时 η 不同的配置不可区分，并顺带看到"η=0"有两种形态（同步 vs
流水化同步）。但没有人问过：**那个基准本身能不能从一条 async trace 里估出来？**

本实验只问四件事：

  Q1 「本 trace 里的 η=0 步」能不能当同步基准？（捷径估计 B0，逐选择偏差）
  Q2 从 async trace 估基准需要哪些可观测字段、哪些区间结构性不可辨识？（B2／B2'）
  Q3 两种 η=0 形态（更新期间引擎暂停 / 不暂停）差多少，η 能不能分开它们？
  Q4 陈旧窗口把 async trace 变成"有 η=0 步的 trace"时，代价是什么、捷径是否因此成立？

模型是**因果**的：引擎只能用已发布的权重代生成（生成开始时读一次版本）；
版本只在 update **完成**时推进。墙钟是任意虚拟秒，不可与任何论文报值相除
（§19.1 规则一）。只用标准库；随机性全部来自 random.Random(seed)。

用法：
    python3 c3_sync_baseline.py             # 全量重跑并覆盖 summary.json
    python3 c3_sync_baseline.py --verify    # 重跑并与归档 summary.json 逐字段比对
    python3 -m unittest test_c3_sync_baseline -v
"""

import heapq
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY_PATH = os.path.join(HERE, 'summary.json')

SEEDS = list(range(30))
STEPS = 200
N_GROUPS = 8      # 一条批装的组数（一条批 = 一个训练步的负载）


# ---------------------------------------------------------------- 事件驱动仿真

def simulate(cfg, seed=0):
    """跑一条 trace。

    cfg 字段：
      K            引擎副本数；每副本每 T 产出一条完整批（产出速率 K/T）
      T            单副本生成一条批的基准时长
      U            一次 update 的基准时长
      paused       True = 更新期间完成的副本不重启（colocate 同步形态）
      arrival_cv / update_cv   生成与更新时长的对数正态扰动（0 = 确定性）
      window       陈旧窗口：版本落后超过 window 的批在消费时被淘汰（None = 不限）
      queue        'lifo'（取最新代、同代取最新到达）或 'fifo'

    因果纪律：副本开始生成时读一次已发布版本；生成期间版本再变不影响这条批。
    版本号 = **已完成的 update 数**，只在 update 完成时 +1（消费只触发 update）。
    """
    rng = random.Random(seed)
    K = int(cfg.get('K', 1))
    T = float(cfg['T'])
    U = float(cfg['U'])
    paused = bool(cfg.get('paused', False))
    acv = float(cfg.get('arrival_cv', 0.0))
    ucv = float(cfg.get('update_cv', 0.0))
    window = cfg.get('window', None)
    queue_mode = cfg.get('queue', 'lifo')

    def draw(base, cv):
        if cv <= 0:
            return base
        return base * math.exp(cv * rng.gauss(0.0, 1.0))

    clock = [0.0]
    events = []
    seq = [0]

    def push(t, kind, payload):
        seq[0] += 1
        heapq.heappush(events, (t, seq[0], kind, payload))

    version = 0            # 已完成的 update 数 = 已发布版本号
    pending = []           # [(version, arrival_time, batch_id)]
    arrivals = []
    steps = []
    batch_id = 0
    last_end = 0.0
    updating = False
    paused_replicas = []
    dropped = 0
    idle_waits = 0

    def start_generation(replica):
        nonlocal batch_id
        bid = batch_id
        batch_id += 1
        gen_ver = version
        push(clock[0] + draw(T, acv), 'gen', (replica, bid, gen_ver))

    for i in range(K):
        start_generation(i)

    n_steps = 0
    limit = int(cfg.get('steps', STEPS))
    while n_steps < limit and events:
        t, _, kind, payload = heapq.heappop(events)
        clock[0] = t
        restart = None

        if kind == 'gen':
            replica, bid, gen_ver = payload
            pending.append((gen_ver, t, bid))
            arrivals.append({'t': t, 'version': gen_ver})
            restart = replica

        elif kind == 'update':
            updating = False
            version += 1                       # 版本只在 update 完成时推进
            for replica in paused_replicas:
                start_generation(replica)
            paused_replicas = []

        # 训练侧：不更新、队列里够一条批就消费。陈旧淘汰统一在消费侧做（与 §21.7 一致）。
        if not updating:
            if window is not None:
                keep = []
                for p in pending:
                    if version - p[0] > window:
                        dropped += 1
                    else:
                        keep.append(p)
                pending = keep

            if pending:
                if queue_mode == 'fifo':
                    take = min(pending, key=lambda p: (p[0], p[1], p[2]))
                else:
                    take = max(pending, key=lambda p: (p[0], p[1], p[2]))
                pending = [p for p in pending if p is not take]
                consume_ver = version
                eta = consume_ver - take[0]
                start = t
                waited = start > last_end + 1e-12
                u = draw(U, ucv)
                update_end = start + u
                updating = True
                push(update_end, 'update', None)
                steps.append({
                    'idx': n_steps,
                    'start': start, 'end': update_end, 'wall': update_end - last_end,
                    'wait': start - last_end, 'u': u,
                    'eta': eta, 'waited': waited,
                })
                last_end = update_end
                n_steps += 1
            else:
                idle_waits += 1

        # 本时刻的事件已全部处理完，才决定刚才完成的副本能否重启：
        # colocate 暂停形态下，训练侧此刻已经开始 update，副本必须等 update 结束。
        if restart is not None:
            if paused and updating:
                paused_replicas.append(restart)
            else:
                start_generation(restart)

    return {'steps': steps, 'arrivals': arrivals, 'horizon': last_end,
            'dropped': dropped, 'idle_waits': idle_waits, 'cfg': dict(cfg)}


# ---------------------------------------------------------------- 小工具


def quantile(xs, q):
    ys = sorted(xs)
    if not ys:
        return float('nan')
    k = (len(ys) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(ys) - 1)
    return ys[lo] + (ys[hi] - ys[lo]) * (k - lo)


def summarize(items, keys):
    out = {}
    for k in keys:
        vals = [it[k] for it in items if it[k] == it[k]]
        if not vals:
            out[k] = {'mean': float('nan'), 'std': float('nan'),
                      'min': float('nan'), 'max': float('nan'), 'n': 0}
            continue
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1)
        out[k] = {'mean': m, 'std': var ** 0.5, 'min': min(vals),
                  'max': max(vals), 'n': len(vals)}
    return out


def trace_stats(tr):
    steps = tr['steps']
    n = len(steps)
    eta0 = [s for s in steps if s['eta'] == 0]
    eta0_later = [s for s in eta0 if s['idx'] > 0]
    return {
        'n_steps': n,
        'horizon': tr['horizon'],
        'n_arrivals': len(tr['arrivals']),
        'dropped': tr['dropped'],
        'idle_waits': tr['idle_waits'],
        'eta0_frac': len(eta0) / float(n) if n else float('nan'),
        'eta0_frac_later': len(eta0_later) / float(n) if n else float('nan'),
        'waited_frac': sum(1 for s in steps if s['waited']) / float(n) if n else float('nan'),
        'eta_max_mean': max(s['eta'] for s in steps) if steps else float('nan'),
        'eta_mean': sum(s['eta'] for s in steps) / float(n) if n else float('nan'),
        'step_wall_mean': sum(s['wall'] for s in steps) / float(n) if n else float('nan'),
        'step_wall_p50': quantile([s['wall'] for s in steps], 0.5),
        'step_wall_min': min(s['wall'] for s in steps) if steps else float('nan'),
        'no_wait_frac': sum(1 for s in steps if not s['waited']) / float(n) if n else float('nan'),
    }


# ---------------------------------------------------------------- 估计量


def b0_eta0_step_wall(tr):
    """捷径 B0：把「本 trace 里 η=0 的步」的整步墙钟当同步基准。"""
    walls = [s['wall'] for s in tr['steps'] if s['eta'] == 0]
    if not walls:
        return None, 0
    return sum(walls) / len(walls), len(walls)


def b0_split(tr):
    """把 B0 拆成「第一步」与「其余 η=0 步」——两者读的是完全不同的东西。"""
    first = [s['wall'] for s in tr['steps'] if s['eta'] == 0 and s['idx'] == 0]
    later = [s['wall'] for s in tr['steps'] if s['eta'] == 0 and s['idx'] > 0]
    return {'first': (first[0] if first else float('nan')),
            'later_mean': (sum(later) / len(later) if later else float('nan')),
            'n_later': len(later)}


def b2_count_minus_idle(tr, K):
    """B2：到达计数估 T、无等待步估 U，合成 T + U。

    条件：① 到达计数完整（否则 T̂ 有偏）② 存在"无等待步"（否则 Û 无定义）。
    """
    st = trace_stats(tr)
    if st['n_arrivals'] == 0 or st['horizon'] <= 0:
        return None
    t_hat = K * tr['horizon'] / float(st['n_arrivals'])
    nw = [s['wall'] for s in tr['steps'] if not s['waited']]
    if not nw:
        return None
    u_hat = quantile(nw, 0.5)
    return {'t_hat': t_hat, 'u_hat': u_hat, 'baseline': t_hat + u_hat}


def b2_naive_min_wall(tr, K):
    """B2-naive：用「最小整步墙钟」当 U（没有无等待步时的常见替代，含等待污染）。"""
    st = trace_stats(tr)
    if st['n_arrivals'] == 0 or st['horizon'] <= 0:
        return None
    t_hat = K * tr['horizon'] / float(st['n_arrivals'])
    u_hat = st['step_wall_min']
    return {'t_hat': t_hat, 'u_hat': u_hat, 'baseline': t_hat + u_hat}


def b2_median_gap(tr, K):
    """B2'：到达间隔中位数 × K 估 T（错误做法的对照；K 个过程叠加后中位数远小于 T/K）。"""
    ts = sorted(a['t'] for a in tr['arrivals'])
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    if not gaps:
        return None
    t_hat = K * quantile(gaps, 0.5)
    nw = [s['wall'] for s in tr['steps'] if not s['waited']]
    u_hat = quantile(nw, 0.5) if nw else st_min_wall(tr)
    return {'t_hat': t_hat, 'u_hat': u_hat, 'baseline': t_hat + u_hat}


def st_min_wall(tr):
    return min(s['wall'] for s in tr['steps'])


def arrivals_during_updates(tr):
    """暂停策略的可观测指纹：有多少条批是在某次更新窗口内到达的。"""
    total = 0
    for s in tr['steps']:
        for a in tr['arrivals']:
            if s['start'] < a['t'] < s['end'] - 1e-12:
                total += 1
    return total


# ---------------------------------------------------------------- 场景


def cfg_of(K, u_over_t, **kw):
    T = 1.0
    cfg = {'K': K, 'T': T, 'U': u_over_t * T}
    cfg.update(kw)
    return cfg


ASYNC_KS = [1, 2, 4, 8]
ASYNC_US = [('u0.1', 0.1), ('u0.3', 0.3), ('u0.7', 0.7), ('u2.0', 2.0)]


def scenario_b0_bias():
    """Q1：把「本 trace 的 η=0 步」当基准会偏多少、往哪偏。

      truth      = T + U（同步基准的定义值）
      speed_true = truth / cycle（真实 async 相对同步的倍数）
      B0         = η=0 步的平均墙钟
      speed_B0   = truth / B0（拿 B0 当基准时报出的倍数）
    """
    out = {}
    for k in ASYNC_KS:
        for uname, r in ASYNC_US:
            key = 'K%d_%s' % (k, uname)
            per_seed = []
            for seed in SEEDS:
                tr = simulate(cfg_of(k, r, queue='lifo', window=None), seed)
                st = trace_stats(tr)
                truth = 1.0 + r
                b0, n0 = b0_eta0_step_wall(tr)
                sp = b0_split(tr)
                cycle = st['step_wall_mean']
                per_seed.append({
                    'truth': truth, 'cycle': cycle,
                    'b0': b0 if b0 is not None else float('nan'),
                    'b0_defined': b0 is not None,
                    'b0_first': sp['first'], 'b0_later': sp['later_mean'],
                    'n_eta0': n0, 'n_eta0_later': sp['n_later'],
                    'eta0_frac': st['eta0_frac'], 'eta0_frac_later': st['eta0_frac_later'],
                    'speed_true': truth / cycle if cycle > 0 else float('nan'),
                    'speed_b0': truth / b0 if b0 else float('nan'),
                    'eta_max': st['eta_max_mean'],
                })
            out[key] = summarize(per_seed,
                                 ['truth', 'cycle', 'b0', 'b0_first', 'b0_later',
                                  'n_eta0', 'n_eta0_later', 'eta0_frac', 'eta0_frac_later',
                                  'speed_true', 'speed_b0', 'eta_max'])
            out[key]['b0_defined_frac'] = sum(1 for p in per_seed if p['b0_defined']) / float(len(per_seed))
            out[key]['speed_ratio_b0_over_true'] = (
                out[key]['speed_b0']['mean'] / out[key]['speed_true']['mean']
                if out[key]['speed_true']['mean'] else float('nan'))
    return out


def scenario_two_eta0_forms():
    """Q3：两种 η=0 形态——更新期间引擎暂停 vs 不暂停。

    K=1 与 K=4 各扫 U：gap 应为 min(T, U) 与 min(T/K, U)。
    """
    out = {}
    for k in (1, 4):
        for uname, r in (('u0.1', 0.1), ('u0.5', 0.5), ('u2.0', 2.0)):
            for form, paused in (('paused', True), ('pipelined', False)):
                key = 'K%d_%s_%s' % (k, form, uname)
                per_seed = []
                for seed in SEEDS:
                    tr = simulate(cfg_of(k, r, paused=paused), seed)
                    st = trace_stats(tr)
                    per_seed.append({
                        'step_wall_mean': st['step_wall_mean'],
                        'eta_max_mean': st['eta_max_mean'],
                        'eta_mean': st['eta_mean'],
                        'eta0_frac': st['eta0_frac'],
                        'arrivals_in_update': arrivals_during_updates(tr),
                        'n_arrivals': st['n_arrivals'],
                    })
                out[key] = summarize(per_seed, ['step_wall_mean', 'eta_max_mean', 'eta_mean',
                                                'eta0_frac', 'arrivals_in_update', 'n_arrivals'])
    out['gap'] = {}
    out['regime'] = {}
    for k in (1, 4):
        for uname, r in (('u0.1', 0.1), ('u0.5', 0.5), ('u2.0', 2.0)):
            a = out['K%d_paused_%s' % (k, uname)]['step_wall_mean']['mean']
            b = out['K%d_pipelined_%s' % (k, uname)]['step_wall_mean']['mean']
            key = 'K%d_%s' % (k, uname)
            out['gap'][key] = a - b
            # 绑定约束：U 与 T/K 谁更大。更新受限时暂停不损失墙钟（队列吸收），
            # 生成受限时暂停才把等待显式化。
            out['regime'][key] = 'update-limited' if r >= 1.0 / k else 'generation-limited'
    out['eta_cannot_separate'] = {
        'K%d_%s' % (k, uname): (
            abs(out['K%d_paused_%s' % (k, uname)]['eta_max_mean']['mean'] -
                out['K%d_pipelined_%s' % (k, uname)]['eta_max_mean']['mean']) < 1e-9 and
            abs(out['K%d_paused_%s' % (k, uname)]['eta_mean']['mean'] -
                out['K%d_pipelined_%s' % (k, uname)]['eta_mean']['mean']) < 1e-9)
        for k in (1, 4) for uname in ('u0.1', 'u0.5', 'u2.0')}
    return out


def scenario_estimators():
    """Q2：B2／B2-naive／B2' 在各区的偏差，与「无等待步」的可辨识边界。"""
    out = {}
    cases = [('det', 0.0, 0.0), ('jitter_T', 0.3, 0.0),
             ('jitter_U', 0.0, 0.3), ('jitter_both', 0.3, 0.3)]
    for cname, acv, ucv in cases:
        for uname, r in (('u0.1', 0.1), ('u0.7', 0.7)):
            for k in (1, 4):
                key = '%s_K%d_%s' % (cname, k, uname)
                per_seed = []
                for seed in SEEDS:
                    cfg = cfg_of(k, r, arrival_cv=acv, update_cv=ucv)
                    tr = simulate(cfg, seed)
                    st = trace_stats(tr)
                    truth = 1.0 + r
                    e1 = b2_count_minus_idle(tr, k)
                    e2 = b2_median_gap(tr, k)
                    e3 = b2_naive_min_wall(tr, k)
                    per_seed.append({
                        'truth': truth,
                        'b2': (e1['baseline'] if e1 else float('nan')),
                        'b2_defined': e1 is not None,
                        'b2_u': (e1['u_hat'] if e1 else float('nan')),
                        'b2m': (e2['baseline'] if e2 else float('nan')),
                        'b2m_t': (e2['t_hat'] if e2 else float('nan')),
                        'b2n': (e3['baseline'] if e3 else float('nan')),
                        'no_wait_frac': st['no_wait_frac'],
                        'waited_frac': st['waited_frac'],
                        'cycle': st['step_wall_mean'],
                    })
                out[key] = summarize(per_seed, ['truth', 'b2', 'b2_u', 'b2m', 'b2m_t',
                                                'b2n', 'no_wait_frac', 'waited_frac', 'cycle'])
                out[key]['b2_defined_frac'] = sum(1 for p in per_seed if p['b2_defined']) / float(len(per_seed))
    return out


def scenario_regime_boundary():
    """Q2 的结构部分：K=4、T=1，扫 U/T 过相变点 U = T/K。

    给出「无等待步占比」（Û 能否观测）、B0 偏差、B2/B2-naive 的误差。
    """
    out = {}
    for uname, r in (('u0.05', 0.05), ('u0.2', 0.2), ('u0.24', 0.24),
                     ('u0.26', 0.26), ('u0.3', 0.3), ('u0.6', 0.6), ('u1.2', 1.2)):
        per_seed = []
        for seed in SEEDS:
            tr = simulate(cfg_of(4, r), seed)
            st = trace_stats(tr)
            truth = 1.0 + r
            b0, n0 = b0_eta0_step_wall(tr)
            e1 = b2_count_minus_idle(tr, 4)
            e3 = b2_naive_min_wall(tr, 4)
            per_seed.append({
                'truth': truth,
                'no_wait_frac': st['no_wait_frac'],
                'eta0_frac': st['eta0_frac'],
                'n_eta0': n0,
                'cycle': st['step_wall_mean'],
                'speed_true': truth / st['step_wall_mean'],
                'b0': b0 if b0 is not None else float('nan'),
                'b0_defined': b0 is not None,
                'b2': (e1['baseline'] if e1 else float('nan')),
                'b2_defined': e1 is not None,
                'b2n': (e3['baseline'] if e3 else float('nan')),
            })
        out[uname] = summarize(per_seed, ['truth', 'no_wait_frac', 'eta0_frac', 'n_eta0',
                                          'cycle', 'speed_true', 'b0', 'b2', 'b2n'])
        out[uname]['b2_defined_frac'] = sum(1 for p in per_seed if p['b2_defined']) / float(len(per_seed))
        out[uname]['b2_err'] = out[uname]['b2']['mean'] / out[uname]['truth']['mean'] - 1.0
        out[uname]['b2n_err'] = out[uname]['b2n']['mean'] / out[uname]['truth']['mean'] - 1.0
    return out


def scenario_window_escape():
    """Q4：用陈旧窗口把 async trace 变成「有 η=0 步的 trace」，代价是什么。

    window=0 时所有旧批被淘汰，训练侧只能等新代批 ⇒ η=0，但生成量被浪费；
    此时 B0 反而等于真值——因为这条 trace 已经被窗口改造成了同步节奏。
    """
    out = {}
    for wname, w in (('w_none', None), ('w8', 8), ('w2', 2), ('w0', 0)):
        per_seed = []
        for seed in SEEDS:
            cfg = cfg_of(4, 0.3, queue='fifo', window=w)
            tr = simulate(cfg, seed)
            st = trace_stats(tr)
            truth = 1.0 + 0.3
            b0, n0 = b0_eta0_step_wall(tr)
            per_seed.append({
                'truth': truth,
                'n_gen': st['n_arrivals'],
                'n_dropped': st['dropped'],
                'drop_frac': st['dropped'] / float(st['n_arrivals'] + st['dropped'])
                             if (st['n_arrivals'] + st['dropped']) else float('nan'),
                'idle_waits': st['idle_waits'],
                'cycle': st['step_wall_mean'],
                'speed_true': truth / st['step_wall_mean'],
                'eta0_frac': st['eta0_frac'],
                'n_eta0': n0,
                'b0': b0 if b0 is not None else float('nan'),
                'b0_defined': b0 is not None,
            })
        out[wname] = summarize(per_seed, ['truth', 'n_gen', 'n_dropped', 'drop_frac',
                                          'idle_waits', 'cycle', 'speed_true',
                                          'eta0_frac', 'n_eta0', 'b0'])
        out[wname]['b0_defined_frac'] = sum(1 for p in per_seed if p['b0_defined']) / float(len(per_seed))
        out[wname]['b0_err'] = out[wname]['b0']['mean'] / out[wname]['truth']['mean'] - 1.0
    return out


def scenario_b0_on_sync():
    """对照：同一个 B0 估计量用在同步 trace 上，读到的就是基准本身。"""
    out = {}
    for form, paused in (('paused', True), ('pipelined', False)):
        per_seed = []
        for seed in SEEDS:
            tr = simulate(cfg_of(1, 0.3, paused=paused), seed)
            st = trace_stats(tr)
            b0, n0 = b0_eta0_step_wall(tr)
            per_seed.append({'truth': 1.3,
                             'b0': b0 if b0 is not None else float('nan'),
                             'eta0_frac': st['eta0_frac'],
                             'step_wall_mean': st['step_wall_mean']})
        block = summarize(per_seed, ['truth', 'b0', 'eta0_frac', 'step_wall_mean'])
        block['b0_minus_truth'] = block['b0']['mean'] - block['truth']['mean']
        out[form] = block
    return out


def scenario_sampling_error():
    """Q2 的后半截：短 trace 上「η=0 步」有多稀有、T̂ 的采样误差有多大。

    再跑一遍 §21.7 的教训——一个量在长期极限里"几乎总是"成立，不代表在
    实际报值长度（几十到几百步）里可观测。
    """
    out = {}
    for steps in (20, 50, 200):
        per_seed = []
        for seed in SEEDS:
            cfg = cfg_of(4, 0.3)
            cfg['steps'] = steps
            tr = simulate(cfg, seed)
            st = trace_stats(tr)
            truth = 1.3
            e1 = b2_count_minus_idle(tr, 4)
            per_seed.append({
                'truth': truth,
                'n_eta0': None,
                'eta0_frac': st['eta0_frac'],
                'b2_defined': e1 is not None,
                'b2': (e1['baseline'] if e1 else float('nan')),
                'b2_t': (e1['t_hat'] if e1 else float('nan')),
                'n_arrivals': st['n_arrivals'],
            })
            b0, n0 = b0_eta0_step_wall(tr)
            per_seed[-1]['n_eta0'] = n0
        block = summarize(per_seed, ['truth', 'eta0_frac', 'b2', 'b2_t', 'n_arrivals', 'n_eta0'])
        block['b2_defined_frac'] = sum(1 for p in per_seed if p['b2_defined']) / float(len(per_seed))
        block['b2_err_mean'] = (
            (block['b2']['mean'] - block['truth']['mean']) / block['truth']['mean']
            if block['b2']['n'] else float('nan'))
        block['b2_t_err_mean'] = (
            (block['b2_t']['mean'] - 1.0) if block['b2_t']['n'] else float('nan'))
        block['t_hat_std'] = block['b2_t']['std']
        out['steps%d' % steps] = block
    return out


def scenario_warmup_dependence():
    """Q1 的机制：那条 η=0 的步是不是稳态量？换个 trace 起点再测。

    本模型里 η=0 在结构上只能出现在第 0 步（引擎在训练侧还没更新时就产出了
    第 0 代批）。于是「丢弃前 R 步」这个纯读法选择就决定了 B0 是否存在：
      R=0 → 有一步 η=0，B0 = T + U（冷启动的墙钟恰好就是同步基准）
      R≥1 → 一步都没有，B0 无定义
    若 B0 只在某一种起点选择下存在，它就不是可以跨系统复用的基准。
    """
    out = {}
    for discard in (0, 1, 5):
        for k in (1, 4):
            for uname, r in (('u0.3', 0.3), ('u2.0', 2.0)):
                key = 'discard%d_K%d_%s' % (discard, k, uname)
                per_seed = []
                for seed in SEEDS:
                    tr = simulate(cfg_of(k, r), seed)
                    steps = tr['steps'][discard:]
                    n0 = sum(1 for s in steps if s['eta'] == 0)
                    b0 = (sum(s['wall'] for s in steps if s['eta'] == 0) / n0) if n0 else None
                    cycle = sum(s['wall'] for s in steps) / len(steps) if steps else float('nan')
                    per_seed.append({
                        'n_eta0': n0,
                        'b0_defined': b0 is not None,
                        'b0': b0 if b0 is not None else float('nan'),
                        'cycle': cycle,
                        'truth': 1.0 + r,
                        'speed_true': (1.0 + r) / cycle if cycle else float('nan'),
                        'speed_b0': (1.0 + r) / b0 if b0 else float('nan'),
                    })
                out[key] = summarize(per_seed, ['n_eta0', 'b0', 'cycle', 'truth',
                                                'speed_true', 'speed_b0'])
                out[key]['b0_defined_frac'] = sum(1 for p in per_seed if p['b0_defined']) / float(len(per_seed))
    return out


def run_all():
    return {
        'config': {'seeds': len(SEEDS), 'steps': STEPS, 'n_groups': N_GROUPS},
        'b0_bias': scenario_b0_bias(),
        'two_eta0_forms': scenario_two_eta0_forms(),
        'estimators': scenario_estimators(),
        'regime_boundary': scenario_regime_boundary(),
        'window_escape': scenario_window_escape(),
        'sampling_error': scenario_sampling_error(),
        'warmup_dependence': scenario_warmup_dependence(),
        'b0_on_sync': scenario_b0_on_sync(),
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
