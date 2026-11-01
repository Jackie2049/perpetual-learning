#!/usr/bin/env python3
"""§21.2 C2 事件边界拆分约定的可执行性实证（rollout 专项，2026-11-01 M1）。

支撑 notebook/rollout-performance-research.md §21.4。

模拟一个最小 async-RL 交接：训练侧产生新权重（B3 链）、引擎侧要先把 replica 的
KV 迁回来才能吃满一批（B2 链）。C2 用三个事件时间戳定义切分
    t0 = 训练侧 update 返回
    t1 = 新权重引擎侧生效
    t2 = 引擎恢复满批
即 B3 = t1 − t0、B2 = t2 − t1。本实验问三件事：

  Q1 t1 在引擎侧报不出来（它看不见权重广播何时结束）。换成可观测的代理，
     B2/B3 偏多少，会不会把两者孰大孰小读反（order flip）。
  Q2 C2 的"无法分离时整体记 B2∪B3"，相对真实 B2+B3 偏多少（保守规则的代价）。
  Q3 拿运行期 microbenchmark 常数当 B2（C2 第九条明文禁止），代价是多少。

墙钟是任意虚拟秒，不可与任何论文报值相除（§19.1 规则一）。
只用标准库；随机性全部来自 random.Random(seed)。

用法：
    python3 c2_split_ablation.py             # 全量重跑并覆盖 summary.json
    python3 c2_split_ablation.py --verify    # 重跑并与归档 summary.json 逐字段比对
    python3 -m unittest test_c2_split_ablation -v
"""

import json
import math
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY_PATH = os.path.join(HERE, 'summary.json')

SEEDS = list(range(30))
BATCH_STEPS = 40
BATCH_MEAN = 50
BATCH_CV = 0.6

# b2b3: 两条链的标称 base 时长；pressure: 每多一个在途回迁 token 多等多少
CONFIGS = {
    'near-balance': {'b2': 8.0, 'b3': 8.0, 'pressure': 0.10, 'note': 'B2、B3 量级相当'},
    'update-heavy': {'b2': 6.0, 'b3': 12.0, 'pressure': 0.15, 'note': '更新侧主导'},
    'gen-heavy': {'b2': 12.0, 'b3': 6.0, 'pressure': 0.15, 'note': '生成侧主导'},
}


class Task(object):
    __slots__ = ('name', 'dur', 'deps', 'busy')

    def __init__(self, name, dur, deps, busy=True):
        self.name = name
        self.dur = dur
        self.deps = deps
        self.busy = busy


def jitter(rng, x, frac):
    return x * (1.0 + frac * (2.0 * rng.random() - 1.0))


def build_scenario(cfg, rng):
    """时间原点 t0 = 0。

    prefill 只依赖新权重（KV 冷启动），KV 回迁是并行的另一件事；
    批请求要等 prefill 与 kv_migrate 都完成 —— 于是 t2 与 t1 之间才有非零的 B2。
    """
    tasks = [
        Task('weight_pack', jitter(rng, 0.4, 0.30), [], busy=False),
        Task('weight_bcast', jitter(rng, cfg['b3'], 0.15), ['weight_pack']),
        Task('replica_sleep', jitter(rng, 0.3, 0.30), [], busy=False),
        Task('engine_init', jitter(rng, 0.5, 0.30), ['replica_sleep'], busy=False),
        # 预热用的 prefill 只要新权重就绪就能开；它本身是引擎算力，不进 B2
        Task('prefill', jitter(rng, 1.0, 0.10), ['weight_bcast']),
        Task('kv_migrate', jitter(rng, cfg['b2'], 0.15), ['replica_sleep']),
    ]
    for i in range(BATCH_STEPS):
        # 批必须在新权重生效（weight_bcast）、本机 KV 就绪、且预热 prefill 跑完之后才开跑。
        # 只依赖 kv_migrate 会让批抢在 t1 之前开跑，否定 C2"t2 在 t1 之后"的前提
        tasks.append(Task('batch_%03d' % i, 1.0, ['weight_bcast', 'kv_migrate', 'prefill']))
    return tasks


def simulate(tasks, pressure):
    """确定性事件仿真，早开始者先跑、同时刻按名字排序。"""
    end = {}
    events = []
    ready = sorted([t for t in tasks if not t.deps], key=lambda t: t.name)
    while ready:
        ready.sort(key=lambda t: (max([end[d] for d in t.deps] or [0.0]), t.name))
        t = ready.pop(0)
        start = max([end[d] for d in t.deps] or [0.0])
        inflight = sum(1 for e in events if e['busy'] and e['start'] < start < e['end'])
        end[t.name] = start + t.dur * (1.0 + pressure * inflight)
        events.append({'name': t.name, 'start': start, 'end': end[t.name], 'busy': t.busy})
        for other in tasks:
            if t.name in other.deps and all(d in end for d in other.deps):
                ready.append(other)
    events.sort(key=lambda e: (e['start'], e['name']))
    # 批起点是"引擎恢复满批"的 t2，不是批跑完的时刻
    end['_batch_start'] = min(e['start'] for e in events if e['name'].startswith('batch_'))
    return events, end


def batch_len(rng):
    return max(1, int(rng.lognormvariate(math.log(BATCH_MEAN) - 0.5 * BATCH_CV ** 2, BATCH_CV)))


def t1_of(end):
    """t1 = 新权重在引擎侧生效的时刻，只由权重广播决定。
    KV 回迁跑得比广播久并不推迟权重生效，它推迟的是 t2 —— 这正是 B2 的来源。"""
    return end['weight_bcast']


def truth_of(end):
    """真值：B3 = t1 − t0，B2 = t2 − t1，t2 取第一个批请求的起点。
    prefill 与 KV 回迁并行，故 t2 早于两条链的朴素串行和——这正是要测的那部分。"""
    t0 = 0.0
    t1 = t1_of(end)
    return {'B2': end['_batch_start'] - t1, 'B3': t1 - t0,
            'B2U3': end['_batch_start'] - t0}


def observable(events, end, tasks):
    """引擎侧真能看见的事件（不看仿真真值）：
    预热 prefill 的起止、广播结束 ack、KV 回迁结束、批起点序列。"""
    return {
        'prefill_start': [e for e in events if e['name'] == 'prefill'][0]['start'],
        'prefill_end': end['prefill'],
        'bcast_end': end['weight_bcast'],
        'kv_end': end['kv_migrate'],
        'batch_starts': sorted(e['start'] for e in events if e['name'].startswith('batch_')),
    }


def readings(obs, n_batch, cfg):
    """七个读法。只用可观测事件与标称步数，不碰 truth_of。

    所有读法都按 §21.2 的 C2 原式，切点固定为"引擎开始吃新权重"（t_start，可观测），
    只有"切点是哪一刻"和"在途的那段算给谁"不同。
    """
    t0 = 0.0
    t_start = obs['prefill_start']
    t_held = obs['batch_starts'][min(n_batch, len(obs['batch_starts'])) - 1]
    return {
        # ① 用标称步数代替测量：既没测 B2 也没测 B3
        'naive_sep': {'B2': cfg['b2'], 'B3': cfg['b3']},
        # ② 把 prefill 起点当作 t1：这是引擎真能观测到的时刻，但在 KV 回迁更慢时偏早
        'held_sep': {'B2': obs['kv_end'] - t_start, 'B3': t_start},
        # ③ 用广播 ack 当 t1：无先验时最常见的误取（ack 早于权重真正生效）
        'ack_sep': {'B2': obs['bcast_end'] + 0.05 - t_start, 'B3': t_start},
        # ④ 把"批被拖住的时刻"当作 t1 —— 本文件不采用，留在 test 里作负向对照
        # ⑤ C2 的保守规则：从同一起点整体记 B2∪B3，不猜切点
        'union': {'B2U3': t_held},
        # ⑤ microbenchmark 常数
        'micro': {'B2U3': 0.85 * cfg['b2']},
    }


def rel(x, y, scale=None):
    """相对误差。B2 可能恰为 0（两条链同时结束），此时用整个交接时窗做分母。"""
    denom = y if y else scale
    return (x - y) / denom if denom else float('nan')


def run_config(cfg):
    rows = []
    for seed in SEEDS:
        rng = random.Random(seed)
        tasks = build_scenario(cfg, rng)
        events, end = simulate(tasks, cfg['pressure'])
        obs = observable(events, end, tasks)
        n = batch_len(rng)
        rows.append({'seed': seed,
                     'batch_len': n,
                     'truth': truth_of(end),
                     'readings': readings(obs, n, cfg)})
    return rows


def aggregate(rows):
    agg = {'truth': {k: statistics.mean(r['truth'][k] for r in rows) for k in ('B2', 'B3', 'B2U3')}}
    for key in ('naive_sep', 'held_sep', 'ack_sep'):
        e2 = [rel(r['readings'][key]['B2'], r['truth']['B2'], r['truth']['B2U3']) for r in rows]
        e3 = [rel(r['readings'][key]['B3'], r['truth']['B3']) for r in rows]
        agg[key] = {
            'B2_mean': statistics.mean(r['readings'][key]['B2'] for r in rows),
            'B3_mean': statistics.mean(r['readings'][key]['B3'] for r in rows),
            'B2_rel_err_mean': statistics.mean(e2),
            'B3_rel_err_mean': statistics.mean(e3),
            'B2_negative': sum(1 for r in rows if r['readings'][key]['B2'] < 0),
            'order_flipped': sum(1 for r in rows
                                 if (r['readings'][key]['B2'] > r['readings'][key]['B3'])
                                 != (r['truth']['B2'] > r['truth']['B3'])),
        }
    # 整体读法的真值：t0 起算到满批，即 B3 + 真实 B2（B2U3 = t2 − t0）
    for r in rows:
        r['truth']['handoff'] = r['truth']['B2U3']
    agg['truth']['handoff'] = agg['truth']['B2U3']
    for key in ('union', 'micro'):
        e = [rel(r['readings'][key]['B2U3'], r['truth']['handoff']) for r in rows]
        agg[key] = {
            'B2U3_mean': statistics.mean(r['readings'][key]['B2U3'] for r in rows),
            'rel_err_mean': statistics.mean(e),
        }
    return agg


def build_summary():
    out = {
        'meta': {
            'title': 'C2 事件边界拆分约定的可执行性实证',
            'doc': 'notebook/rollout-performance-research.md §21.4',
            'seeds': len(SEEDS),
            'unit': '虚拟秒；与任何论文报值不可相除（§19.1 规则一）',
            'caveat': '进程内事件图仿真，无真实 GPU 或引擎；t1 由仿真真值给出，'
                      '各读法只用可观测事件',
        },
        'configs': {},
    }
    for name, cfg in CONFIGS.items():
        rows = run_config(cfg)
        out['configs'][name] = {'config': cfg, 'aggregate': aggregate(rows), 'per_seed': rows}
    return out


def main(argv):
    out = build_summary()
    text = json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True)
    if '--verify' in argv:
        with open(SUMMARY_PATH) as f:
            ref = json.load(f)
        ok = json.dumps(ref, ensure_ascii=False, indent=1, sort_keys=True) == text
        print('VERIFY OK: 逐字段相同' if ok else 'VERIFY FAIL: 与归档不一致')
        return 0 if ok else 1
    with open(SUMMARY_PATH, 'w') as f:
        f.write(text + '\n')
    for name in CONFIGS:
        a = out['configs'][name]['aggregate']
        print('%-13s truth B2=%.2f B3=%.2f | naive B2 %+.0f%% flips=%d | held B2 %+.0f%% flips=%d '
              '| ack B2 %+.0f%% | union %+.0f%%'
              % (name, a['truth']['B2'], a['truth']['B3'],
                 100 * a['naive_sep']['B2_rel_err_mean'], a['naive_sep']['order_flipped'],
                 100 * a['held_sep']['B2_rel_err_mean'], a['held_sep']['order_flipped'],
                 100 * a['ack_sep']['B2_rel_err_mean'],
                 100 * a['union']['rel_err_mean']))
    print('wrote', SUMMARY_PATH)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
