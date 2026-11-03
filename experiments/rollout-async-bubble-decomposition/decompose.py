#!/usr/bin/env python3
"""§11.8 验证：async 下的非计算墙钟能否被 B2／B5 完全解释，η 是否与之正交

背景：§11.7 推出「async 下与 B1 对偶的不是新气泡，是 B2 的显性化」，但未验证。

**§11.8 第一版的错误（留档）**：最初用稳态公式算利用率
（`engine_util = 吞吐/容量`），得到 `both_idle_B2 ≡ 0`，因为训练侧在稳态公式里
永远是满的。错在**把"利用率"当成"时间占比"**——B2 问的是「有多少比例的墙钟
同时满足两侧都没满载」，这必须逐时刻看，不能用稳态平均。

本版改为事件驱动：维护"引擎空闲槽位数"与"训练侧是否在跑"两个状态，
按事件推进时间，**在每个时间片上判定它属于哪一类**。

用法：
    python3 decompose.py
    python3 decompose.py --verify
"""

import heapq
import json
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent
SUMMARY = HERE / 'summary.json'
SEEDS = list(range(30))


def simulate(cfg, seed):
    """事件驱动，返回三级时间分解（秒）。

    三类非计算墙钟：
      B2  两侧都未满载（引擎有槽位空闲 且 训练侧没在算）
      B5  引擎全闲（没有任何槽位在产）而训练侧在算 —— 训练侧在空等数据
      comp 其余时间（至少有一侧在工作，且不是 B2/B5）
    """
    rng = random.Random(seed)
    K, T, U = cfg['K'], cfg['T'], cfg['U']
    rate = cfg.get('gen_rate', 1.0)
    steps = cfg.get('steps', 200)
    cv = cfg.get('cv', 0.0)

    def draw(base):
        return base * (1.0 + rng.gauss(0, cv)) if cv else base

    events = []
    seq = [0]

    def push(t, kind, payload):
        seq[0] += 1
        heapq.heappush(events, (t, seq[0], kind, payload))

    version = 0
    pending = []
    clock = 0.0
    batch_id = 0
    train_free_at = 0.0
    n_steps = 0
    last_t = 0.0

    # 时间记账
    t_b2 = 0.0      # 两侧都闲
    t_b5 = 0.0      # 引擎全闲、训练在等
    t_comp = 0.0    # 其余

    busy_slots = 0      # 当前在产的槽位数

    def start_gen():
        nonlocal batch_id, busy_slots
        bid = batch_id
        batch_id += 1
        busy_slots += 1
        push(clock + draw(T) / rate, 'gen', (bid, version))

    for _ in range(K):
        start_gen()

    while n_steps < steps and events:
        t, _, kind, payload = heapq.heappop(events)

        # --- 在 [last_t, t) 这段时间里，判定当时的状态 ---
        # 状态必须在**上一个事件处理完之后**读取：用 [last_t, t) 区间开始时的
        # busy_slots / train_busy。初版把这段写在"处理当前事件"之后，读到的是
        # 下一个状态，导致 B2/B5 恒为 0。
        dt = t - last_t
        if dt > 0:
            train_busy = last_t < train_free_at - 1e-12
            engine_has_free = busy_slots < K
            engine_all_idle = busy_slots == 0
            if engine_has_free and not train_busy:
                t_b2 += dt
            elif engine_all_idle and train_busy:
                t_b5 += dt
            else:
                t_comp += dt

        last_t = t
        clock = t

        if kind == 'gen':
            busy_slots -= 1
            bid, gv = payload
            pending.append((gv, t))
            start_gen()

        elif kind == 'update':
            version += 1

        if clock >= train_free_at - 1e-12 and pending:
            pending.pop(0)
            u = draw(U)
            train_free_at = clock + u
            push(train_free_at, 'update', None)
            n_steps += 1

    total = t_b2 + t_b5 + t_comp
    return {'b2': t_b2, 'b5': t_b5, 'comp': t_comp, 'total': total,
            'b2_share': t_b2 / total if total else 0.0,
            'b5_share': t_b5 / total if total else 0.0,
            'comp_share': t_comp / total if total else 0.0}


def scenario_speed_ratio():
    """引擎/训练速度比扫描：B2 与 B5 如何随它变化。"""
    out = {}
    for rate in (0.3, 0.5, 1.0, 2.0, 4.0):
        cfg = {'K': 4, 'T': 1.0, 'U': 0.5, 'gen_rate': rate}
        rows = [simulate(cfg, s) for s in SEEDS]
        out['rate%.1f' % rate] = {
            'b2_share': sum(r['b2_share'] for r in rows) / len(rows),
            'b5_share': sum(r['b5_share'] for r in rows) / len(rows),
            'comp_share': sum(r['comp_share'] for r in rows) / len(rows),
        }
    return out


def scenario_capacity():
    """并行度扫描（K 变化），固定速度比。"""
    out = {}
    for K in (1, 2, 4, 8):
        cfg = {'K': K, 'T': 1.0, 'U': 0.2, 'gen_rate': 1.0}
        rows = [simulate(cfg, s) for s in SEEDS]
        out['K%d' % K] = {
            'b2_share': sum(r['b2_share'] for r in rows) / len(rows),
            'b5_share': sum(r['b5_share'] for r in rows) / len(rows),
            'comp_share': sum(r['comp_share'] for r in rows) / len(rows),
        }
    return out


def scenario_partition_check():
    """Q1 核心：三类之和是否恒为 1（即有没有"第四类"）。"""
    out = {}
    worst = 0.0
    for rate in (0.3, 1.0, 4.0):
        for K in (1, 4):
            cfg = {'K': K, 'T': 1.0, 'U': 0.5, 'gen_rate': rate}
            for s in SEEDS[:5]:
                r = simulate(cfg, s)
                dev = abs(r['b2_share'] + r['b5_share'] + r['comp_share'] - 1.0)
                worst = max(worst, dev)
    out['max_deviation_from_1'] = worst
    out['covers_completely'] = worst < 1e-9
    return out


def run_all():
    return {
        'config': {'seeds': len(SEEDS)},
        'speed_ratio': scenario_speed_ratio(),
        'capacity': scenario_capacity(),
        'partition_check': scenario_partition_check(),
    }


def _canon(x):
    if isinstance(x, dict):
        return {k: _canon(v) for k, v in sorted(x.items())}
    if isinstance(x, list):
        return [_canon(v) for v in x]
    if isinstance(x, float):
        return round(x, 9)
    return x


def verify():
    if not SUMMARY.exists():
        print('summary.json 不存在，先运行一次全量')
        return 1
    archived = json.loads(SUMMARY.read_text(encoding='utf-8'))
    fresh = _canon(run_all())
    if _canon(archived) != fresh:
        print('差异：', [k for k in fresh if _canon(archived.get(k)) != fresh[k]])
        return 1
    print('--verify 通过：与归档 summary.json 逐字段一致')
    return 0


def main():
    if '--verify' in sys.argv:
        return verify()
    SUMMARY.write_text(json.dumps(_canon(run_all()), ensure_ascii=False,
                                  indent=1, sort_keys=True), encoding='utf-8')
    print('写出', SUMMARY)
    return 0


if __name__ == '__main__':
    sys.exit(main())
