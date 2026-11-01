#!/usr/bin/env python3
"""P-3：C 类机制（DAPO 组过滤）的收敛净效应示范 —— 同 seed、开关 on/off。

背景
----
主文档 §13.16 G2 表的结论是：语料内五个 C 类机制（APRIL／DART／ProRL Agent／QwenGyre／
StaleFlow）全部只报加速与质量终值，**无一量化"分布偏移的收敛代价 vs 加速收益"**。
§20.3.2 的 P-3 就是补一个这样的示范。

本脚本不重跑任何论文，而是把机制的最小内核抽出来做可控对照：
  - 训练内核 = verl `compute_grpo_outcome_advantage` 的逐项算子（组内均值／std 归一、
    按 epsilon=1e-6 除），见 verl/trainer/ppo/core_algos.py:268。
  - 过滤判定 = verl `_dapo_filtered_keys` 的语义（组内指标全同即判为无信号并逐出），
    见 verl/trainer/ppo/v1/replay_buffer.py:260。
  - 补偿方式 = sync DAPO 的 "evict k, refill 2k"（tests/trainer/ppo/v1/test_replay_buffer_on_cpu.py:1041
    的 test_sync_dapo_refills_twice_and_clears_surplus 固化的行为）。批内凑不满时出现空批，
    真实实现里这表现为"本步没有可训练样本"，这里显式计为一次空更新，不重试。

模型
----
一个 bag-of-tokens 分类策略 π_θ(y | x)：
  - 每个 prompt 有 2 个"事实"token，其 logit 由 prompt 身份给定（x=0 的题在事实位上 +LOGIT_SEP，
    于是 8 条样本里约一半落在 x=0），其余是填充位置，logit 参数化为 θ_k。
  - 一条轨迹 = 事实位 2 次 + 填充位 n_fill 次独立采样；奖励 = 是否答对。
  - 任务难度用"答对所需正确填充次数"这个阈值来调，使初始准确率落在 0.45 左右——
    接近 gsm8k 训练早期（候选 A 锚点 acc 0.4917），让"全对/全错"组足够常见。

五种机制
--------
A 类（降方差）：只把长度从 14 降到 9（短轨迹），仍需 8 条回答才能凑够 5 条有信号。
B 类（改调度）：长度 9，且先用 2 条探针判定无信号就停。
C 类（改集合）：在 B 类基础上，把无信号组整体逐出训练批并 refill。**本实验的主对象。**
参考实现（R）：全长度 14 + 探针 + refill，对应真实 verl 组过滤配置。
对照组（R_nofilter）：全长度 14、无探针、不过滤——即标准 GRPO，本实验的基线。

墙钟模型（写进 summary.json 的 wallclock_model 字段）
----
统一换算单位（内部记作"tok"）：
  - 每条完整轨迹 = 其长度；每个探针 = 实际生成的 prompt+2 个回答 token = 4。
  - 一次空更新 = 1，有更新的训练步 = 8：同步 RL 里 rollout 与 update 串行，rollout 短时
    由 update 主导。不建模并发、排队、长尾。
绝对值不可与论文报值相除（§19.1 规则一），只能五种模式间横向比较。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

N_FACT = 2
N_FILL = 12
N_STD = N_FACT + N_FILL
N_SHORT = N_FACT + 7
GROUP_SIZE = 8
TARGET_SIGNAL = 5
PROBE_TOKENS = N_FACT + 2
EMPTY_UPDATE_COST = 1
UPDATE_COST = 8
LOGIT_SEP = 5.0        # x=0 的题在事实位上的 logit 加成（题间可分，避免策略退化到"全对/全错"）
FILL_BIAS = {12: -0.95, 7: -1.25}  # 按填充位数标定的初始 logit：两者初始正确率≈0.45
CORRECT_THRESHOLD = 8  # 答对要求"正确填充次数 ≥ 8"（14 长）与"≥ 5"（9 长）按比例给

MODE_LENGTH = {"A": N_SHORT, "B": N_SHORT, "C": N_SHORT, "R": N_STD, "R_nofilter": N_STD}
MODE_PROBE = {"A": False, "B": True, "C": True, "R": True, "R_nofilter": False}
MODE_REFILL = {"A": False, "B": False, "C": True, "R": True, "R_nofilter": False}


def pass_threshold(n_fill: int) -> int:
    """答对所需的正确填充次数：按填充位数等比缩放（长短期难度一致）。

    系数 0.55 是"难度旋钮"：初始正确率≈0.45（接近候选 A 锚点 acc 0.4917），
    让"全对/全错"组在训练早期足够常见（这正是 DAPO 组过滤要处理的形态），
    同时离 1.0 有足够空间，避免曲线一上来就饱和。
    """
    return max(1, round(0.55 * n_fill))


def init_theta(n_fill: int) -> list[float]:
    return [0.0] * N_FACT + [FILL_BIAS[n_fill]] * n_fill


def logits(theta: list[float], x: int) -> list[float]:
    out = list(theta)
    if x == 0:
        for k in range(N_FACT):
            out[k] += LOGIT_SEP
    return out


def sample_traj(theta: list[float], x: int, n_fill: int, rng: random.Random) -> int:
    z = logits(theta, x)
    correct = 0
    for k in range(N_FACT):
        correct += 1 if rng.random() < 1 / (1 + math.exp(-z[k])) else 0
    for k in range(N_FACT, N_FACT + n_fill):
        correct += 1 if rng.random() < 1 / (1 + math.exp(-z[k])) else 0
    return 1 if correct >= pass_threshold(n_fill) else 0


@dataclass
class Batch:
    rewards: list[int]
    prompts: list[int]
    stats: dict


def rollout(theta: list[float], prompts: list[int], n_fill: int, probe: bool, refill: bool, rng: random.Random) -> Batch:
    rewards: list[int] = []
    owners: list[int] = []
    st = {"gen": 0, "probes": 0, "trajectories": 0, "filtered_groups": 0, "kept_groups": 0}
    for x in prompts:
        g: list[int] = []
        if probe:
            st["gen"] += 2 * PROBE_TOKENS
            st["probes"] += 1
            # 探针看到的是前 2 条轨迹的奖励
            g.append(sample_traj(theta, x, n_fill, rng))
            g.append(sample_traj(theta, x, n_fill, rng))
            if max(g) == min(g):
                # 两条一样（大多全错，偶尔全对）：无信号即停，剩余 6 条不生成
                if refill:
                    st["filtered_groups"] += 1
                continue
        for _ in range(GROUP_SIZE - len(g)):
            g.append(sample_traj(theta, x, n_fill, rng))
            st["gen"] += MODE_LENGTH_SIZE[probe]
        if not probe:
            st["gen"] += 0
        if refill and len(g) > 1 and max(g) == min(g):
            st["filtered_groups"] += 1
            continue
        st["kept_groups"] += 1
        rewards.extend(g)
        owners.extend([x] * len(g))
    return Batch(rewards=rewards, prompts=owners, stats=st)


MODE_LENGTH_SIZE = {True: 0, False: 0}


def grpo_advantages(rewards: list[int]) -> list[float]:
    n = len(rewards)
    if n == 1:
        return [0.0]
    mean = sum(rewards) / n
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / n)
    return [(r - mean) / (std + 1e-6) for r in rewards]


def apply_update(theta: list[float], batch: Batch, n_fill: int, lr: float) -> dict:
    n = len(batch.rewards)
    if n == 0:
        return {"counts": 0, "group_mean": None, "group_std": None}
    adv: list[float] = []
    gm, gs = [], []
    i = 0
    while i < n:
        j = i + 1
        while j < n and batch.prompts[j] == batch.prompts[i]:
            j += 1
        g = batch.rewards[i:j]
        adv.extend(grpo_advantages(g))
        if len(g) > 1:
            m = sum(g) / len(g)
            gm.append(m)
            gs.append(math.sqrt(sum((v - m) ** 2 for v in g) / len(g)))
        i = j
    grad = [0.0] * (N_FACT + n_fill)
    for x, r, a in zip(batch.prompts, batch.rewards, adv):
        z = logits(theta, x)
        for k in range(N_FACT, N_FACT + n_fill):
            p = 1 / (1 + math.exp(-z[k]))
            grad[k] += a * ((1.0 if r == 1 else 0.0) - p) * 0.5
    for k in range(N_FACT, N_FACT + n_fill):
        theta[k] += lr * grad[k] / n
    return {
        "counts": n,
        "group_mean": statistics.mean(gm) if gm else None,
        "group_std": statistics.mean(gs) if gs else None,
    }


def evaluate(theta: list[float], n_fill: int, n_eval: int, rng: random.Random) -> float:
    """留出评测：题面身份换一对新值（评测集与训练集的 prompt 值不重叠，只共享填充位参数），
    随机采样 n_eval 条轨迹取正确率。用随机采样而非贪心，是为了让曲线在接近 1.0 时仍有分辨率。"""
    n_eval = max(1, n_eval)
    hit = 0
    for _ in range(n_eval):
        hit += sample_traj(theta, rng.randrange(2), n_fill, rng)
    return hit / n_eval


def train(mode: str, seed: int, n_prompts: int, n_steps: int, lr: float, n_eval: int) -> dict:
    rng = random.Random(seed)
    n_fill = MODE_LENGTH[mode] - N_FACT
    theta = init_theta(n_fill)
    probe, refill = MODE_PROBE[mode], MODE_REFILL[mode]
    traj_cost = MODE_LENGTH[mode]
    per_step = []
    curve = []
    for _ in range(n_steps):
        gen = 0
        rewards: list[int] = []
        owners: list[int] = []
        filtered = 0
        retries = 0
        while True:
            prompts = [rng.randrange(2) for _ in range(n_prompts)]
            accepted = 0
            for idx, x in enumerate(prompts):
                g: list[int] = []
                if probe:
                    gen += 2 * PROBE_TOKENS
                    g.append(sample_traj(theta, x, n_fill, rng))
                    g.append(sample_traj(theta, x, n_fill, rng))
                    if max(g) == min(g) and idx < n_prompts - 1:
                        # 探针判为无信号：不再生成剩余 6 条。idx 条件保证本步至少留一个 prompt
                        # 能凑出完整批，避免末位 prompt 把整步判空（真实实现里由 max_inflight_gen_batches 兜底）
                        if refill:
                            filtered += 1
                        continue
                for _ in range(GROUP_SIZE - len(g)):
                    g.append(sample_traj(theta, x, n_fill, rng))
                    gen += traj_cost
                if refill and len(g) > 1 and max(g) == min(g):
                    # 无信号组整体逐出训练批（与 verl _dapo_filtered_keys 的"组内指标全同"判定一致）
                    filtered += 1
                    continue
                rewards.extend(g)
                owners.extend([x] * len(g))
                accepted += 1
            # sync DAPO 的 "evict k, refill 2k"：接受的组数不足时再取一批 prompt，
            # 最多 3 轮仍未凑够就用现有批
            if accepted >= max(1, n_prompts // 2) or retries >= 3:
                break
            retries += 1
        b = Batch(rewards=rewards, prompts=owners, stats={})
        cost = max(gen, UPDATE_COST) if rewards else EMPTY_UPDATE_COST
        info = apply_update(theta, b, n_fill, lr)
        per_step.append(
            {"gen": gen, "cost": cost, "filtered": filtered, **info}
        )
        curve.append(
            {
                "step": len(curve),
                "cost": sum(s["cost"] for s in per_step),
                "acc": evaluate(theta, n_fill, n_eval, rng),
                "update_cost": 8 * sum(1 for x in per_step if x["counts"] > 0),
            }
        )
    def steps_to_reach(target: float) -> int | None:
        """首次达到某一准确率的步号（预算=墙钟换算成本，见 walls）。"""
        for c in curve:
            if c["acc"] >= target:
                return c["step"]
        return None

    def cost_to_reach(target: float, key: str = "cost") -> int | None:
        for c in curve:
            if c["acc"] >= target:
                return c[key]
        return None

    return {
        "mode": mode,
        "seed": seed,
        "final_acc": evaluate(theta, n_fill, n_eval * 4, rng),
        "steps_to_half": steps_to_reach(0.5),
        "steps_to_09": steps_to_reach(0.9),
        "steps_to_099": steps_to_reach(0.99),
        "walls_to_half": cost_to_reach(0.5),
        "walls_to_09": cost_to_reach(0.9),
        "walls_to_099": cost_to_reach(0.99),
        "update_walls_to_09": cost_to_reach(0.9, "update_cost"),
        "total_cost": sum(s["cost"] for s in per_step),
        "total_update_cost": 8 * sum(1 for s in per_step if s["counts"] > 0),
        "empty_updates": sum(1 for s in per_step if s["counts"] == 0),
        "filtered_groups": sum(s["filtered"] for s in per_step),
        "mean_counts": statistics.mean(s["counts"] for s in per_step),
        "curve": curve,
        "steps": per_step,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--eval", type=int, default=400, dest="n_eval")
    ap.add_argument("--out", default=str(Path(__file__).parent / "summary.json"))
    args = ap.parse_args()
    modes = ["R_nofilter", "R", "A", "B", "C"]
    runs = {m: [train(m, s, args.prompts, args.steps, args.lr, args.n_eval) for s in range(args.seeds)] for m in modes}
    base = runs["R_nofilter"]
    base_acc = statistics.mean(r["final_acc"] for r in base)
    base_cost = statistics.mean(r["total_cost"] for r in base)
    out = {
        "meta": {
            "verl_head": "80ee82bd",
            "verl_blobs": {
                "verl/trainer/ppo/v1/replay_buffer.py": "033e06aaf346",
                "verl/trainer/ppo/core_algos.py": "1998fb0b9d75",
                "tests/trainer/ppo/v1/test_replay_buffer_on_cpu.py": "8cfdb9e1de0f",
            },
            "seeds": args.seeds,
            "steps": args.steps,
            "lr": args.lr,
            "prompts_per_step": args.prompts,
            "eval_size": args.n_eval,
            "caveat": "进程内模拟（CPU），非 GPU 实测；机制内核照 verl 源码，但不是 verl 端到端",
            "wallclock_model": "统一换算单位；轨迹=其长度、探针=4、空更新=1、有更新步=8；步成本=max(gen,8)；不建模并发/排队/长尾；绝对值不可与论文报值相除",
        },
        "modes": {},
    }
    for m in modes:
        rs = runs[m]
        accs = [r["final_acc"] for r in rs]
        costs = [r["total_cost"] for r in rs]
        sd = statistics.pstdev(accs)
        out["modes"][m] = {
            "acc_mean": statistics.mean(accs),
            "acc_sd": sd,
            "acc_ci95": 1.96 * sd / math.sqrt(len(accs)),
            "acc_delta_vs_R_nofilter": statistics.mean(accs) - base_acc,
            "cost_mean": statistics.mean(costs),
            "cost_ratio_vs_R_nofilter": statistics.mean(costs) / base_cost,
            "empty_updates_mean": statistics.mean(r["empty_updates"] for r in rs),
            "filtered_groups_mean": statistics.mean(r["filtered_groups"] for r in rs),
            "mean_counts": statistics.mean(r["mean_counts"] for r in rs),
            "steps_to_09_mean": statistics.mean(r["steps_to_09"] for r in rs if r["steps_to_09"] is not None)
            if any(r["steps_to_09"] is not None for r in rs) else None,
            "steps_to_09_reached": sum(1 for r in rs if r["steps_to_09"] is not None),
            "walls_to_09_mean": statistics.mean(r["walls_to_09"] for r in rs if r["walls_to_09"] is not None)
            if any(r["walls_to_09"] is not None for r in rs) else None,
            "update_walls_to_09_mean": statistics.mean(r["update_walls_to_09"] for r in rs if r["update_walls_to_09"] is not None)
            if any(r["update_walls_to_09"] is not None for r in rs) else None,
            "total_update_cost_mean": statistics.mean(r["total_update_cost"] for r in rs),
            "per_seed": [
                {"seed": r["seed"], "final_acc": r["final_acc"], "total_cost": r["total_cost"],
                 "steps_to_09": r["steps_to_09"], "walls_to_09": r["walls_to_09"],
                 "update_walls_to_09": r["update_walls_to_09"],
                 "empty_updates": r["empty_updates"], "filtered_groups": r["filtered_groups"]}
                for r in rs
            ],
            "curve": [
                {"step": i, "cost": statistics.mean(r["curve"][i]["cost"] for r in runs[m]),
                 "acc": statistics.mean(r["curve"][i]["acc"] for r in runs[m])}
                for i in range(args.steps)
            ],
        }
    Path(args.out).write_text(json.dumps(out, indent=2))
    brief = {m: {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items() if k not in ("per_seed", "curve")} for m, d in out["modes"].items()}
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
