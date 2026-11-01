#!/usr/bin/env python3
"""P-3：C 类机制（DAPO 组过滤）的收敛净效应示范 —— 同 seed、开关 on/off。

背景
----
主文档 §13.16 G2 表的结论是：语料内五个 C 类机制（APRIL／DART／ProRL Agent／QwenGyre／
StaleFlow）全部只报加速与质量终值，**无一量化"分布偏移的收敛代价 vs 加速收益"**。
§20.3.2 的 P-3 就是补一个这样的示范。

本脚本不重跑任何论文，而是把机制的最小内核抽出来做可控对照：
  - 训练内核 = verl `compute_grpo_outcome_advantage` 的逐项算子（组内均值／std 归一、
    按 epsilon=1e-6 除）。
  - 过滤判定 = verl `_dapo_filtered_keys` 的语义（组内指标全同即判为无信号并逐出）。
  - 补偿方式 = sync DAPO 的 "evict k, refill 2k"。
  三处源码出处见 readme.md 的"机制取自真实源码"表（固定 GitHub commit 链接）。

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

复跑即得同值
------------
全部随机性来自 `random.Random(seed)`；没有任何依赖进程状态的隐式随机源。`summary.json`
保存 30 个种子的逐步曲线，故复跑可与归档逐位比对：

    python3 p3_group_filter_net_effect.py --verify summary.json

2026-11-01 复核批首次实跑复现时本文件还带运行期死代码（`MODE_LENGTH_SIZE` 恒为 0、未被
调用的 `rollout()`／`Batch.stats`），清理后重跑与当时产物逐位相同——死代码路径从未执行。
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
PROBE_TOKENS = N_FACT + 2
EMPTY_UPDATE_COST = 1
UPDATE_COST = 8
LOGIT_SEP = 5.0        # x=0 的题在事实位上的 logit 加成（题间可分，避免策略退化到"全对/全错"）
FILL_BIAS = {12: -0.95, 7: -1.25}  # 按填充位数标定的初始 logit：两者初始正确率≈0.45

MODE_LENGTH = {"A": N_SHORT, "B": N_SHORT, "C": N_SHORT, "R": N_STD, "R_nofilter": N_STD}
MODE_PROBE = {"A": False, "B": True, "C": True, "R": True, "R_nofilter": False}
MODE_REFILL = {"A": False, "B": False, "C": True, "R": True, "R_nofilter": False}
MODES = ["R_nofilter", "R", "A", "B", "C"]


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


def grpo_advantages(rewards: list[int]) -> list[float]:
    n = len(rewards)
    if n == 1:
        return [0.0]
    mean = sum(rewards) / n
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / n)
    return [(r - mean) / (std + 1e-6) for r in rewards]


def apply_update(theta: list[float], batch: Batch, lr: float) -> dict:
    n = len(batch.rewards)
    if n == 0:
        return {"counts": 0}
    adv: list[float] = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and batch.prompts[j] == batch.prompts[i]:
            j += 1
        adv.extend(grpo_advantages(batch.rewards[i:j]))
        i = j
    grad = [0.0] * len(theta)
    for x, r, a in zip(batch.prompts, batch.rewards, adv):
        z = logits(theta, x)
        for k in range(N_FACT, len(theta)):
            p = 1 / (1 + math.exp(-z[k]))
            grad[k] += a * ((1.0 if r == 1 else 0.0) - p) * 0.5
    for k in range(N_FACT, len(theta)):
        theta[k] += lr * grad[k] / n
    return {"counts": n}


def curve_walls(step_costs: list[int], step_has_update: list[bool]) -> list[dict]:
    """把每步成本换算成累计墙钟与累计训练步成本（两者都只增不减）。"""
    out, cumulative, update_steps = [], 0, 0
    for cost, has_update in zip(step_costs, step_has_update):
        cumulative += cost
        update_steps += 1 if has_update else 0
        out.append({"cost": cumulative, "update_cost": UPDATE_COST * update_steps})
    return out


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
        info = apply_update(theta, Batch(rewards=rewards, prompts=owners), lr)
        cost = max(gen, UPDATE_COST) if rewards else EMPTY_UPDATE_COST
        per_step.append({"gen": gen, "cost": cost, "filtered": filtered, **info})
        curve.append(
            {
                "step": len(curve),
                "acc": evaluate(theta, n_fill, n_eval, rng),
            }
        )
    walls = curve_walls([s["cost"] for s in per_step], [s["counts"] > 0 for s in per_step])

    def first_reaching(target: float, key: str) -> int | None:
        for i, c in enumerate(curve):
            if c["acc"] >= target:
                return i if key == "step" else walls[i][key]
        return None

    return {
        "mode": mode,
        "seed": seed,
        "final_acc": evaluate(theta, n_fill, n_eval * 4, rng),
        "steps_to_half": first_reaching(0.5, "step"),
        "steps_to_09": first_reaching(0.9, "step"),
        "steps_to_099": first_reaching(0.99, "step"),
        "walls_to_half": first_reaching(0.5, "cost"),
        "walls_to_09": first_reaching(0.9, "cost"),
        "walls_to_099": first_reaching(0.99, "cost"),
        "update_walls_to_09": first_reaching(0.9, "update_cost"),
        "total_cost": sum(s["cost"] for s in per_step),
        "total_update_cost": UPDATE_COST * sum(1 for s in per_step if s["counts"] > 0),
        "empty_updates": sum(1 for s in per_step if s["counts"] == 0),
        "filtered_groups": sum(s["filtered"] for s in per_step),
        "mean_counts": statistics.mean(s["counts"] for s in per_step),
        # 逐步成本与策略、采样无关，故只存每步增量、累计由 curve_walls 重建；
        # 逐种子只存准确率序列，体积从 O(种子×步×3) 降到 O(种子×步)
        "step_costs": [s["cost"] for s in per_step],
        "step_has_update": [s["counts"] > 0 for s in per_step],
        "acc_curve": [round(c["acc"], 6) for c in curve],
    }


def summarize(runs: dict, n_steps: int, meta: dict) -> dict:
    base = runs["R_nofilter"]
    base_acc = statistics.mean(r["final_acc"] for r in base)
    base_cost = statistics.mean(r["total_cost"] for r in base)
    out: dict = {"meta": meta, "modes": {}}
    for m in MODES:
        rs = runs[m]
        walls = [curve_walls(r["step_costs"], r["step_has_update"]) for r in rs]
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
                {
                    "seed": r["seed"],
                    "final_acc": r["final_acc"],
                    "total_cost": r["total_cost"],
                    "steps_to_09": r["steps_to_09"],
                    "walls_to_09": r["walls_to_09"],
                    "update_walls_to_09": r["update_walls_to_09"],
                    "empty_updates": r["empty_updates"],
                    "filtered_groups": r["filtered_groups"],
                    "step_costs": r["step_costs"],
                    "step_has_update": r["step_has_update"],
                    "acc_curve": r["acc_curve"],
                }
                for r in rs
            ],
            # 跨种子平均曲线：[累计墙钟成本, 累计训练步成本, 准确率]
            "curve": [
                [
                    round(statistics.mean(w[i]["cost"] for w in walls), 3),
                    round(statistics.mean(w[i]["update_cost"] for w in walls), 3),
                    round(statistics.mean(rr["acc_curve"][i] for rr in rs), 6),
                ]
                for i in range(n_steps)
            ],
        }
    return out


def run_all(seeds: int, steps: int, lr: float, prompts: int, n_eval: int) -> dict:
    runs = {m: [train(m, s, prompts, steps, lr, n_eval) for s in range(seeds)] for m in MODES}
    meta = {
        "seeds": seeds,
        "steps": steps,
        "lr": lr,
        "prompts_per_step": prompts,
        "eval_size": n_eval,
        "python": "3.9.6（复现执行机）",
        "caveat": "进程内模拟（CPU），非 GPU 实测；机制内核照 verl 源码，但不是 verl 端到端",
        "wallclock_model": "统一换算单位；轨迹=其长度、探针=4、空更新=1、有更新步=8；步成本=max(gen,8)；不建模并发/排队/长尾；绝对值不可与论文报值相除",
        "curve_format": "modes[].curve 每步为 [累计墙钟成本, 累计训练步成本, 准确率]（跨种子平均）；per_seed[] 存 step_costs（每步增量成本，确定性）与 acc_curve（准确率，6 位小数），累计值可由二者重建",
        "source_provenance": "机制内核源码出处见 readme.md「机制取自真实源码」表（GitHub 固定 commit 链接）",
    }
    return summarize(runs, steps, meta)


def compare(archived: dict, fresh: dict) -> list[str]:
    """逐字段比对两份 summary；返回差异描述（空表示逐位相同）。"""
    diffs = []
    if archived.get("meta") != fresh.get("meta"):
        for k in set(archived.get("meta", {})) | set(fresh.get("meta", {})):
            if archived.get("meta", {}).get(k) != fresh.get("meta", {}).get(k):
                diffs.append(f"meta.{k}: {archived.get('meta', {}).get(k)!r} != {fresh.get('meta', {}).get(k)!r}")
    for m in MODES:
        a, b = archived.get("modes", {}).get(m, {}), fresh.get("modes", {}).get(m, {})
        for k in set(a) | set(b):
            if a.get(k) != b.get(k):
                diffs.append(f"modes.{m}.{k} 不一致")
    return diffs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--eval", type=int, default=400, dest="n_eval")
    ap.add_argument("--out", default=str(Path(__file__).parent / "summary.json"))
    ap.add_argument("--verify", metavar="ARCHIVE", default=None,
                    help="照参数复跑并与给定归档逐字段比对（不写文件）")
    args = ap.parse_args()
    summary = run_all(args.seeds, args.steps, args.lr, args.prompts, args.n_eval)
    if args.verify:
        archived = json.loads(Path(args.verify).read_text())
        diffs = compare(archived, summary)
        if diffs:
            print(f"复跑核对失败：{len(diffs)} 处差异")
            for d in diffs[:20]:
                print(" ", d)
            raise SystemExit(1)
        print(f"复跑核对通过：与 {args.verify} 逐字段相同（{args.seeds} 种子 × {args.steps} 步 × {len(MODES)} 模式）")
        return
    # info.json：跑完用 --verify 可逐字段复核（README 的"复跑"节引用它）
    # 先写临时文件再替换：跑一半中断时不会留下被截断的归档
    out = Path(args.out)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(summary, indent=2))
    tmp.replace(out)
    brief = {m: {k: round(v, 4) if isinstance(v, float) else v
                 for k, v in d.items() if k not in ("per_seed", "curve")}
             for m, d in summary["modes"].items()}
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
