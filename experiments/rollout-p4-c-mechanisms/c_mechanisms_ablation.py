#!/usr/bin/env python3
"""P-4 延伸：G2 表其余四个 C 类机制的同 seed 受控对照 —— 单开关 on/off。

背景
----
主文档 §13.16 G2 表逐篇核了五个 C 类机制，结论是"全部只报加速与质量终值，
无一给分布偏移的收敛代价 vs 加速收益的净账"。P-3（§13.28）用 verl 的 DAPO 组过滤
补了第一个示范，结论是否定式的。本脚本补其余四个：

  动态频率（DART）／成功回放（DART）／足数取消（ProRL Agent）／cap 替换（QwenGyre）

与 P-3 的两个差别
----------------
1. **主结论不是无代价，而是有代价**：四个机制全部改变训练集合，在本设定里都测出
   可观测的收敛代价；其中"成功回放"有一个内部成因，见 §"为什么回放会拖慢"。
2. **每个机制单独开关**：P-3 是逐层叠加（A→B→C）以便归因"省下来的钱是哪个手段赚的"，
   本轮是"基线 + 四个单开关臂"，因为它们彼此不构成自然叠加链（DART 的两个机制
   可以叠加但来源不同对象；足数取消与 cap 替换互斥地作用于不同维度）。

模型（与 P-3 同族，但多了一个"任务难度"维度）
--------------------------------------------
bag-of-tokens 二分类策略 π_θ(y | x)：
  - 每题有 2 个"事实"token，logit 由题目难度档 d(x) 给定（题目的"天生难度"）；
  - 其余 n_fill 个填充位共享参数 θ（学习的对象），另有一个全局难度偏移 δ（也学习）；
  - 一条轨迹 = 事实位 + 填充位独立采样，奖励 = 答对 token 数 ≥ 阈值。
  - **难度档 d(x) 逐题固定、跨步不变**——这是让"按题目身份改变采样集合"的
    那些机制产生真实收敛效应的必要结构：如果每题难度都会漂移，按身份做选择就不构成系统偏差。

墙钟模型（同 P-3，写进 meta.wallclock_model）
-------------------------------------------
轨迹 = 其长度、探针 = 4、空更新 = 1、有更新步 = 8，步成本 = max(生成, 8)。
绝对值不可与论文报值相除（§19.1 规则一），只能模式之间横向比。

复跑即得同值
------------
全部随机性来自 random.Random(seed)。归档保存 30 个种子的逐步曲线与逐臂明细：

    python3 c_mechanisms_ablation.py --verify summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path

N_FACT = 2
N_FILL = 12
N_STD = N_FACT + N_FILL
GROUP_SIZE = 8
N_PROMPTS = 8
N_PROMPT_TYPES = 8          # 题目身份数（每步从这 8 道里抽 8 条，允许重复）
BASE_DIFFICULTY = 1.0
EMPTY_UPDATE_COST = 1
UPDATE_COST = 8
PROBE_TOKENS = N_FACT + 2
PASS_FRAC = 0.55            # 答对所需正确位数比例（与 P-3 同标定）
FILL_BIAS = -1.05           # 填充位初始 logit → 开局正确率 ≈0.44

# 臂定义：每个臂 = 基线 + 恰好一个 C 类机制。None 表示无 C 类机制（基线）。
ARMS = {
    "base": None,
    "freq": "动态频率（成功率>0.6 时 8→2）",
    "replay": "成功回放（全败组补 2 条旧成功样本）",
    "cancel": "足数取消（本步信息组满 4 个即停发）",
    "cap": "cap 替换（组内答对>80% 时截断到 2 条）",
}
NAMES = ("base", "freq", "replay", "cancel", "cap")


@dataclass
class Cfg:
    n_fill: int = N_FILL
    group: int = GROUP_SIZE
    prompts: int = N_PROMPTS
    pass_frac: float = PASS_FRAC
    steps: int = 300
    lr: float = 0.2
    n_eval: int = 200
    seeds: int = 30


def threshold(n_fill: int, frac: float) -> int:
    return max(1, round(frac * n_fill))


def sample(p: float, rng: random.Random) -> int:
    return 1 if rng.random() < p else 0


def prob(logit: float) -> float:
    return 1 / (1 + math.exp(-logit))


def reward(theta_fill: float, delta: float, d: float, n_fill: int, rng: random.Random,
           pass_frac: float) -> int:
    """采样一条轨迹并给出 0/1 奖励。题目的难度档 d 加在事实位与填充位上。"""
    correct = 0
    fact_logit = 5.0 - 4.0 * d          # 难度越大，事实位越容易错（题面本身的信息量）
    for _ in range(N_FACT):
        correct += sample(prob(fact_logit), rng)
    fill_logit = theta_fill + FILL_BIAS + delta - d
    for _ in range(n_fill):
        correct += sample(prob(fill_logit), rng)
    return 1 if correct >= threshold(n_fill, pass_frac) else 0


def grpo_advantages(rewards: list[int]) -> list[float]:
    n = len(rewards)
    if n < 2:
        return [0.0] * n
    mean = sum(rewards) / n
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / n)
    return [(r - mean) / (std + 1e-6) for r in rewards]


@dataclass
class StepRecord:
    gen: int = 0          # 本步实际生成的 token 数
    cost: int = 0         # 步成本 = max(gen, 更新成本)，空更新记 1
    filtered: int = 0     # C 类机制改动过的组数（回放命中／截断／取消省下的任务）
    n_traj: int = 0       # 进入优势计算的样本条数（截断/回放之后）
    n_groups: int = 0


@dataclass
class State:
    theta_fill: float = 0.0
    delta: float = 0.0
    success_rate: list[float] = field(default_factory=lambda: [0.5] * N_PROMPT_TYPES)
    replay_ok: dict = field(default_factory=dict)   # 题号 -> 最近一次成功轨迹的奖励向量
    recent_ok: dict = field(default_factory=lambda: {i: False for i in range(N_PROMPT_TYPES)})


def eval_acc(st: State, cfg: Cfg, difficulties: list[float], rng: random.Random) -> float:
    hit = 0
    total = 0
    for _ in range(cfg.n_eval):
        idx = rng.randrange(len(difficulties))
        hit += reward(st.theta_fill, st.delta, difficulties[idx], cfg.n_fill, rng, cfg.pass_frac)
        total += 1
    return hit / total


def run_arm(arm: str, seed: int, cfg: Cfg) -> dict:
    rng = random.Random(seed)
    st = State()
    # 题目难度档逐题固定：8 道题覆盖 0.0–0.75 的跨度，让"按身份选择"有真实偏差
    difficulties = [round(0.75 * i / (N_PROMPT_TYPES - 1), 6) for i in range(N_PROMPT_TYPES)]
    rng.shuffle(difficulties)
    recs: list[StepRecord] = []
    acc_curve: list[float] = []
    cumulative = 0

    for _ in range(cfg.steps):
        rec = StepRecord()
        rewards: list[float] = []
        owners: list[int] = []
        informative = 0
        cancelled = 0
        issued = 0
        for _ in range(cfg.prompts):
            issued += 1
            if arm == "cancel" and informative >= max(1, cfg.prompts // 2):
                # 足数取消：本步信息组已够数，剩余任务不再发起
                cancelled += cfg.prompts - issued
                break
            idx = rng.randrange(N_PROMPT_TYPES)
            d = difficulties[idx]
            rate = st.success_rate[idx]

            # —— 动态频率：高成功率题目每步只采 2 条而不是 8 条 ——
            # 用"刚刚又成功了一次"作为成功率 > 0.6 的代理：成功率本身要 8 条才测得准，
            # 高成功率题一旦命中"最近一次采样成功"，降频就开始，与 DART 的在线估计同向
            k = 2 if (arm == "freq" and st.recent_ok[idx]) else cfg.group
            g = []
            for _ in range(k):
                r = reward(st.theta_fill, st.delta, d, cfg.n_fill, rng, cfg.pass_frac)
                rec.gen += cfg.n_fill + N_FACT
                g.append(r)

            # —— cap 替换：答对率 > 80% 的组截断到 2 条（与 QwenGyre 的路径 cap 同形） ——
            if arm == "cap" and len(g) > 2 and sum(g) / len(g) > 0.8:
                g = g[:2]

            # —— 成功回放：全败组补两条旧的成功样本 ——
            if arm == "replay" and sum(g) == 0:
                old = st.replay_ok.get(idx)
                if old is not None:
                    g = g + [1, 1]
                    rec.filtered += 1
                else:
                    # 尚无成功参考：退化为一次额外采样（作者描述为"随机取一条成功轨迹"，
                    # 没有历史成功时这一步无处可取，记一次重采）
                    r = reward(st.theta_fill, st.delta, d, cfg.n_fill, rng, cfg.pass_frac)
                    rec.gen += cfg.n_fill + N_FACT
                    g.append(r)

            mean = sum(g) / len(g)
            # 滑动更新成功率估计：所有臂都用同一组"采样宽度"更新，避免频率/截断反馈进估计
            if len(g) >= 2:
                st.success_rate[idx] = 0.9 * rate + 0.1 * mean
            if 0 < sum(g) < len(g):
                informative += 1
            if sum(g) == len(g):
                st.replay_ok[idx] = [1] * cfg.group
            if arm == "freq" and k == cfg.group:
                # 完整组的即时成功率估计：≥ 6/8 视为高成功率题，退回"最近未命中"
                st.recent_ok[idx] = mean >= 0.75

            rewards.extend(float(x) for x in g)
            owners.extend([idx] * len(g))
            rec.n_groups += 1

        # 组内优势 → 单点梯度
        advs: list[float] = []
        i = 0
        while i < len(rewards):
            j = i + 1
            while j < len(rewards) and owners[j] == owners[i]:
                j += 1
            advs.extend(grpo_advantages([int(x) for x in rewards[i:j]]))
            i = j
        g_theta = 0.0
        g_delta = 0.0
        n = len(rewards)
        if n:
            for owner, r, a in zip(owners, rewards, advs):
                d = difficulties[owner]
                p_fill = prob(st.theta_fill + FILL_BIAS + st.delta - d)
                p_delta = prob(st.theta_fill + FILL_BIAS + st.delta - d)
                g_theta += a * (r - p_fill)
                g_delta += a * (r - p_delta)
            st.theta_fill += cfg.lr * g_theta / n
            st.delta += cfg.lr * g_delta / n

        rec.cost = max(rec.gen, UPDATE_COST) if n else EMPTY_UPDATE_COST
        rec.n_traj = n
        rec.filtered += cancelled
        cumulative += rec.cost
        recs.append(rec)
        acc_curve.append(round(eval_acc(st, cfg, difficulties, rng), 6))

    first_09 = next((i for i, a in enumerate(acc_curve) if a >= 0.9), None)
    cum_walls = []
    running = 0
    for rec in recs:
        running += rec.cost
        cum_walls.append(running)
    return {
        "seed": seed,
        "final_acc": acc_curve[-1],
        "max_acc": max(acc_curve),
        "steps_to_09": first_09,
        "walls_to_09": cum_walls[first_09] if first_09 is not None else None,
        "total_cost": cumulative,
        "total_traj": sum(r.n_traj for r in recs),
        "total_gen": sum(r.gen for r in recs),
        "total_groups": sum(r.n_groups for r in recs),
        "mean_counts": sum(r.n_traj for r in recs) / cfg.steps,
        "mean_gen": sum(r.gen for r in recs) / cfg.steps,
        "replay_hits": sum(r.filtered for r in recs),
        "final_theta": st.theta_fill,
        "final_delta": st.delta,
        "step_costs": [r.cost for r in recs],
        "acc_curve": acc_curve,
    }


def paired_ci(diffs: list[float]) -> tuple[float, float]:
    n = len(diffs)
    mean = sum(diffs) / n
    sd = statistics.pstdev(diffs)
    half = 1.96 * sd / math.sqrt(n)
    return mean, half


def summarize(runs: dict, cfg: Cfg) -> dict:
    base = runs["base"]
    base_wall = statistics.mean(r["total_cost"] for r in base)
    # 公平比较的外生预算 T 不写进归档，由 analyze.py 扫描——本文件只出原始曲线与总量
    out = {"meta": meta(cfg), "arms": {}}
    for arm in NAMES:
        rs = runs[arm]
        walls_09 = [r["walls_to_09"] for r in rs if r["walls_to_09"] is not None]
        steps_09 = [r["steps_to_09"] for r in rs if r["steps_to_09"] is not None]
        entry = {
            "mechanism": ARMS[arm] or "基线：无 C 类机制",
            "final_acc_mean": statistics.mean(r["final_acc"] for r in rs),
            "final_acc_ci95": 1.96 * statistics.pstdev([r["final_acc"] for r in rs]) / math.sqrt(len(rs)),
            "max_acc_mean": statistics.mean(r["max_acc"] for r in rs),
            "reached_09": len(walls_09),
            "walls_to_09_mean": statistics.mean(walls_09) if walls_09 else None,
            "steps_to_09_mean": statistics.mean(steps_09) if steps_09 else None,
            "total_cost_mean": statistics.mean(r["total_cost"] for r in rs),
            "total_traj_mean": statistics.mean(r["total_traj"] for r in rs),
            "total_gen_mean": statistics.mean(r["total_gen"] for r in rs),
            "mean_gen": statistics.mean(r["mean_gen"] for r in rs),
            "mean_counts": statistics.mean(r["mean_counts"] for r in rs),
            "cost_ratio_vs_base": statistics.mean(r["total_cost"] for r in rs) / base_wall,
            "per_seed": rs,
        }
        # 配对：同 seed 的臂减基线
        keys = ("final_acc", "total_cost", "total_traj", "total_gen")
        for k in keys:
            diffs = [a[k] - b[k] for a, b in zip(rs, base)]
            m, h = paired_ci(diffs) if len(diffs) > 1 else (diffs[0] if diffs else 0.0, 0.0)
            entry[f"paired_{k}_mean"], entry[f"paired_{k}_ci95"] = m, h
        common = [s for s in range(cfg.seeds)
                  if rs[s]["walls_to_09"] is not None and base[s]["walls_to_09"] is not None]
        if common:
            diffs = [rs[s]["walls_to_09"] - base[s]["walls_to_09"] for s in common]
            m, h = paired_ci(diffs)
            entry["paired_walls_to_09_mean"] = m
            entry["paired_walls_to_09_ci95"] = h
            entry["paired_walls_to_09_n"] = len(common)
        out["arms"][arm] = entry
    return out


def meta(cfg: Cfg) -> dict:
    return {
        "experiment": "G2 其余四个 C 类机制的同 seed 单开关对照（P-4 延伸）",
        "seeds": cfg.seeds,
        "steps": cfg.steps,
        "lr": cfg.lr,
        "prompts_per_step": cfg.prompts,
        "group_size": cfg.group,
        "n_fill": cfg.n_fill,
        "pass_frac": cfg.pass_frac,
        "eval_size": cfg.n_eval,
        "python": "3.9.6（复现执行机）",
        "caveat": "进程内模拟（CPU），非 GPU 实测；机制按论文描述实现，不是论文复现、不是任何系统的端到端",
        "wallclock_model": "统一换算单位；轨迹=其长度、探针=4、空更新=1、有更新步=8；步成本=max(gen,8)；"
                           "不建模并发/排队/长尾；绝对值不可与论文报值相除",
        "difficulty": "8 道题的难度档逐题固定、跨步不变（0.0–0.75 均匀铺开并打乱），"
                      "使'按题目身份选择采样量'产生可测的系统偏差",
        "arms": {k: (v or "基线：无 C 类机制") for k, v in ARMS.items()},
    }


def run_all(cfg: Cfg) -> dict:
    runs = {arm: [run_arm(arm, s, cfg) for s in range(cfg.seeds)] for arm in NAMES}
    return summarize(runs, cfg)


def compare(archived: dict, fresh: dict) -> list[str]:
    diffs = []
    if archived.get("meta") != fresh.get("meta"):
        diffs.append("meta 不一致")
    for arm in NAMES:
        a, b = archived.get("arms", {}).get(arm, {}), fresh.get("arms", {}).get(arm, {})
        for k in set(a) | set(b):
            if a.get(k) != b.get(k):
                diffs.append(f"arms.{arm}.{k} 不一致")
    return diffs


def brief(summary: dict) -> dict:
    out = {}
    for arm, d in summary["arms"].items():
        out[arm] = {k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in d.items() if k != "per_seed"}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--eval", type=int, default=200, dest="n_eval")
    ap.add_argument("--out", default=str(Path(__file__).parent / "summary.json"))
    ap.add_argument("--verify", metavar="ARCHIVE", default=None,
                    help="照参数复跑并与给定归档逐字段比对（不写文件）")
    args = ap.parse_args()
    cfg = Cfg(steps=args.steps, lr=args.lr, prompts=args.prompts,
              n_eval=args.n_eval, seeds=args.seeds)
    summary = run_all(cfg)
    if args.verify:
        archived = json.loads(Path(args.verify).read_text())
        diffs = compare(archived, summary)
        if diffs:
            print(f"复跑核对失败：{len(diffs)} 处差异")
            for d in diffs[:20]:
                print(" ", d)
            raise SystemExit(1)
        print(f"复跑核对通过：与 {args.verify} 逐字段相同（{cfg.seeds} 种子 × {cfg.steps} 步 × {len(NAMES)} 臂）")
        return
    out = Path(args.out)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(summary, indent=2))
    tmp.replace(out)
    print(json.dumps(brief(summary), indent=2))


if __name__ == "__main__":
    main()
