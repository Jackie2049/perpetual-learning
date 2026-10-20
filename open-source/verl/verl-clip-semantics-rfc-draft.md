# verl：澄清不同策略损失中 clip_ratio 的含义

<!-- contribution-summary:start -->
**【仓库】** [verl](https://github.com/verl-project/verl)（后训练框架）

**【贡献点】** 澄清不同策略损失中 clip_ratio 的含义。[源码](https://github.com/verl-project/verl/blob/main/verl/trainer/ppo/core_algos.py) · [验证脚本](../../experiments/scripts/verify_gmpo_clip_semantics.py)

**【类型】** 文档改进

**【价值评估】** 7 / 10 — 直接影响训练配置理解，有源码与数值对照；尚未获得社区反馈。

**【进展】** 证据已整理（2026-10-19）：源码、论文和数值对照完成，RFC 文案未发布，暂无上游编号。

**【大白话讲明白】**

几个训练方法都叫某个参数 clip_ratio，但裁剪的是不同量。同一个数值换到另一种方法上，限制强度未必相同。我们希望把这些差别写清，减少照抄配置造成的误解。

**【What–Why–How】**

- **What：** geo_mean、GSPO 与 SAPO 对策略更新幅度的限制语义及示例参数。
- **Why：** 有的限制对数，有的限制概率比，还有的使用平滑门控；同名参数不足以表达这些差异。
- **How：** 整理公式与实现对照，补充 docstring 或示例说明；数值证明支持语义差异，不代表已经证明某次训练失败的原因。
<!-- contribution-summary:end -->

## 详细记录

> 以下保留原始分析与历史进展；当前结论见上方速览。

> 发布形态：verl issue（英文，RFC/文档澄清）。内容证据链：分析 → 源码核对 → 数值证明（experiments/scripts/verify_gmpo_clip_semantics.py）。发布需用户授权。

---

**Title:** [RFC] clip_ratio semantics differ across policy losses (GSPO ratio-space vs geo_mean log-space); example defaults may be misleadingly tight

**Body:**

`actor_rollout_ref.actor.clip_ratio_low/high` is a shared config field, but the three recent policy losses interpret it in different spaces, so the same numeric value produces very different effective trust regions:

| Loss | Implementation (core_algos.py) | Clip space | Effective window for `clip_ratio=3e-4` |
|---|---|---|---|
| GSPO | `clamp(seq_importance_ratio, 1 - eps_low, 1 + eps_high)` | ratio space | ratio in [0.9997, 1.0003] (KL +-0.0004 nats) |
| geo_mean (GMPO) | `clamp(negative_approx_kl, -eps_low, eps_high)` then exp | log-ratio space | log-ratio in [-3e-4, 4e-4] -> ratio ~ [0.9997, 1.0004] |
| SAPO | `sigmoid(tau*(x-1))*4/tau` gate | ratio space (smooth) | temperature-controlled, no hard window |

Two consequences that have bitten users:

1. **Paper-following users get a different algorithm than intended.** The GMPO paper clips in probability space with eps=0.4 (ratio window ~ [0.67, 1.49]). Passing 0.4 to verl's geo_mean clamps the *log* ratio at +-0.4 (ratio ~ [0.67, 1.49] — coincidentally close), but passing the same value to GSPO clamps ratio at 1+-0.4 (window [0.6, 1.4]) — the two losses disagree on what "0.4" means. For small values the discrepancy is larger: `clip_ratio=3e-4` is a wide-ish log-space window but an extremely tight ratio-space window (ratio 1+-0.0003).

2. **The GSPO example defaults (3e-4/4e-4) are ~500x tighter than PPO-typical 0.2.** `examples/gspo_trainer/run_qwen3_30b_a3b_megatron.sh` ships `CLIP_RATIO_LOW=3e-4, CLIP_RATIO_HIGH=4e-4`. In ratio space that is 1+-0.0004 — most tokens saturate the clip once per-token KL exceeds ~4e-4 nats, gradients vanish for those tokens, and after a drift event the policy cannot recover. We have observed this failure-to-recover pattern in async multi-turn training runs using these defaults (see verl#8250 discussion).

Note: `log(min(rho, eps_hi)) == min(log rho, log eps_hi)`, so the GSPO/geo_mean implementations are mathematically equivalent *given corresponding windows* (verified numerically); the issue is purely the config semantics — one number, multiple meanings.

**Proposed direction (open to maintainers):**
- Document the clip space per loss in the config schema/docstrings (ActorConfig.clip_ratio_low/high), and/or
- Normalize example defaults to the intended space with a comment, and/or
- Add a per-loss semantic (e.g., geo_mean clipping in probability space like the paper).

Happy to contribute the doc change or a validation warning if maintainers agree on the direction.
