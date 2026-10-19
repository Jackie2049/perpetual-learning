# Draft: verl policy-loss clip semantics are inconsistent across losses

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
