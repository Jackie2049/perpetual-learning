# GRPO Training Debugging Guide: Step-by-Step Diagnosis & Fix

**Date**: 2026-07-15 (Session 10)
**Purpose**: Practical debugging reference for GRPO training issues — from symptom to root cause to fix
**Sources**: 7 framework bug patterns, 10 fork PRs, cross-framework-avoidance skill, grpo-debug skill

---

## 1. Diagnosis Flowchart

```
GRPO Training Issue → Which symptom?

1. Loss/advantages near zero     → Section 2 (grouped rewards and actual updates)
2. NaN in loss/advantages         → Section 3 (NaN diagnosis)
3. Loss spikes wildly             → Section 4 (gradient explosion)
4. Responses never end (no EOS)   → Section 5 (EOS token bug)
5. All responses identical        → Section 6 (KL too strong / collapse)
6. Training OOM                   → Section 7 (memory management)
7. Intermittent corruption        → Section 8 (CUDA stream race)
8. Slow convergence               → Section 9 (advantage/loss choice)
```

---

## 2. Loss Values and Zero Advantages

### Symptoms
- Loss stays at 0.0 throughout training
- Advantages all equal 0.0
- No improvement in reward over steps

### Root Cause
```
For a multi-sample group (n >= 2), identical finite outcome scores
→ centered scores = 0 → A = 0 / (0 + epsilon) = 0
→ that group's GRPO policy-gradient advantage is zero

Singleton groups are a separate implementation case, not the same condition.
```

### Diagnosis
```python
# Check: print advantage statistics
print(f"advantages mean={advantages.mean()}, std={advantages.std()}")
# std=0 alone is insufficient: a constant nonzero advantage also has zero std.
print(f"group_size={len(rewards_per_group)}")
# Check actual UID counts and reward/advantage values before diagnosing.
```

### Fix
```
Inspect groups with n >= 2: are their total rewards actually distinguishable?
Check reward parsing, UID grouping and response masks before changing rollout.n.
Choose sampling count/reward design against the task and budget, not a universal
"n >= 4" or "MoE n >= 8" rule derived from this implementation.
```

2026-10-22 更正：固定verl `1876b06d` 的loop/vectorized GRPO都把singleton设为mean=0、std=1。默认标准化时优势为 `score/(1+epsilon)`，关闭std归一化时为score，再按response_mask广播；不是自动归零，也不发生该分支的除零。原4组等价性测试及12组双实现CPU对照见[源码证据](frameworks/verl-grpo-core-algos-reading.md#单样本与同分组的不同回退)。旧“NEVER n=1、n≥4/8是硬门槛”的结论撤回。是否能有效学习仍需任务/训练证据，非零优势或loss标量为零都不能单独判定参数是否更新。

---

## 3. NaN in Loss/Advantages

### Symptoms
- Loss becomes NaN after a few steps
- Advantages contain NaN or Inf values
- Training crashes with "NaN detected in gradients"

### Root Causes (ordered by likelihood)

```
1. MoE FP16 gating softmax (★★★★★★★★ MOST COMMON for MoE models)
   → logits computation in FP16 → overflow before softmax shifting
   → Fix: compute gating in FP32: logits.float() → softmax → result.to(dtype)

2. gradient_clipping = 0.0 (★★★★★★★★ DeepSpeed default!)
   → No clipping → gradient explosion → NaN in optimizer step
   → Fix: gradient_clipping = 1.0

3. overlap_comm = True on single GPU (★★★★★ DeepSpeed #8061)
   → Multi-stream race → stale data → NaN
   → Fix: overlap_comm = False for dp=1

4. Check group statistics and non-finite inputs (see Section 2)
   → default GRPO uses epsilon and explicit singleton fallback
   → equal finite rewards in an n>=2 group give zero advantage, not automatic NaN
   → inspect the first non-finite value and actual estimator before changing group size

5. FP16 model dtype (★ overflow in activations)
   → BF16 range = FP32 range, FP16 range = [-65504, 65504]
   → Fix: use bf16, NOT fp16
```

### Diagnosis Steps
```python
# Step 1: Check where NaN appears
loss_val = loss.item()
if math.isnan(loss_val):
    # Check advantages
    print(f"advantages: nan={advantages.isnan().sum()}, inf={advantages.isinf().sum()}")
    # Check logits (MoE gating)
    print(f"gating_logits: max={logits.max()}, min={logits.min()}")
    # Check gradients
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.isnan().any():
            print(f"NaN gradient in {name}")
```

### Fix Priority
```
1. FP32 MoE gating softmax (universal, prevents NaN at source)
2. gradient_clipping = 1.0 (prevents gradient explosion)
3. overlap_comm = False (prevents stream race)
4. bf16 model dtype (prevents activation overflow)
5. Validate UID/reward/mask statistics; group size alone is not a universal NaN fix
```

---

## 4. Loss Spikes / Gradient Explosion

### Symptoms
- Loss jumps to very large values periodically
- Gradient norm > 100 or 1000
- Training oscillates between stable and unstable

### Root Causes

```
1. gradient_clipping = 0.0 → no bound on updates
2. Large rewards with REINFORCE → unbounded A = R → huge gradient
3. KL penalty too small → policy drifts too far → large ratio → large loss
4. Learning rate too high → overshooting optimum
```

### Diagnosis
```python
# Check gradient norm before clipping
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
print(f"gradient norm: {total_norm.item()}")
# If > 100 → explosion
# If consistently > 1.0 → clipping always active → need smaller lr or better advantage
```

### Fix
```
1. gradient_clipping = 1.0 (★★★ MUST)
2. Use GRPO (bounded advantages, std=1) instead of REINFORCE
3. kl_coef ≥ 0.01 (prevent drift)
4. Reduce learning rate by 2-5× (typical: 1e-5 for GRPO, 3e-4 for supervised)
```

### Async 熵爆炸 → 坍塌模式（verl #8250 案例，2026-10）

**2026-10-21纠正**：旧版把观测次序写成了已确证机制，并过早排除mask/管线问题；这些结论撤回。原报告的多组run并非受控消融，结果奖励为0时总奖励仍可非零且组内有差异。

`rollout_corr/kl` 在报告版本是采样token上 rollout_log_prob-old_log_prob 的masked mean，不是reference KL；具体阈值不能直接跨任务套用。标准有限、n>1、std归一GRPO有组内优势幅度界；不能断言grad爆炸必然伴随优势离群。

先对齐总奖励各分项、uid/group、优势、有效mask、ratio分位数与clip方向、梯度测量位置及各模型版本。A=0只保证孤立PG项为零；A非零仍可能因clip/mask等产生零梯度，不能仅据优势std判唯一根因。

固定完整token序列和版本的actor/重载checkpoint/live rollout三路logprob对照可以缩小范围，但数值差异不唯一指向weight-sync/cache。未补证前不推荐通用KL系数或0.05守卫作为已验证治疗。见 [更新后的诊断与数值反例](../open-source/verl/candidate-8250-async-entropy-collapse.md)。

---

## 5. Responses Never End (No EOS Token)

### Symptoms
- Generated responses keep going beyond expected length
- Never produce EOS/end-of-sequence token
- Response length >> max_new_tokens

### Root Cause

```
LoRA rank ≥ 64 (★★★★★★★★ vLLM #6782)
→ LoRA weights distort EOS token probability
→ Model loses ability to predict EOS
→ Responses continue indefinitely

Specific: lora_expand with block_n=128 on sm_90 (Hopper) → NaN in LoRA output → EOS probability corrupted
```

### Diagnosis
```python
# Check EOS probability during generation
eos_prob = F.softmax(logits[:, -1, :], dim=-1)[:, eos_token_id]
print(f"EOS probability: {eos_prob.item()}")
# If near 0 → EOS token lost

# Check LoRA config
print(f"lora_rank={lora_config.r}, lora_alpha={lora_config.alpha}")
# If rank >= 64 → problematic
```

### Fix
```
★★★ MUST: LoRA rank = 32, alpha = 64 (NEVER rank = 64!)
★★★ MUST: On sm_90 (Hopper): block_n = 32 for lora_expand (not default 128)
★★★ MUST: enforce_eager = True for training (no CUDA graph corruption)
```

---

## 6. All Responses Identical (Policy Collapse)

### Symptoms
- All generated responses are exactly the same
- Low diversity in rollout outputs
- Reward variance within groups → 0 (σ = 0 → A = 0 → no learning)

### Root Causes

```
1. KL penalty too strong → policy forced to match reference exactly
   → kl_coef too high (e.g., 0.1 or higher)
   → All responses identical → σ=0 → A=0 → no learning

2. Temperature = 0 during generation → greedy decoding
   → All group samples identical → σ=0 → A=0

3. PPO-clip ε too small → ratio barely changes → outputs barely change
```

### Diagnosis
```python
# Check response diversity
unique_responses = len(set([tuple(r.tolist()) for r in generated_responses]))
print(f"Unique responses in group: {unique_responses}/{group_size}")
# If 1 → all identical → collapse

# Check KL penalty magnitude
print(f"KL penalty: {kl_penalty.item()}")
# If > 0.1 → too strong
```

### Fix
```
1. Reduce kl_coef: 0.01 → 0.001 (start small)
2. Temperature ≥ 0.7 during rollout generation (NOT greedy!)
3. ε = 0.2 (standard, not 0.05)
4. Use UP-GRPO (allows positive A to flow freely → more diversity)
```

---

## 7. Training OOM

### Symptoms
- RuntimeError: CUDA out of memory
- Training crashes during forward/backward pass

### Root Causes (ordered by memory usage)

```
1. FSDP2 (★★★★★★★★ verl #6468 + #7016)
   → CPU memory leak in all_gather → OOM after many steps
   → MoE backward crash (#7016)
   → Fix: FSDP1 ONLY (NOT FSDP2!)

2. PPO with critic (★★★★★★★★)
   → 2× model parameters → 2× memory → OOM on 24 GiB
   → Fix: GRPO (no critic) → half memory

3. No bypass_mode (★★★★★★★)
   → old_log_prob forward → 18Ψ activations → OOM
   → Fix: bypass_mode = True → 18Ψ→3.8Ψ

4. ZeRO-3 on dp=1 (★★★★★)
   → No memory savings + overhead → pure waste
   → Fix: ZeRO-2 + CPU_Adam

5. overlap_comm = True (★★★★)
   → Communication buffers + gradient partitions → extra memory
   → Fix: overlap_comm = False on dp=1
```

### Memory Budget (RTX 4090, 24 GiB)
```
Config                           GPU Used    Headroom
ZeRO-2 + bypass + GRPO          19.2 GiB    4.8 GiB  ← VIABLE
ZeRO-2 + no bypass + GRPO       33.4 GiB    OOM!     ← FAILS
FSDP2 + bypass + GRPO           19.2 GiB    4.8 GiB  ← VIABLE but LEAKS
ZeRO-3 + bypass + GRPO          19.2 GiB    4.8 GiB  ← VIABLE but WASTE
PPO (with critic) + bypass      33.2 GiB    OOM!     ← FAILS
```

---

## 8. Intermittent Corruption (CUDA Stream Race)

### Symptoms
- Occasional NaN or wrong values (not every step)
- Bug depends on exact kernel timing → hard to reproduce
- torch.compile triggers it (changes kernel ordering)

### Root Cause

```
CUDA stream use-after-free (★★★★★★★★ #8061/#5788/#45552)
→ Buffer freed on stream A → stream B still reading → stale/garbage data
→ Caching allocator recycles memory → consumer reads wrong data
→ Intermittent: depends on exact kernel timing and allocator state
```

### Diagnosis
```bash
# Run compute-sanitizer (if available)
compute-sanitizer --tool memcheck python train.py
# Look for: "Invalid access to freed memory"

# Quick test: run with overlap_comm=True vs False
# If NaN with overlap_comm=True and no NaN with overlap_comm=False → stream race
```

### Fix
```
★★★ MUST: overlap_comm = False on single GPU (RTX 4090)
★★★ MUST: record_stream before freeing buffers (for multi-GPU)
★★★ MUST: torch.cuda.synchronize() before CuMem unmap (#45552)
```

---

## 9. Slow Convergence (Advantage/Loss Choice)

### Symptoms
- Training converges but very slowly
- Reward improves gradually but takes many steps
- Loss decreases but plateaus early

### Root Causes

```
1. PPO-clip suppresses positive advantages (upper clip blocks gradient)
2. REINFORCE: high variance → noisy gradient direction
3. token-mean aggregation: long responses dominate gradient
4. Outcome reward: too noisy for small group sizes
```

### Fix Priority
```
1. UP-GRPO loss (★★★★★★★★ BEST for GRPO — positive A always gets gradient)
   → Our PR #9 on Jackie2049/verl and Jackie2049/trl

2. CISPO loss (★★★★★★★ Alternative — ALL tokens keep gradient)
   → Detached clamp = weight, not gate → no zero-gradient zones

3. seq-mean-token-mean aggregation (★★★★★ Fair weighting per trajectory)
   → Short and long responses contribute equally

4. Shaped rewards (★★★★★ Not flat outcome rewards)
   → Per-token or per-feature rewards → more granular signal

5. Choose rollout.n against reward statistics and sampling budget; no universal threshold is established here
```

---

## 10. RTX 4090 GRPO Debugging Checklist

```
Before training:
  ★ Check UID counts, reward differences and masks; no universal group_size>=4/8 requirement (Section 2)
  ★ gradient_clipping = 1.0 (NOT 0.0!)
  ★ overlap_comm = False
  ★ ZeRO-2 + cpu_adam (NOT ZeRO-3)
  ★ FSDP1 (NOT FSDP2)
  ★ bypass_mode = True
  ★ LoRA rank = 32, alpha = 64 (NEVER 64!)
  ★ enforce_eager = True
  ★ bf16 model dtype + fp32 gating softmax
  ★ sleep_level = 1 (NEVER 2)
  ★ kl_coef = 0.01 (not too high, not zero)
  ★ temperature ≥ 0.7 during rollout

During training:
  ★ Monitor: advantages.mean() ≈ 0, advantages.std() ≈ 1
  ★ Monitor: gradient_norm ≈ 0.5-2.0 (not 0, not 100+)
  ★ Monitor: loss decreasing (not NaN, not spiking)
  ★ Monitor: KL penalty < 0.1 (not exploding)
  ★ Monitor: unique responses ≥ gs/2 (diverse rollouts)

If problems:
  1. NaN → check MoE gating FP32, grad_clip=1.0, overlap_comm=False
  2. Loss=0 → check group_size, shaped rewards, temperature
  3. OOM → check bypass_mode, ZeRO-2 vs ZeRO-3, no critic
  4. Slow → try UP-GRPO or CISPO loss, increase group_size
  5. No EOS → check LoRA rank, enforce_eager
```

---

## Session Stats
- **9 failure modes** diagnosed with root causes, diagnosis steps, and fixes
- **RTX 4090 checklist**: 11 before-training + 5 during-training checks
- **Priority ordering**: each fix rated by likelihood and impact
- **Cross-references**: all fixes linked to specific bug IDs and fork PRs
