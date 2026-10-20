"""Verify AReaL prompt_mean split-dependence: faithful numeric check.

AReaL chain (loss_aggregation.py::PromptMean + loss_reduction.py::_bind_prompt_mean
+ train_engine.py::normalize_loss):
  token_weights  = loss_mask / group_token_count            (prepared on full batch)
  step_scale     = len(microbatches)/active_groups          (= PPO minibatch count N / G)
  engine_weight  = rows / global_rows                       (mb share of a minibatch's responses)
  mb_loss        = (sum loss_t*w_t) * step_scale / engine_weight   # PromptMean.aggregate
  final          = sum_mb normalize_loss(mb_loss, engine_weight, total_weight=1)

Hypothesis being tested: final normalized loss scales with N (minibatch count).
We compare against token_mean (no step factor) as an invariant baseline, and
against the "ideal" prompt_mean that would be truly split-invariant.
"""
import numpy as np

# faithful PromptMean
def prompt_mean_loss(group_sizes, seq_len, n_mbs, rng):
    n_seq = sum(group_sizes)
    mask = np.ones((n_seq, seq_len), dtype=bool)
    ids = np.repeat(np.arange(len(group_sizes)), group_sizes)
    counts = np.zeros(len(group_sizes), dtype=np.int64)
    np.add.at(counts, ids, mask.sum(axis=-1))
    tw = mask.astype(np.float32) / np.maximum(counts[ids], 1)[:, None]   # 1/group_token_count
    loss = rng.standard_normal((n_seq, seq_len)).astype(np.float32)
    active_groups = int((counts > 0).sum())                              # G
    step_scale = n_mbs / active_groups                                   # N / G
    idx = np.linspace(0, n_seq, n_mbs + 1).astype(int)
    rows = [np.arange(idx[i], idx[i+1]) for i in range(n_mbs)]
    total = 0.0
    for mb in rows:
        global_rows = len(mb)                        # this minibatch's response count
        for s in mb:                                 # each sequence = one microbatch
            num = (np.where(mask[s], loss[s], 0.0).astype(np.float32) * tw[s]).sum()
            eng = 1.0 / global_rows                   # rows/global_rows
            local = num * step_scale / eng           # aggregate divides by eng
            total += local * (eng / 1.0)              # normalize_loss, total_weight=1
    return total

# invariant baseline: token_mean over the whole batch (one PPO step)
def token_mean_baseline(group_sizes, seq_len, rng):
    n_seq = sum(group_sizes)
    mask = np.ones((n_seq, seq_len), dtype=bool)
    loss = rng.standard_normal((n_seq, seq_len)).astype(np.float32)
    return loss.astype(np.float32).sum() / mask.sum()   # split-invariant by construction

gs = [4, 3, 2]
base_pm = prompt_mean_loss(gs, 5, 1, np.random.default_rng(0))
tm = token_mean_baseline(gs, 5, np.random.default_rng(0))
print(f"group_sizes={gs}, seq_len=5, active_groups(G)=3")
print(f"token_mean baseline (invariant reference): {tm:.6f}")
for n_mbs in [1, 2, 3, 4, 6]:
    v = prompt_mean_loss(gs, 5, n_mbs, np.random.default_rng(0))
    print(f"  prompt_mean N={n_mbs}: {v:.6f}  ratio_to_N1={v/base_pm:.4f}  (=N since G=3 only if N multiple of 3)")

# Special case: N == G (one group per minibatch) -> ratio should be N/G = G/G = 1? test with G groups
print("--- N == G case (G=3): ratio should collapse to 1 if design is 'one group per step' ---")
print("   (AReaL default ppo_n_minibatches typically != group count)")
