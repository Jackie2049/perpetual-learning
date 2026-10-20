"""Verify AReaL prompt_mean split-dependence with a faithful normalisation chain.

Faithful chain (traced from real source, AReaL main 3b93679):

  actor.py:569        prepare_policy_gradient_batch() prepares token_weights on the
                      FULL batch: token_weights = loss_mask / group_token_count,
                      and returns _PromptMeanBatch(active_group_count=G).
  actor.py:590        split_training_batch_into_microbatches(n_mbs=ppo_n_minibatches)
                      splits the full batch into a DP-synchronised schedule whose
                      *real* length is effective_n_mbs (not necessarily == ppo_n_minibatches,
                      data.py:1144-1147). Some slots may be transport dummies.
  actor.py:600        prepared_loss.for_steps(mb_inputs) sets
                      step_scale = len(microbatches)/active_groups = effective_n_mbs / G
                      (loss_reduction.py:115).
  actor.py:609-636    each minibatch i runs engine.train_batch() -> one optimizer_step.
  engine/fsdp_engine.py:872  compute_total_loss_weight() sums engine_weight over the
                      micro-batches of ONE minibatch, DP all_reduce -> = 1
                      (each mb i: engine_weight = rows_i/global_rows_i, DP sum over
                      ranks of rows_i = global_rows_i, so total_weight == 1).
  engine/fsdp_engine.py:887  normalize_loss(loss, local_weight, total_weight,
                      loss_multiplier=dp_size)  ->  scalar path loss*(local/total*dp_size).

  PromptMean.aggregate (loss_aggregation.py:96-110):
      mb_loss = (sum_mb loss_t*w_t) * step_scale / engine_weight
  normalize_loss then cancels engine_weight and divides by total_weight=1:
      final_mb = (sum_mb loss_t*w_t) * step_scale * dp_size
  N minibatches accumulate in the optimizer:
      grad_total = dp_size * step_scale * sum_all loss_t*w_t
                = dp_size * (effective_n_mbs/G) * sum_all loss_t*w_t

  Under fixed dp_size this scales linearly with effective_n_mbs.
  token_mean has NO step factor and NO dp cancellation of a minibatch count, so it is
  split-invariant; seq_mean/constant likewise. So prompt_mean is the ONLY reduction
  whose gradient grows with the minibatch count.

Hypothesis tested: total gradient is proportional to effective_n_mbs (split-sensitive),
independently of how the minibatch boundaries cut across prompt groups.
"""
import numpy as np


def group_token_counts(group_sizes, mask):
    """faithful token_weights = mask / group_token_count (loss_reduction.py:167-173)."""
    n_seq = mask.shape[0]
    ids = np.repeat(np.arange(len(group_sizes)), group_sizes)
    counts = np.zeros(len(group_sizes), dtype=np.int64)
    np.add.at(counts, ids, mask.sum(axis=-1))
    tw = mask.astype(np.float32) / np.maximum(counts[ids], 1)[:, None]
    return tw, int((counts > 0).sum()), counts


def prompt_mean_total(group_sizes, seq_len, n_mbs, rng, dp_size=1):
    """Faithful end-to-end gradient for one PPO update under DP=dp_size (DP=1 default)."""
    n_seq = sum(group_sizes)
    mask = np.ones((n_seq, seq_len), dtype=bool)
    loss = rng.standard_normal((n_seq, seq_len)).astype(np.float32)
    tw, G, _ = group_token_counts(group_sizes, mask)
    # effective_n_mbs is the DP-synchronised schedule length. Under DP=1 it equals n_mbs.
    effective_n_mbs = n_mbs
    step_scale = effective_n_mbs / G                       # loss_reduction.py:115
    # split the full batch into n_mbs row-blocks (minibatch i covers rows idx[i]:idx[i+1])
    idx = np.linspace(0, n_seq, n_mbs + 1).astype(int)
    blocks = [(idx[i], idx[i + 1]) for i in range(n_mbs)]
    total = 0.0
    for (lo, hi) in blocks:
        # one minibatch i -> engine.train_batch -> optimizer_step
        # total_weight over the micro-batches of this minibatch (DP all_reduce) == 1
        total_weight = 1.0
        # sum_mb loss_t*w_t over this minibatch's sequences
        seg_sum = float((loss[lo:hi].astype(np.float32) * tw[lo:hi]).sum())
        final_mb = seg_sum * step_scale * dp_size           # normalize_loss
        total += final_mb
    return total, G


def token_mean_total(group_sizes, seq_len, rng):
    """token_mean: sum(loss)/valid_token_count -> split-invariant, no step factor."""
    n_seq = sum(group_sizes)
    mask = np.ones((n_seq, seq_len), dtype=bool)
    loss = rng.standard_normal((n_seq, seq_len)).astype(np.float32)
    return loss.astype(np.float32).sum() / mask.sum()


def run(gs, seq_len, dp=1):
    rng0 = np.random.default_rng(0)
    tm = token_mean_total(gs, seq_len, rng0)
    print(f"group_sizes={gs}, seq_len={seq_len}, G={sum(1 for g in gs if g>0)}, dp_size={dp}")
    print(f"token_mean baseline (split-invariant): {tm:.6f}")
    base = None
    for n in [1, 2, 3, 4, 6]:
        v, G = prompt_mean_total(gs, seq_len, n, np.random.default_rng(0), dp)
        if base is None:
            base = v
        print(f"  prompt_mean n_mbs={n}: {v:.6f}  ratio_to_n_mbs1={v/base:.4f}  (G={G})")
    print(f"  theory: grad_total ~ effective_n_mbs/G * dp * sum_all loss*w; "
          f"token_mean independent of n_mbs\n")


# Case 1: uniform groups, DP=1  (default AReaL-style row splitting)
run([4, 3, 2], 5)
# Case 2: uneven groups, more sequences, verify the scaling holds regardless of cut points
run([7, 1, 4, 5], 6)
# Case 3: DP=2 (loss_multiplier=dp_size multiplies everything; relative prompt_mean scaling in N persists)
run([4, 3, 2], 5, dp=2)
