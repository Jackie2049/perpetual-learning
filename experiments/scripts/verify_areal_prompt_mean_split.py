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

Additional faithful finding (Case 4): transport-dummy slots count toward step_scale's M
(len(microbatches), loss_reduction.py:115) even though their loss_mask is zeroed
(make_transport_dummy, data.py:886-887) and engine_weight=0, so they contribute nothing
to the gradient but still inflate step_scale by one unit per dummy -> extra gradient
scale proportional to the number of dummy slots in the synchronized schedule.
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


def prompt_mean_total_with_dummy(group_sizes, seq_len, n_mbs, rng, dp_size=1):
    """Faithful prompt_mean gradient when some schedule slots are transport dummies.

    Mirrors real AReaL (data.py:1176-1180): after DP sync, effective_n_mbs slots are
    filled and the remaining slots become transport dummies. The dummy retains a
    *zeroed* loss_mask (make_transport_dummy, data.py:886-887) and engine_weight=0
    (loss_reduction.py:200), so it contributes 0 to the gradient numerator and 0 to
    total_weight; but it still counts toward step_scale's M (loss_reduction.py:115).
    """
    n_seq = sum(group_sizes)
    mask = np.ones((n_seq, seq_len), dtype=bool)
    loss = rng.standard_normal((n_seq, seq_len)).astype(np.float32)
    tw, G, _ = group_token_counts(group_sizes, mask)
    # effective_n_mbs = real (concrete) mbs + dummy padding to a sync multiple.
    real_mbs = n_mbs
    # pad to a larger synchronized schedule (simulates a rank needing one extra slot)
    effective_n_mbs = real_mbs + 1
    step_scale = effective_n_mbs / G  # M now INCLUDES the dummy slot
    idx = np.linspace(0, n_seq, real_mbs + 1).astype(int)
    blocks = [(idx[i], idx[i + 1]) for i in range(real_mbs)]
    total = 0.0
    for (lo, hi) in blocks:
        total_weight = 1.0
        seg_sum = float((loss[lo:hi] * tw[lo:hi]).sum())
        total += seg_sum * step_scale * dp_size  # dummy adds nothing to seg_sum
    # the extra dummy slot: numerator=0 (zeroed loss_mask), contributes nothing
    return total, effective_n_mbs, G


def run_dummy(gs, seq_len, dp=1):
    """Show that a dummy slot inflates step_scale (M) but contributes no gradient."""
    rng0 = np.random.default_rng(0)
    base, G = prompt_mean_total(gs, seq_len, 1, np.random.default_rng(0), dp)
    print(f"[dummy-slot] group_sizes={gs}, G={G}, dp={dp}")
    for real in [1, 2, 3]:
        v, eff_M, Gv = prompt_mean_total_with_dummy(
            gs, seq_len, real, np.random.default_rng(0), dp
        )
        print(f"  real_mbs={real} eff_M={eff_M} (1 dummy): {v:.6f}  "
              f"ratio_to_M1={v/base:.4f}  (extra factor {(eff_M/real):.2f} from dummy)")
    print(f"  expected: gradient = eff_M/G*dp*sum_all loss*w; dummy raises eff_M "
          f"but not the numerator, so a dummy slot ~1 extra M factor\n")


def megatron_prompt_mean_total(group_sizes, seq_len, n_mbs, rng, dp_size=1, p_inner=2):
    """Faithful end-to-end prompt_mean gradient for the MEGATRON backend (non per-token path).

    Megatron shares actor.py:600 `_PromptMeanBatch.for_steps` with FSDP2, so the SAME
    `step_scale = len(microbatches)/active_groups = M/G` applies (loss_reduction.py:115).
    The difference is in the engine's train_batch normalisation:

      megatron_engine.py:2382  loss_multiplier = dp_size * len(mb_list)
                                (non per_token_loss path; mb_list is this ONE ppo-minibatch
                                 re-split into P inner pipeline micro-batches by
                                 _prepare_mb_list, len(mb_list)=P)
      megatron_engine.py:4328  normalize_loss(loss, local_weight, total_weight, loss_multiplier)
                                -> scalar path loss*(local/total*multiplier)

    For each of the P inner pipeline mb of one ppo-minibatch i:
        PromptMean.aggregate = (sum loss_t*w_t)_p * step_scale / engine_weight_p
        normalize -> (sum loss_t*w_t)_p * step_scale * dp_size * P
        (engine_weight_p cancels; total_weight==1 as in FSDP2)
    Summing the P inner mb (additive over sequences) and then the M ppo-minibatches:
        grad_total = dp_size * step_scale * P * sum_all loss*w
                  = dp_size * (M/G) * P * sum_all loss*w
    So Megatron shares the M-linear split sensitivity with FSDP2 (step_scale=M/G),
    and additionally multiplies the absolute scale by P (inner pipeline mb count,
    fixed by pp_size / mb_spec, NOT by ppo_n_minibatches).
    """
    n_seq = sum(group_sizes)
    mask = np.ones((n_seq, seq_len), dtype=bool)
    loss = rng.standard_normal((n_seq, seq_len)).astype(np.float32)
    tw, G, _ = group_token_counts(group_sizes, mask)
    effective_n_mbs = n_mbs
    step_scale = effective_n_mbs / G
    idx = np.linspace(0, n_seq, n_mbs + 1).astype(int)
    total = 0.0
    for i in range(n_mbs):
        lo, hi = idx[i], idx[i + 1]
        # total_weight over this ppo-minibatch's inner mb (DP all_reduce) == 1
        total_weight = 1.0
        # split this ppo-minibatch into P inner pipeline micro-batches (additive)
        inner = np.linspace(lo, hi, p_inner + 1).astype(int)
        minibatch_sum = 0.0
        for j in range(p_inner):
            ilo, ihi = inner[j], inner[j + 1]
            seg_sum = float((loss[ilo:ihi].astype(np.float32) * tw[ilo:ihi]).sum())
            minibatch_sum += seg_sum * step_scale * dp_size * p_inner  # normalize_loss
        total += minibatch_sum
    return total, G


def run_megatron(gs, seq_len, dp=1, p_inner=2):
    """Show Megatron shares FSDP2's M-linear prompt_mean sensitivity, plus a constant P."""
    rng0 = np.random.default_rng(0)
    tm = token_mean_total(gs, seq_len, rng0)
    print(f"[Megatron backend] group_sizes={gs}, seq_len={seq_len}, G={sum(1 for g in gs if g>0)}, "
          f"dp_size={dp}, P(inner pipeline mb)= {p_inner}")
    print(f"token_mean baseline (split-invariant): {tm:.6f}")
    base = None
    for n in [1, 2, 4]:
        v, G = megatron_prompt_mean_total(gs, seq_len, n, np.random.default_rng(0), dp, p_inner)
        if base is None:
            base = v
        print(f"  prompt_mean n_mbs={n}: {v:.6f}  ratio_to_n_mbs1={v/base:.4f}  (G={G})")
    # show P scales the absolute value but not the M-ratio
    p1 = megatron_prompt_mean_total(gs, seq_len, 4, np.random.default_rng(0), dp, 1)[0]
    p2 = megatron_prompt_mean_total(gs, seq_len, 4, np.random.default_rng(0), dp, 2)[0]
    print(f"  P-ratio at fixed n_mbs=4: P=2 / P=1 = {p2/p1:.3f} (theory 2.0) -> "
          f"P is a constant absolute-scale factor, NOT a ppo split term")
    print(f"  theory: grad_total ~ effective_n_mbs/G * dp * P * sum_all loss*w; "
          f"M-linear sensitivity shared with FSDP2, plus constant P\n")


# Case 1: uniform groups, DP=1  (default AReaL-style row splitting)
run([4, 3, 2], 5)
# Case 2: uneven groups, more sequences, verify the scaling holds regardless of cut points
run([7, 1, 4, 5], 6)
# Case 3: DP=2 (loss_multiplier=dp_size multiplies everything; relative prompt_mean scaling in N persists)
run([4, 3, 2], 5, dp=2)
# Case 4: dummy transport slots inflate step_scale's M without adding gradient
run_dummy([4, 3, 2], 5)
run_dummy([7, 1, 4, 5], 6, dp=2)
def megatron_per_token_prompt_mean_total(group_sizes, seq_len, n_mbs, rng, dp_size=1, p_inner=2):
    """Faithful end-to-end prompt_mean gradient for the MEGATRON *per-token-loss* path.

    Traced from AReaL source (main 3b93679) + Megatron-Core 0.19.0 default
    (`calculate_per_token_loss: bool = False`, so this is the OPT-IN path taken only
    when a user sets calculate_per_token_loss=True in the MCore model config):

      megatron_engine.py:2364-2369  elif per_token_loss:   (prompt_mean is NOT a named
                      normalizer -> is_named_loss_normalizer(loss_weight_fn)==False)
                        total_loss_weight = None
                        loss_multiplier = self._per_token_loss_multiplier(...)
                                        = T_global / W_global
                        (per-token count over weight, all-reduced over DP; megatron_engine.py:2414)
      megatron_engine.py:4124-4132  native_named = per_token_loss and isinstance(local_weight, dict)
                        -> False for prompt_mean (local_weight=engine_weight is a scalar)
                        token_count = self._per_token_count(...) = loss_mask.count_nonzero()
      megatron_engine.py:4330      _build_per_token_loss_output(loss, engine_weight,
                                        loss_multiplier=T_global/W_global, token_count):
                        loss_numerator = loss * engine_weight * (T_global/W_global) / cp_size
      MCore finalize_model_grads    sums loss_numerator over DP+CP and divides by TOTAL tokens T_global.

    Per ppo-minibatch i, over its P inner pipeline mb (each cp replica has full sequences):
        loss_numerator_p = (sum_t loss*w)_p * step_scale/engine_weight_p * engine_weight_p
                             * (T_global_i/W_global_i) / cp_size
        finalize:  sum_P loss_numerator_p / T_global_i
                 = (sum_t loss*w)_p * step_scale / W_global_i     (engine_weight cancels; T cancels)
        W_global_i = all-reduced sum of engine_weight over the inner mbs = 1 (as in FSDP2/§10.2)
      Summing over M ppo-minibatches:
        grad_total = step_scale * sum_all loss*w = (M/G) * sum_all loss*w

    Result: the M-linear split-sensitivity (step_scale=M/G) SURVIVES the per-token path;
    the constant factors change (no dp_size, no P) because MCore's /T_global cancels
    the *multiplied* T_global, but the split-dependence in M is identical to FSDP2 and to
    the Megatron default path. Closes the §10.5 "per_token_loss path unverified" gap.
    """
    n_seq = sum(group_sizes)
    mask = np.ones((n_seq, seq_len), dtype=bool)
    loss = rng.standard_normal((n_seq, seq_len)).astype(np.float32)
    tw, G, _ = group_token_counts(group_sizes, mask)
    effective_n_mbs = n_mbs
    step_scale = effective_n_mbs / G
    idx = np.linspace(0, n_seq, n_mbs + 1).astype(int)
    total = 0.0
    for i in range(n_mbs):
        lo, hi = idx[i], idx[i + 1]
        W_global_i = 1.0  # all-reduced sum of engine_weight over inner mbs (as in §10.2)
        inner = np.linspace(lo, hi, p_inner + 1).astype(int)
        minibatch_sum = 0.0
        for j in range(p_inner):
            ilo, ihi = inner[j], inner[j + 1]
            seg_sum = float((loss[ilo:ihi].astype(np.float32) * tw[ilo:ihi]).sum())
            # loss_numerator then /T_global by MCore; T_global cancels, W_global_i==1:
            minibatch_sum += seg_sum * step_scale / W_global_i
        total += minibatch_sum
    return total, G


def run_megatron_per_token(gs, seq_len, dp=1, p_inner=2):
    """Show the Megatron per-token-loss path KEEPS the M-linear prompt_mean sensitivity."""
    rng0 = np.random.default_rng(0)
    tm = token_mean_total(gs, seq_len, rng0)
    print(f"[Megatron per-token-loss path] group_sizes={gs}, seq_len={seq_len}, "
          f"G={sum(1 for g in gs if g>0)}, dp_size={dp}, P(inner pipeline mb)={p_inner}")
    print(f"token_mean baseline (split-invariant): {tm:.6f}")
    base = None
    for n in [1, 2, 4]:
        v, G = megatron_per_token_prompt_mean_total(
            gs, seq_len, n, np.random.default_rng(0), dp, p_inner
        )
        if base is None:
            base = v
        print(f"  prompt_mean n_mbs={n}: {v:.6f}  ratio_to_n_mbs1={v/base:.4f}  (G={G})")
    # P should NOT affect the per-token result (T_global cancels; no P multiplier survives)
    p1 = megatron_per_token_prompt_mean_total(gs, seq_len, 4, np.random.default_rng(0), dp, 1)[0]
    p2 = megatron_per_token_prompt_mean_total(gs, seq_len, 4, np.random.default_rng(0), dp, 2)[0]
    print(f"  P-ratio at fixed n_mbs=4: P=2 / P=1 = {p2/p1:.3f} "
          f"(theory 1.0: per-token path cancels P via /T_global)")
    print(f"  theory: grad_total ~ (M/G) * sum_all loss*w; M-linear sensitivity "
          f"identical to FSDP2 & Megatron default path; constants (dp_size, P) drop out\n")


# Case 5: Megatron backend shares step_scale=M/G (cross-backend); P is a constant factor
run_megatron([4, 3, 2], 5)
run_megatron([7, 1, 4, 5], 6, dp=2)
# Case 6: Megatron per-token-loss path (calculate_per_token_loss=True, opt-in) keeps M-linear sensitivity
run_megatron_per_token([4, 3, 2], 5)
run_megatron_per_token([7, 1, 4, 5], 6, dp=2)
