#!/usr/bin/env python3
"""
#7913 verification: GMPO geo_mean microbatch-split invariance.

Extracts masked_sum/masked_mean/agg_loss semantics verbatim from verl
(verl/trainer/ppo/core_algos.py, verl/utils/torch_functional.py) and the
geo_mean loss (pre- and post-patch), then reproduces the issue's
experiment: one minibatch of 8 sequences vs 8 microbatches of 1.

Method: autograd gradients of the accumulated microbatch losses w.r.t.
log_prob, compared with the single-minibatch gradient (the FSDP engine
accumulates per-microbatch backward() with no rescaling).
"""

import torch

torch.manual_seed(42)


def masked_sum(values, mask, axis=None):
    if mask.dtype == torch.bool:
        mask = mask.float()
    return (values * mask).sum(dim=axis)


def masked_mean(values, mask, axis=None):
    if mask.dtype == torch.bool:
        mask = mask.float()
    return (values * mask).sum(dim=axis) / mask.sum(dim=axis)


# ---- agg_loss, seq-mean-token-sum branch (verbatim from core_algos.py:1140) ----
def agg_loss_seq_mean_token_sum(loss_mat, loss_mask, dp_size=1, global_batch_size=None):
    """seq-mean-token-sum: token-sum per sequence, mean over global batch sequences."""
    seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
    seq_mask = (torch.sum(loss_mask, dim=-1) > 0).float()
    if global_batch_size is None:
        global_batch_size = seq_mask.sum()
    loss = masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size
    return loss


# ---- geo_mean pre-patch (verbatim core_algos.py:1928, simplified args) ----
def compute_geo_mean_pre(old_log_prob, log_prob, advantages, response_mask, clip_ratio_low=0.2, clip_ratio_high=0.2,
                        **unused):
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl_clamp = torch.clamp(negative_approx_kl, -clip_ratio_low, clip_ratio_high)
    sgn_advantage = torch.sign(advantages)
    negative_approx_kl_min = torch.min(sgn_advantage * negative_approx_kl, sgn_advantage * negative_approx_kl_clamp)
    negative_approx_kl_min = sgn_advantage * negative_approx_kl_min
    response_mask_sum = response_mask.sum(dim=-1)
    ratio = torch.exp((negative_approx_kl_min * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8))
    advantage = (advantages * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
    pg_losses = -advantage * ratio
    pg_loss = torch.mean(pg_losses)  # <-- the bug: microbatch-local mean
    return pg_loss


# ---- geo_mean post-patch: agg_loss seq-mean-token-sum with global batch info ----
def compute_geo_mean_post(old_log_prob, log_prob, advantages, response_mask, dp_size=1, global_batch_size=None,
                         clip_ratio_low=0.2, clip_ratio_high=0.2):
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl_clamp = torch.clamp(negative_approx_kl, -clip_ratio_low, clip_ratio_high)
    sgn_advantage = torch.sign(advantages)
    negative_approx_kl_min = torch.min(sgn_advantage * negative_approx_kl, sgn_advantage * negative_approx_kl_clamp)
    negative_approx_kl_min = sgn_advantage * negative_approx_kl_min
    response_mask_sum = response_mask.sum(dim=-1)
    ratio = torch.exp((negative_approx_kl_min * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8))
    advantage = (advantages * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
    pg_losses = -advantage * ratio
    seq_has_tokens = (response_mask.sum(dim=-1) > 0).float()
    pg_loss = agg_loss_seq_mean_token_sum(
        loss_mat=pg_losses.unsqueeze(1),
        loss_mask=seq_has_tokens.unsqueeze(1),
        dp_size=dp_size,
        global_batch_size=global_batch_size,
    )
    return pg_loss


def make_batch(bs=8, resp_len=16, dtype=torch.float64):
    old_log_prob = torch.randn(bs, resp_len, dtype=dtype) * 0.05
    log_prob = old_log_prob + torch.randn(bs, resp_len, dtype=dtype) * 0.1
    advantages = torch.randn(bs, resp_len, dtype=dtype)
    response_mask = torch.ones(bs, resp_len, dtype=dtype)
    # give a couple sequences fewer valid tokens to exercise weighting
    response_mask[0, -4:] = 0
    response_mask[3, -8:] = 0
    return old_log_prob, log_prob, advantages, response_mask


def grad_wrt_logprob(fn, old_log_prob, log_prob, advantages, response_mask, **kw):
    """Gradient of loss w.r.t. log_prob (the only 'parameter' here)."""
    lp = log_prob.clone().requires_grad_(True)
    loss = fn(old_log_prob, lp, advantages, response_mask, **kw)
    loss.backward()
    return lp.grad, loss


def relative_l2(a, b):
    return (a - b).norm() / b.norm()


def run(split, fn, global_batch_size, dp_size=1, data=None):
    """Simulate engine: per-microbatch loss, accumulate gradients (aligned by position).
    `data` must be shared across comparisons; defaults to a fresh make_batch()."""
    if data is None:
        data = make_batch()
    old_log_prob, log_prob, advantages, response_mask = data
    total_grad = torch.zeros_like(log_prob)
    start = 0
    for bs in split:
        idx = slice(start, start + bs)
        start += bs
        g, _ = grad_wrt_logprob(fn, old_log_prob[idx], log_prob[idx], advantages[idx], response_mask[idx],
                                global_batch_size=global_batch_size, dp_size=dp_size)
        total_grad[idx] += g
    g_full, _ = grad_wrt_logprob(fn, old_log_prob, log_prob, advantages, response_mask,
                                 global_batch_size=global_batch_size, dp_size=dp_size)
    return total_grad, g_full


def main():
    print("== #7913 geo_mean microbatch invariance (fp64, 8 seqs, resp_len 16) ==")
    n = 8
    for name, fn in [("PRE-patch (torch.mean)", compute_geo_mean_pre), ("POST-patch (agg_loss seq-mean)", compute_geo_mean_post)]:
        print(f"\n--- {name} ---")
        data = make_batch()
        g_full, _ = run([8], fn, global_batch_size=n, data=data)
        for split, label in [([1] * 8, "8 microbatches of 1"),
                             ([3, 5], "2 microbatches 3+5"),
                             ([2, 2, 4], "3 microbatches 2+2+4")]:
            g_split, _ = run(split, fn, global_batch_size=n, data=data)
            print(f"  split {str(split):<16} rel L2 grad vs [8]: {relative_l2(g_split, g_full):.6e}")

    # also check the patch preserves the exact single-batch loss value (equivalence when no split)
    old_log_prob, log_prob, advantages, response_mask = make_batch()
    _, loss_pre = grad_wrt_logprob(compute_geo_mean_pre, old_log_prob, log_prob, advantages, response_mask)
    _, loss_post = grad_wrt_logprob(compute_geo_mean_post, old_log_prob, log_prob, advantages, response_mask,
                                    global_batch_size=8, dp_size=1)
    print(f"\nSingle-batch loss: pre={loss_pre.item():.10f} post={loss_post.item():.10f} "
          f"(should match; relative diff {relative_l2(loss_post, loss_pre).item():.2e})")

    # dp>1: two ranks of 4, global_batch_size=8; FSDP averages rank grads.
    old, logp, adv, mask = make_batch(bs=4)
    g_r1, _ = grad_wrt_logprob(compute_geo_mean_post, old, logp, adv, mask, global_batch_size=8, dp_size=2)
    g_r2, _ = grad_wrt_logprob(compute_geo_mean_post, old, logp, adv, mask, global_batch_size=8, dp_size=2)
    _, g_dp1 = grad_wrt_logprob(compute_geo_mean_post, old, logp, adv, mask, global_batch_size=8, dp_size=1)
    # same 4 sequences on one rank of a dp=2 world vs on the single rank of dp=1
    print(f"dp=2 (rank of 4, global 8): rel L2 grad vs dp=1 same local data: {relative_l2(g_r1, g_dp1):.6e}")


if __name__ == "__main__":
    main()
