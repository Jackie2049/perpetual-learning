#!/usr/bin/env python3
"""
Verify GSPO gradient-path difference: verl combined-ratio (detached seq
part) vs paper full-sequence ratio.

Claim (notebook/papers/gmpo.md 延伸问题 2 / gspo.md):
- verl GSPO: log s~_t = log rho_t (live) + sg[mean_t' log rho_t'] (detached)
  -> gradient w.r.t. log rho flows ONLY through the current token (identity)
- paper GSPO: log s_i = sum_t log rho_t (full theta-dependence)
  -> gradient w.r.t. log rho is all-ones (every token's ratio gradient
     includes ALL tokens' contributions)
- additionally the seq-ratio SCALE differs: mean (verl) vs sum (paper),
  length-dependent

Run: python3 verify_gspo_detach_gradient.py
"""

import torch

torch.manual_seed(0)


def grad_log_ratio_verl(log_rho):
    """verl combined ratio: log_prob - log_prob.detach() + seq_kl.detach().
    The live part is the current token only (identity Jacobian)."""
    seq_kl = log_rho.mean(-1, keepdim=True)
    log_combined = log_rho - log_rho.detach() + seq_kl.detach()
    return log_combined


def grad_log_ratio_paper(log_rho):
    """paper sequence ratio: log s = sum_t log rho_t (all tokens live)."""
    seq_log = log_rho.sum(-1, keepdim=True)
    return seq_log + log_rho * 0  # broadcast to token level for comparison


def main():
    B, L = 4, 8
    log_rho = torch.randn(B, L) * 0.3

    for name, fn in [("verl (detached seq part)", grad_log_ratio_verl),
                     ("paper (full sequence product)", grad_log_ratio_paper)]:
        x = log_rho.detach().requires_grad_(True)
        out = fn(x)
        out.sum().backward()
        jac_diag = x.grad  # d(out)/d(log_rho) per element (summed over the seq dim)
        print(f"{name}:")
        print(f"  d(sum log_combined)/d log_rho_t (first sequence): {x.grad[0].tolist()}")
        # per-token gradient of the token's own contribution vs cross terms
        print()

    # explicit Jacobian comparison at token level
    x = log_rho.detach().requires_grad_(True)
    out_v = grad_log_ratio_verl(x)
    J_v = torch.autograd.functional.jacobian(lambda v: grad_log_ratio_verl(v).flatten(), log_rho.flatten())
    x = log_rho.detach().requires_grad_(True)
    out_p = grad_log_ratio_paper(x)
    J_p = torch.autograd.functional.jacobian(lambda v: grad_log_ratio_paper(v).reshape(1, -1).squeeze(0), log_rho.flatten())
    print("verl Jacobian (first 4x4 block, should be identity):")
    print(torch.round(J_v[:4, :4], decimals=2))
    print("paper Jacobian (first 4x4 block, should be all-ones):")
    print(torch.round(J_p[:4, :4], decimals=2))

    # scale difference: mean vs sum for different lengths
    print("\nseq-ratio scale: mean (verl) vs sum (paper) for L=8 and L=64:")
    for L in (8, 64):
        lr = torch.randn(1, L) * 0.05
        print(f"  L={L}: verl seq-kl = {lr.mean().item():+.4f} (per-token), paper seq-kl = {lr.sum().item():+.4f} (total)")


if __name__ == "__main__":
    main()
