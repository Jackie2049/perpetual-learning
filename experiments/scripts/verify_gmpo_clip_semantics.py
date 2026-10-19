#!/usr/bin/env python3
"""
Verify GMPO clip-space equivalence: paper rho-space clip vs verl log-space clamp.

Claim (notebook/papers/gmpo.md): with matched windows,
  log(min(rho, eps_hi)) == min(log rho, log eps_hi)
so the paper's rho-space clipping and verl's log-space clamping produce
identical forward values AND gradients (up to measure-zero boundary
convention). The real difference is config semantics: the same numeric
epsilon means a rho-space window in the paper but a log-space (KL)
window in verl.

Run: python3 verify_gmpo_clip_semantics.py
"""

import math

import torch


def grad_of(fn, x):
    x = x.detach().requires_grad_(True)
    fn(x).mean().backward()
    return x.grad


def main():
    a, b = 0.4, 0.4  # verl log-space window [-a, b]; paper rho window [e^-a, e^b]

    # 1) forward equivalence across tight + outlier regions
    log_rho = torch.linspace(-1.0, 1.0, 401)
    paper_clip = torch.clamp(log_rho.exp(), min=math.exp(-a), max=math.exp(b))
    verl_clamp = torch.clamp(log_rho, min=-a, max=b).exp()
    print("1) forward max |diff|:", (paper_clip - verl_clamp).abs().max().item())

    # 2) gradient equivalence (interior; boundary points excluded - torch
    #    clamp's inclusive-at-edge convention differs at exactly x == edge)
    log_rho = torch.linspace(-0.999, 0.999, 501)
    gp = grad_of(lambda lr: torch.clamp(lr.exp(), min=math.exp(-a), max=math.exp(b)), log_rho)
    gv = grad_of(lambda lr: torch.clamp(lr, min=-a, max=b).exp(), log_rho)
    print("2) interior grad allclose (1e-12):", torch.allclose(gp, gv, atol=1e-12),
          "| max diff:", (gp - gv).abs().max().item())

    # random-set sanity (no exact boundary hits)
    torch.manual_seed(7)
    lr = torch.randn(10000) * 0.5
    gp2 = grad_of(lambda x: torch.clamp(x.exp(), min=math.exp(-a), max=math.exp(b)), lr)
    gv2 = grad_of(lambda x: torch.clamp(x, min=-a, max=b).exp(), lr)
    print("   random-set allclose:", torch.allclose(gp2, gv2, atol=1e-10))

    # 3) sequence-level geo-mean equivalence
    B, L = 4, 32
    torch.manual_seed(0)
    log_rho_seq = torch.randn(B, L) * 0.3
    paper_seq = (torch.clamp(log_rho_seq.exp(), min=math.exp(-a), max=math.exp(b))
                 .clamp_min(1e-12).log().sum(-1) / L).exp()
    verl_seq = (torch.clamp(log_rho_seq, min=-a, max=b).sum(-1) / L).exp()
    print("3) sequence geo-mean allclose:", torch.allclose(paper_seq, verl_seq, atol=1e-10))

    # 4) config-semantics trap: same number, different effective windows
    print("\n4) config-semantics trap (effective windows for a given clip_ratio):")
    for clip_ratio in [0.4, 0.0003, 0.0004]:
        print(f"   clip_ratio={clip_ratio:<8} -> rho window "
              f"[{math.exp(-clip_ratio):.6f}, {math.exp(clip_ratio):.6f}] "
              f"(KL +-{clip_ratio} nats)")


if __name__ == "__main__":
    main()
