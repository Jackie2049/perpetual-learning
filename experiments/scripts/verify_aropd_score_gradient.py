#!/usr/bin/env python3
"""Exact three-action CPU reference for the raw AROPD score-function formula.

Python >=3.10, CPU torch. Self-authored math, not original UniRL execution/tests.
Fixed prefix, positive full-support p/q, frozen teacher, p_old=p_new, raw gap only.
No sampling noise, trajectory gradients, gap normalization, model or device probe.
"""
import sys

import torch


def objective(logits, teacher):
    logp = logits.log_softmax(0)
    return (logp.exp() * (logp - teacher.log())).sum()


for title, student, teacher in [
    ('different policies', [.2, .3, .5], [.7, .2, .1]),
    ('student equals teacher', [.2, .3, .5], [.2, .3, .5]),
]:
    logits = torch.tensor(student, dtype=torch.float64, device='cpu').log().requires_grad_()
    q = torch.tensor(teacher, dtype=torch.float64, device='cpu')
    logp = logits.log_softmax(0)
    p = logp.exp()
    # Detached p weights enumerate the expectation of a sampled-token gradient.
    weights = p.detach()
    gap = (q.log() - logp.detach()).detach()
    losses = {
        'reverse_KL': objective(logits, q),
        'raw_score_surrogate': -(weights * gap * logp).sum(),
        'naive_sampled_gap': (weights * (logp - q.log())).sum(),
        'teacher_cross_entropy': -(p * q.log()).sum(),
    }
    gradients = {name: torch.autograd.grad(loss, logits, retain_graph=True)[0]
                 for name, loss in losses.items()}
    torch.testing.assert_close(gradients['raw_score_surrogate'], gradients['reverse_KL'],
                               rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(gradients['naive_sampled_gap'], torch.zeros_like(logits),
                               rtol=0, atol=1e-12)
    assert not torch.allclose(gradients['teacher_cross_entropy'], gradients['reverse_KL'])
    entropy_term = (p * logp).sum()
    entropy_gradient = torch.autograd.grad(entropy_term, logits)[0]
    torch.testing.assert_close(gradients['teacher_cross_entropy'] + entropy_gradient,
                               gradients['reverse_KL'], rtol=1e-12, atol=1e-12)
    eps = 1e-5
    basis = torch.eye(3, dtype=torch.float64, device='cpu')
    finite_diff = torch.stack([(objective(logits.detach() + eps * v, q)
                               - objective(logits.detach() - eps * v, q)) / (2 * eps) for v in basis])
    torch.testing.assert_close(finite_diff, gradients['reverse_KL'], rtol=1e-8, atol=1e-9)
    if title == 'student equals teacher':
        torch.testing.assert_close(gradients['reverse_KL'], torch.zeros_like(logits), rtol=0, atol=1e-12)
        assert gradients['teacher_cross_entropy'].norm() > .1
    print(title, 'KL=', losses['reverse_KL'].item(), 'surrogate_value=', losses['raw_score_surrogate'].item())
    for name, gradient in gradients.items():
        print(name, [round(value, 9) for value in gradient.tolist()])
print('PASS: two exact expectations, entropy identity and central finite differences')
print('Reference math only; not AROPD/default-normalization/LoRA/FSDP/training validation')
print('Python', sys.version.split()[0], 'torch', torch.__version__)
