"""CPU math control, not a verl patch or the paper's original JAX test suite.

PyTorch translation of arXiv:2610.31642v1 Sections 2/3, for finite dense
logits and a frozen teacher. Only the student's first derivative is injected.
Run: python verify.py --output result.json
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch
import torch.nn.functional as F


def paper_log_softmax(z):
    index = z.argmax(dim=-1, keepdim=True)
    mask = torch.zeros_like(z, dtype=torch.bool).scatter(-1, index, True)
    shifted = torch.where(mask, 0.0, z - z.gather(-1, index))
    others = shifted.masked_fill(mask, -torch.inf)
    return shifted - F.softplus(torch.logsumexp(others, dim=-1, keepdim=True))


def injected_ce(student, teacher):
    log_q = paper_log_softmax(student)
    q, p = log_q.exp(), paper_log_softmax(teacher).exp()
    delta = student - teacher
    d = delta - (p * delta).sum(dim=-1, keepdim=True)
    s = d - torch.log1p((p * torch.expm1(d)).sum(dim=-1, keepdim=True))
    u = p * torch.expm1(s)
    g = u - q * u.sum(dim=-1, keepdim=True)
    forward = -(p * log_q.detach()).sum(dim=-1)
    return forward + (g.detach() * (student - student.detach())).sum(dim=-1)


def ordinary_ce(student, teacher):
    return -(teacher.softmax(-1) * student.log_softmax(-1)).sum(-1)


def value_grad(fn, student, teacher):
    x = student.detach().clone().requires_grad_()
    loss = fn(x, teacher)
    grad = torch.autograd.grad(loss, x)[0]
    return loss.item(), grad.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    teacher = torch.tensor([2.0, 1.0, 0.0], dtype=torch.float32)
    cases = {
        "identical": teacher.clone(),
        "paper_near": teacher + torch.tensor([-1e-7, 1e-7, 0.0]),
        "constant_shift": teacher + 1.0,
    }
    result = {
        "scope": "PyTorch translation; dense finite logits; frozen teacher; CPU only",
        "paper": "https://arxiv.org/html/2610.31642v1#S3",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": {},
    }
    for name, student in cases.items():
        # Cast the already-rounded FP32 inputs, not the intended decimal values.
        _, reference = value_grad(ordinary_ce, student.double(), teacher.double())
        row = {"student": student.tolist(), "reference_fp64_grad": reference.tolist()}
        for label, fn in (("ordinary", ordinary_ce), ("injected", injected_ce)):
            value, grad = value_grad(fn, student, teacher)
            assert torch.isfinite(grad).all()
            row[label] = {
                "loss": value,
                "grad": grad.tolist(),
                "grad_sum": grad.sum().item(),
                "max_abs_error": (grad.double() - reference).abs().max().item(),
            }
        if name == "identical":
            assert row["injected"]["grad"] == [0.0, 0.0, 0.0]
        if name == "paper_near":
            assert row["injected"]["max_abs_error"] < row["ordinary"]["max_abs_error"]
        result["cases"][name] = row

    # A detached gradient injection is first-order only, not a Hessian recipe.
    t64 = teacher.double()
    x = t64.clone().requires_grad_()
    grad = torch.autograd.grad(injected_ce(x, t64), x, create_graph=True)[0]
    h_injected = torch.autograd.functional.hessian(lambda z: injected_ce(z, t64), t64)
    h_ordinary = torch.autograd.functional.hessian(lambda z: ordinary_ce(z, t64), t64)
    result["higher_order"] = {
        "injected_first_grad_requires_grad": grad.requires_grad,
        "functional_hessian_strict_false_injected": h_injected.tolist(),
        "ordinary_hessian": h_ordinary.tolist(),
        "max_hessian_difference": (h_injected - h_ordinary).abs().max().item(),
    }
    assert not grad.requires_grad
    assert torch.count_nonzero(h_injected).item() == 0
    assert (h_injected - h_ordinary).abs().max().item() > 0.1

    # Same distributions need not be stationary for an unnormalized top-k loss.
    p = t64.softmax(-1)
    ids = torch.tensor([0, 1])
    log_p = t64.log_softmax(-1)[ids]

    def truncated(z):
        return (p[ids] * (log_p - z.log_softmax(-1)[ids])).sum()

    loss, actual = value_grad(lambda z, _: truncated(z), t64, t64)
    expected = p[ids].sum() * p
    expected[ids] -= p[ids]
    assert torch.allclose(actual, expected, atol=1e-15, rtol=0)
    assert actual.abs().max().item() > 0.05
    result["topk_objective_control"] = {
        "scope": "unclamped mathematical top-k objective, not a verl test",
        "teacher_mass": p[ids].sum().item(),
        "loss_at_identical_logits": loss,
        "grad": actual.tolist(),
        "analytic_grad": expected.tolist(),
    }
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
