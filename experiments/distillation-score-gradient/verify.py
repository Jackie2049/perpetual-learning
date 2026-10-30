"""Finite-state CPU gradient decomposition, not verl's distillation_loss API.

Uses original kl_penalty/kl_penalty_forward with inactive clamps. The ratio
surrogate is explicitly transcribed here, without PPO or configuration objects.
Sampling weights are detached for expected SGD gradients, live only when
differentiating the complete distributional expectation.
"""

import ast
import hashlib
import json
import os
import platform
import subprocess
import sys

import torch

BASE = "321e4e450fcfc47a3491c946cd01dacd9ea7281a"
HEAD = "b2278006cc65a1cc6f82c14e6cf1d458c576cdef"
CORE = "verl/trainer/ppo/core_algos.py"
LOSS = "verl/trainer/distillation/losses.py"
CONFIG = "verl/workers/config/distillation.py"
YAML = "verl/trainer/config/distillation/distillation.yaml"


def main(repo):
    torch.set_num_threads(1)
    sources = {}
    for revision in (BASE, HEAD):
        for path in (CORE, LOSS, CONFIG, YAML):
            source = subprocess.check_output(["git", "-C", repo, "show", f"{revision}:{path}"],
                                             env={**os.environ, "GIT_NO_LAZY_FETCH": "1"})
            sources[revision, path] = source
            if path.endswith(".py"):
                compile(source, path, "exec")
    for path in (CORE, LOSS, YAML):
        assert sources[BASE, path] == sources[HEAD, path], path
    defaults = {}
    for revision in (BASE, HEAD):
        tree = ast.parse(sources[revision, CONFIG])
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "DistillationLossConfig")
        field = next(node for node in cls.body if isinstance(node, ast.AnnAssign)
                     and isinstance(node.target, ast.Name) and node.target.id == "use_policy_gradient")
        defaults[revision] = ast.literal_eval(field.value)
    assert defaults == {BASE: True, HEAD: False}
    assert b"use_policy_gradient: false" in sources[BASE, YAML]

    tree = ast.parse(sources[BASE, CORE])
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {"kl_penalty", "kl_penalty_forward"}]
    assert len(selected) == 2 and all(not node.decorator_list for node in selected)
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=selected, type_ignores=[]), CORE, "exec"), namespace)
    penalty = namespace["kl_penalty"]
    cases = [("equal", [0.5, 0.5], [0.5, 0.5]),
             ("mild", [0.6, 0.4], [0.55, 0.45]),
             ("opposing", [0.8, 0.2], [0.2, 0.8]),
             ("three_actions", [0.2, 0.3, 0.5], [0.5, 0.4, 0.1])]
    output = []
    for label, student, teacher in cases:
        p = torch.tensor(student, dtype=torch.float64, device="cpu")
        q = torch.tensor(teacher, dtype=torch.float64, device="cpu")
        logits = p.log().requires_grad_()
        logp = logits.log_softmax(-1)
        logq = q.log()
        reverse = (logp.exp() * (logp - logq)).sum()
        forward = (q * (logq - logp)).sum()

        def grad(value):
            return torch.autograd.grad(value, logits, retain_graph=True)[0]

        reverse_g, forward_g = grad(reverse), grad(forward)
        finite_difference = []
        for index in range(len(p)):
            delta = torch.zeros_like(logits)
            delta[index] = 1e-6
            plus, minus = (logits.detach() + delta).log_softmax(-1), (logits.detach() - delta).log_softmax(-1)
            finite_difference.append(((plus.exp() * (plus - logq)).sum()
                                      - (minus.exp() * (minus - logq)).sum()) / 2e-6)
        torch.testing.assert_close(reverse_g, torch.stack(finite_difference), rtol=1e-7, atol=1e-9)
        modes = {}
        for mode in ("k1", "kl", "k3", "low_var_kl"):
            k = penalty(logp, logq, mode)
            unclamped = logp - logq if mode in ("k1", "kl") else (logq - logp).exp() - 1 + logp - logq
            assert (logq - logp).abs().max() < 20 and unclamped.abs().max() < 10
            torch.testing.assert_close(k, unclamped, rtol=1e-12, atol=1e-14)
            # This is a local ratio surrogate, not the original PPO implementation.
            ratio = (logp - logp.detach()).exp()
            assert torch.equal(ratio, torch.ones_like(ratio))
            expected = (logp.exp() * k).sum()
            score_loss = (p.detach() * ratio * k.detach()).sum()
            direct_loss = (p.detach() * k).sum()
            total_g, score_g, direct_g = grad(expected), grad(score_loss), grad(direct_loss)
            torch.testing.assert_close(expected, reverse, rtol=1e-11, atol=1e-14)
            torch.testing.assert_close(total_g, reverse_g, rtol=1e-10, atol=1e-12)
            torch.testing.assert_close(score_g + direct_g, reverse_g, rtol=1e-10, atol=1e-12)
            expected_direct = torch.zeros_like(logits) if mode in ("k1", "kl") else forward_g
            torch.testing.assert_close(direct_g, expected_direct, rtol=1e-10, atol=1e-12)
            torch.testing.assert_close(score_g, reverse_g - expected_direct, rtol=1e-10, atol=1e-12)
            updated_logp = (logits.detach() - 0.01 * score_g).log_softmax(-1)
            after = (updated_logp.exp() * (updated_logp - logq)).sum()
            modes[mode] = {"estimate": expected.item(), "score_gradient": score_g.tolist(),
                           "direct_fixed_sampling_gradient": direct_g.tolist(),
                           "live_expectation_gradient": total_g.tolist(),
                           "reverse_kl_after_score_sgd_001": after.item()}
            if label == "opposing":
                assert after < reverse if mode in ("k1", "kl") else after > reverse
        assert modes["k1"] == modes["kl"] and modes["k3"] == modes["low_var_kl"]
        output.append({"case": label, "student": student, "teacher": teacher,
                       "reverse_kl": reverse.item(), "reverse_gradient": reverse_g.tolist(),
                       "forward_gradient": forward_g.tolist(),
                       "finite_difference": torch.stack(finite_difference).tolist(),
                       "aliases_equal": True, "modes": {key: modes[key] for key in ("k1", "k3")}})
    print(json.dumps({"python": platform.python_version(), "torch": torch.__version__,
                      "torch_git": torch.version.git_version, "base": BASE, "head": HEAD,
                      "sources": {f"{rev}:{path}": {"sha256": hashlib.sha256(source).hexdigest(),
                                     "git_blob": hashlib.sha1(b"blob " + str(len(source)).encode()
                                                             + b"\0" + source).hexdigest()}
                                  for (rev, path), source in sources.items()},
                      "loss_core_yaml_equal_base_head": True, "python_inputs_compile": 6,
                      "python_pg_defaults_ast": defaults, "yaml_pg_default_false_text": True,
                      "case_mode_pairs": 16, "cases": output,
                      "scope": "Four fixed-state finite-action examples, CPU FP64; original two KL functions with inactive clamps, local ratio-surrogate and expected-gradient decomposition. Not original distillation_loss/PPO API, original 25/64 tests, configuration construction/OmegaConf conversion, model, teacher, sampled training, off-policy/clipped/mixed-reward, GPU/NPU or quality/performance verification. Sampling weights detached except for explicitly live distributional expectation. Config default inspected by AST; YAML by text only."}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main(sys.argv[1])
