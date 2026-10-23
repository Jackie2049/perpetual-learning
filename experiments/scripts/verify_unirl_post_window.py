"""CPU-only numerical controls for UniRL #545; no model, engine or reference harness.

Load complete original pure modules from fixed Git objects. Empty package shells
only avoid UniRL's unrelated root initialization; require.py is original too.
The prefix strategy in the solver control is real DPM2, NOT SD3/Flow SDE.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import types

import torch

HEAD = "7c56ed05d46f50bf01a36ba6e36dbbb4114b48f8"
BASE = "c6d4b11b24743fe1e933b61192f9d49558f7171e"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", help="Local UniRL Git object repository")
    args = parser.parse_args()
    env = dict(os.environ, GIT_NO_LAZY_FETCH="1")

    def source(rev, path):
        return subprocess.check_output(
            ["git", "-C", args.repo, "show", f"{rev}:{path}"], env=env
        ).decode()

    hashes = {}

    def module(name, rev, path):
        code = source(rev, path)
        hashes[f"{rev[:8]}:{path}"] = hashlib.sha256(code.encode()).hexdigest()
        result = types.ModuleType(name)
        sys.modules[name] = result
        exec(compile(code, f"{rev}:{path}", "exec"), result.__dict__)
        return result

    for name in ("unirl", "unirl.config"):
        shell = types.ModuleType(name)
        shell.__path__ = []
        sys.modules[name] = shell
    module("unirl.config.require", HEAD, "unirl/config/require.py")
    sampling = module("pl_sampling", HEAD, "unirl/types/sampling.py")
    runtime = module("pl_runtime", HEAD, "unirl/sde/runtime.py")
    old_runtime = module("pl_old_runtime", BASE, "unirl/sde/runtime.py")
    kernels = module("pl_kernels", HEAD, "unirl/sde/kernels.py")

    # Compile every changed Python module at both revisions, without importing
    # model/engine dependencies or discovering an accelerator.
    changed = subprocess.check_output(
        ["git", "-C", args.repo, "diff", "--name-only", f"{BASE}...{HEAD}"], env=env
    ).decode().splitlines()
    compiled = 0
    for path in changed:
        if path.endswith(".py"):
            for rev in (BASE, HEAD):
                compile(source(rev, path), f"{rev}:{path}", "exec")
                compiled += 1

    # Existing DPM math was not rewritten by this candidate.
    names = {
        "_DPMState", "_convert_model_output", "_sigma_to_alpha_sigma_t",
        "_dpm_solver_first_order_update", "_multistep_dpm_solver_second_order_update",
        "_dpm_step", "DPM2Strategy",
    }
    def definitions(rev):
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(source(rev, "unirl/sde/kernels.py")).body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
        }
    assert definitions(BASE) == definitions(HEAD)
    assert len(definitions(HEAD)) == len(names)

    schedule_cases = 0
    max_ratio_one_drift = 0.0
    for total in (2, 3, 10, 25, 50):
        for end in range(total):
            for ratio in (0.02, 0.2, 0.4, 1.0):
                post = sampling.PostWindowODE(compress_ratio=ratio)
                steps = post.num_steps(total, end)
                expected = total if end == total - 1 else end + max(int((total - end) * ratio), 2)
                assert steps == expected and end + 1 <= steps <= total
                for shift in (1.0, 3.0):
                    policy = runtime.FlowMatchSchedulePolicy(shift=shift)
                    kw = dict(num_inference_steps=total, height=256, width=256)
                    full = old_runtime.FlowMatchSchedulePolicy(shift=shift).compute_sigma(**kw)
                    assert torch.equal(policy.compute_sigma(**kw), full)
                    compressed = policy.compute_post_window_sigma(
                        **kw, window_end=end, num_steps=steps, device=torch.device("cpu")
                    )
                    assert len(compressed) == steps + 1
                    assert torch.equal(compressed[:end + 2], full[:end + 2])
                    assert compressed[-1].item() == 0
                    assert torch.isfinite(compressed).all() and (compressed[:-1] > compressed[1:]).all()
                    if ratio == 1:
                        drift = (compressed - full).abs().max().item()
                        max_ratio_one_drift = max(max_ratio_one_drift, drift)
                        torch.testing.assert_close(compressed, full, rtol=1e-6, atol=1e-7)
                    schedule_cases += 1

    # Synthetic CPU vector field. Actual original DPM2 handles the prefix;
    # the candidate wrapper handles the tail. This is NOT FlowSDEStrategy,
    # SD3 replay, an original upstream test, or MixGRPO reference parity.
    solver_cases = 0
    max_euler_error = 0.0
    for total in (10, 25):
        for end in (0, 3, total - 2, total - 1):
            for ratio in (0.2, 1.0):
                for order in (1, 2):
                    for shift in (1.0, 3.0):
                        steps = sampling.PostWindowODE(compress_ratio=ratio).num_steps(total, end)
                        sigmas = runtime.FlowMatchSchedulePolicy(shift=shift).compute_post_window_sigma(
                            num_inference_steps=total, height=256, width=256,
                            window_end=end, num_steps=steps,
                        )
                        strategy = kernels.PostWindowDPMStrategy(
                            kernels.DPM2Strategy(), window_end=end, order=order
                        )
                        prefix_reference = kernels.DPM2Strategy()
                        prefix_reference.init_schedule(sigmas)
                        initial = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3) / 12

                        def run():
                            nonlocal max_euler_error
                            strategy.init_schedule(sigmas)
                            assert strategy._state.lower_order_nums == 0
                            assert all(value is None for value in strategy._state.model_outputs)
                            sample = initial.clone()
                            for index in range(steps):
                                pred = sample * 0.2 + (index + 1) * 0.01
                                result, logp, _ = strategy.denoise(
                                    pred, sample, sigmas[index], sigmas[index + 1], step_index=index
                                )
                                assert logp is None and torch.isfinite(result).all()
                                if index <= end:
                                    expected, _, _ = prefix_reference.denoise(
                                        pred, sample, sigmas[index], sigmas[index + 1], step_index=index
                                    )
                                    assert torch.equal(result, expected)
                                    x0 = sample - sigmas[index] * pred
                                    assert torch.equal(strategy._state.model_outputs[-1], x0)
                                    assert strategy._state.lower_order_nums == min(index + 1, order)
                                elif order == 1:
                                    euler = sample + (sigmas[index + 1] - sigmas[index]) * pred
                                    max_euler_error = max(max_euler_error, (result - euler).abs().max().item())
                                    torch.testing.assert_close(result, euler, rtol=1e-5, atol=1e-6)
                                sample = result
                            return sample

                        first = run()
                        prefix_reference.init_schedule(sigmas)
                        assert torch.equal(first, run()), "init_schedule must reset both strategies"
                        solver_cases += 1

    examples = {}
    for total, end, ratio in ((25, 3, .4), (25, 3, .2), (10, 3, .4), (10, 6, .4), (10, 9, .4)):
        examples[f"T{total}/w{end}/r{ratio}"] = sampling.PostWindowODE(compress_ratio=ratio).num_steps(total, end)
    print(json.dumps(dict(
        torch=torch.__version__, device="cpu", compiled_modules=compiled,
        schedule_cases=schedule_cases, solver_cases=solver_cases,
        max_ratio_one_sigma_drift=max_ratio_one_drift,
        max_order_one_local_euler_error=max_euler_error,
        examples=examples, sha256=hashes,
    ), indent=2))


if __name__ == "__main__":
    main()
