"""Original eight GPG alpha regressions on base/head, plus three own boundaries.

Real CPU Torch estimator only; no trainer, policy loss, model or distributed run.
"""

import argparse
import ast
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

import numpy as np
import pytest
import torch

BASE = "2ab35292039a51175a4b3d3b3b2d76e951a0d3e4"
HEAD = "7f38292d6203706787aa377cf3dce4d067978255"
CORE = "verl/trainer/ppo/core_algos.py"
TEST = "tests/trainer/ppo/test_core_algos_on_cpu.py"
FUNCTION = "compute_gpg_outcome_advantage"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    raw, hashes = {}, {}
    for ref, path in ((BASE, CORE), (HEAD, CORE), (HEAD, TEST)):
        source = subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=args.repo,
                                         env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        compile(ast.parse(source), f"{ref}:{path}", "exec")
        raw[ref, path] = source
        hashes[f"{ref}:{path}"] = hashlib.sha256(source).hexdigest()
    functions, results = {}, {}
    for ref in (BASE, HEAD):
        namespace = dict(torch=torch, np=np, pytest=pytest, defaultdict=defaultdict)
        function = next(n for n in ast.parse(raw[ref, CORE]).body
                        if isinstance(n, ast.FunctionDef) and n.name == FUNCTION)
        function.decorator_list = []  # registry decorator only; original body unchanged
        exec(compile(ast.Module(body=[function], type_ignores=[]), CORE, "exec"), namespace)
        functions[ref] = namespace[FUNCTION]
        test = next(n for n in ast.parse(raw[HEAD, TEST]).body
                    if isinstance(n, ast.FunctionDef) and n.name == "test_gpg_advantage_alpha")
        exec(compile(ast.Module(body=[test], type_ignores=[]), TEST, "exec"), namespace)
        original = namespace[test.name]
        mark, = original.pytestmark
        names = mark.args[0].split(",")
        reports = []
        for parameter in mark.args[1]:
            try:
                original(**dict(zip(names, parameter.values, strict=True)))
            except AssertionError as exc:
                reports.append(dict(case=parameter.id, outcome="assertion_failed", detail=str(exc)))
            else:
                reports.append(dict(case=parameter.id, outcome="passed"))
        results[ref] = reports
    assert sum(r["outcome"] == "passed" for r in results[BASE]) == 2
    assert all(r["outcome"] == "passed" for r in results[HEAD])

    def call(ref, rewards, groups, mask=None, dtype=torch.float32):
        reward = torch.zeros(len(rewards), 4, dtype=dtype)
        reward[:, -1] = torch.tensor(rewards, dtype=dtype)
        mask = torch.ones_like(reward) if mask is None else mask
        before, mask_before = reward.clone(), mask.clone()
        advantage, returns = functions[ref](reward, mask, np.array(groups))
        assert torch.equal(reward, before) and torch.equal(mask, mask_before)
        assert torch.equal(advantage, returns)
        return advantage

    own = []
    uniform = {}
    for ref in (BASE, HEAD):
        observations = []
        for count in (3, 8, 17):
            for dtype in (torch.float32, torch.float64):
                values = call(ref, [0.1] * count, [0] * count, dtype=dtype)
                observations.append(dict(rows=count, dtype=str(dtype), first=float(values[0, 0]),
                                         nonzero_elements=int(torch.count_nonzero(values))))
        uniform[ref] = observations
    assert uniform[BASE] == uniform[HEAD]
    own.append(dict(case="uniform_fractional_centering", observations=uniform,
                    interpretation="Group-validity correction does not change the centering subtraction. "
                    "Exact-zero behavior is reported by row count/dtype, not assumed from valid-group count."))
    nonuniform = {ref: call(ref, [0, 1, 2, 0, 0, 0], [0, 0, 0, 1, 1, 1])[:, 0].tolist()
                  for ref in (BASE, HEAD)}
    assert nonuniform[BASE] == [-3, 0, 3, 0, 0, 0]
    assert nonuniform[HEAD] == [-2, 0, 2, 0, 0, 0]
    own.append(dict(case="zero_centered_member_in_nonuniform_group", first_token=nonuniform,
                    interpretation="The middle member counts with its whole nonuniform group, even though "
                    "that member has zero centered reward; num_valid is not count_nonzero(advantages)."))
    masks = {}
    for ref in (BASE, HEAD):
        short = call(ref, [1, 0, 0, 0], [0] * 4)
        mask = torch.cat([torch.ones(4, 4), torch.zeros(4, 4)])
        with_masked_group = call(ref, [1, 0, 0, 0, 0, 0, 0, 0], [0] * 4 + [1] * 4, mask)
        assert torch.equal(with_masked_group[:4], 2 * short)
        assert torch.count_nonzero(with_masked_group[4:]) == 0
        masks[ref] = dict(without_extra_rows=short[:, 0].tolist(),
                          with_extra_masked_rows=with_masked_group[:, 0].tolist())
    own.append(dict(case="response_mask_does_not_remove_rows_from_scale", observations=masks,
                    interpretation="Both versions count batch rows and rewards before final response-mask "
                    "multiplication. This is an estimator boundary, not proof that a trainer supplies such rows."))
    print(json.dumps(dict(base=BASE, head=HEAD, python=platform.python_version(), torch=torch.__version__,
                         torch_git=torch.version.git_version, numpy=np.__version__, pytest=pytest.__version__,
                         source_sha256=hashes, original_test_cases=results, own_controls=own,
                         scope="Complete original estimator bodies; registry decorators removed only. One "
                         "original test function and parametrize retained, eight cases manually invoked per "
                         "version (not native pytest); base 2 pass/6 assertion failures, head 8 pass. Three "
                         "own boundary groups use real CPU tensors, no data/kernel mocks. Three full source "
                         "inputs compile. No full module/registry, actor policy-loss/backward/optimizer, "
                         "trainer batching/filtering, rejection/resampling, DP aggregation, GPU/NPU, model, "
                         "training or performance executed."), indent=2))


if __name__ == "__main__":
    main()
