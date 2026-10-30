"""Stdlib-only controls for UniRL's merged seed helper; no adapter or engine execution."""

import ast
import copy
import dataclasses
import hashlib
import json
from numbers import Integral
import os
import platform
import subprocess
import sys
import types

REVISION = "5d1da78d4d87593b41a4235a65e9ad9dc3f9b6f8"
MERGE = "ffad0b832707f5f1a87609e96f351426ede3f043"
PREFIX = "unirl/rollout/engine/"
CHANGED = [PREFIX + path for path in ["README.md", "vllm_omni/adapters/hi3.py",
           "vllm_omni/adapters/qwen3_omni.py", "vllm_omni/utils/__init__.py",
           "vllm_omni/utils/sampling.py"]] + ["unirl/trainer/unified_model.py"]
BASE = PREFIX + "vllm_omni/backends/base.py"
DERIVE = PREFIX + "sglang/utils/sampling.py"
HELPER = PREFIX + "vllm_omni/utils/sampling.py"
NATIVE = PREFIX + "vllm_omni/backends/native.py"


def main(repo):
    def read(path, revision=REVISION):
        return subprocess.check_output(["git", "-C", repo, "show", f"{revision}:{path}"],
                                       env={**os.environ, "GIT_NO_LAZY_FETCH": "1"})

    sources = {path: read(path) for path in [*CHANGED, BASE, DERIVE, NATIVE]}
    for path in CHANGED:
        assert sources[path] == read(path, MERGE)
    for path, source in sources.items():
        if path.endswith(".py"):
            compile(source, path, "exec")

    # Full original base module, including the real frozen dataclasses.
    base = types.ModuleType("_pl_unirl_seed_base")
    sys.modules[base.__name__] = base
    exec(compile(sources[BASE], BASE, "exec"), base.__dict__)

    # Original function bodies isolated only to bypass Sample/engine registry imports.
    tree = ast.parse(sources[DERIVE])
    names = {"_validated_base_seed", "derive_sampling_seed"}
    selected = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in names)
                or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
                    and t.id == "_MAX_SGLANG_SAMPLING_SEED" for t in n.targets))]
    assert len(selected) == 3 and sum(isinstance(n, ast.FunctionDef) for n in selected) == 2
    derive_ns = {"hashlib": hashlib, "Integral": Integral}
    exec(compile(ast.Module(body=selected, type_ignores=[]), DERIVE, "exec"), derive_ns)
    tree = ast.parse(sources[HELPER])
    selected = [n for n in tree.body if isinstance(n, (ast.ImportFrom, ast.FunctionDef))
                and (isinstance(n, ast.FunctionDef) or n.module == "__future__")]
    assert sum(isinstance(n, ast.FunctionDef) for n in selected) == 1
    helper_ns = {"replace": dataclasses.replace, "STAGE_KIND_AR": base.STAGE_KIND_AR,
                 "derive_sampling_seed": derive_ns["derive_sampling_seed"]}
    exec(compile(ast.Module(body=selected, type_ignores=[]), HELPER, "exec"), helper_ns)
    seed_calls = helper_ns["seed_ar_calls"]
    derive = derive_ns["derive_sampling_seed"]
    ar = base.StageSampling("ar", {"temperature": 0.7, "stop_token_ids": [2]})
    diffusion = base.StageSampling("diffusion", {"seed": 999, "extra_args": {
        "init_noise_group_ids": ["fixed"], "init_noise_seed": 999}})
    prompts = [{"text": value} for value in ("alpha", "beta", "gamma")]
    calls = [base.GenerateCall(prompts[:2], [ar, diffusion]), base.GenerateCall(prompts[2:], [ar, diffusion])]
    ids = ["a/0", "b/0", "c/0"]
    original = copy.deepcopy([dataclasses.asdict(call) for call in calls])

    # No-seed bypass preserves identity; it does not validate sample-id counts.
    assert seed_calls(calls, None, []) is calls
    output = seed_calls(calls, 17, ids)
    assert len(output) == 3 and all(len(call.prompts) == 1 for call in output)
    mapping = {sid: call.sampling[0].kwargs["seed"] for sid, call in zip(ids, output, strict=True)}
    assert mapping == {sid: derive(17, sid) for sid in ids}
    for index, call in enumerate(output):
        assert call.prompts[0] is prompts[index]
        assert call.sampling[0] is not ar and call.sampling[0].kwargs is not ar.kwargs
        assert call.sampling[0].kwargs["stop_token_ids"] is ar.kwargs["stop_token_ids"]
        assert call.sampling[1] is diffusion
    assert [dataclasses.asdict(call) for call in calls] == original

    def map_rows(order):
        ordered_ids = [ids[i] for i in order]
        call = base.GenerateCall([prompts[i] for i in order], [ar, diffusion])
        result = seed_calls([call], 17, ordered_ids)
        return {sid: row.sampling[0].kwargs["seed"] for sid, row in zip(ordered_ids, result, strict=True)}

    assert map_rows([2, 0, 1]) == mapping
    assert {**map_rows([0]), **map_rows([1, 2])} == mapping
    errors = {}
    for label, bad_ids in [("too_short", ids[:2]), ("too_long", [*ids, "extra"])]:
        try:
            seed_calls(calls, 17, bad_ids)
        except ValueError as error:
            errors[label] = str(error)
        else:
            raise AssertionError(label)
    boundary_seeds = {str(seed): derive(seed, "a/0") for seed in (0, (1 << 63) - 1)}
    rejected = []
    for seed, sid in [(-1, "a"), (1 << 63, "a"), (True, "a"), (1.5, "a"), (17, ""), (17, " ")]:
        try:
            seed_calls([base.GenerateCall([prompts[0]], [ar])], seed, [sid])
        except ValueError:
            rejected.append({"seed": seed, "sample_id": sid})
        else:
            raise AssertionError((seed, sid))
    duplicate = seed_calls([calls[0]], 17, ["same", "same"])
    assert duplicate[0].sampling[0].kwargs["seed"] == duplicate[1].sampling[0].kwargs["seed"]
    assert [dataclasses.asdict(call) for call in calls] == original
    assert "torch" not in sys.modules
    print(json.dumps({"python": platform.python_version(), "revision": REVISION, "merge": MERGE,
                      "sources": {path: {"sha256": hashlib.sha256(source).hexdigest(),
                                         "git_blob": hashlib.sha1(b"blob " + str(len(source)).encode()
                                                                 + b"\0" + source).hexdigest()}
                                  for path, source in sources.items()},
                      "changed_files_equal_merge": len(CHANGED), "python_files_compile": 8,
                      "unseeded_same_list_identity_with_unchecked_ids": True,
                      "split_prompt_order": [call.prompts[0]["text"] for call in output],
                      "seed_mapping": mapping, "reorder_mapping_equal": True, "shard_mapping_equal": True,
                      "ownership": {"new_ar_stage_and_kwargs": True, "nested_ar_values_shared": True,
                                    "diffusion_stage_shared_unchanged": True, "inputs_unchanged": True},
                      "count_errors": errors, "boundary_seeds": boundary_seeds,
                      "invalid_inputs_rejected": rejected,
                      "duplicate_ids_same_seed": True, "torch_imported": False,
                      "scope": "Seven local stdlib helper-control groups, not author 78/27 tests. Full original wire base, AST-isolated original seed helper and derivation functions; caller inputs are local dictionaries. Adapters, Sample lineage, trainer and native generate only source/compile. No old-fail/new-pass adapter comparison, native parameter construction, engine calls, RNG sampling, model outputs, GPU/NPU or throughput validation."}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
