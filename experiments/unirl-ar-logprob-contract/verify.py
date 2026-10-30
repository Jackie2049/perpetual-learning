"""Local CPU wire controls for merged UniRL #631, not the author's harness.

Original build_ar_segment functions are AST-isolated. Five complete original
container modules execute behind import-only package shells; no fake pack,
concat, slice or select. Git objects must already be cached; no lazy fetch.
"""

import ast
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import types
from typing import Any, List, Optional, Sequence

import torch

BASE = "ffad0b832707f5f1a87609e96f351426ede3f043"
MERGE = "5d1da78d4d87593b41a4235a65e9ad9dc3f9b6f8"
HEAD = "e6651631cad409bbea4408da338b699d8b3783a3"
TRACKS = "unirl/rollout/engine/vllm_omni/utils/tracks.py"
MODULES = ["unirl.distributed.tensor.batch", "unirl.types.conditions.base",
           "unirl.types.conditions.text", "unirl.types.segments.base", "unirl.types.segments.text"]


def main(repo):
    torch.set_num_threads(1)
    sources = {}

    def read(revision, path):
        source = subprocess.check_output(["git", "-C", repo, "show", f"{revision}:{path}"],
                                         env={**os.environ, "GIT_NO_LAZY_FETCH": "1"})
        sources[f"{revision}:{path}"] = source
        compile(source, path, "exec")
        return source

    # Empty package shells bypass initialization/transport aggregation only.
    for module_name in MODULES:
        parts = module_name.split(".")
        for index in range(1, len(parts)):
            name = ".".join(parts[:index])
            if name not in sys.modules:
                package = types.ModuleType(name)
                package.__path__ = []
                sys.modules[name] = package
                if index > 1:
                    setattr(sys.modules[".".join(parts[:index - 1])], parts[index - 1], package)
        path = module_name.replace(".", "/") + ".py"
        source = read(MERGE, path)
        assert source == read(BASE, path), path
        module = types.ModuleType(module_name)
        module.__package__ = module_name.rpartition(".")[0]
        sys.modules[module_name] = module
        setattr(sys.modules[module.__package__], module_name.rpartition(".")[2], module)
        exec(compile(source, path, "exec"), module.__dict__)
    TextSegment = sys.modules[MODULES[-1]].TextSegment

    functions = {}
    for label, revision in [("base", BASE), ("merged", MERGE)]:
        source = read(revision, TRACKS)
        tree = ast.parse(source)
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "build_ar_segment"]
        assert len(selected) == 1
        namespace = dict(torch=torch, Any=Any, List=List, Optional=Optional, Sequence=Sequence)
        exec(compile(ast.Module(body=selected, type_ignores=[]), TRACKS, "exec"), namespace)
        functions[label] = namespace["build_ar_segment"]
    assert sources[f"{MERGE}:{TRACKS}"] == read(HEAD, TRACKS)

    def entry(value):
        return types.SimpleNamespace(logprob=value)

    def row(tokens, logprobs):
        return [types.SimpleNamespace(stage_id=0, outputs=[
            types.SimpleNamespace(token_ids=tokens, logprobs=logprobs)])]

    def values(tensor):
        if tensor is None:
            return None
        return [value if not isinstance(value, float) or math.isfinite(value)
                else str(value) for value in tensor.tolist()]

    def describe(segment):
        if segment is None:
            return None
        assert type(segment) is TextSegment
        assert segment.tokens.dtype == torch.int64 and segment.tokens.device.type == "cpu"
        for field in (segment.log_probs, segment.rollout_log_probs):
            if field is not None:
                assert field.dtype == torch.float32 and field.device.type == "cpu"
        return {"batch_size": segment.batch_size, "tokens": values(segment.tokens),
                "cu_seqlens": values(segment.cu_seqlens), "log_probs": values(segment.log_probs),
                "rollout_log_probs": values(segment.rollout_log_probs)}

    cases = [
        ("valid", [row([11, 12], [{11: entry(-0.5)}, {12: entry(-0.75)}])]),
        ("sampled_not_first", [row([11], [{99: entry(-0.125), 11: entry(-0.75)}])]),
        ("mixed_empty", [[], [types.SimpleNamespace(stage_id=1, outputs=[])],
                          [types.SimpleNamespace(stage_id=0, outputs=[])],
                          row([], None), row([], []), row([11], [{11: entry(-0.5)}])]),
        ("all_empty", [[], row([], None), row([], [])]),
        ("short", [row([11, 12], [{11: entry(-0.5)}])]),
        ("long", [row([11], [{11: entry(-0.5)}, {12: entry(-0.75)}])]),
        ("logprobs_on_empty", [row([], [{11: entry(-0.5)}])]),
        ("missing_none", [row([11], None)]),
        ("missing_empty", [row([11], [])]),
        ("not_sequence", [row([11], 7)]),
        ("step_not_dict", [row([11], [None])]),
        ("step_empty_dict", [row([11], [{}])]),
        ("sampled_omitted", [row([11], [{12: entry(-0.25)}])]),
        ("nan", [row([11], [{11: entry(float("nan"))}])]),
        ("negative_infinity", [row([11], [{11: entry(float("-inf"))}])]),
        ("finite_fp64_overflows_fp32", [row([11], [{11: entry(-1e100)}])]),
        ("selected_none", [row([11], [{11: None}])]),
        ("missing_logprob_attribute", [row([11], [{11: types.SimpleNamespace(value=-0.5)}])]),
        ("unconvertible_logprob", [row([11], [{11: entry("not-a-number")}])]),
        ("positive_finite", [row([11], [{11: entry(0.25)}])]),
    ]
    outcomes = {}
    segments = {}
    for name, inputs in cases:
        outcomes[name] = {}
        for label, function in functions.items():
            try:
                segment = function(inputs)
                segments[label, name] = segment
                outcomes[name][label] = {"result": describe(segment)}
            except (RuntimeError, AttributeError, ValueError) as error:
                outcomes[name][label] = {"error": type(error).__name__, "message": str(error)}

    for name in ("valid", "mixed_empty", "all_empty", "positive_finite"):
        assert outcomes[name]["base"] == outcomes[name]["merged"], name
    assert outcomes["valid"]["merged"]["result"]["log_probs"] == [-0.5, -0.75]
    assert outcomes["mixed_empty"]["merged"]["result"]["cu_seqlens"] == [0, 0, 0, 0, 0, 0, 1]
    assert outcomes["all_empty"]["merged"] == {"result": None}
    assert outcomes["sampled_not_first"]["base"]["result"]["log_probs"] == [-0.125]
    assert outcomes["sampled_not_first"]["merged"]["result"]["log_probs"] == [-0.75]
    rejected = [name for name, _ in cases[4:17]]
    assert len(rejected) == 13
    for name in rejected:
        assert outcomes[name]["merged"]["error"] == "RuntimeError", name
    for name, expected in {"short": [-0.5, 0.0], "long": [-0.5],
                           "step_not_dict": [0.0], "step_empty_dict": [0.0],
                           "sampled_omitted": [-0.25], "nan": ["nan"],
                           "negative_infinity": ["-inf"],
                           "finite_fp64_overflows_fp32": ["-inf"]}.items():
        assert outcomes[name]["base"]["result"]["log_probs"] == expected, name
    for name in ("missing_none", "missing_empty", "not_sequence"):
        assert outcomes[name]["base"]["result"]["log_probs"] is None
    assert outcomes["logprobs_on_empty"]["base"] == {"result": None}
    assert outcomes["selected_none"]["base"]["error"] == "AttributeError"
    for name, kind in [("missing_logprob_attribute", "AttributeError"),
                       ("unconvertible_logprob", "ValueError")]:
        assert outcomes[name]["base"]["error"] == outcomes[name]["merged"]["error"] == kind

    # Original native container operations, not an interception of pack arguments.
    mixed = segments["merged", "mixed_empty"]
    joined = TextSegment.concat([mixed.slice(0, 3), mixed.slice(3, 6)])
    assert describe(joined) == describe(mixed)
    selected = mixed.select([5, 0, 5])
    assert describe(selected) == {"batch_size": 3, "tokens": [11, 11],
                                 "cu_seqlens": [0, 1, 1, 2], "log_probs": [-0.5, -0.5],
                                 "rollout_log_probs": [-0.5, -0.5]}
    old_mixed = functions["base"]([row([11], [{11: entry(-0.5)}]), row([12], None)])
    assert old_mixed.batch_size == 2 and old_mixed.log_probs is None
    try:
        TextSegment.concat([segments["base", "valid"], old_mixed])
    except ValueError as error:
        concat_error = str(error)
        assert "TextSegment.log_probs" in concat_error and "missing" in concat_error
    else:
        raise AssertionError("Expected native mixed-anchor concat rejection")
    assert not any(name in sys.modules for name in ("ray", "hydra", "omegaconf", "vllm", "vllm_omni"))
    print(json.dumps({"python": platform.python_version(), "torch": torch.__version__,
                      "torch_git": torch.version.git_version, "base": BASE, "merge": MERGE, "head": HEAD,
                      "sources": {key: {"sha256": hashlib.sha256(source).hexdigest(),
                                        "git_blob": hashlib.sha1(b"blob " + str(len(source)).encode()
                                                                + b"\0" + source).hexdigest()}
                                  for key, source in sources.items()},
                      "full_container_modules_equal_base": 5, "merged_tracks_equal_head": True,
                      "local_wire_cases": len(cases), "new_runtime_error_cases": rejected,
                      "cases": outcomes, "native_slice_concat_roundtrip": True,
                      "native_repeated_select": describe(selected),
                      "base_missing_row_drops_batch_anchors": describe(old_mixed),
                      "native_base_mixed_anchor_concat_error": concat_error,
                      "scope": "20 local wire cases, not the author's 16-case harness or upstream pytest. Original AST-isolated build_ar_segment with SimpleNamespace responses, real Torch CPU and five full original container modules; import-only package shells bypass initialization. No adapter, actual engine response, Sample/Part, GSPO replay, joint HI3 update, worker transport, GPU/NPU, training or performance validation. Positive finite logprobs are accepted; missing attributes and conversion failures are not normalized to RuntimeError."}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main(sys.argv[1])
