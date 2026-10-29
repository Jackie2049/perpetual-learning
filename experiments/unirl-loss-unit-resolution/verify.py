"""Five original-helper CPU controls; backend collectives and segment types are explicit fixtures."""

import __future__
import ast
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace

import torch

HEAD = "9dded4e2fd92732dd9de40252a4ae5b342b164e0"
BASE = "6fd82cba13f52f394747832344039408699ea378"
ROOT = Path(sys.argv[1])
hashes = {}


def read(rev, path):
    raw = subprocess.check_output(["git", "show", f"{rev}:{path}"], cwd=ROOT,
                                  env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
    hashes[f"{rev}:{path}"] = hashlib.sha256(raw).hexdigest()
    tree = ast.parse(raw, filename=path)
    compile(tree, path, "exec")
    return raw, tree


def execute(nodes, ns):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "original-helper", "exec",
                 flags=__future__.annotations.compiler_flag), ns)


mode_raw, _ = read(HEAD, "unirl/types/loss_agg.py")
ns = {"torch": torch, "dataclass": dataclass, "TextSegment": SimpleNamespace}
exec(compile(mode_raw, "original-loss-agg.py", "exec"), ns)
_, algorithm_tree = read(HEAD, "unirl/algorithms/base.py")
norm = next(n for n in algorithm_tree.body if isinstance(n, ast.ClassDef) and n.name == "LossNorm")
exec(compile(ast.Module(body=[norm], type_ignores=[]), "original-LossNorm", "exec", dont_inherit=True), ns)
algorithm = next(n for n in algorithm_tree.body if isinstance(n, ast.ClassDef) and n.name == "StageAlgorithm")
units = next(n for n in algorithm.body if isinstance(n, ast.FunctionDef) and n.name == "loss_units")
execute([units], ns)
AlgorithmFixture = type("AlgorithmFixture", (), {"loss_units": ns["loss_units"]})
versions = {}
for label, rev in [("base", BASE), ("head", HEAD)]:
    _, tree = read(rev, "unirl/train/stack/loss_scale.py")
    local = dict(ns)
    execute([n for n in tree.body if isinstance(n, ast.FunctionDef)], local)
    versions[label] = local["resolve_loss_scales"]

positive = SimpleNamespace(cu_seqlens=torch.tensor([0, 2, 5, 6], device="cpu"),
                           lengths=torch.tensor([2, 3, 1], device="cpu"),
                           loss_mask=torch.tensor([1., 0., 1., 1., 1., 0.], device="cpu"))
zero = SimpleNamespace(cu_seqlens=torch.tensor([0, 1, 4], device="cpu"),
                       lengths=torch.tensor([1, 3], device="cpu"), loss_mask=torch.zeros(4, device="cpu"))
cases = [
    ("sample_share_sp2", "seq-mean-token-mean", None, [(0, 2), (2, 3)], None, 99., 2,
     {"scales": [2 / 3, 1 / 3], "global_units": None, "events": []}),
    ("token_reordered_unequal", "token-mean", positive, [(0, 1), (1, 3)], [1, 0, 2], 6., 1,
     {"scales": [0.6, 0.2], "global_units": 10., "events": [["all_reduce_loss_sums", [4.]], ["gradient_world", 2]]}),
    ("local_zero_peer_positive", "token-mean", zero, [(0, 1), (1, 2)], None, 6., 1,
     {"scales": [0., 0.], "global_units": 6., "events": [["all_reduce_loss_sums", [0.]], ["gradient_world", 2]]}),
    ("global_zero", "token-mean", zero, [(0, 1), (1, 2)], None, 0., 1,
     {"scales": [0., 0.], "global_units": 0., "events": [["all_reduce_loss_sums", [0.]]]}),
    ("token_sp2_rejected", "token-mean", positive, [(0, 1), (1, 3)], [1, 0, 2], 6., 2,
     {"error": "ValueError", "events": []}),
]
results = []
for name, mode, segment, micros, order, peer_sum, sp, expected in cases:
    actor = AlgorithmFixture()
    actor.loss_agg_mode = mode
    part = SimpleNamespace(segment=segment)
    raw_units = actor.loss_units(part)
    observed = {}
    for version, resolver in versions.items():
        events = []

        def reduce(values):
            events.append(["all_reduce_loss_sums", values])
            return [values[0] + peer_sum]

        def world():
            events.append(["gradient_world", 2])
            return 2

        try:
            scales, total = resolver(actor, part, micros=micros,
                                     order=torch.tensor(order, device="cpu") if order is not None else None,
                                     backend=SimpleNamespace(all_reduce_loss_sums=reduce, gradient_average_world_size=world),
                                     rank_info=SimpleNamespace(sp_size=sp), owner="fixture")
            result = {"scales": scales, "global_units": total, "events": events}
        except ValueError as error:
            assert "sequence parallelism (sp_size=2)" in str(error)
            result = {"error": type(error).__name__, "events": events}
        assert result == expected, (name, version, result)
        observed[version] = result
    assert observed["head"] == observed["base"]
    results.append({"case": name, "mode": mode, "sp_size_fixture": sp, "peer_units_fixture": peer_sum,
                    "head_row_units": None if raw_units is None else raw_units.tolist(), "observed": observed})

assert "unirl" not in sys.modules
print(json.dumps({"head": HEAD, "base": BASE, "python": platform.python_version(), "torch": torch.__version__,
                  "source_sha256": hashes, "controls_passed": len(results), "results": results,
                  "scope": "Original loss-mode module, LossNorm, loss_units method and both resolve_loss_scales bodies. SimpleNamespace substitutes for TextSegment/Part/rank info; deterministic peer-sum and gradient-world backend fixtures. No real segment construction, native suite, process group, collective, FSDP, backward, optimizer, GPU/NPU or training. Five local controls do not establish rank agreement or general bit-identity."}, indent=2))
