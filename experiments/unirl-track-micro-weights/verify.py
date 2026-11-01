#!/usr/bin/env python3
"""Fixed #301 original-method slice with own CPU carriers, not native UniRL.

../deepinfer-eplb-venv/bin/python verify.py ../unirl-readonly > result.json
Requires cached Git objects; no lazy fetch, installs, models or device probes.
"""
import ast
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import os
import platform
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

HEAD = "6cd3c09a90e34643d167ae428b6d6aa5751186b3"
BASE = "683a59b3cb1f04bea3d93cc79a79cd2552769902"
API_BASE = "da7506200829f0ac9fde2e6c7a71d34fc72732ba"
MAIN = "b8e9740e981a68a5556fbeb8225558040a4d47a4"
PATHS = ["unirl/train/unified_model_stack.py", "unirl/utils/misc.py", "unirl/train/stack/planner/types.py", "unirl/train/stack/base.py", "unirl/algorithms/base.py"]
METHODS = {"__init__", "_optimizer_step_slices", "prepare_segment", "_backward_part"}


def read(repo, rev, path):
    raw = subprocess.check_output(["git", "-C", repo, "show", f"{rev}:{path}"], env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
    compile(raw, path, "exec")
    return raw.decode(), hashlib.sha256(raw).hexdigest()


def selected(source, names):
    nodes = [n for n in ast.parse(source).body if getattr(n, "name", None) in names]
    assert {n.name for n in nodes} == set(names)
    return nodes


def load(repo, rev):
    src, hashes = {}, {}
    for path in PATHS:
        src[path], hashes[path] = read(repo, rev, path)
    module = ModuleType("fixed_" + rev[:8])
    sys.modules[module.__name__] = module
    module.__dict__.update(dict(torch=torch, dataclass=dataclass, Any=Any, Dict=Dict, List=List, Mapping=Mapping, Optional=Optional, Tuple=Tuple))
    # Remote initialization, dispatch decorators, imports and all other methods are excluded.
    exec("class Remote: pass", module.__dict__)
    nodes = []
    nodes += selected(src[PATHS[2]], {"_positive_int", "_update_ranges", "_build_micro_batch_slices"})
    nodes += selected(src[PATHS[4]], {"AlgorithmStepResult"})
    nodes += selected(src[PATHS[3]], {"TrainStepResult"})
    nodes += selected(src[PATHS[1]], {"aggregate_numeric_metrics"})
    cls = selected(src[PATHS[0]], {"UnifiedModelTrainStack"})[0]
    methods = [n for n in cls.body if getattr(n, "name", None) in METHODS]
    assert {n.name for n in methods} == METHODS
    cls.body = methods
    cls.decorator_list = []
    nodes.append(cls)
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), rev, "exec"), module.__dict__)
    return module, hashes


class Carrier:
    """Own row carrier, not Sample/Part/packed Segment or TensorRef."""
    def __init__(self, ids, lengths=None, values=None):
        self.conditions = torch.as_tensor(ids, dtype=torch.long, device="cpu")
        self.batch_size = len(ids)
        self.segment = SimpleNamespace(lengths=lengths, values=values, anchor=None)
        self.advantages = torch.ones(self.batch_size, dtype=torch.float64, device="cpu")

    def slice(self, start, end):
        lengths = None if self.segment.lengths is None else self.segment.lengths[start:end]
        values = None if self.segment.values is None else self.segment.values[start:end]
        return Carrier(self.conditions[start:end], lengths, values)


def make_stack(module, **kwargs):
    algo = SimpleNamespace(supports_multi_update=True)
    return module.UnifiedModelTrainStack(fsdp_backend=None, ar_algorithm=algo, image_algorithm=algo, micro_batch_size=2, max_grad_norm=1, **kwargs)


def geometry(module, candidate):
    stack = make_stack(module)
    fallback = stack._optimizer_step_slices(5)
    assert fallback == [[(0, 2), (2, 4), (4, 5)]]
    if not candidate:
        return {"shared_ragged": fallback}
    assert stack.micro_batch_sizes == {"ar": 2, "image": 2}
    stack = make_stack(module, ar_micro_batch_size=1, image_micro_batch_size=3)
    ar = stack._optimizer_step_slices(5, micro_batch_size=stack.micro_batch_sizes["ar"])
    image = stack._optimizer_step_slices(10, micro_batch_size=stack.micro_batch_sizes["image"])
    assert ar == [[(i, i + 1) for i in range(5)]]
    assert image == [[(0, 3), (3, 6), (6, 9), (9, 10)]]
    assert fallback == stack._optimizer_step_slices(5, micro_batch_size=2)
    updates = make_stack(module, num_updates_per_batch=2)._optimizer_step_slices(6, micro_batch_size=2)
    assert updates == [[(0, 2), (2, 3)], [(3, 5), (5, 6)]]
    failures = []
    for name, call in [
        ("zero_ar", lambda: make_stack(module, ar_micro_batch_size=0)),
        ("negative_image", lambda: make_stack(module, image_micro_batch_size=-1)),
        ("zero_override", lambda: stack._optimizer_step_slices(5, micro_batch_size=0)),
        ("indivisible_update", lambda: make_stack(module, num_updates_per_batch=2)._optimizer_step_slices(5)),
    ]:
        try:
            call()
        except ValueError:
            failures.append(name)
        else:
            raise AssertionError(name)
    part = Carrier(range(5))
    calls = []

    class AnchorRecorder:
        anchor_fields = ("anchor",)
        def recomputes_anchor(self):
            return True
        def prepare_segment(self, *, conditions, segment):
            calls.append(conditions.tolist())
            segment.anchor = torch.stack([conditions, torch.full_like(conditions, len(conditions))], dim=1)

    stack.prepare_segment(AnchorRecorder(), part, micro_batch_size=2)
    assert calls == [[0, 1], [2, 3], [4]]
    assert part.segment.anchor.tolist() == [[0, 2], [1, 2], [2, 2], [3, 2], [4, 1]]
    class MissingAnchor(AnchorRecorder):
        def prepare_segment(self, **kwargs):
            pass
    try:
        stack.prepare_segment(MissingAnchor(), Carrier(range(5)), micro_batch_size=2)
    except RuntimeError as error:
        assert "produced None" in str(error)
        failures.append("missing_anchor")
    else:
        raise AssertionError("missing anchor accepted")
    return {"shared_ragged": fallback, "ar": ar, "image": image, "two_updates": updates, "anchor_calls": calls, "packed_anchor": part.segment.anchor.tolist(), "rejections": failures}


def metric_controls(module, candidate):
    reduce = module.aggregate_numeric_metrics
    rows = [{"x": 1.5}, {"x": 3.5}, {"x": 5.0}]
    assert abs(reduce(rows)["x"] - 10 / 3) < 1e-12
    if not candidate:
        return {"unweighted_ragged": reduce(rows)}
    weighted = reduce(rows, weights=[2, 2, 1])
    assert weighted == {"x": 3.0}
    assert reduce(rows, weights=[1, 1, 1]) == reduce(rows)
    assert reduce([{"x": 1}, {}, {"x": 5}], weights=[2, 100, 1]) == {"x": 7 / 3}
    assert reduce([{"x": "skip"}, {"x": 3}], weights=[100, 1]) == {"x": 3}
    assert reduce([{"x": 2}, {"x": 6}], weights=[0, 0]) == {"x": 4}
    coerced = reduce([{"b": True, "v": torch.tensor([1., 3.]), "s": torch.tensor(2.)}, {"b": False, "v": torch.tensor([3., 5.]), "s": torch.tensor(6.)}], weights=[3, 1])
    assert coerced == {"b": .75, "v": 2.5, "s": 3.0}
    assert reduce([], weights=[]) == {}
    try:
        reduce(rows, weights=[1])
    except ValueError:
        pass
    else:
        raise AssertionError("weight count mismatch accepted")
    return {"unweighted_ragged": reduce(rows), "weighted_ragged": weighted, "tensor_bool_coercion": coerced, "checks": 9}


def backward(module, mode, width):
    lengths = torch.tensor([1, 3, 2, 4, 1], dtype=torch.long, device="cpu")
    values = torch.arange(1, 6, dtype=torch.float64, device="cpu")
    theta = torch.ones(5, dtype=torch.float64, device="cpu", requires_grad=True)
    part = Carrier(range(5), lengths, values)
    stack = make_stack(module)
    calls = []
    class OwnMeanAlgorithm:
        def compute_loss_and_backward(self, *, conditions, segment, advantages, training_progress, loss_scale):
            row_losses = theta[conditions] * segment.values
            local = (row_losses * segment.lengths).sum() / segment.lengths.sum() if mode == "token" else row_losses.mean()
            (local * loss_scale).backward()
            calls.append({"ids": conditions.tolist(), "scale": loss_scale, "tokens": int(segment.lengths.sum())})
            return module.AlgorithmStepResult(loss=float(local.detach()), metrics={"local_mean": float(local.detach())}, num_steps_or_tokens=int(segment.lengths.sum()), has_backward=True)
    slices = [(s, min(5, s + width)) for s in range(0, 5, width)]
    result, did_backward = stack._backward_part(OwnMeanAlgorithm(), part, slices, training_progress=.5)
    assert did_backward
    reference_theta = torch.ones(5, dtype=torch.float64, device="cpu", requires_grad=True)
    row_losses = reference_theta * values
    reference = (row_losses * lengths).sum() / lengths.sum() if mode == "token" else row_losses.mean()
    reference.backward()
    error = float((theta.grad - reference_theta.grad).abs().max())
    if mode == "row" or width == 5:
        assert error < 1e-12
    else:
        assert error > .1
    return {"mode": mode, "width": width, "loss": result.loss, "metric": dict(result.metrics), "gradient": theta.grad.tolist(), "reference_loss": float(reference.detach()), "reference_gradient": reference_theta.grad.tolist(), "gradient_max_error": error, "calls": calls}


def main():
    repo = sys.argv[1]
    revisions = OrderedDict()
    for label, rev in (("ancestor", BASE), ("candidate", HEAD)):
        module, hashes = load(repo, rev)
        revisions[label] = {"revision": rev, "source_sha256": hashes, "geometry": geometry(module, label == "candidate"), "metrics": metric_controls(module, label == "candidate"), "backward": [backward(module, mode, width) for mode in ("row", "token") for width in (1, 2, 5)]}
    for old, new in zip(revisions["ancestor"]["backward"], revisions["candidate"]["backward"]):
        assert old["gradient"] == new["gradient"] and old["loss"] == new["loss"]
    main_source, main_hash = read(repo, MAIN, "unirl/train/stack/unified.py")
    assert "resolve_loss_scales" in main_source and "prepare_segment_anchors" in main_source
    print(json.dumps({"head": HEAD, "actual_merge_base": BASE, "api_target_base": API_BASE, "current_main": MAIN, "main_unified_sha256_compile_only": main_hash, "environment": {"python": platform.python_version(), "torch": torch.__version__, "torch_git": torch.version.git_version, "device": "cpu"}, "revisions": revisions, "scope": {"original": "4 complete methods, 4 complete helpers, original 2 result dataclasses; main compile only", "own": "Remote empty base; row Carrier; anchor recorder; linear row/token-mean algorithm with real CPU autograd", "not_run": ["author uncommitted 24-assertion harness", "native Part/packed Segment/TensorRef", "real GRPO/FlowGRPO/LoRA/model forward", "train_track dispatch or backend/optimizer/DP/collective", "ratio parity, hardware memory/time, training quality"]}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
