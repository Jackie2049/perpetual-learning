"""#313 fixed-branch original request/client/service, own CPU carrier/scorer.

python verify.py /cached/UniRL-git
No native Sample/Part traversal, Ray Handle, TensorRef transport or training.
"""
import __future__
import ast
from dataclasses import dataclass, fields, replace
import hashlib
import json
import logging
import math
import os
import platform
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from abc import ABC
from typing import ClassVar

import torch

HEAD = "fbee2850f8e5127745b84f1d60286111d2911d2a"
BASE = "8fc283e0eaee9de9f5a7de085872f7a046760ce6"
MAIN = "b8e9740e981a68a5556fbeb8225558040a4d47a4"


@dataclass
class PartCarrier:
    primitives: dict
    sampling_params: object
    segment: object
    trajectory: object
    sample_ids: list
    group_ids: list
    primitive_metadata: dict
    rewards: object = None
    component_rewards: object = None

    @property
    def batch_size(self):
        return len(self.sample_ids)


@dataclass
class SampleCarrier:
    parts: list
    conditions: object
    metadata: list

    def conditioning(self):
        return [self.conditions]

    def root_metadata(self, index):
        assert index == -1
        return self.metadata

    def with_parts(self, parts):
        return replace(self, parts=parts)


def main(cache):
    hashes = {}
    env = dict(os.environ, GIT_NO_LAZY_FETCH="1")

    def read(path, ref=HEAD):
        raw = subprocess.check_output(["git", "-C", cache, "show", f"{ref}:{path}"], env=env)
        hashes[f"{ref}:{path}"] = hashlib.sha256(raw).hexdigest()
        compile(raw, path, "exec")
        return ast.parse(raw)

    def named(tree, name):
        return next(node for node in tree.body if getattr(node, "name", None) == name)

    ns = ModuleType("cpu_reward_boundary")
    sys.modules[ns.__name__] = ns
    ns.__dict__.update(torch=torch, dataclass=dataclass, dc_fields=fields, math=math,
                       ABC=ABC, ClassVar=ClassVar, replace=replace, logging=logging,
                       logger=logging.getLogger("cpu_reward_boundary"))

    def execute(nodes):
        tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
        exec(compile(tree, "original_ast", "exec", flags=__future__.annotations.compiler_flag), ns.__dict__)

    merge_base = subprocess.check_output(["git", "-C", cache, "merge-base", BASE, HEAD], env=env).decode().strip()
    assert merge_base == BASE
    execute(read("unirl/distributed/tensor/batch.py").body)
    reward = read("unirl/types/reward.py")
    execute([node for node in reward.body if not (isinstance(node, ast.ImportFrom)
             and (node.module == "PIL" or node.module == "unirl.distributed.tensor.batch"))])
    sampling = read("unirl/types/sampling.py")
    execute([named(sampling, name) for name in ("_coerce_type", "BaseSamplingParams", "ARSamplingParams")])
    sample = read("unirl/types/sample.py")
    execute([named(sample, "_part_with_field")])
    # Own modality mapper and Sample/Part carrier seams; actual primitives are CPU tensors.
    ns.__dict__["primitive_modality_key"] = lambda primitive: "image"
    client_tree = read("unirl/reward/client.py")
    execute([named(client_tree, "_build_reward_request"), named(client_tree, "RewardClient")])
    service_tree = read("unirl/reward/service.py")
    service = named(service_tree, "RewardService")
    service.bases = [ast.Name(id="object", ctx=ast.Load())]  # No Ray Remote construction.
    for node in service.body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = [item for item in node.decorator_list if not (
                isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and item.func.id == "distributed")]
    execute([service])

    Req, Resp = ns.RewardRequest, ns.RewardResponse
    generated = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    conditions = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    part = PartCarrier({"image": generated}, ns.ARSamplingParams(max_new_tokens=6),
                       SimpleNamespace(lengths=torch.tensor([2, 4, 6, 8])), object(),
                       ["a.0", "b.0", "a.1", "b.1"], ["a", "b", "a", "b"], {})
    sample = SampleCarrier([object(), part], conditions, [None, {"row": 1}, None, {"row": 3}])
    request = ns._build_reward_request(sample)
    assert request.generated is not part.primitives and request.generated["image"] is generated
    assert request.primitives["image"] is conditions
    assert request.response_lengths == [2, 4, 6, 8] and request.max_new_tokens == 6
    assert request.group_ids == part.group_ids and request.sample_ids == part.sample_ids
    assert not {"trajectory", "segment", "sampling_params", "rewards"} & set(vars(request))
    chunks = request.chunk(2)
    restored = Req.concat(chunks)
    assert restored.sample_ids == request.sample_ids
    assert torch.equal(restored.generated["image"], generated)
    assert restored.metadata == request.metadata and restored.response_lengths == request.response_lengths
    assert chunks[0].group_ids == ["a", "b"] and chunks[1].group_ids == ["a", "b"]
    selected = request.select(torch.tensor([3, 0]))
    assert selected.sample_ids == ["b.1", "a.0"] and selected.response_lengths == [8, 2]
    assert torch.equal(selected.generated["image"], generated[[3, 0]])
    assert torch.equal(selected.primitives["image"], conditions[[3, 0]])
    assert selected.metadata == [sample.metadata[3], sample.metadata[0]]
    assert request.slice(1, 3).sample_ids == ["b.0", "a.1"]

    class Scorer:
        preferred_input_kind = "image"
        response = Resp([1.0] * 4, {"quality": [0.5] * 4}, [True] * 4, [None] * 4)

        def get_model_name(self):
            return "own_cpu_recorder"

        def compute_rewards(self, req):
            self.last_request = req
            return self.response

    shaping = {}
    for mode, expected in (("zero", [1, 1, 0, 0]), ("keep", [1, 1, 1, 1]), ("soft", [1, 1, 0, -1])):
        worker = ns.RewardService(Scorer(), truncated_reward=mode, overlong_buffer_len=2)
        response = worker.score(request)
        assert response.rewards == expected
        assert response.component_rewards == {"quality": [0.5] * 4}
        shaping[mode] = response.rewards
    fixed_type = type("OwnFixedAR", (ns.ARSamplingParams,), {"emits_fixed_length": True})
    fixed_request = ns._build_reward_request(replace(sample, parts=[replace(part, sampling_params=fixed_type())]))
    assert fixed_request.response_lengths is None and fixed_request.max_new_tokens is None
    assert ns.RewardService(Scorer()).score(fixed_request).rewards == [1] * 4
    short_lengths = replace(part, segment=SimpleNamespace(lengths=torch.tensor([2, 4])))
    assert ns._build_reward_request(replace(sample, parts=[short_lengths])).response_lengths is None

    worker = ns.RewardService(Scorer(), truncated_reward="keep")

    class HandleRecorder:
        dp_size = 2

        def score(self, req):
            return worker.score(req)  # Direct call, no actual DP dispatch.

        def score_differentiable(self, *args, **kwargs):
            return args, kwargs

        get_memory_stats = score_differentiable

    client = ns.RewardClient(HandleRecorder())
    scored = client.score_and_attach(sample)
    frontier = scored.parts[-1]
    assert scored is not sample and frontier is not part and part.rewards is None
    assert scored.parts[0] is sample.parts[0] and frontier.trajectory is part.trajectory
    assert frontier.segment is part.segment and frontier.primitives is part.primitives
    assert scored.conditions is conditions and frontier.rewards.dtype == torch.float32
    assert frontier.rewards.device.type == "cpu" and frontier.rewards.tolist() == [1] * 4
    assert client.dp_size == 2 and client.get_memory_stats(1, tag="x") == ((1,), {"tag": "x"})
    assert client.score_differentiable(2, tag="y") == ((2,), {"tag": "y"})

    rejected = {}

    def reject(name, kind, fn):
        try:
            fn()
        except kind as exc:
            rejected[name] = str(exc)
        else:
            raise AssertionError(name)

    reject("metadata_rows", ValueError, lambda: replace(request, metadata=[None]))
    reject("paired_length_metadata", ValueError, lambda: replace(request, max_new_tokens=None))
    reject("concat_schema", ValueError, lambda: Req.concat([request, replace(request, generated={"video": generated})]))
    reject("no_parts", ValueError, lambda: client.score_and_attach(replace(sample, parts=[])))
    reject("precomputed", RuntimeError, lambda: client.score_and_attach(replace(sample, parts=[replace(part, rewards=torch.ones(4))])))
    reject("no_generated", ValueError, lambda: client.score_and_attach(replace(sample, parts=[replace(part, primitives={})])))
    failed_scorer = Scorer()
    failed_scorer.response = Resp([1] * 4, {}, [True, False, True, True], [None, "bad row", None, None])
    reject("worker_failure", RuntimeError, lambda: ns.RewardService(failed_scorer).score(request))
    failed_scorer.response = Resp([1] * 3, {}, [True] * 4, [None] * 4)
    reject("worker_reward_count", RuntimeError, lambda: ns.RewardService(failed_scorer).score(request))
    reject("input_kind", ValueError, lambda: worker.score(replace(request, generated={"video": generated})))

    geometry = {}
    for label, ref in (("base", BASE), ("head", HEAD)):
        tree = read("unirl/trainer/diffusion.py", ref)
        functions = [named(tree, "_validate_prompt_tree_dp_geometry"), named(tree, "_validate_dp_geometry")]
        if label == "head":
            functions.insert(1, named(tree, "_validate_reward_dp_geometry"))
        execute(functions)
        rows = []
        for rollout_dp, reward_dp in ((1, 2), (1, 4), (2, 2)):
            try:
                ns._validate_dp_geometry(batch_size=3, samples_per_prompt=2, num_updates_per_batch=1,
                                         rollout_dp_size=rollout_dp, reward_dp_size=reward_dp, train_dp_size=1)
            except ValueError as exc:
                rows.append({"rollout_dp": rollout_dp, "reward_dp": reward_dp,
                             "status": "rejected", "error": str(exc)})
            else:
                rows.append({"rollout_dp": rollout_dp, "reward_dp": reward_dp, "status": "accepted"})
        geometry[label] = rows
    assert [row["status"] for row in geometry["base"]] == ["rejected"] * 3
    assert [row["status"] for row in geometry["head"]] == ["accepted", "rejected", "rejected"]
    current_request = named(read("unirl/types/reward.py", MAIN), "RewardRequest")
    current_fields = [node.target.id for node in current_request.body if isinstance(node, ast.AnnAssign)]
    assert "conditioning" in current_fields and "primitives" not in current_fields

    print(json.dumps({"head": HEAD, "actual_merge_base": merge_base, "source_sha256": hashes,
                      "python": platform.python_version(), "torch": torch.__version__,
                      "projection": {"generated_value_identity": True, "heavy_fields_absent": True,
                                     "response_lengths": request.response_lengths, "max_new_tokens": 6},
                      "native_request_rows": {"chunk_concat_exact": True, "select_ids": selected.sample_ids,
                                              "chunk_group_ids": [chunk.group_ids for chunk in chunks]},
                      "original_service_shaping": shaping, "fixed_ar_omits_length_shaping": True,
                      "mismatched_segment_length_omits_metadata": True,
                      "original_copy_helper_on_own_carrier": {"input_unchanged": True, "nonreward_identity": True},
                      "client_forwarding": True, "rejected_controls": rejected,
                      "original_dp_geometry_three_cases_per_version": geometry,
                      "current_main_static_only": {"sha": MAIN, "request_fields": current_fields},
                      "scope": "Own CPU tensor/carrier/scorer/Handle fixtures; original Batch/RewardRequest/Response, "
                               "AR class, client, copy helper, two-version geometry functions and service with "
                               "Remote base/distributed decorators removed. Current-main request AST only. "
                               "Not native Sample/Part/conditioning, TensorRef views/transport, DP/Ray, author 11 tests, "
                               "real scoring, bandwidth, gradients or training."}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
