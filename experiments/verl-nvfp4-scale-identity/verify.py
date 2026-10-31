"""Original collector only: own CPU module-tree controls and two-rank Gloo metadata.

No ModelOpt/CT imports, quantization, Bridge, Megatron execution, or original test bodies.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile

import torch

BASE = "8e48d72de4766228c36623c90f9f5c49ee5b45f3"
HEAD = "ba2fbf9d52c0ac24b4d78ab20ff3e77afb95a8cf"
MODEL = "verl/utils/qat/modelopt.py"
CT = "verl/utils/qat/compressed_tensors.py"
ENGINE = "verl/workers/engine/megatron/transformer_impl.py"
TEST = "tests/utils/test_modelopt_mxfp8_on_gpu.py"


def source(repo, ref, path):
    return subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=repo,
                                   env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))


def collector(raw):
    tree = ast.parse(raw)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "collect_nvfp4_input_global_scales")
    namespace = {"torch": torch, "nn": torch.nn}
    exec(compile(ast.Module(body=[node], type_ignores=[]), MODEL, "exec"), namespace)
    return namespace[node.name]


def chunks_for(numbers_by_chunk, wrapped=False, experts=None, multiplier=1):
    chunks, expected = [], {}
    for numbers in numbers_by_chunk:
        chunk = torch.nn.Module()
        chunk.decoder = torch.nn.Module()
        chunk.decoder.layers = torch.nn.ModuleList()
        for number in numbers:
            layer = torch.nn.Module()
            layer.layer_number = number
            layer.mlp = torch.nn.Module()
            layer.mlp.experts = torch.nn.Module()
            if experts is None:
                targets = [(None, layer.mlp.experts)]
            else:
                layer.mlp.local_expert_indices = experts
                layer.mlp.experts.local_experts = torch.nn.ModuleList(
                    [torch.nn.Module() for _ in experts])
                targets = list(zip(experts, layer.mlp.experts.local_experts))
            for expert_id, target in targets:
                for projection, key, factor in (("linear_fc1", "gate_up", 1), ("linear_fc2", "down", 2)):
                    linear = torch.nn.Module()
                    linear.input_quantizer = torch.nn.Module()
                    value = float(number * factor * multiplier * (1 if expert_id is None else expert_id + 1))
                    linear.input_quantizer._verl_input_global_scale = torch.tensor([value])
                    setattr(target, projection, linear)
                    suffix = "" if expert_id is None else f".{expert_id}"
                    expected[f"layers.{number - 1}.mlp.experts{suffix}.{key}"] = value
            chunk.decoder.layers.append(layer)
        chunks.append(torch.nn.Sequential(chunk) if wrapped else chunk)
    return chunks, expected


def scalar_map(scales):
    return {key: value.item() for key, value in scales.items()}


def pipeline_worker(rank, repo, init_file, virtual_chunks, queue):
    torch.set_num_threads(1)
    dist = torch.distributed
    collect = collector(source(repo, HEAD, MODEL))
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        chunks, _ = chunks_for(tuple((vp * 2 + rank + 1,) for vp in range(virtual_chunks)))
        observers = [module for chunk in chunks for _, module in chunk.named_modules()
                     if hasattr(module, "_verl_input_global_scale")]
        for observer in observers:
            del observer._verl_input_global_scale
        assert collect(chunks, pp_group=dist.group.WORLD) == {}
        snapshots = []
        for step in (1, 2):
            for vp in range(virtual_chunks):
                number = vp * 2 + rank + 1
                for offset, factor in enumerate((1, 2)):
                    observers[vp * 2 + offset]._verl_input_global_scale = torch.tensor([float(number * factor * step)])
            actual = scalar_map(collect(chunks, pp_group=dist.group.WORLD))
            _, expected = chunks_for((tuple(range(1, virtual_chunks * 2 + 1)),), multiplier=step)
            assert actual == expected
            snapshots.append(actual)
        queue.put({"rank": rank, "empty": {}, "snapshots": snapshots})
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    hashes, funcs = {}, {}
    for ref, paths in ((BASE, (MODEL, CT)), (HEAD, (MODEL, CT, ENGINE, TEST))):
        for path in paths:
            raw = source(args.repo, ref, path)
            compile(ast.parse(raw), path, "exec")
            hashes[f"{ref}:{path}"] = hashlib.sha256(raw).hexdigest()
            if path == MODEL:
                funcs[ref] = collector(raw)
    identity, sequential = [], []
    for numbers in (((1, 2),), ((5, 6),), ((1, 2), (5, 6)), ((5, 6), (9, 10))):
        for wrapped in (False, True):
            chunks, expected = chunks_for(numbers, wrapped)
            outputs = {ref: scalar_map(func(chunks)) for ref, func in funcs.items()}
            assert outputs[HEAD] == expected
            identity.append({"numbers": numbers, "wrapped": wrapped, "expected": expected,
                             "base": outputs[BASE], "head": outputs[HEAD], "base_matches": outputs[BASE] == expected})
    assert sum(item["base_matches"] for item in identity) == 1
    for experts in ((0, 1), (4, 5)):
        chunks, expected = chunks_for(((5,),), experts=experts)
        outputs = {ref: scalar_map(func(chunks)) for ref, func in funcs.items()}
        assert outputs[HEAD] == expected and outputs[BASE] != expected
        sequential.append({"experts": experts, "expected": expected, "base": outputs[BASE], "head": outputs[HEAD]})
    pipeline = []
    for count in (1, 2):
        ctx = torch.multiprocessing.get_context("spawn")
        queue = ctx.SimpleQueue()
        with tempfile.TemporaryDirectory(prefix="nvfp4-metadata-") as temporary:
            torch.multiprocessing.spawn(pipeline_worker,
                                       args=(str(args.repo.resolve()), str(Path(temporary) / "init"), count, queue),
                                       nprocs=2, join=True)
            ranks = sorted([queue.get(), queue.get()], key=lambda item: item["rank"])
        queue.close()
        pipeline.append({"virtual_chunks": count, "ranks": ranks})
    print(json.dumps({"python": platform.python_version(), "torch": torch.__version__,
                      "torch_git": torch.version.git_version, "base": BASE, "head": HEAD,
                      "source_sha256": hashes, "own_layer_identity_controls": identity,
                      "own_sequential_expert_controls": sequential, "own_two_rank_gloo_controls": pipeline,
                      "scope": "Six full source inputs compile. Exact original collector AST only; own nn.Module "
                      "trees with manually assigned observer scalars, not ModelOpt observers. Eight layer/wrapper "
                      "controls (base 1 match/7 mismatch, head 8 match), two expert controls (base 2 mismatch/head "
                      "2 match). Two real CPU Gloo processes for each of 1/2 virtual chunks; empty observation and "
                      "two refreshed scalar snapshots on both ranks. No original test bodies or pytest suite, "
                      "CT lookup/compression/refresh method, ModelOpt/QAT, Bridge export, Megatron pipeline, "
                      "TP/EP, GPU/NPU, training, quality or performance execution."}, indent=2))


if __name__ == "__main__":
    main()
