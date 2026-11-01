"""#8436 original async methods with explicit request/queue recorders.

python verify.py /path/to/pre-cached/verl-git
No Ray, TransferQueue, engine, tokenizer, model, GPU, or original tests run.
"""

import __future__
import ast
import asyncio
import copy
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from types import MethodType, SimpleNamespace

import numpy as np

HEAD = "04280af2b88453fb7e53bb28b73298954a1805f7"
BASE = "2ab35292039a51175a4b3d3b3b2d76e951a0d3e4"
LEGACY = "verl/experimental/agent_loop/agent_loop.py"
TQ = "verl/trainer/ppo/v1/agent_loop_tq.py"
SERVER = "verl/workers/rollout/vllm_rollout/vllm_async_server.py"


class NonTensorData:
    def __init__(self, data):
        self.data = data


class NonTensorStack:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        return NonTensorData(self.values[index])


class TQBatch(dict):
    def __len__(self):
        return 2


class LegacyBatch:
    def __init__(self, validate):
        self.meta_info = dict(validate=validate, global_steps=0)
        self.non_tensor_batch = dict(index=np.arange(2), __do_sample__=np.array([True, False]))

    def __len__(self):
        return 2


async def trajectory_info(_step, indices, validate):
    return [dict(step=0, sample_index=index, rollout_n=0, validate=validate) for index in indices]


def main(cache):
    hashes = {}

    def read(ref, path):
        raw = subprocess.check_output(["git", "-C", cache, "show", f"{ref}:{path}"],
                                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        hashes[f"{ref}:{path}"] = hashlib.sha256(raw).hexdigest()
        return ast.parse(raw)

    def execute(nodes, ns):
        tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
        exec(compile(tree, "original_ast", "exec", flags=__future__.annotations.compiler_flag), ns)

    merge_base = subprocess.check_output(["git", "-C", cache, "merge-base", HEAD, BASE],
                                        env=dict(os.environ, GIT_NO_LAZY_FETCH="1")).decode().strip()
    assert merge_base == BASE
    server = read(HEAD, SERVER)
    default_nodes = [node for node in ast.walk(server) if isinstance(node, ast.Expr)
                     and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                     and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == "sampling_params"
                     and node.value.func.attr == "setdefault" and node.value.args
                     and isinstance(node.value.args[0], ast.Constant)
                     and node.value.args[0].value == "repetition_penalty"]
    assert len(default_nodes) == 1

    def apply_server_default(params, config):
        result = copy.deepcopy(params)
        execute(default_nodes, dict(sampling_params=result, self=SimpleNamespace(config=config)))
        return result

    namespaces = {}
    for ref in (BASE, HEAD):
        ns = dict(asyncio=asyncio, np=np, logger=logging.getLogger("isolated_repetition"),
                  NonTensorData=NonTensorData, NonTensorStack=NonTensorStack,
                  torch=SimpleNamespace(Tensor=type("UnusedTensorMarker", (), {})),
                  get_trajectory_info=trajectory_info,
                  RolloutTraceConfig=SimpleNamespace(get_instance=lambda: SimpleNamespace(max_samples_per_step_per_worker=None)))
        for path, cls_name, method_names in (
            (LEGACY, "AgentLoopWorker", {"generate_sequences"}),
            (TQ, "AgentLoopWorkerTQ", {"generate_sequences", "_run_prompt"}),
        ):
            tree = read(ref, path)
            cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls_name)
            methods = [node for node in cls.body if getattr(node, "name", None) in method_names]
            assert len(methods) == len(method_names)
            helpers = [node for node in tree.body if getattr(node, "name", None)
                       in {"apply_greedy_sampling_params", "_settle_session_tasks"}]
            execute([*helpers, ast.ClassDef(name=cls_name, bases=[], keywords=[], body=methods, decorator_list=[])], ns)
        namespaces[ref] = ns

    async def run_case(ref, path, penalty, validate):
        ns, captured, statuses = namespaces[ref], [], []
        config = SimpleNamespace(temperature=.8, top_p=.9, top_k=20, calculate_log_probs=True,
                                 repetition_penalty=penalty, topk_log_probs=4, n=1,
                                 agent=SimpleNamespace(default_agent_loop="single_turn_agent"),
                                 val_kwargs=SimpleNamespace(temperature=.2, top_p=.7, top_k=3, n=1))

        async def record(params, *args, **kwargs):
            assert "__do_sample__" not in kwargs
            captured.append(copy.deepcopy(params))

        async def queue_record(**kwargs):
            statuses.append(kwargs["tag"]["status"])

        worker = SimpleNamespace(rollout_config=config, config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=config)),
                                 _run_agent_loop=record, _postprocess=lambda outputs, **kwargs: outputs,
                                 background_tasks=set())
        if path == "legacy":
            await ns["AgentLoopWorker"].generate_sequences(worker, LegacyBatch(validate))
        else:
            ns["tq"] = SimpleNamespace(async_kv_put=queue_record)
            worker._run_prompt = MethodType(ns["AgentLoopWorkerTQ"]._run_prompt, worker)
            batch = TQBatch(validate=validate, index=NonTensorStack([0, 1]), global_steps=NonTensorData(0),
                            uid=NonTensorStack(["own-0", "own-1"]), __do_sample__=NonTensorStack([True, False]))
            await ns["AgentLoopWorkerTQ"].generate_sequences(worker, batch)
            while worker.background_tasks:
                await asyncio.gather(*tuple(worker.background_tasks))
            assert statuses.count("running") == statuses.count("finished") == 2 and "failure" not in statuses
        assert len(captured) == 2
        records = []
        for index, params in enumerate(captured):
            assert ("repetition_penalty" in params) == (ref == BASE)
            greedy = not validate and index == 1
            expected_tpk = (0, 1., -1) if greedy else ((.2, .7, 3) if validate else (.8, .9, 20))
            assert (params["temperature"], params["top_p"], params["top_k"]) == expected_tpk
            assert params["logprobs"] is True
            assert ("topk_log_probs" in params) == (path == "tq" and not validate and not greedy)
            effective = apply_server_default(params, {"repetition_penalty": penalty})
            assert effective["repetition_penalty"] == (1. if ref == BASE else penalty)
            records.append(dict(per_sample_do_sample=index == 0, request=params,
                                vllm_default_result=effective["repetition_penalty"]))
        return dict(version=ref, path=path, configured_penalty=penalty, validate=validate, samples=records)

    async def matrix():
        return [await run_case(ref, path, penalty, validate) for ref in (BASE, HEAD)
                for path in ("legacy", "tq") for penalty in (1., 1.2, 1.5) for validate in (False, True)]

    matrix_result = asyncio.run(matrix())
    explicit = []
    for request in ({}, {"repetition_penalty": 1.}, {"repetition_penalty": 1.7}, {"repetition_penalty": None}):
        result = apply_server_default(request, {"repetition_penalty": 1.2})
        assert result["repetition_penalty"] == request.get("repetition_penalty", 1.2)
        explicit.append(dict(request=request, configured=1.2, result=result["repetition_penalty"]))
    assert apply_server_default({}, {})["repetition_penalty"] == 1.
    other_backends = {}
    for path in ("verl/workers/rollout/sglang_rollout/async_sglang_server.py",
                 "verl/workers/rollout/trtllm_rollout/trtllm_async_server.py"):
        tree = read(HEAD, path)
        mentions = sum((isinstance(node, ast.Constant) and node.value == "repetition_penalty")
                       or (isinstance(node, ast.Attribute) and node.attr == "repetition_penalty")
                       for node in ast.walk(tree))
        assert mentions == 0
        other_backends[path] = dict(exact_field_ast_mentions=mentions,
                                   scope="Static field search only; not execution or proof about indirect/native defaults.")
    print(json.dumps(dict(base=BASE, head=HEAD, actual_merge_base=merge_base, python=platform.python_version(),
                         numpy=np.__version__, source_sha256=hashes, own_async_matrix=matrix_result,
                         own_server_default_controls=explicit, missing_config_fallback=1.,
                         other_backend_source_only=other_backends,
                         scope="Two complete original generate_sequences methods, original TQ _run_prompt, "
                         "greedy/session-settling helpers, real asyncio tasks, one original vLLM setdefault statement. "
                         "All batch/wrapper/trace/trajectory/queue/agent/postprocess boundaries are explicit own "
                         "recorders; no original tests, real DataProto/TensorDict, Ray/TQ, SamplingParams, "
                         "inference, penalty math, logprob normalization, model, device or training execution."), indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
