"""Pinned SGLang sampling/wire contracts; no SGLang, tokenizer or real Sample execution."""

import argparse
import ast
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import ModuleType, SimpleNamespace

REF = "b8e9740e981a68a5556fbeb8225558040a4d47a4"
PREFIX = "unirl/rollout/engine/sglang/"
PATHS = [PREFIX + suffix for suffix in (
    "utils/sampling.py", "adapters/text.py", "backends/native.py", "backends/http.py",
    "utils/conversations.py", "adapters/vlm.py")]
PATHS.append("unirl/rollout/harness/tool_agent.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    args = parser.parse_args()
    trees, hashes = {}, {}
    for path in PATHS:
        raw = subprocess.check_output(["git", "show", f"{REF}:{path}"], cwd=args.repo,
                                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        trees[path] = ast.parse(raw)
        compile(trees[path], path, "exec")
        hashes[path] = hashlib.sha256(raw).hexdigest()

    sampling_tree = trees[PATHS[0]]
    sampling_tree.body = [n for n in sampling_tree.body if not (
        isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("unirl."))]
    module = ModuleType("isolated_sampling_scope")
    sys.modules[module.__name__] = module
    module.Sample = module.ARSamplingParams = object  # Only type/frontier markers; not real constructors.
    exec(compile(sampling_tree, PATHS[0], "exec"), module.__dict__)
    namespace = dict(module.__dict__)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    adapter = next(n for n in trees[PATHS[1]].body if isinstance(n, ast.ClassDef) and n.name == "TextLMAdapter")
    base_payload = next(n for n in adapter.body if isinstance(n, ast.FunctionDef) and n.name == "base_payload")
    native_nodes = [n for n in trees[PATHS[2]].body if
                    (isinstance(n, ast.FunctionDef) and n.name == "payload_to_generate_kwargs") or
                    (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_GENERATE_PASSTHROUGH"
                                                     for t in n.targets))]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, base_payload, *native_nodes], type_ignores=[])),
                 "original-wire-definitions", "exec"), namespace)
    adapter_scope = SimpleNamespace(_response_forbidden_token_ids=())
    rows = []
    last_resolved = None
    for requested in (None, False, True):
        control = {"stop": ["</tool_call>"], "stop_token_ids": [17], "skip_special_tokens": False}
        if requested is not None:
            control["no_stop_trim"] = requested
        ar = SimpleNamespace(seed=None, temperature=1., max_new_tokens=16, top_p=1., top_k=0)
        parent = SimpleNamespace(control={"ar": control}, sample_ids=["parent"])
        gen = SimpleNamespace(sampling_params=ar, sample_ids=["generated"])
        sample = SimpleNamespace(parts=[parent, gen], frontier_gen_part=lambda _type: gen)
        config = SimpleNamespace(engine_kwargs={}, system_instruction=None)
        resolved = module.resolve_sampling(config, sample)
        payload = namespace["base_payload"](adapter_scope, resolved)
        native = namespace["payload_to_generate_kwargs"]({**payload, "input_ids": [1, 2]})
        assert "no_stop_trim" not in resolved.block
        for key in ("stop", "stop_token_ids", "skip_special_tokens"):
            assert resolved.block[key] == control[key]
        assert payload["sampling_params"] == native["sampling_params"] == resolved.block
        rows.append({"requested_no_stop_trim": requested, "resolved_block": resolved.block,
                     "native_sampling_block": native["sampling_params"]})
        last_resolved = resolved

    # Explicit injection AFTER the resolver, not a patch or an in-tree reachable configuration claim.
    injected = replace(last_resolved, block={**last_resolved.block, "no_stop_trim": True})
    payload = namespace["base_payload"](adapter_scope, injected)
    native = namespace["payload_to_generate_kwargs"]({**payload, "input_ids": [1, 2]})
    assert payload["sampling_params"]["no_stop_trim"] is True
    assert native["sampling_params"]["no_stop_trim"] is True
    print(json.dumps({"python": platform.python_version(), "ref": REF, "controls": rows,
                      "injected_after_resolver_preserved_by_adapter_and_native_mapper": True,
                      "source_sha256": hashes,
                      "scope": "Three original resolver calls with explicit SimpleNamespace frontier/config fixtures; "
                      "original base_payload and native kwargs mapper; one injected post-resolver block. "
                      "Seven complete modules AST/compile only. No real Sample/ARSamplingParams validation, "
                      "tokenizer/processor, HTTP/native SGLang engine, generation, replay, image loop, "
                      "training, device or performance validation."}, indent=2))


if __name__ == "__main__":
    main()
