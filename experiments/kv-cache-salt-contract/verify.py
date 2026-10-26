#!/usr/bin/env python3
"""Narrow original helper checks for verl #8323 at 4e93e476.

Input: an isolated source directory containing the seven pinned files below.
No network, Ray/engine imports, config construction, request submission or device
query. Only the original salt formatter and connector checker execute; the
SimpleNamespace holds their required fields, not a simulated engine or cache.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Optional

HASHES = {
    "tests/workers/rollout/test_prefix_cache_versioning_on_cpu.py": "fa542182d0f0d27447f97dd5b1c538662361ff5cddd09067581ae2b885595265",
    "verl/checkpoint_engine/base.py": "5c2c1eb86e572310dd6b61ae338483a4e9daf922a9a5294b02082d6925c62644",
    "verl/workers/config/rollout.py": "166034674ca9b62e2c7663bc22d3f801cfde9b362e1e25fc1cf6b6ce1b7dd118",
    "verl/workers/rollout/replica.py": "ec938575ca5202bbc1b443752bd8568289450a07558e10434b0f835b9ca146c0",
    "verl/workers/rollout/sglang_rollout/async_sglang_server.py": "9378e51959ae078583294c1b9166e2c3b2e4c0bfc45553e7a3c4ff3f20ed9c71",
    "verl/workers/rollout/trtllm_rollout/trtllm_async_server.py": "5b321daf80343f4df5305d5c57860b4a6a572c993128c76eb834f5b7eb4b851d",
    "verl/workers/rollout/vllm_rollout/vllm_async_server.py": "c8f4ee7dd53b71093015eb48cbcbc696f518c404966c50e0939c4423cde67330",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    root = parser.parse_args().source
    trees = {}
    for path, expected in HASHES.items():
        data = (root / path).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            parser.error(f"source hash mismatch: {path}")
        trees[path] = ast.parse(data, filename=path)
        compile(trees[path], path, "exec")
    selected = []
    for path, names in [
        ("verl/workers/config/rollout.py", {"_SALTED_VLLM_KV_CONNECTORS", "_check_salted_kv_connectors"}),
        ("verl/workers/rollout/replica.py", {"prefix_cache_salt"}),
    ]:
        for node in trees[path].body:
            name = node.name if isinstance(node, ast.FunctionDef) else None
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
            if name in names:
                selected.append(node)
    assert len(selected) == 3
    namespace = {"json": json, "RolloutConfig": SimpleNamespace, "Optional": Optional}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "original_helpers", "exec"), namespace)
    salt = namespace["prefix_cache_salt"]
    config = lambda mode, run: SimpleNamespace(external_kv_cache_invalidation=mode, prefix_cache_namespace=run)
    formatted = {
        "reset": [salt(config("reset", "run-a"), n) for n in [3, 4]],
        "salt": [salt(config("salt", "run-a"), n) for n in [3, 4]],
        "other_run": salt(config("salt", "run-b"), 3),
        "same_version_again": salt(config("salt", "run-a"), 3),
        "missing_version_at_helper": salt(config("salt", "run-a"), None),
    }
    assert formatted == {"reset": [None, None], "salt": ["run-a:3", "run-a:4"],
                         "other_run": "run-b:3", "same_version_again": "run-a:3",
                         "missing_version_at_helper": "run-a:None"}
    try:
        salt(config("salt", None), 3)
    except AssertionError:
        formatted["missing_namespace"] = "AssertionError"
    else:
        raise AssertionError("missing namespace accepted")
    transfers = [
        ({"kv_connector": "MooncakeStoreConnector", "kv_role": "kv_both"}, True),
        ('{"kv_connector":"OffloadingConnector","kv_role":"kv_both"}', True),
        ({"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}, False),
        ({"kv_connector": "MultiConnector", "kv_connector_extra_config": {"connectors": [
            {"kv_connector": "LMCacheMPConnector"}, {"kv_connector": "LMCacheConnectorV1"}]}}, False),
    ]
    connector_results = []
    for key in ["kv_transfer_config", "kv-transfer-config"]:
        for index, (transfer, expected) in enumerate(transfers):
            try:
                namespace["_check_salted_kv_connectors"]({"vllm": {key: transfer}})
                accepted = True
            except ValueError as error:
                assert "LMCacheConnectorV1" in str(error)
                accepted = False
            assert accepted == expected
            connector_results.append({"key": key, "original_case": index, "accepted": accepted})
    print(json.dumps({"python": sys.version.split()[0], "revision": "4e93e4761ddde2ea5bf9d3032b86b6c3844126fd",
                      "scope": "two original pure helpers only; not full config or cache execution",
                      "source_sha256": HASHES, "formatting": formatted,
                      "connector_checks": connector_results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
