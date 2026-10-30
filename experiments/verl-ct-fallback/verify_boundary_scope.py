"""Probe #8463's original classifier with explicit token/decoder fixtures, not a model."""

import ast
import dataclasses
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import types

repo = sys.argv[1]
ref = "d9d1f9165c550b3effc4aebc0fd5d94365a70273"
env = {**os.environ, "GIT_NO_LAZY_FETCH": "1"}

def source(path):
    raw = subprocess.check_output(["git", "-C", repo, "show", f"{ref}:{path}"], env=env)
    compile(raw, path, "exec")
    print(hashlib.sha256(raw).hexdigest(), path)
    return ast.parse(raw, filename=path)

core = source("verl/utils/tokenizer/continuous_token.py")
check = source("verl/utils/tokenizer/continuous_token_check.py")
source("verl/experimental/agent_loop/agent_loop.py")
source("verl/utils/tokenizer/continuous_token_wiring.py")
module = types.ModuleType("isolated_ct_check")
sys.modules[module.__name__] = module
module.__dict__.update(dataclass=dataclasses.dataclass, field=dataclasses.field)
merge = next(n for n in core.body if isinstance(n, ast.ClassDef) and n.name == "MergeResult")
future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
exec(compile(ast.fix_missing_locations(ast.Module(body=[future, merge], type_ignores=[])), "merge", "exec"), module.__dict__)
# Only the relative type import is omitted; checker functions and constants stay original.
check.body = [n for n in check.body if not (isinstance(n, ast.ImportFrom) and n.level)]
exec(compile(check, "original_checker", "exec"), module.__dict__)

class CharacterDecoder:
    def decode(self, ids):
        return "".join(map(chr, ids))

def ids(text):
    return list(map(ord, text))

cases = [
    ("exact", "oldZ", "oldZcd", "oldZcd", 0, "match", False),
    ("history", "oldZ", "oldZcd", "newerZcd", 0, "history", False),
    ("separator", "oldZ", "oldZcd", "oldZ\ncd", 0, "boundary", True),
    ("content", "oldZ", "oldZXd", "oldZcd", 0, "boundary", False),
    ("removed_stop", "oldZZ", "oldZcd", "newerZcd", 1, "history", False),
]
results = []
for name, runtime, merged, expected, removed, status, whitespace in cases:
    merge_result = module.MergeResult(ids(merged), appended_token_count=2, removed_prefix_token_count=removed)
    result = module._classify(CharacterDecoder(), name, ids(runtime), merge_result, ids(expected))
    assert (result.status, result.whitespace_only) == (status, whitespace)
    results.append([name, result.status, result.whitespace_only])

assert module._simulate_generation([1, 2], [1, 2, 3, 9, 10], [1, 2, 3, 9, 10, 4], {9}) == [1, 2, 3, 9]
assert module._simulate_generation([1, 2], [1, 2, 3], [1, 2, 3, 9, 4], {9}) == [1, 2, 3, 9]

class UnsupportedFixtureBuilder:
    tokenizer = types.SimpleNamespace(eos_token_id=9)
    def build_initial_tokens(self, *args, **kwargs):
        raise ValueError("fixture render unsupported")

builder = UnsupportedFixtureBuilder()
skipped = module.check_continuous_token_builder(builder, stop_token_ids={9})
assert len(skipped) == 8 and all(r.status == "skipped" for r in skipped)
warnings = []
class Capture(logging.Handler):
    def emit(self, record):
        warnings.append(record.getMessage())
handler = Capture()
module.logger.addHandler(handler)
try:
    returned = module.verify_continuous_token_builder(builder, mode="strict", hf_model_type="fixture", generation_config=None)
finally:
    module.logger.removeHandler(handler)
assert returned is None and len(warnings) == 1 and "Could not verify" in warnings[0]
print(json.dumps({"python": platform.python_version(), "classification": results, "stop_simulations": 2,
                  "strict_all_skipped": {"skipped": len(skipped), "returns_none": returned is None,
                                         "warning": "Could not verify"}}, sort_keys=True))
