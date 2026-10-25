"""Check fixed verl teacher-logit hooks on CPU, without importing verl or vLLM."""

import ast
import hashlib
import json
import os
import subprocess
import sys
from types import MethodType, SimpleNamespace
from typing import Optional

import torch

VERSIONS = {
    "main": "3c6d9b43b67c74ec343d77874cd8fa6e841523da",
    "pre_fp32": "d44e8847c4e9312290ac0271f888c83e12cc61a7",
    "head": "2920c0144f2cb8e0d416f7ad35ce28034795fdc9",
}
HOOK_PATH = "verl/workers/rollout/vllm_rollout/utils.py"
TEST_PATH = "tests/workers/rollout/test_vllm_teacher_temperature_on_cpu.py"


def blob(repo, revision, path):
    return subprocess.check_output(
        ["git", "-C", repo, "show", revision + ":" + path],
        env=dict(os.environ, GIT_NO_LAZY_FETCH="1"),
    )


def load_node(raw, name, namespace):
    node = next(
        node for node in ast.parse(raw).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name
    )
    if isinstance(node, ast.FunctionDef):
        node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, "original_ast", "exec"), namespace)


def verify(repo):
    test_raw = blob(repo, VERSIONS["head"], TEST_PATH)
    result = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "versions": VERSIONS,
        "test_sha256": hashlib.sha256(test_raw).hexdigest(),
        "defaults": {},
        "small_temperature": {},
        "hook_sha256": {},
    }
    for label, revision in VERSIONS.items():
        raw = blob(repo, revision, HOOK_PATH)
        compile(ast.parse(raw), HOOK_PATH, "exec")
        namespace = {"torch": torch, "Optional": Optional, "MethodType": MethodType}
        load_node(raw, "monkey_patch_compute_logits", namespace)
        load_node(test_raw, "FakeModel", namespace)
        namespace["utils"] = SimpleNamespace(
            monkey_patch_compute_logits=namespace["monkey_patch_compute_logits"]
        )
        load_node(test_raw, "test_default_temperature_preserves_rollout_logits", namespace)
        for dtype in [torch.float16, torch.bfloat16, torch.float32]:
            namespace["test_default_temperature_preserves_rollout_logits"](dtype)
        result["defaults"][label] = 3
        result["hook_sha256"][label] = hashlib.sha256(raw).hexdigest()
        if label == "main":
            continue
        cases = []
        for dtype in [torch.float16, torch.bfloat16]:
            logits = torch.tensor([[20., 19., -20., 80., 100.]], dtype=dtype)
            model = namespace["FakeModel"]()
            namespace["monkey_patch_compute_logits"](
                model, vocab_size=4, banned_token_ids=[3], logits_temperature=0.0001
            )
            scaled = model.compute_logits(logits)
            actual = scaled.float().log_softmax(-1)
            expected = (logits[:, :3].float() / 0.0001).log_softmax(-1)
            finite = bool(torch.isfinite(actual[:, :3]).all())
            error = float((actual[:, :3] - expected).abs().max()) if finite else None
            masked = bool(torch.isneginf(actual[:, 3:]).all())
            if label == "head":
                torch.testing.assert_close(actual[:, :3], expected)
                assert finite and masked and scaled.dtype == torch.float32
            elif dtype == torch.float16:
                assert not finite
            else:
                assert error > 0
            cases.append({
                "dtype": str(dtype),
                "scaled_dtype": str(scaled.dtype),
                "valid_logprobs_finite": finite,
                "max_abs_error": error,
                "masked_scores_neg_inf": masked,
            })
        result["small_temperature"][label] = cases
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify.py /path/to/verl-git")
    print(json.dumps(verify(sys.argv[1]), indent=2))

