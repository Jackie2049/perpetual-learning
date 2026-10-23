"""Check #517 row views with native CPU Sample/Part/request types, not a reward service.

Usage: Python3.12+ verify_unirl_reward_rows.py /path/to/UniRL /path/to/Pillow-target
Needs torch/Pillow. Complete original types load from fixed Git objects; only
tensor package initialization skips Ray transport and re-exports original helpers.
Four original functions are AST-loaded; local shard labels are synthetic scores.
No trainer, dispatch wrapper, scorer, model, TensorRef hydration or device proof.
"""
import ast
import dataclasses
import hashlib
import importlib.abc
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

HEAD = "fca5a55a0e5b30d108289f7394a7c50850593bf2"
BASE = "7a5303647d0c1ec27807124f70939cbb8edde6e9"
repo = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(Path(sys.argv[2]).resolve()))
sys.dont_write_bytecode = True


def git(*args):
    return subprocess.check_output(["git", *args], cwd=repo,
                                  env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))


paths = set(git("ls-tree", "-r", "--name-only", HEAD).decode().splitlines())


class Originals(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "unirl" and not fullname.startswith("unirl."):
            return None
        stem = fullname.replace(".", "/")
        path = stem + "/__init__.py" if stem + "/__init__.py" in paths else stem + ".py"
        if path not in paths:
            return None
        spec = importlib.util.spec_from_loader(fullname, self, is_package=path.endswith("/__init__.py"))
        spec.loader_state = path
        return spec

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        if module.__name__ == "unirl.distributed.tensor":
            from unirl.distributed.tensor import batch, ref
            for source in (batch, ref):
                for name in dir(source):
                    if not name.startswith("_"):
                        setattr(module, name, getattr(source, name))
            return
        path = module.__spec__.loader_state
        exec(compile(git("show", HEAD + ":" + path), HEAD + ":" + path, "exec"), module.__dict__)


sys.meta_path.insert(0, Originals())
import torch
import PIL
from unirl.types.sample import Part, Sample
from unirl.types.sample_id import ancestor_id
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams
from unirl.types.primitives import Images, Texts, primitive_modality_key
from unirl.types.reward import RewardRequest

torch.set_default_device("cpu")
namespace = dict(dataclasses=dataclasses, torch=torch, Sample=Sample, Part=Part,
                 ancestor_id=ancestor_id, Texts=Texts, RewardRequest=RewardRequest,
                 primitive_modality_key=primitive_modality_key)
for path, names in (
    ("unirl/trainer/diffusion.py", {"_reward_row_view", "_attach_row_rewards", "_validate_prompt_tree_dp_geometry"}),
    ("unirl/reward/service.py", {"_build_reward_request"}),
):
    raw = git("show", HEAD + ":" + path)
    compile(raw, path, "exec")
    print(path, hashlib.sha256(raw).hexdigest())
    nodes = [node for node in ast.parse(raw).body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.Module(future + nodes, type_ignores=[]), path, "exec"), namespace)
for rev in (BASE, HEAD):
    for path in ("unirl/trainer/diffusion.py", "unirl/train_diffusion.py"):
        compile(git("show", rev + ":" + path), path, "exec")
row_view, attach, validate, request = [namespace[name] for name in (
    "_reward_row_view", "_attach_row_rewards", "_validate_prompt_tree_dp_geometry", "_build_reward_request")]

root = Part.input(["A", "B"], metadata=[{"domain": "a"}, {"domain": "b"}],
                  primitives={"text": Texts(["original A", "original B"]),
                              "image": Images.from_dense(torch.arange(24.).reshape(2, 3, 2, 2) / 23)})
middle_ids = ["A/2", "B/0", "A/0", "B/2", "A/1", "B/1"]
middle = Part(sample_ids=middle_ids, sampling_params=ARSamplingParams(),
              primitives={"text": Texts(["generation " + sid for sid in middle_ids])})
leaf_ids = ["B/2/0", "A/0/0", "B/1/0", "A/2/0", "A/1/0", "B/0/0"]
frontier = Part(sample_ids=leaf_ids, sampling_params=DiffusionSamplingParams(),
                primitives={"text": Texts(["output " + sid for sid in leaf_ids])})
sample = Sample(parts=[root, middle, frontier])
view = row_view(sample)
assert sample.batch_size == 2 and view.batch_size == 6
assert all(part.batch_size == 6 for part in view.parts)
assert len(view.split()) == 6 and all(shard.batch_size == 1 for shard in view.split())
print("PASS native reroot: 2 roots/6 frontier rows -> 6 complete roots")

before, after = request(sample, "text"), request(view, "text")
for field in ("original_prompt", "generation_prompt"):
    assert getattr(before, field).texts == getattr(after, field).texts
assert before.metadata == after.metadata
assert before.texts == after.texts
assert torch.equal(before.conditioning["image"].packed_pixels, after.conditioning["image"].packed_pixels)
assert before.sample_ids != after.sample_ids and before.group_ids != after.group_ids
assert len(set(after.group_ids)) == 6
print("PASS native RewardRequest content/conditioning/metadata retained, scorer IDs intentionally changed")

shards = []
for start in range(0, 6, 2):
    shard = view.slice(start, start + 2)
    rewards = torch.arange(start, start + 2, dtype=torch.float32)
    shards.append(shard.replace_frontier(dataclasses.replace(
        shard.parts[-1], rewards=rewards, component_rewards={"fixture": rewards + 10})))
scored = Sample.concat(shards)
restored = attach(sample, scored)
assert restored.parts[0] is root and restored.parts[1] is middle
assert restored.parts[-1].sample_ids == leaf_ids
assert restored.parts[-1].primitives is frontier.primitives
assert restored.parts[-1].rewards.tolist() == list(range(6))
assert restored.parts[-1].component_rewards["fixture"].tolist() == list(range(10, 16))
assert frontier.rewards is None
print("PASS native slice/concat and attach: synthetic score labels restored; ancestors/input untouched")

try:
    attach(sample, Sample.concat(list(reversed(shards))))
except RuntimeError as exc:
    assert "order or identity" in str(exc)
else:
    raise AssertionError("reordered scores must not attach")
print("PASS reordered shard identities rejected")

for mode, rollout_dp, reward_dp, expected in (
    ("row", 1, 32, True), ("prompt_tree", 1, 32, False),
    ("row", 1, 48, False), ("row", 32, 32, False),
):
    try:
        validate(batch_size=8, samples_per_prompt=8, rollout_dp_size=rollout_dp,
                 reward_dp_size=reward_dp, reward_dispatch=mode, context="CPU fixture")
    except ValueError:
        assert not expected
    else:
        assert expected
print("PASS original geometry helper: row reward DP32 allowed, prompt-tree/DP48/rollout-DP32 rejected")
assert not torch.distributed.is_initialized() and "ray" not in sys.modules
print("5 native CPU groups; 4 original functions; no upstream tests/trainer/service/transport execution")
print("torch", torch.__version__, "Pillow", PIL.__version__, "Python", sys.version.split()[0])
