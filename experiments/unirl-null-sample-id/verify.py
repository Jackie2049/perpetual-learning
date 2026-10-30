"""UniRL #637: original Dataset/normalizer and isolated guards/eval RNG on CPU.

Run: python -B verify.py /path/to/UniRL
Reads fixed local Git blobs with lazy fetch disabled. AST selection preserves
whole definitions, but skips framework imports. Guard receivers and FlowMatchSFT
configuration are SimpleNamespace fixtures, not native Part/Sample/model objects.
No trainer, distributed service, weights, media decode or GPU access.
Original definitions: Tencent-Hunyuan/UniRL, Apache-2.0.
"""

import __future__
import ast
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from types import SimpleNamespace

import torch

BASE = "b8e9740e981a68a5556fbeb8225558040a4d47a4"
HEAD = "62514b42afbcee2672028c205c82554f4d160bf0"
hashes = {}


def source(rev, path):
    raw = subprocess.check_output(["git", "show", rev + ":" + path], cwd=sys.argv[1],
                                  env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
    hashes[rev + ":" + path] = hashlib.sha256(raw).hexdigest()
    return ast.parse(raw)


def load(tree, names, ns):
    nodes = [n for n in tree.body if getattr(n, "name", None) in names or
             isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    assert len(nodes) == len(names), names
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "original-definitions", "exec",
                 flags=__future__.annotations.compiler_flag), ns)


def method(tree, cls, name, ns):
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    load(original, {name}, ns)
    return ns[name]


def main():
    torch.set_num_threads(1)
    ns = dict(os=os, json=json, logger=logging.getLogger("null-id-control"), dataclass=dataclass,
              torch=torch, hashlib=hashlib)
    media = source(HEAD, "unirl/types/media.py")
    load(media, {"SUPPORTED_MEDIA_MODALITIES", "SUPPORTED_MEDIA_ROLES", "normalize_media_uri", "MediaRef"}, ns)
    datasets = source(HEAD, "unirl/data/datasets.py")
    load(datasets, {"_LEGACY_EMBEDDING_FIELDS", "_normalize_media_refs", "_resolve_media_uri"}, ns)
    sft = source(HEAD, "unirl/algorithms/sft.py")
    load(sft, {"flow_shift_sigma", "draw_shifted_sigma", "sample_eval_seed"}, ns)
    draw_sigma = method(sft, "FlowMatchSFT", "_draw_sigma", ns)
    eval_draws = method(sft, "FlowMatchSFT", "_eval_draws", ns)
    sampler = SimpleNamespace(eval_seed=42, timestep_sampling="logit_normal", logit_mean=0.,
                              logit_std=1., timestep_shift=1., sigma_min=1e-4)
    sampler._draw_sigma = lambda *args: draw_sigma(sampler, *args)
    sample = source(HEAD, "unirl/types/sample.py")
    part_guard = method(sample, "Part", "__post_init__", ns)
    sample_guard = method(sample, "Sample", "__post_init__", ns)
    cases = [({}, "fallback"), ({"sample_id": None}, "fallback"),
             ({"prompt_id": None}, "fallback"), ({"sample_id": None, "prompt_id": "p7"}, "p7"),
             ({"sample_id": None, "prompt_id": None}, "fallback"),
             ({"sample_id": "s1", "prompt_id": "p7"}, "s1"),
             ({"prompt_id": "p7"}, "p7"), ({"sample_id": 42}, "42"),
             ({"sample_id": ""}, ""), ({"sample_id": 0}, "0"), ({"sample_id": False}, "False")]
    rows = [{"prompt": "different row " + str(i), "sample_id": None} for i in range(3)]
    rows += [{"prompt": "fourth", "sample_id": "explicit"}, {"prompt": "fifth"}]
    results = {}
    with tempfile.TemporaryDirectory(prefix="unirl-null-id-") as directory:
        path = Path(directory) / "manifest.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        for label, rev in [("base", BASE), ("head", HEAD)]:
            load(source(rev, "unirl/data/sft.py"), {"_SUPERVISED_EXCLUDED_KEYS", "_normalize_message_content",
                 "normalize_supervised_example", "SupervisedDataset"}, ns)
            direct = [ns["normalize_supervised_example"]({"prompt": "p", **kw}, default_sample_id="fallback")["sample_id"]
                      for kw, _ in cases]
            ids = [r["sample_id"] for r in ns["SupervisedDataset"](str(path)).records]
            assert ids == (["None"] * 3 + ["explicit", "manifest.jsonl:4"] if label == "base" else
                           ["manifest.jsonl:" + str(i) for i in range(3)] + ["explicit", "manifest.jsonl:4"])
            p = SimpleNamespace(sample_ids=ids, primitives={}, primitive_metadata={}, init_noise_group_ids=[], is_root=True)
            part_guard(p)  # Original guard has no uniqueness check on this empty-payload fixture.
            error = None
            try:
                sample_guard(SimpleNamespace(parts=[p]))
            except ValueError as exc:
                error = str(exc)
            assert (error is not None) == (label == "base")
            sigma, noise = eval_draws(sampler, torch.zeros((5, 2, 3)), ids)
            pair_equal = torch.equal(sigma[0], sigma[1]) and torch.equal(noise[0], noise[1])
            assert pair_equal == (label == "base")
            # Changing microbatch partition must not change draws for the same IDs/shapes.
            split = [eval_draws(sampler, torch.zeros((len(chunk), 2, 3)), chunk) for chunk in [ids[:2], ids[2:]]]
            assert torch.equal(sigma, torch.cat([x[0] for x in split]))
            assert torch.equal(noise, torch.cat([x[1] for x in split]))
            results[label] = dict(direct_ids=direct, manifest_ids=ids, unique_ids=len(set(ids)),
                                 isolated_part_guard="accepted", isolated_sample_guard_error=error,
                                 first_two_draws_identical=pair_equal, split_batch_draws_identical=True,
                                 seeds=[str(ns["sample_eval_seed"](42, key)) for key in ids],
                                 sigma=sigma.tolist(), noise_first_component=noise[:, 0, 0].tolist())
    assert results["head"]["direct_ids"] == [expected for _, expected in cases]
    assert [i for i, (a, b) in enumerate(zip(results["base"]["direct_ids"], results["head"]["direct_ids"])) if a != b] == [1, 2, 3, 4]
    print(json.dumps(dict(environment=dict(python=platform.python_version(), torch=torch.__version__, device="cpu"),
                         source_sha256=hashes, direct_case_count=len(cases), results=results,
                         scope="Original definitions with AST isolation; fixture guards/config, no native Part/Sample construction or training/eval model loss"),
                     indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
