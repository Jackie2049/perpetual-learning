"""Fixed #8355 local safetensors I/O controls, not a distributed engine test.

Original exporter and filesystem module are complete. Two original base classes
are AST-loaded to avoid Ray/worker imports. Only get_device_name is pinned to
cpu. send_weights, manager, collectives, FSDP and shared mounts are NOT executed.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from abc import ABC, abstractmethod
from typing import Any, Generator

import safetensors
import torch
from safetensors import safe_open

HEAD = "7d981cde954fce079d8c973379738d6e8c258ac2"
BASE = "a4b9354968705a441fb7da7367924b15b902e771"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", help="Local verl Git object repository")
    args = parser.parse_args()
    env = dict(os.environ, GIT_NO_LAZY_FETCH="1")

    def git(*argv):
        return subprocess.check_output(["git", "-C", args.repo, *argv], env=env).decode()

    def source(rev, path):
        return git("show", f"{rev}:{path}")

    ancestor = git("merge-base", BASE, HEAD).strip()
    hashes = {}

    def original(name, path):
        code = source(HEAD, path)
        hashes[path] = hashlib.sha256(code.encode()).hexdigest()
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        exec(compile(code, f"{HEAD}:{path}", "exec"), mod.__dict__)
        return mod

    for name in ("verl", "verl.checkpoint_engine", "verl.utils", "verl.utils.checkpoint"):
        shell = types.ModuleType(name)
        shell.__path__ = []
        sys.modules[name] = shell
    base_path = "verl/checkpoint_engine/base.py"
    base_code = source(HEAD, base_path)
    hashes[base_path] = hashlib.sha256(base_code.encode()).hexdigest()
    selected = [node for node in ast.parse(base_code).body
                if isinstance(node, ast.ClassDef)
                and node.name in {"CheckpointEngineRegistry", "CheckpointEngine"}]
    assert len(selected) == 2
    base_mod = types.ModuleType("verl.checkpoint_engine.base")
    base_mod.__dict__.update(ABC=ABC, abstractmethod=abstractmethod, Any=Any, Generator=Generator, torch=torch)
    sys.modules[base_mod.__name__] = base_mod
    exec(compile(ast.Module(body=selected, type_ignores=[]), base_path, "exec"), base_mod.__dict__)
    device_mod = types.ModuleType("verl.utils.device")
    device_mod.get_device_name = lambda: "cpu"
    sys.modules[device_mod.__name__] = device_mod
    helper = original("verl.utils.checkpoint.hf_safetensors", "verl/utils/checkpoint/hf_safetensors.py")
    engine_mod = original("verl.checkpoint_engine.filesystem_checkpoint_engine",
                          "verl/checkpoint_engine/filesystem_checkpoint_engine.py")
    Engine = engine_mod.FileSystemCheckpointEngine

    compiled = 0
    for status_line in git("diff", "--name-status", f"{BASE}...{HEAD}").splitlines():
        status, path = status_line.split("\t")
        if path.endswith(".py"):
            revisions = (HEAD,) if status == "A" else (ancestor, HEAD)
            for revision in revisions:
                compile(source(revision, path), f"{revision}:{path}", "exec")
                compiled += 1

    # Each caller consumes its own stream. Local sequential invocations are
    # NOT ranks/processes and do not validate actual collective ordering.
    def stream(consumed):
        buffer = torch.empty(4)
        for index in range(5):
            buffer.fill_(index + 1)
            consumed.append(f"p{index}")
            yield f"p{index}", buffer
        consumed.append("large")
        yield "large", torch.arange(20, dtype=torch.float32)
        consumed.append("counter")
        yield "counter", torch.tensor([7, 9], dtype=torch.int64)
        consumed.append("alias")
        yield "alias", torch.ones(20)
        buffer.fill_(999)

    expected = {f"p{i}": torch.full((4,), float(i + 1), dtype=torch.float32 if i == 4 else torch.bfloat16)
                for i in range(5)}
    expected.update(large=torch.arange(20, dtype=torch.bfloat16), counter=torch.tensor([7, 9]))

    def export(path, rank=0, writers=1):
        consumed = []
        result = helper.export_hf_safetensors(
            stream(consumed), str(path), max_shard_bytes=24, rank=rank, num_writers=writers,
            exclude_names={"alias"}, target_dtype=torch.bfloat16, preserve_dtype_names={"p4"},
        )
        assert len(consumed) == 8  # even non-owned/excluded entries are drained
        index = json.loads((path / helper.HF_WEIGHTS_INDEX_NAME).read_text())
        assert result == (104, 4, 7)
        assert index["metadata"]["total_size"] == 104
        return index

    async def collect(engine, step):
        return {name: tensor async for name, tensor in engine.receive_weights(step)}

    with tempfile.TemporaryDirectory(prefix="pl-filesystem-export-") as scratch:
        root = Path(scratch)
        single = root / "single"
        single.mkdir()
        index = export(single)
        assert len(list(single.glob("*.safetensors"))) == 4
        expected_files = sorted(set(index["weight_map"].values()))
        payloads = []
        for file_name in expected_files:
            with safe_open(single / file_name, framework="pt") as opened:
                payloads.append(sum(opened.get_tensor(key).nbytes for key in opened.keys()))
                for key in opened.keys():
                    assert torch.equal(opened.get_tensor(key), expected[key])
                    assert opened.get_tensor(key).dtype == expected[key].dtype
        assert payloads == [24, 24, 40, 16]  # a whole oversized tensor is not split

        destinations = []
        for rank in range(3):
            target = root / f"writer-{rank}"
            target.mkdir()
            assert export(target, rank, 3) == index
            actual = sorted(path.name for path in target.glob("*.safetensors"))
            assert actual == [name for i, name in enumerate(expected_files) if i % 3 == rank]
            destinations.append(target)

        mounted = root / "local-mounted-directory"
        version = mounted / "global_step_17"
        version.mkdir(parents=True)
        for destination in destinations:
            for file_path in destination.glob("*.safetensors"):
                engine_mod._copy_file(str(file_path), str(version / file_path.name))
        receiver = Engine(bucket_size=24, path=str(mounted))
        try:
            asyncio.run(collect(receiver, 17))
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Reader must not manufacture a missing index")
        engine_mod._copy_file(str(destinations[0] / helper.HF_WEIGHTS_INDEX_NAME),
                              str(version / helper.HF_WEIGHTS_INDEX_NAME))
        restored = asyncio.run(collect(receiver, 17))
        assert restored.keys() == expected.keys()
        for key in expected:
            assert torch.equal(restored[key], expected[key])
            assert restored[key].dtype == expected[key].dtype

        # Negative fixture: a readable index alone cannot certify shard presence.
        incomplete = mounted / "global_step_18"
        incomplete.mkdir()
        engine_mod._copy_file(str(single / helper.HF_WEIGHTS_INDEX_NAME),
                              str(incomplete / helper.HF_WEIGHTS_INDEX_NAME))
        try:
            asyncio.run(collect(receiver, 18))
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Index-only directory must not load successfully")

        # Direct finalize-method controls, not a manager completion test:
        # exported_dir is explicitly supplied because send_weights is not run.
        receiver.exported_dir = str(version)
        receiver.finalize()
        assert version.is_dir() and receiver.exported_dir == str(version)
        owner = Engine(bucket_size=24, path=str(mounted), is_master=True)
        owner.exported_dir = str(version)
        owner.finalize()
        assert not version.exists() and owner.exported_dir is None
        assert incomplete.is_dir()

        empty = root / "empty"
        empty.mkdir()
        assert helper.export_hf_safetensors([], str(empty), 24) == (0, 0, 0)
        assert json.loads((empty / helper.HF_WEIGHTS_INDEX_NAME).read_text())["weight_map"] == {}
        assert not list(empty.glob("*.safetensors"))
        try:
            Engine(bucket_size=24, path="hdfs://example/weights")
        except AssertionError:
            pass
        else:
            raise AssertionError("URI must be rejected; a mounted path is required")

    print(json.dumps(dict(head=HEAD, target_base=BASE, merge_base=ancestor,
                          torch=torch.__version__, safetensors=safetensors.__version__, device="cpu",
                          compiled_modules=compiled, total_payload=104, shard_payloads=payloads,
                          writer_shard_counts=[2, 1, 1], tensors_restored=len(restored),
                          sha256=hashes), indent=2))


if __name__ == "__main__":
    main()
