#!/usr/bin/env python3
"""Run #7957's four original CPU tests through unchanged AST-isolated classes/functions.

Python 3.12+, torch, numpy, tensordict, OmegaConf, pytest; fixed local Git objects.
Original DataProto uses its numpy serialization path, never device discovery.
No native verl import, Ray, TransferQueue, trainer, worker or model execution.
"""

import argparse
import ast
import __future__
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from typing import Any, Callable, Optional

BASE = "23a341e361e18063b4ec53e1f7cc4eae68f102cf"
HEAD = "1eb80e2212f24a5b71ca113c3e42797734961e7d"
MAIN = "a4b9354968705a441fb7da7367924b15b902e771"
TEST = "tests/trainer/test_pf_ppo_v1_on_cpu.py"
REVISIONS = {"base": BASE, "head": HEAD, "main": MAIN}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    parser.add_argument("--case", choices=REVISIONS)
    args = parser.parse_args()
    if not args.case:
        for case in REVISIONS:
            subprocess.run([sys.executable, "-B", __file__, str(args.source_repo), "--case", case], check=True)
        return
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ["VERL_DATAPROTO_SERIALIZATION_METHOD"] = "numpy"
    sys.dont_write_bytecode = True
    import numpy as np
    import torch
    import tensordict
    from tensordict import TensorDict
    import omegaconf
    import pytest
    assert torch.get_default_device().type == "cpu"

    env = dict(os.environ, GIT_NO_LAZY_FETCH="1")

    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(args.source_repo.resolve()), *arguments], env=env)

    revision = REVISIONS[args.case]
    print("case", args.case, revision, flush=True)
    print("Python", sys.version.split()[0], "torch", torch.__version__, "numpy", np.__version__,
          "tensordict", tensordict.__version__, "OmegaConf", omegaconf.__version__,
          "pytest", pytest.__version__, flush=True)

    def load(name, path, names, extra=None):
        source = git("show", revision + ":" + path)
        tree = ast.parse(source, path)
        nodes = []
        found = set()
        for node in tree.body:
            ids = {node.name} if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else set()
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                ids = {node.target.id}
            if ids & names:
                nodes.append(node)
                found.update(ids & names)
        assert found == names, (path, found, names)
        module = types.ModuleType(name)
        module.__dict__.update(dict(torch=torch, np=np, tensordict=tensordict, TensorDict=TensorDict,
                                    dataclass=dataclass, field=field, Enum=Enum, os=os,
                                    Any=Any, Optional=Optional, Callable=Callable))
        module.__dict__.update(extra or {})
        sys.modules[name] = module
        exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec",
                     flags=__future__.annotations.compiler_flag), module.__dict__)
        print(path, "SHA256", hashlib.sha256(source).hexdigest(), flush=True)
        return module

    protocol = load("verl.protocol", "verl/protocol.py", {
        "DataProto", "serialize_single_tensor", "serialize_tensordict",
        "deserialize_single_tensor", "deserialize_tensordict",
    })
    functional = load("verl.utils.torch_functional", "verl/utils/torch_functional.py",
                      {"masked_sum", "masked_mean", "masked_var", "masked_whiten"})
    core = load("verl.trainer.ppo.core_algos", "verl/trainer/ppo/core_algos.py", {
        "AdvantageEstimator", "ADV_ESTIMATOR_REGISTRY", "register_adv_est",
        "compute_gae_advantage_return", "compute_pf_ppo_reweight_data",
    }, {"verl_F": functional})
    trainer = load("verl.trainer.ppo.ray_trainer", "verl/trainer/ppo/ray_trainer.py",
                   {"compute_advantage"}, {"core_algos": core, "AdvantageEstimator": core.AdvantageEstimator})
    load("verl.trainer.ppo.v1.utils", "verl/trainer/ppo/v1/utils.py",
         {"compute_advantage_for_multi_trajectories"},
         {"core_algos": core, "compute_advantage": trainer.compute_advantage})

    class Reports:
        passed = 0
        xfailed = 0
        test_module = None

        def pytest_collection_modifyitems(self, items):
            self.test_module = items[0].module
            if args.case != "head":
                for item in items:
                    if item.name == "test_use_pf_ppo_is_rejected_on_v1_advantage_path":
                        item.add_marker(pytest.mark.xfail(strict=True, raises=pytest.fail.Exception,
                                                        reason="fixed version lacks the rejection"))

        def pytest_runtest_logreport(self, report):
            if report.when == "call":
                self.passed += int(report.passed)
                self.xfailed += int(report.skipped and hasattr(report, "wasxfail"))

    tests = git("show", HEAD + ":" + TEST)
    print("original test SHA256", hashlib.sha256(tests).hexdigest(), flush=True)
    report = Reports()
    with tempfile.TemporaryDirectory(prefix="perpetual-pf-ppo-") as directory:
        path = Path(directory) / Path(TEST).name
        path.write_bytes(tests)
        code = pytest.main([str(path), "-q", "-p", "no:cacheprovider"], plugins=[report])
    expected = (4, 0) if args.case == "head" else (3, 1)
    assert code == 0 and (report.passed, report.xfailed) == expected
    if args.case == "main":
        padding = load("pf_ppo_original_padding", "verl/workers/utils/padding.py", {"response_to_nested"})
        data = report.test_module._batch()
        b, t = data.batch.batch_size[0], report.test_module.RESPONSE_LEN
        data.batch["responses"] = torch.arange(b).unsqueeze(1).repeat(1, t)
        lengths = torch.arange(b) % t + 1
        data.batch["response_mask"] = torch.arange(t).unsqueeze(0) < lengths.unsqueeze(1)
        data.meta_info = {"row_ids": list(range(b)), "constant": "same"}
        before = data.batch.clone()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            out = core.compute_pf_ppo_reweight_data(data, "pow", 2.0)
        indices = torch.tensor([int(uid.removeprefix("uid")) for uid in out.non_tensor_batch["uid"]])
        for key in before.keys():
            assert torch.equal(out.batch[key], before[key][indices])
            assert torch.equal(data.batch[key], before[key])
        assert out.meta_info["row_ids"] == indices.tolist()
        assert out.meta_info["constant"] == "same" and data.meta_info["row_ids"] == list(range(b))
        assert len(set(indices.tolist())) < b
        old_mask = torch.nested.as_nested_tensor(
            [before["response_mask"][i, :lengths[i]] for i in range(b)], layout=torch.jagged)
        wrong = padding.response_to_nested(out.batch["responses"], old_mask)
        right = padding.response_to_nested(out.batch["responses"], torch.nested.as_nested_tensor(
            [out.batch["response_mask"][i, :lengths[indices[i]]] for i in range(b)], layout=torch.jagged))
        assert wrong.offsets().diff().tolist() == lengths.tolist()
        assert right.offsets().diff().tolist() == lengths[indices].tolist()
        assert not torch.equal(wrong.offsets(), right.offsets())
        print("Additional CPU row/length control: source rows", indices.tolist(),
              "old lengths", lengths.tolist(), "source lengths", lengths[indices].tolist(), flush=True)
    assert not any(name.split(".")[0] in {"ray", "transfer_queue", "vllm", "sglang"} for name in sys.modules)
    print("Original tests and stated CPU control only; no native package import or TransferQueue execution.", flush=True)


if __name__ == "__main__":
    main()
