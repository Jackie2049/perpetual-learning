"""Original TQ tensor-list packing + verl retag, with tiny real CPU tensors.

AST-extracts complete original methods, not a TQ service or TensorDict integration.
The verl method receives a dict adapter; annotations are deferred. No fallback
container is mocked: an unexpected NonTensorStack branch fails instead.
The explicit native builder is a local semantic reference, not a repair patch.
Inputs: TransferQueue (Apache-2.0) fixed full simple_storage_manager.py and cached
verl Git objects. No network/device discovery/queue construction/model execution.
"""

import __future__
import argparse
import ast
import copy
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import subprocess

import torch


VERL = "a08ea02a784d7d4982b1d81df36d188a476192bb"
TQ = "5c15a2c8d65f05a52514830da6de4c5ec4336c1c"
TQ_BLOB = "d14e67c3c2f38bd4e829b78bac354d20924a454a"
UTIL = "verl/utils/tensordict_utils.py"


def definition(raw, name, namespace):
    tree = ast.parse(raw)
    nodes = [copy.deepcopy(n) for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1
    nodes[0].decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 "selected_original_definition", "exec", flags=__future__.annotations.compiler_flag), namespace)


def layout(nt):
    return {"ragged_idx": nt._ragged_idx, "values_shape": list(nt.values().shape),
            "offsets": nt.offsets().tolist(),
            "public_shape": [s if isinstance(s, int) else "j" for s in nt.shape]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verl_repo", type=Path)
    parser.add_argument("tq_module", type=Path)
    args = parser.parse_args()
    tq_raw = args.tq_module.read_bytes()
    assert hashlib.sha1(b"blob " + str(len(tq_raw)).encode() + b"\0" + tq_raw).hexdigest() == TQ_BLOB
    verl_raw = subprocess.check_output(["git", "show", VERL + ":" + UTIL], cwd=args.verl_repo,
                                     env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
    namespace = {"torch": torch, "logger": logging.getLogger("original_tq_packer")}
    definition(tq_raw, "_pack_field_values", namespace)
    definition(verl_raw, "maybe_fix_3d_position_ids", namespace)
    torch.set_num_threads(1)
    results = {}
    for name, lengths in [("equal", [7, 7, 7]), ("unequal", [5, 7, 9]),
                          ("single", [7]), ("square", [8, 8])]:
        samples = [torch.arange(i * 1000, i * 1000 + 4 * length, device="cpu").reshape(4, length)
                   for i, length in enumerate(lengths)]
        nt = namespace["_pack_field_values"](samples)
        before = layout(nt)
        assert all(torch.equal(x, y) for x, y in zip(nt.unbind(), samples, strict=True))
        data = {"position_ids": nt}  # data-only adapter, not real TensorDict
        namespace["maybe_fix_3d_position_ids"](data)
        after = layout(nt)
        try:
            unbound = nt.unbind()
        except RuntimeError as error:
            outcome = {"status": "RuntimeError", "message": str(error)}
        else:
            matches = len(unbound) == len(samples) and all(torch.equal(x, y) for x, y in zip(unbound, samples))
            outcome = {"status": "correct" if matches else "wrong_samples",
                       "sample_shapes": [list(x.shape) for x in unbound], "exact_samples_match": matches}
        expected = {"equal": "RuntimeError", "single": "RuntimeError",
                    "square": "wrong_samples", "unequal": "correct"}[name]
        assert outcome["status"] == expected, outcome
        offsets = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="cpu")
        reference = torch.nested.nested_tensor_from_jagged(
            torch.cat(samples, dim=-1), offsets=offsets, jagged_dim=2)
        namespace["maybe_fix_3d_position_ids"]({"position_ids": reference})
        assert all(torch.equal(x, y) for x, y in zip(reference.unbind(), samples, strict=True))
        results[name] = {"lengths": lengths, "components": 4, "before_retag": before,
                         "original_layout_roundtrip": True, "after_retag": after,
                         "retag_outcome": outcome, "explicit_reference_layout": layout(reference),
                         "explicit_reference_roundtrip": True}
    print(json.dumps({"verl_revision": VERL, "tq_revision": TQ, "tq_blob": TQ_BLOB,
        "source_sha256": {"tq_module": hashlib.sha256(tq_raw).hexdigest(),
                          "verl_tensordict_utils": hashlib.sha256(verl_raw).hexdigest()},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": "cpu", "threads": 1}, "cases": results,
        "scope": {"tq_method": "original _pack_field_values pure tensor branch; no fallback mocks",
            "verl_input": "plain dict adapter, not TensorDict",
            "reference": "native explicit jagged_dim=2 construction, not candidate #7901",
            "not_run": ["TQ service", "_merge_tensors_to_tensordict", "pickle", "microbatch/trainer",
                        "#7901 tests", "community 63-case suite", "GPU/NPU", "training"]}},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
