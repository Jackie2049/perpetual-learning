"""Pinned UniRL SDE constructor controls; no engine, model or device imports.

Run with Python 3.12 and NumPy: python -B verify_bounds.py /path/to/UniRL
Requires local Git blobs; disables implicit fetch. Executes complete original
modules. Package shells only isolate unrelated UniRL root initialization.
"""

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import numpy

BASE = "8cfbac17765004afb68fef08a4e011a4c8daf555"
MAIN = "5d1da78d4d87593b41a4235a65e9ad9dc3f9b6f8"
WINDOW = "d31017aea1c91f881043d896637b7c3a6db4bc6b"
SAMPLING = "523d245e61dec961b714ba96ef9663ab877d36d4"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    args = parser.parse_args()
    hashes = {}

    def source(rev, path):
        raw = subprocess.check_output(
            ["git", "show", f"{rev}:{path}"], cwd=args.repo,
            env=dict(os.environ, GIT_NO_LAZY_FETCH="1"),
        )
        hashes[f"{rev}:{path}"] = hashlib.sha256(raw).hexdigest()
        return raw

    def module(name, rev, path):
        result = ModuleType(name)
        sys.modules[name] = result
        exec(compile(source(rev, path), f"{rev}:{path}", "exec"), result.__dict__)
        return result

    for name in ("unirl", "unirl.config"):
        shell = ModuleType(name)
        shell.__path__ = []
        sys.modules[name] = shell
    module("unirl.config.require", BASE, "unirl/config/require.py")
    old_window = module("bounds_old_window", BASE, "unirl/sde/index_schedule.py")
    new_window = module("bounds_new_window", WINDOW, "unirl/sde/index_schedule.py")
    old_sampling = module("bounds_old_sampling", BASE, "unirl/types/sampling.py")
    new_sampling = module("bounds_new_sampling", SAMPLING, "unirl/types/sampling.py")
    for path in ("unirl/sde/index_schedule.py", "unirl/types/sampling.py"):
        assert source(MAIN, path) == source(BASE, path)

    rows = []

    def check(name, make, expected_old, expected_new):
        values = []
        for candidate in (False, True):
            try:
                values.append(make(candidate))
            except ValueError:
                values.append("ValueError")
        assert values == [expected_old, expected_new], (name, values)
        rows.append({"case": name, "base": values[0], "candidate": values[1]})

    def window(candidate, total, strategy, size, start):
        m = new_window if candidate else old_window
        return sorted(m.WindowScheduler(total, m.WindowConfig(
            strategy=strategy, window_size=size, init_timestep=start,
        )).get_sde_indices(0))

    check("633 partial upper overflow", lambda c: window(c, 10, "progressive", 4, 9),
          [9, 10, 11, 12], "ValueError")
    check("633 oversized random window", lambda c: window(c, 1, "random", 4, 0),
          [0, 1, 2, 3], "ValueError")
    check("633 negative start remains accepted", lambda c: window(c, 10, "progressive", 4, -1),
          [-1, 0, 1, 2], [-1, 0, 1, 2])
    check("633 all ignores window geometry", lambda c: window(c, 10, "all", 100, 100),
          list(range(10)), list(range(10)))

    def sampling(candidate, **kwargs):
        m = new_sampling if candidate else old_sampling
        return m.DiffusionSamplingParams(**kwargs)

    for name, indices in (("past end", [10, 11]), ("negative", [-1])):
        check(f"634 static {name}", lambda c, x=indices: sampling(
            c, num_inference_steps=10, sde_indices=x).get_sde_indices(0), indices, "ValueError")
    check("634 string normalization preserved", lambda c: sampling(
        c, num_inference_steps=10, sde_indices=["3"]).get_sde_indices(0), [3], [3])
    check("634 empty override with oversized dormant scheduler", lambda c: sampling(
        c, num_inference_steps=10, sde_indices=[], scheduler=old_window.AllSDEScheduler(20)
    ).get_sde_indices(0), [], "ValueError")
    check("634 oversized scheduler with in-range filtered pool", lambda c: sampling(
        c, num_inference_steps=10, scheduler=old_window.AllSDEScheduler(20, timestep_fraction=0.5)
    ).get_sde_indices(0), list(range(10)), "ValueError")
    check("634 does not validate returned window indices", lambda c: sampling(
        c, num_inference_steps=10, scheduler=old_window.WindowScheduler(
            10, old_window.WindowConfig(strategy="progressive", window_size=4, init_timestep=9)
        )).get_sde_indices(0), [9, 10, 11, 12], [9, 10, 11, 12])
    check("634 shorter scheduler prefix", lambda c: sampling(
        c, num_inference_steps=10, scheduler=old_window.AllSDEScheduler(6)
    ).get_sde_indices(0), list(range(6)), list(range(6)))
    check("634 replace unrelated height", lambda c: replace(sampling(
        c, num_inference_steps=10, sde_indices=[9]), height=512).get_sde_indices(0), [9], [9])
    check("634 replace shrunk steps", lambda c: replace(sampling(
        c, num_inference_steps=10, sde_indices=[9]), num_inference_steps=8
    ).get_sde_indices(0), [9], "ValueError")
    check("634 replace cleared eval-like fields", lambda c: replace(sampling(
        c, num_inference_steps=10, sde_indices=[9]), num_inference_steps=8,
        eta=0.0, sde_indices=[], scheduler=None).get_sde_indices(0), [], [])

    assert not any(n.split(".")[0] in {"torch", "ray", "vllm", "sglang"} for n in sys.modules)
    print(json.dumps({"python": sys.version.split()[0], "numpy": numpy.__version__,
                      "cases": rows, "source_sha256": hashes,
                      "scope": "14 original constructor/selection controls, not engine execution or upstream tests"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
