#!/usr/bin/env python3
"""Execute pinned original index schedules on CPU; no UniRL package or model import."""

import argparse
import ast
from dataclasses import replace
import hashlib
from itertools import groupby
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import numpy
import yaml

REV = "6fd82cba13f52f394747832344039408699ea378"
SOURCE = "unirl/sde/index_schedule.py"
RECIPES = [
    "examples/diffusion/sd3/sd3_mixgrpo.yaml",
    "examples/diffusion/wan21/wan21_t2v_mixgrpo.yaml",
    "examples/diffusion/wan22/wan22_t2v_14b_mixgrpo.yaml",
    "examples/diffusion/qwen_image/qwen_image_mixgrpo.yaml",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("unirl", type=Path, help="read-only Git checkout with pinned blobs")
    args = parser.parse_args()
    hashes = {}

    def blob(path):
        raw = subprocess.check_output(["git", "show", f"{REV}:{path}"], cwd=args.unirl,
                                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        hashes[path] = hashlib.sha256(raw).hexdigest()
        return raw

    raw = blob(SOURCE)
    module = ModuleType("pinned_original_index_schedule")
    sys.modules[module.__name__] = module
    exec(compile(ast.parse(raw, filename=SOURCE), SOURCE, "exec"), module.__dict__)

    def sweep(timesteps, config):
        scheduler = module.WindowScheduler(timesteps, config)
        first = tuple(sorted(scheduler.get_sde_indices(0)))
        values = [first]
        for step in range(1, 1000):
            current = tuple(sorted(scheduler.get_sde_indices(step)))
            if current == first and values[-1] != first:
                break
            values.append(current)
        else:
            raise AssertionError("No complete sweep observed within this bounded control")
        runs = [[list(window), len(list(items))] for window, items in groupby(values)]
        period = len(values)
        for step in (period + 1, 0, period - 1, 1, period):
            assert tuple(sorted(scheduler.get_sde_indices(step))) == values[step % period]
        stopped = module.WindowScheduler(timesteps, replace(config, roll_back=False))
        assert tuple(sorted(stopped.get_sde_indices(period + 20))) == values[-1]
        assert all(len(window) == config.window_size and min(window) >= 0
                   and max(window) < timesteps for window in values)
        return {"runs": runs, "period": period, "wrap_window": list(first),
                "stop_window": list(values[-1])}

    config = module.WindowConfig(strategy="progressive", window_size=4, overlap_size=1,
                                 iters_per_window=25, max_iters_per_window=10,
                                 min_iters_per_window=1, roll_back=True)
    controls = {
        "progressive": sweep(10, config),
        "decay": sweep(10, replace(config, strategy="decay")),
        "exp_threshold_3": sweep(10, replace(config, strategy="exp_decay", exp_decay_threshold=3)),
        "exp_default_13": sweep(10, replace(config, strategy="exp_decay")),
    }
    expected = {"progressive": [25, 25, 25], "decay": [10, 7, 4],
                "exp_threshold_3": [25, 25, 19], "exp_default_13": [25, 25, 25]}
    for name, counts in expected.items():
        assert [count for _, count in controls[name]["runs"]] == counts

    recipes = []
    for path in RECIPES:
        parsed = yaml.safe_load(blob(path))
        sampling = parsed["sampling"]
        timesteps = sampling["num_inference_steps"]
        fields = dict(sampling["scheduler"]["config"])
        fields.pop("_target_")
        assert isinstance(timesteps, int) and fields["strategy"] == "progressive"
        actual = module.WindowConfig(**fields)
        exp = replace(actual, strategy="exp_decay")
        recipes.append({"path": path, "num_inference_steps": timesteps,
                        "configured_strategy": actual.strategy,
                        "actual": sweep(timesteps, actual),
                        "counterfactual_exp_default": sweep(timesteps, exp)})

    print(json.dumps({
        "revision": REV, "python": sys.version.split()[0],
        "numpy": numpy.__version__, "pyyaml": yaml.__version__,
        "controls_T10_W4_overlap1": controls, "recipes": recipes,
        "source_sha256": hashes,
        "scope": "original scheduler module and explicit YAML scalars; no Hydra/engine/SDE/model/reference parity"
    }, indent=2))


if __name__ == "__main__":
    main()
