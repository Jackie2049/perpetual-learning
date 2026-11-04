#!/usr/bin/env python3
"""Offline cross-check of the rollout-bound A/B parsers against *real* logs.

Usage: python3 crosscheck-real-logs.py    -> exit 0 iff every check passes

Why: the A/B tooling is only exercised on synthetic fixtures until a GPU window
opens, but the two things it assumes about real console output -- the
`np.float64(...)` value reprs and the exact step-line shape -- can be checked
today against the logs of the *previous* separate_async experiment.  Those logs
came from the same verl mode, the same host and the same launch script family;
only the load differs (n=4 / resp<=256 / sync_step=4 instead of n=16 /
resp<=1024 / sync_step=1), so they prove the parsers and config extraction,
not the rollout-bound regime.

Three independent parsers are compared:
  verify-bound.py         splits on " - " and fullmatches np.float64(...)
  make-summary-bound.py   one regex over the whole line
  summary.json            produced earlier by make-summary-separate.py
A green run means all three agree on every steady-window mean, so a mismatch
in the window is data -- not the tooling.
"""
import importlib.util
import json
import pathlib
import re
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
SEP = HERE.parent / "verl-v1-poll-quantum-separate"
RUNS = ["sa20a", "sa01a", "sa20b", "sa01b"]
WINDOW = range(3, 9)
KEYS = ["timing_s/gen", "timing_s/step", "timing_s/old_log_prob",
        "timing_s/update_actor", "timing_s/update_weights", "response_length/mean"]
fails = []


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'} {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


def window_mean(steps, key):
    vals = [steps[s][key] for s in sorted(steps) if s in WINDOW and key in steps[s]]
    return statistics.fmean(vals) if len(vals) == len(list(WINDOW)) else None


def main():
    vb = load("vb", HERE / "verify-bound.py")
    ms = load("ms", HERE / "make-summary-bound.py")
    ref = json.loads((SEP / "summary.json").read_text())

    for run in RUNS:
        path = SEP / "raw" / f"{run}.txt"
        text = path.read_text(errors="ignore")
        cfg, steps_v = vb.parse(path)
        resolved, logged, steps_m = ms.parse(path)
        want_poll = 2.0 if "20" in run else 0.1
        check(f"{run}: both parsers see 8 steps",
              sorted(steps_v) == list(range(1, 9)) and sorted(steps_m) == list(range(1, 9)),
              f"verify={sorted(steps_v)} summary={sorted(steps_m)}")
        check(f"{run}: one step line per step (no console contamination)",
              logged == 8 and len(steps_m) == 8, f"logged={logged}")
        check(f"{run}: config extraction from the real dump",
              cfg.get("trainer_mode") == "separate_async"
              and cfg.get("poll_interval") == want_poll
              and cfg.get("parameter_sync_step") == 4
              and cfg.get("max_response_length") == 256,
              str({k: cfg.get(k) for k in ("trainer_mode", "poll_interval",
                                           "parameter_sync_step", "max_response_length")}))
        check(f"{run}: resolved poll_interval={want_poll}", resolved == want_poll, str(resolved))
        n_repr = len(re.findall(r"np\.(?:float64|int64)\(", text))
        for key in KEYS:
            va, ma = window_mean(steps_v, key), window_mean(steps_m, key)
            ra = ref["runs"][run]["stats"][key]["mean"]
            ok = va is not None and ma is not None and abs(va - ma) < 1e-9 and abs(va - ra) < 1e-9
            check(f"{run}: {key} mean agrees 3-way", ok,
                  f"verify={va} summary={ma} ref={ra}")
        print(f"INFO {run}: {n_repr} np.float64/np.int64 reprs in this log")

    print(f"\n{len(fails)} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
