#!/usr/bin/env python3
"""Judge the critic-side three-arm probe from the arm JSON summaries alone (no GPU, no server).

This is the pre-registered verdict for `run-critic-pdbg.sh`. The rule was written before the
run exists, so the window cannot choose its own scoring after seeing the numbers.

What the run asks
-----------------
The static finding (`open-source/verl/evidence/precision-debugger-stage-producers`) is:
the whitelist accepts six stages, but `critic_update` and `compute_values` have no
`@DistProfiler.annotate(role=...)` anywhere -> configuring them collects nothing and says
nothing. `actor_update` does have decorators, so it is the positive control.

Validity (a run failing these proves nothing either way)
--------------------------------------------------------
V1  every arm exited 0
V2  every arm ran the full `--min-steps` (default 2) steps
V3  the critic really trained in the arms under test: `update_critic` appears in the log
    (v1 logs `timing_s/update_critic`), so "collected nothing" cannot mean "no critic"
V4  the tree carries the sdpa site patch: `flash_attention_2_left == 0` in the recorded
    model.py state. A run whose tree lost the patch dies at value-head build.

Pre-registered outcomes (gated)
-------------------------------
P1  positive control: the `au` arm wrote >= 1 file under `step_<n>/actor_update/`, i.e. the
    plumbing works on this tree and this msprobe install
P2  the `cu` arm wrote 0 dump files
P3  the `cv` arm wrote 0 dump files

Reported, not gated
-------------------
* `train_batch_warnings` per arm -- the static note says TrainingWorker.train_batch's role is
  not in the whitelist and warns on every call. With a real critic that should be > 0 in all
  three arms, including `au`, because the warning ignores the user's stage list. If it is 0,
  say so and do not quote the claim.
* per-arm dump inventory (path + size), log bytes, window timestamps.

Exit codes
----------
0  valid run, all pre-registered outcomes met
1  invalid run (V1-V4): nothing can be concluded; fix and re-run
2  refuted: a dump appeared where the static finding says there is no producer
4  --selftest failure

Usage
-----
    python3 verify-critic-pdbg.py --out <OUT>          # OUT holds arm-au.json / arm-cu.json / arm-cv.json
    python3 verify-critic-pdbg.py --selftest
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

ARMS = ("au", "cu", "cv")
STAGE_OF = {"au": "actor_update", "cu": "critic_update", "cv": "compute_values"}
EXPECTED_DUMP_NAMES = {"construct.json", "stack.json", "dump.json"}


def load_arms(out_dir, arms=ARMS):
    data, missing = {}, []
    for arm in arms:
        p = Path(out_dir) / f"arm-{arm}.json"
        if not p.is_file():
            missing.append(str(p))
            continue
        data[arm] = json.loads(p.read_text())
    return data, missing


def judge(data, missing, min_steps=2):
    checks, reported = [], {}

    def ck(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    ck("arms_present", not missing, f"missing: {missing}" if missing else "3/3")
    if missing:
        return checks, reported

    for arm in ARMS:
        d = data[arm]
        ck(f"V1:{arm}:exit0", d.get("exit_code") == 0, f"exit_code={d.get('exit_code')}")
        ck(f"V2:{arm}:steps", d.get("steps_seen", 0) >= min_steps,
           f"steps_seen={d.get('steps_seen')} (need >= {min_steps})")
        ck(f"V4:{arm}:sdpa_patch", (d.get("fa2") or {}).get("flash_attention_2_left") == 0,
           f"flash_attention_2_left={(d.get('fa2') or {}).get('flash_attention_2_left')}")
        reported[f"{arm}_train_batch_warnings"] = d.get("train_batch_warnings")
        reported[f"{arm}_unknown_stage_warnings"] = d.get("unknown_stage_warnings")

    for arm in ("cu", "cv"):
        ck(f"V3:{arm}:critic_ran", data[arm].get("update_critic_occurrences", 0) >= 1,
           f"update_critic_occurrences={data[arm].get('update_critic_occurrences')}")
    reported["au_update_critic_occurrences"] = data["au"].get("update_critic_occurrences")

    # --- gated outcomes -----------------------------------------------------------------
    au_files = [f for f in data["au"].get("dump_files", [])]
    au_stage_files = [f for f in au_files if f"/{STAGE_OF['au']}/" in f"/{f['path']}"]
    ck("P1:au_dump_present", len(au_stage_files) >= 1,
       f"{len(au_stage_files)} file(s) under step_*/actor_update/ of {len(au_files)} total")
    ck("P1:au_expected_names",
       EXPECTED_DUMP_NAMES <= {Path(f["path"]).name for f in au_stage_files},
       f"names={sorted({Path(f['path']).name for f in au_stage_files})}")

    for arm in ("cu", "cv"):
        files = data[arm].get("dump_files", [])
        ck(f"P{2 if arm == 'cu' else 3}:{arm}_no_dump", len(files) == 0,
           f"{len(files)} dump file(s)" + (f", e.g. {files[0]['path']}" if files else ""))
    return checks, reported


def report(checks, reported):
    failed = [c for c in checks if not c["ok"]]
    for c in checks:
        if not c["ok"]:
            print(f"  [FAIL] {c['check']}: {c['detail']}")
    for c in checks:
        if c["ok"]:
            print(f"  [ ok ] {c['check']}: {c['detail']}")
    print("  -- reported, not gated --")
    for k, v in sorted(reported.items()):
        print(f"  {k} = {v}")
    if any(c["check"].startswith(("P2:", "P3:")) for c in failed):
        print("[verdict] REFUTED: a dump appeared for a stage that has no decorator anywhere; "
              "re-check the staging->stage mapping and the tree before quoting anything")
        return 2
    if failed:
        print("[verdict] INVALID RUN: preconditions failed, the three arms prove nothing")
        return 1
    print("[verdict] VALID RUN, pre-registered outcome met: actor_update dumps, "
          "critic_update/compute_values write nothing")
    return 0


def clean_arm(arm, out_root=None, **over):
    stage = STAGE_OF[arm]
    dumps = []
    if arm == "au":
        base = f"step_1/{stage}/step0/rank0"
        dumps = [{"path": f"{base}/construct.json", "size": 2},
                 {"path": f"{base}/stack.json", "size": 851814},
                 {"path": f"{base}/dump.json", "size": 12811506}]
    d = {
        "arm": arm, "stages": f"[{stage}]", "src": "/tmp/tree", "gpu": "4",
        "prof": f"/tmp/out/prof-{arm}", "log": f"/tmp/out/run-{arm}.log",
        "log_bytes": 4096, "start_utc": "2026-11-05T00:00:00Z", "end_utc": "2026-11-05T00:10:00Z",
        "exit_code": 0, "steps_seen": 2, "update_critic_occurrences": 4,
        "unknown_stage_warnings": 4, "train_batch_warnings": 4,
        "dump_file_count": len(dumps), "dump_files": dumps,
        "fa2": {"state": "already-patched", "model_py_sha256": "x", "flash_attention_2_left": 0},
        "deps": {"msprobe_config": True, "trl_target": True, "model": True,
                 "train_file": True, "test_file": True},
    }
    d.update(over)
    return d


def selftest():
    results = []

    def run_case(name, mutate, want):
        with tempfile.TemporaryDirectory() as td:
            arms = {a: clean_arm(a) for a in ARMS}
            mutate(arms)
            missing = []
            for a in ARMS:
                if arms[a] is None:
                    missing.append(a)
                    continue
                (Path(td) / f"arm-{a}.json").write_text(json.dumps(arms[a]))
            data, miss = load_arms(td)
            checks, reported = judge(data, miss)
            rc = report(checks, reported)
            results.append((name, rc == want, rc, want))

    run_case("clean", lambda a: None, 0)
    run_case("au_no_dump", lambda a: a["au"].update(dump_files=[], dump_file_count=0), 1)
    run_case("au_wrong_stage_dir",
             lambda a: a["au"].update(dump_files=[{"path": "step_1/critic_update/step0/rank0/dump.json",
                                                  "size": 1}]), 1)
    run_case("cu_dump_appeared",
             lambda a: a["cu"].update(dump_files=[{"path": "step_1/critic_update/step0/rank0/dump.json",
                                                  "size": 1}], dump_file_count=1), 2)
    run_case("cv_dump_appeared",
             lambda a: a["cv"].update(dump_files=[{"path": "step_1/compute_values/step0/rank0/dump.json",
                                                  "size": 1}], dump_file_count=1), 2)
    run_case("arm_missing", lambda a: a.update(cu=None), 1)
    run_case("arm_failed", lambda a: a["cu"].update(exit_code=1), 1)
    run_case("short_run", lambda a: a["cu"].update(steps_seen=1), 1)
    run_case("no_critic", lambda a: a["cu"].update(update_critic_occurrences=0), 1)
    run_case("tree_lost_patch", lambda a: a["au"].update(fa2={"flash_attention_2_left": 2}), 1)

    failed = [r for r in results if not r[1]]
    for name, ok, got, want in results:
        print(f"[selftest] {'PASS' if ok else 'FAIL'} {name}: got={got} want={want}")
    print(f"[selftest] {len(results) - len(failed)}/{len(results)} passed")
    return 0 if not failed else 4


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", help="directory holding arm-au.json / arm-cu.json / arm-cv.json")
    ap.add_argument("--min-steps", type=int, default=2)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.out:
        ap.error("--out is required unless --selftest")
    data, missing = load_arms(args.out)
    checks, reported = judge(data, missing, args.min_steps)
    return report(checks, reported)


if __name__ == "__main__":
    sys.exit(main())
