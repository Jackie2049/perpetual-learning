#!/usr/bin/env python3
"""Offline self-test for the rollout-bound A/B tooling (no GPU, no server).

Usage: python3 selftest-bound.py   -> prints PASS/FAIL lines, exit 0 iff all PASS

Why this exists: the A/B itself needs three GPUs, but the *parsers and the
regime gate* can be wrong in ways that only surface after an expensive window
has already been spent.  Each case below builds a synthetic console log in the
same format verl prints, runs the two tools as black boxes, and pins their
verdict:

  docs-ok      regime flipped            -> verify exit 0, delta step > 0.2 s
  no-flip      both arms ~0.03 s of gen  -> verify exit 2 (gate), never 0
  short-run    one run has 7 steps       -> verify exit 1 (mechanical)
  broken-gap   step != sum(phases)+gap   -> verify exit 1 (mechanical)

The fixtures live in a temp dir and are deleted; nothing here is measurement.
"""
import json
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
RUNS = ["sb20a", "sb01a", "sb20b", "sb01b"]
fails = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'} {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


def log_text(poll, gen, phases, steps=8, step_pad=0.0, drop_last=False):
    """One console log: resolved-config lines + one 'step:' line per step."""
    lines = [
        "{'trainer': {'v1': {'trainer_mode': 'separate_async', "
        "'sampler': {'poll_interval': " + repr(poll) + ", 'max_off_policy_threshold': 8}}}}",
        "{'actor_rollout_ref': {'rollout': {'max_response_length': 1024, "
        "'checkpoint_engine': {'backend': 'nccl'}}}, 'parameter_sync_step': 1}",
    ]
    for s in range(1, steps + 1):
        if drop_last and s == steps:
            continue
        # noise so that the two arms are not bit-identical runs
        gen_s = gen + (0.02 if s % 2 else -0.02)
        steps_total = gen_s + sum(phases.values()) + step_pad
        parts = [f"step:{s}", f"timing_s/gen:{round(gen_s, 3)}"]
        parts += [f"timing_s/{k}:{v}" for k, v in phases.items()]
        parts += [f"timing_s/step:{round(steps_total, 3)}",
                  "response_length/mean:410.0",
                  "training/off_policy/trajectory_spans/mean:1.0",
                  "training/off_policy/trajectory_staleness/mean:1.0",
                  "critic/rewards/mean:0.30", "perf/throughput:100.0"]
        lines.append("(TaskRunnerV1 pid=1) " + " - ".join(parts))
    return "\n".join(lines) + "\n"


PHASES = {"old_log_prob": 2.0, "adv": 0.1, "update_actor": 1.5, "update_weights": 0.8}


def build(root, gen20, gen01, **kw):
    raw = pathlib.Path(root) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name in RUNS:
        poll = 2.0 if "20" in name else 0.1
        gen = gen20 if poll == 2.0 else gen01
        # only the last run of the corrupt case is shortened
        drop = kw.get("drop_last") and name == "sb01b"
        pad = kw.get("step_pad", 0.0) if poll == 2.0 else 0.0
        (raw / f"{name}.txt").write_text(
            log_text(poll, gen, PHASES, step_pad=pad, drop_last=drop))
    return raw


def run(script, raw):
    return subprocess.run([sys.executable, str(HERE / script), str(raw)],
                          capture_output=True, text=True)


def case(name):
    root = tempfile.mkdtemp(prefix=f"bound-{name}-")
    return root


def main():
    # 1. regime flipped: 2.0s arm quantized to 4.0 s, 0.1 arm sees the true 3.1 s
    root = case("ok")
    raw = build(root, gen20=4.0, gen01=3.1)
    r = run("verify-bound.py", raw)
    check("regime flipped -> verify exit 0", r.returncode == 0, f"rc={r.returncode}")
    check("regime flipped -> both arms pass the gate", "GATE-FAIL" not in r.stdout,
          r.stdout.strip().splitlines()[-1] if r.stdout else "no output")
    s = run("make-summary-bound.py", raw)
    check("summary tool runs", s.returncode == 0, f"rc={s.returncode}")
    if s.returncode == 0:
        out = json.loads(s.stdout)
        d_step = out["delta"]["timing_s/step"]["delta"]
        d_gen = out["delta"]["timing_s/gen"]["delta"]
        check("summary delta step == 0.9 s", abs(d_step - 0.9) < 1e-9, f"{d_step:+.6f}")
        check("summary delta gen == 0.9 s", abs(d_gen - 0.9) < 1e-9, f"{d_gen:+.6f}")

    # 2. load did not flip the regime: both arms ~0.03 s of gen
    root = case("noflip")
    raw = build(root, gen20=0.03, gen01=0.03)
    r = run("verify-bound.py", raw)
    check("no-flip -> verify exit 2 (gate, not 0)", r.returncode == 2, f"rc={r.returncode}")
    check("no-flip -> gate names the 2.0 arm", "GATE-FAIL" in r.stdout and "sb20a" in r.stdout)
    check("no-flip -> no mechanical failure", "mechanical: 0 failure(s)" in r.stdout)

    # 3. one run lost its last step
    root = case("short")
    raw = build(root, gen20=4.0, gen01=3.1, drop_last=True)
    r = run("verify-bound.py", raw)
    check("short run -> verify exit 1 (mechanical)", r.returncode == 1, f"rc={r.returncode}")

    # 4. accounting does not close: +0.5 s unexplained on every 2.0-arm step
    root = case("gap")
    raw = build(root, gen20=4.0, gen01=3.1, step_pad=0.5)
    r = run("verify-bound.py", raw)
    check("broken gap -> verify exit 1 (mechanical)", r.returncode == 1, f"rc={r.returncode}")

    print(f"\n{len(fails)} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
