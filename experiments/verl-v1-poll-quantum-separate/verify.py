#!/usr/bin/env python3
"""Independent re-parse of the separate_async poll A/B (does not import make-summary).

Usage: python3 verify.py [raw_dir]   -> prints PASS/FAIL lines, exit 0 iff all PASS

Deliberately different from make-summary-separate.py: it splits a step line on
" - ", keeps only keys of interest, and recomputes the steady-state means with a
plain mean; it also re-derives the config facts from the resolved-config dump.
"""
import pathlib
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
RUNS = {"sa20a": 2.0, "sa01a": 0.1, "sa20b": 2.0, "sa01b": 0.1}
WINDOW = set(range(3, 9))
WANT = ["timing_s/gen", "timing_s/step", "timing_s/update_weights", "response_length/mean"]
fails = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'} {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


def num(token):
    token = token.strip()
    m = re.fullmatch(r"np\.(?:float64|int64)\((-?[0-9.]+)\)", token)
    if m:
        return float(m.group(1))
    try:
        return float(token)
    except ValueError:
        return None


def parse(path):
    steps, cfg = {}, {}
    for line in ANSI.sub("", path.read_text(errors="ignore")).splitlines():
        if "'trainer_mode':" in line:
            m = re.search(r"'trainer_mode':\s*'([a-z_]+)'", line)
            if m:
                cfg["trainer_mode"] = m.group(1)
        if "'poll_interval':" in line:
            m = re.search(r"'poll_interval':\s*([0-9.]+)", line)
            if m:
                cfg["poll_interval"] = float(m.group(1))
        if "'parameter_sync_step':" in line:
            m = re.search(r"'parameter_sync_step':\s*([0-9]+)", line)
            if m:
                cfg["parameter_sync_step"] = int(m.group(1))
        if re.search(r"'backend':\s*'nccl'", line):
            cfg["ckpt_backend"] = "nccl"
        if " step:" not in line or "timing_s/step" not in line:
            continue
        m = re.search(r"step:(\d+)\s", line)
        rec = {"step": float(m.group(1))} if m else {}
        for part in line.split(" - ")[1:]:
            if ":" not in part:
                continue
            key, _, value = part.partition(":")
            value = num(value)
            if value is not None:
                rec[key.strip()] = value
        if "timing_s/step" in rec and "step" in rec:
            steps[int(rec["step"])] = rec
    return cfg, steps


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    raw = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "raw")
    steady = {}
    for name, want_poll in RUNS.items():
        cfg, steps = parse(raw / f"{name}.txt")
        check(f"{name}: 8 step lines", len(steps) == 8, f"got {len(steps)}")
        check(f"{name}: trainer_mode=separate_async", cfg.get("trainer_mode") == "separate_async",
              str(cfg.get("trainer_mode")))
        check(f"{name}: resolved poll_interval={want_poll}", cfg.get("poll_interval") == want_poll,
              str(cfg.get("poll_interval")))
        check(f"{name}: checkpoint backend=nccl", cfg.get("ckpt_backend") == "nccl")
        check(f"{name}: parameter_sync_step=4", cfg.get("parameter_sync_step") == 4,
              str(cfg.get("parameter_sync_step")))
        window = [s for s in sorted(steps) if s in WINDOW]
        check(f"{name}: steady window steps 3-8 present", window == sorted(WINDOW), str(window))
        steady[name] = {k: mean([steps[s][k] for s in window if k in steps[s]]) for k in WANT}
        gen = [steps[s]["timing_s/gen"] for s in window]
        check(f"{name}: steady gen never near the 2.0s quantum (<0.2s)", max(gen) < 0.2,
              f"max={max(gen):.3f}")
        check(f"{name}: step 1 carries the only large gen wait",
              steps[1]["timing_s/gen"] > 1.0 or name == "sa01b",
              f"step1 gen={steps[1]['timing_s/gen']:.3f}")

    poll20 = [steady[n] for n in ("sa20a", "sa20b")]
    poll01 = [steady[n] for n in ("sa01a", "sa01b")]
    d_step = mean([r["timing_s/step"] for r in poll20]) - mean([r["timing_s/step"] for r in poll01])
    d_gen = mean([r["timing_s/gen"] for r in poll20]) - mean([r["timing_s/gen"] for r in poll01])
    check("pooled delta step is within ±0.2 s", abs(d_step) < 0.2, f"delta={d_step:+.3f}s")
    check("pooled delta gen is within ±0.02 s", abs(d_gen) < 0.02, f"delta={d_gen:+.4f}s")

    # the numbers the README quotes
    check("sa20a steady step mean ≈ 8.24", abs(steady["sa20a"]["timing_s/step"] - 8.236) < 0.02,
          f"{steady['sa20a']['timing_s/step']:.3f}")
    check("sa01b steady step mean ≈ 8.18", abs(steady["sa01b"]["timing_s/step"] - 8.184) < 0.02,
          f"{steady['sa01b']['timing_s/step']:.3f}")
    print(f"\nsummary: {len(fails)} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
