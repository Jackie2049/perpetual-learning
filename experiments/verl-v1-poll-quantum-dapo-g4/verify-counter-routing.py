#!/usr/bin/env python3
"""Does `training/sampler/blocking_polls` reach the console line that read-polls.py parses?

The counter patch can be correct in isolation and still be useless: the value has to
survive three hops before anyone can read it

    sample() return -> metrics dict -> MetricsAggregator -> logger.log(console)

and the key it writes has to be the same string the reader looks for.  Each hop can
break silently -- the run still exits 0, the counter is just absent from the log, and
the GPU window is spent.  This checks the hops on the tree that will actually run.

Two tree states are recognised:

  pristine      the counter is not installed (checks that must survive patching:
                prefix, trainer merge path, one sample() per step, val discard)
  instrumented  the counter is installed (additionally: emit inside sample(), reset at
                the top of sample(), increment inside _wait_for_next_poll, and the key
                registered in MetricsAggregator's "sum" list)

Exit codes: 0 all checks pass, 2 unusable input, 3 a routing check failed.

Usage:
  verify-counter-routing.py --tree /tmp/verl-24f25b03
  verify-counter-routing.py --tree /tmp/m5-g4i1/verl --json raw/counter-routing.json
  verify-counter-routing.py --tree /tmp/m5-g4i1/verl --log raw/rollout-poll-dapo-g4baseR.txt
  verify-counter-routing.py --selftest
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

KEY_SUFFIX = "/sampler/blocking_polls"
READER_DEFAULT = Path(__file__).resolve().parent / "read-polls.py"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def read(path, what):
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"FATAL: missing {what}: {p}")
    return p.read_text()


def block(src, header):
    """Return a method body: from `header` to the next method or class at the same level."""
    start = src.find(header)
    if start < 0:
        return None
    ends = [i for i in (src.find("\n    def ", start + len(header)),
                        src.find("\n\nclass ", start + len(header))) if i > 0]
    end = min(ends) if ends else len(src)
    return src[start:end]


def prefix_map(replay_src):
    """Execute the real `_metrics_prefix` source (it is three lines and version-stable)."""
    m = re.search(r"( {4}@staticmethod\n)? {4}def _metrics_prefix\(partition_id: str\) -> str:\n(?: {8}.*\n)+", replay_src)
    if not m:
        return None
    body = m.group(0)
    body = "\n".join(line[4:] if line.startswith("    ") else line for line in body.splitlines())
    body = body.replace("@staticmethod\n", "")
    ns = {}
    exec(compile(body, "<metrics_prefix>", "exec"), ns)  # noqa: S102 - fixed literal source
    return {p: ns["_metrics_prefix"](p) for p in ("train", "val")}


def sum_list(utils_src):
    """Return the literal strings in the `"sum": [...]` list of _init_aggregation_rules."""
    start = utils_src.find('"sum": [')
    if start < 0:
        return None
    end = utils_src.find("]", start)
    return set(re.findall(r'"([^"]+)"', utils_src[start:end]))


def reader_key(path):
    src = read(path, "reader (read-polls.py)")
    m = re.search(r'^POLL_KEY = "([^"]+)"', src, re.M)
    return m.group(1) if m else None


def sync_block_is_empty(yaml_src):
    """`v1.sync` must not set parameter_sync_step, so the default 1 applies."""
    lines = yaml_src.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^    sync:\s*(\{\})?\s*$", line):
            inline = re.match(r"^    sync:\s*(\{\})\s*$", line)
            if inline:
                return True, "v1.sync: {}"
            indent = len(line) - len(line.lstrip())
            for nxt in lines[i + 1 :]:
                if not nxt.strip() or nxt.lstrip().startswith("#"):
                    continue
                cur = len(nxt) - len(nxt.lstrip())
                if cur <= indent:
                    break
                if "parameter_sync_step" in nxt:
                    return False, nxt.strip()
            return True, "v1.sync has no parameter_sync_step"
    return None, "no `v1.sync` block found"


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def checks(tree, reader=READER_DEFAULT):
    root = Path(tree)
    replay = read(root / "verl/trainer/ppo/v1/replay_buffer.py", "replay_buffer.py")
    trainer = read(root / "verl/trainer/ppo/v1/trainer_base.py", "trainer_base.py")
    utils = read(root / "verl/trainer/ppo/v1/utils.py", "utils.py")
    config = read(root / "verl/trainer/config/ppo_trainer.yaml", "ppo_trainer.yaml")

    out = []

    def add(name, ok, detail):
        out.append({"check": name, "ok": bool(ok), "detail": detail})

    pm = prefix_map(replay)
    add(
        "R1.prefix_train",
        pm == {"train": "training", "val": "validation"},
        pm if pm is not None else "could not extract _metrics_prefix",
    )
    key = (pm or {}).get("train", "<prefix?>") + KEY_SUFFIX

    sample = block(replay, "    def sample(")
    wait = block(replay, "    def _wait_for_next_poll(")
    instrumented = "self._blocking_polls" in replay
    add("R2.state", True, {"instrumented": instrumented})

    if instrumented:
        emit = "{self._metrics_prefix(partition_id)}" + KEY_SUFFIX
        add(
            "R3.emit_in_sample",
            sample is not None and emit in sample,
            {"pattern": emit, "in_sample": sample is not None and emit in sample},
        )
        add(
            "R4.reset_per_sample",
            sample is not None and "self._blocking_polls = 0" in sample,
            {"reset": sample is not None and "self._blocking_polls = 0" in sample},
        )
        add(
            "R5.increment_in_wait",
            wait is not None and "self._blocking_polls += 1" in wait,
            {"in_wait_for_next_poll": wait is not None and "self._blocking_polls += 1" in wait},
        )
        registered = sum_list(utils)
        add(
            "R6.aggregation_rule",
            registered is not None and key in registered,
            {"key": key, "in_sum_list": None if registered is None else key in registered},
        )
    else:
        add("R3.emit_in_sample", True, "pristine tree: counter not installed")
        add("R4.reset_per_sample", True, "pristine tree")
        add("R5.increment_in_wait", True, "pristine tree")
        add("R6.aggregation_rule", True, "pristine tree")

    add(
        "R7.reader_key",
        reader_key(reader) == key,
        {"reader": str(reader), "POLL_KEY": reader_key(reader), "computed": key},
    )

    hop = {
        "sample_returns_metrics": "off_policy_metrics = self.replay_buffer.sample(" in trainer,
        "iter_metrics_updated": "metrics.update(off_policy_metrics)" in trainer,
        "aggregator_added": "metrics_aggregator.add_step_metrics(iter_metrics" in trainer,
        "aggregated_into_step": "metrics.update(metrics_aggregator.get_aggregated_metrics())" in trainer,
        "console_log": "self.logger.log(data=metrics, step=self.global_steps)" in trainer,
    }
    add("R8.trainer_merge_path", all(hop.values()), hop)

    empty, detail = sync_block_is_empty(config)
    one = '.get("parameter_sync_step", 1)' in trainer
    add("R9.one_sample_per_step", bool(one and empty), {"default_is_1": one, "sync_block": detail})

    add(
        "R10.val_discards",
        "batch, _ = self.replay_buffer.sample(" in trainer,
        {"discarded": "batch, _ = self.replay_buffer.sample(" in trainer},
    )
    return {"tree": str(root), "key": key, "instrumented": instrumented, "checks": out}


def empirical(log_path, key):
    """The sibling check: an archived log proves the same eviction_metrics dict is logged."""
    src = read(log_path, "archived console log")
    sibling = "training/filter_groups/evicted_samples"
    return {
        "check": "R11.archived_sibling_key",
        "ok": sibling in src and key not in src,
        "detail": {
            "log": str(log_path),
            "sibling_present": sibling in src,
            "counter_present": key in src,
            "note": "sibling comes from the same eviction_metrics dict; counter is post-patch only",
        },
    }


# ---------------------------------------------------------------------------
# selftest: each mutation must turn its own check red, baseline stays green
# ---------------------------------------------------------------------------
def selftest(tmproot):
    base = checks(tmproot / "clean")
    assert base["instrumented"] is False, base
    ctl = checks(tmproot / "run")
    assert ctl["instrumented"] is True and all(c["ok"] for c in ctl["checks"]), ctl
    assert all(c["ok"] for c in base["checks"]), base

    def mutate(name, rel, old, new, expect_red, rootdir):
        dst = Path(rootdir) / name
        shutil.copytree(rootdir / "run", dst)
        p = dst / rel
        src = p.read_text()
        assert old in src, (name, rel, old)
        p.write_text(src.replace(old, new, 1))
        got = checks(dst)
        bad = sorted(c["check"] for c in got["checks"] if not c["ok"])
        return {"case": name, "expect_red": sorted(expect_red), "red": bad,
                "ok": bad == sorted(expect_red)}

    results = []
    results.append(mutate("prefix", "verl/trainer/ppo/v1/replay_buffer.py",
                          'return "training" if partition_id == "train"',
                          'return "train" if partition_id == "train"',
                          {"R1.prefix_train", "R6.aggregation_rule", "R7.reader_key"}, tmproot))
    results.append(mutate("emit", "verl/trainer/ppo/v1/replay_buffer.py",
                          f'eviction_metrics[f"{{self._metrics_prefix(partition_id)}}{KEY_SUFFIX}"] = self._blocking_polls',
                          "pass  # emit removed",
                          {"R3.emit_in_sample"}, tmproot))
    results.append(mutate("registration", "verl/trainer/ppo/v1/utils.py",
                          f'                "training/sampler/blocking_polls",\n', "",
                          {"R6.aggregation_rule"}, tmproot))
    results.append(mutate("merge", "verl/trainer/ppo/v1/trainer_base.py",
                          "metrics.update(off_policy_metrics)",
                          "pass  # merge removed",
                          {"R8.trainer_merge_path"}, tmproot))
    # reader mutation needs its own copy (the reader lives outside the tree)
    rd = Path(tmproot) / "read-polls-mut.py"
    rd.write_text(READER_DEFAULT.read_text().replace('POLL_KEY = "training/sampler/blocking_polls"',
                                                     'POLL_KEY = "train/sampler/blocking_polls"', 1))
    bad = [c["check"] for c in checks(tmproot / "run", reader=rd)["checks"] if not c["ok"]]
    bad = sorted(bad)
    results.append({"case": "reader", "expect_red": ["R7.reader_key"], "red": bad,
                    "ok": bad == ["R7.reader_key"]})
    return results


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree")
    ap.add_argument("--reader", default=str(READER_DEFAULT))
    ap.add_argument("--log", help="archived console log for the sibling-key check")
    ap.add_argument("--json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        with tempfile.TemporaryDirectory() as td:
            # Build a clean and an instrumented copy of the *same* tree so the mutation
            # targets line up. The caller passes the clean tree.
            root = Path(td)
            shutil.copytree(args.tree, root / "clean")
            shutil.copytree(args.tree, root / "run")
            patch = Path(__file__).resolve().parent.parent / "verl-v1-poll-quantum-polls" / "poll-count.patch"
            proc = subprocess.run(["patch", "-p1", "--fuzz=0", "-d", str(root / "run"), "-i", str(patch)],
                                  capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"FATAL: counter patch did not apply in selftest: {proc.stdout}{proc.stderr}")
                return 2
            results = selftest(root)
        bad = [r for r in results if not r["ok"]]
        print(json.dumps({"selftest": results, "failed": len(bad)}, indent=1))
        print(f"selftest: {len(results) - len(bad)}/{len(results)} passed")
        return 0 if not bad else 3

    if not args.tree:
        ap.error("--tree is required (or --selftest)")
    reading = checks(args.tree, reader=args.reader)
    if args.log:
        reading["checks"].append(empirical(args.log, reading["key"]))
    failed = [c for c in reading["checks"] if not c["ok"]]
    reading["failed"] = len(failed)
    if args.json:
        Path(args.json).write_text(json.dumps(reading, indent=1) + "\n")
    for c in reading["checks"]:
        print(f"{'PASS' if c['ok'] else 'FAIL'}  {c['check']:<24} {json.dumps(c['detail'])}")
    print(f"key={reading['key']} instrumented={reading['instrumented']} failed={len(failed)}")
    return 0 if not failed else 3


if __name__ == "__main__":
    sys.exit(main())
