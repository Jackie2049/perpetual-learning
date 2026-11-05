#!/usr/bin/env python3
"""Is the 4-card `inflight=1` arm still runnable, and still the same experiment?

TM-008 has one unmeasured arm: 4 cards with `max_inflight_gen_batches=1` (the DAPO
default). It needs four idle cards, the scarcest resource here, so a run that dies at
Hydra composition — or that silently trains a different config — costs a whole
window. The arm was last prepared against verl `24f25b03`; main has since moved.

The chain this checks is the one that actually runs on the server:

    run-poll-dapo.sh  --env-->  examples/grpo_trainer/run_qwen3_4b_fsdp.sh  --argv-->  main_ppo

Nothing in that chain is retyped here. A stub `python3` on PATH makes it print the
argv instead of training, so the capture *is* the wrapper's output: an env var that
got renamed upstream, or an override the recipe dropped, shows up as a changed
override list rather than as a run that trains something else. The captured argv is
then composed for real (Hydra, CPU only, no verl import, no GPU) and the effective
values are read back against the pre-registered table below.

Two readings worth knowing about:

  * The base recipe stopped pinning `enable_chunked_prefill=False` between the
    archived tree and main, and the rollout default is True. The wrapper now pins it;
    `--compare-script` on the archived tree's recipe proves the captured override
    lists are then identical.
  * The recipe uses `set -xeuo pipefail` with an empty `EXTRA=()`. Under bash 3.2
    (macOS) `"${EXTRA[@]}"` is an unbound-variable error, so the local run stages a
    copy with `-u` dropped. That rewrite must keep `-o pipefail`: `set -xe pipefail`
    would overwrite the script's positional parameters and silently drop every
    trailing override. The substitution used is recorded in the reading.

Usage:
  preflight-g4.py --tree /tmp/verl-31cfccb4 --out raw/preflight-g4.json
  preflight-g4.py --tree /tmp/verl-31cfccb4 --compare-script /tmp/base_24f25b03.sh
  preflight-g4.py --selftest
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
POLL_PATCH = REPO / "experiments" / "verl-v1-poll-quantum" / "poll-interval-config.patch"
COUNTER_PATCH = REPO / "experiments" / "verl-v1-poll-quantum-polls" / "poll-count.patch"
SIM_LOOP = REPO / "experiments" / "verl-v1-poll-quantum-polls" / "sim-loop.py"
WRAPPER = HERE / "run-poll-dapo.sh"
BASE_SCRIPT_REL = "examples/grpo_trainer/run_qwen3_4b_fsdp.sh"
COUNTER_FILES = ("verl/trainer/ppo/v1/replay_buffer.py", "verl/trainer/ppo/v1/utils.py")

# The arm under test: four cards in the 2+2 placement the archived batch used.
WRAPPER_ENV = {"MODE": "pf-g4i1", "GPUS": "0,1,4,5", "INFLIGHT": "1"}

# What the wrapper must still hand the recipe, as it appears in the captured argv.
# This is the arm's definition; if one of these disappears the experiment changed,
# even if everything still composes.
REQUIRED = {
    "algorithm.filter_groups.enable": "True",
    "algorithm.filter_groups.metric": "acc",
    "algorithm.filter_groups.max_inflight_gen_batches": "1",
    "data.train_batch_size": "32",
    "data.max_prompt_length": "512",
    "data.max_response_length": "256",
    "actor_rollout_ref.actor.ppo_mini_batch_size": "16",
    "actor_rollout_ref.rollout.n": "4",
    "actor_rollout_ref.rollout.gpu_memory_utilization": "0.20",
    "actor_rollout_ref.rollout.enable_chunked_prefill": "False",
    "+actor_rollout_ref.model.override_config.attn_implementation": "sdpa",
    "+actor_rollout_ref.rollout.max_model_len": "2048",
}

# What the composed config must resolve to; read back after Hydra composition.
EXPECTED = {
    "algorithm.filter_groups.enable": True,
    "algorithm.filter_groups.metric": "acc",
    "algorithm.filter_groups.max_inflight_gen_batches": 1,
    "data.train_batch_size": 32,
    "data.max_prompt_length": 512,
    "data.max_response_length": 256,
    "actor_rollout_ref.actor.ppo_mini_batch_size": 16,
    "actor_rollout_ref.rollout.name": "vllm",
    "actor_rollout_ref.rollout.n": 4,
    "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
    "actor_rollout_ref.rollout.gpu_memory_utilization": 0.2,
    "actor_rollout_ref.rollout.enable_chunked_prefill": False,
    "actor_rollout_ref.rollout.max_model_len": 2048,
    "actor_rollout_ref.model.override_config.attn_implementation": "sdpa",
    "trainer.n_gpus_per_node": 4,
    "trainer.total_epochs": 1,
    "trainer.logger": ["console"],
}

# Reported, not asserted: context for reading the numbers.
REPORT = [
    "actor_rollout_ref.rollout.max_num_batched_tokens",
    "actor_rollout_ref.rollout.enable_prefix_caching",
    "actor_rollout_ref.rollout.enforce_eager",
    "actor_rollout_ref.actor.use_dynamic_bsz",
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu",
    "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes",
    "trainer.v1.sampler.poll_interval",
    "trainer.critic_warmup",
]

NEGATIVE_CONTROL = "algorithm.filter_groups.no_such_field=1"

STUB = """#!/usr/bin/env bash
# Fail `-c` probes so DEVICE resolves to gpu; print the main_ppo argv otherwise.
if [ "${1:-}" = "-c" ]; then exit 1; fi
if [ "${1:-}" = "-m" ] && [ "${2:-}" = "verl.trainer.main_ppo" ]; then
  shift 2; printf '%s\\n' "$@" > "$CAPTURE_OUT"; exit 0
fi
echo "stub: unexpected invocation: $*" >&2; exit 9
"""


# ----------------------------------------------------------------- capture


def bash_major():
    out = subprocess.run(["bash", "-c", 'echo "${BASH_VERSINFO[0]}"'],
                         capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else 5


def stage_base_script(src, workdir, name, drop_nounset):
    """Stage a base recipe under `<workdir>/<name>`, optionally bash-3.2 safe."""
    text = Path(src).read_text()
    note = "verbatim"
    if drop_nounset:
        fixed = text.replace("set -xeuo pipefail", "set -xeo pipefail")
        if fixed == text:
            note = "no-nounset-line-found"
        else:
            note = "dropped -u only (bash 3.2 cannot expand an empty array under set -u)"
            text = fixed
    dst = Path(workdir) / name / BASE_SCRIPT_REL
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text)
    dst.chmod(0o755)
    return Path(workdir) / name, note


def capture_from_wrapper(tree, workdir, name="ex", base_src=None):
    """Run run-poll-dapo.sh end to end against a stub python3; return the argv."""
    stub_dir = Path(workdir) / "stub"
    stub_dir.mkdir(parents=True, exist_ok=True)
    stub = stub_dir / "python3"
    stub.write_text(STUB)
    stub.chmod(0o755)

    src = base_src or (Path(tree) / BASE_SCRIPT_REL)
    stage, note = stage_base_script(src, workdir, name, drop_nounset=bash_major() < 4)

    out = Path(workdir) / f"argv-{name}.txt"
    env = {"HOME": os.environ.get("HOME", "/tmp"),
           "PATH": str(stub_dir) + ":/usr/bin:/bin",
           "CAPTURE_OUT": str(out),
           "SRC": str(stage),
           # The stage carries only the base recipe, by design: this half checks the
           # argv, not the runtime tree. The instrumentation guard is exercised on real
           # trees further down (`wrapper_guard`).
           "CHECK_TREE": "0"}
    env.update(WRAPPER_ENV)
    proc = subprocess.run(["bash", str(WRAPPER)], cwd=str(workdir), env=env,
                          capture_output=True, text=True)
    res = {"base_script": str(src), "staged_as": name, "substitution": note,
           "returncode": proc.returncode}
    if not out.exists():
        res["error"] = (proc.stderr or proc.stdout or "").strip().splitlines()[-4:]
        res["overrides"] = []
        return res
    res["overrides"] = [line for line in out.read_text().splitlines() if line]
    return res


def parse_overrides(overrides):
    """Last occurrence wins, matching Hydra. `+key=value` adds a new key."""
    out = {}
    for item in overrides:
        key, _, value = item.partition("=")
        out[key.lstrip("+")] = value
    return out


def check_capture(overrides, required=None):
    """The captured argv must still carry every override the arm is defined by."""
    required = REQUIRED if required is None else required
    got = parse_overrides(overrides)
    raw_keys = {item.partition("=")[0] for item in overrides}
    problems = []
    for key, want in required.items():
        bare = key.lstrip("+")
        if bare not in got:
            problems.append(f"{bare}: not passed (wanted {want!r})")
        elif got[bare] != want:
            problems.append(f"{bare}: {got[bare]!r} != {want!r}")
        # A key that does not exist upstream needs '+', otherwise Hydra refuses it.
        if key.startswith("+") and key not in raw_keys:
            problems.append(f"{key}: passed without the '+' prefix (a new key needs it)")
    return {"n_overrides": len(overrides), "problems": problems}


def diff_overrides(old, new):
    """Override-level diff: what changed between the archived recipe and today's."""
    a, b = parse_overrides(old), parse_overrides(new)
    return {
        "removed": sorted(f"{k}={a[k]}" for k in a.keys() - b.keys()),
        "added": sorted(f"{k}={b[k]}" for k in b.keys() - a.keys()),
        "changed": sorted(f"{k}: {a[k]!r} -> {b[k]!r}" for k in a.keys() & b.keys() if a[k] != b[k]),
        "same": a == b,
    }


# ------------------------------------------------------------------ compose


def compose(config_dir, overrides):
    from hydra import compose as hydra_compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return hydra_compose(config_name="ppo_trainer", overrides=overrides)


def readback(cfg, keys):
    from omegaconf import OmegaConf

    out = {}
    for key in keys:
        node = cfg
        try:
            for part in key.split("."):
                node = node[part]
            out[key] = OmegaConf.to_container(node, resolve=True) if OmegaConf.is_config(node) else node
        except Exception as exc:
            out[key] = f"<missing: {type(exc).__name__}>"
    return out


def stage_patch_files(tree, workdir, name):
    stage = Path(workdir) / name
    shutil.rmtree(stage, ignore_errors=True)
    for rel in ("verl/trainer/config/ppo_trainer.yaml",
                "verl/trainer/ppo/v1/trainer_base.py",
                "verl/trainer/ppo/v1/trainer_separate_async.py"):
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(tree) / rel, dst)
    return stage


def poll_patch_status(tree, workdir):
    """Does the poll_interval patch still apply to this tree (or is it already on)?"""
    already = instrumentation(tree)["present"]
    already_applied = already["poll_interval_default"] and already["poll_interval_wiring"]
    stage = stage_patch_files(tree, workdir, "patchcheck")
    proc = subprocess.run(["patch", "-p1", "--dry-run", "--fuzz=0", "-i", str(POLL_PATCH)],
                          cwd=str(stage), capture_output=True, text=True)
    return {"applies": proc.returncode == 0, "returncode": proc.returncode,
            "already_applied": already_applied, "usable": proc.returncode == 0 or already_applied,
            "files": ["verl/trainer/config/ppo_trainer.yaml",
                      "verl/trainer/ppo/v1/trainer_base.py",
                      "verl/trainer/ppo/v1/trainer_separate_async.py"],
            "stderr": proc.stderr.strip().splitlines()}


def patched_config_dir(tree, workdir):
    """A config dir whose `ppo_trainer.yaml` carries the real patch hunk."""
    if instrumentation(tree)["present"]["poll_interval_wiring"]:
        # The tree already carries the key (an instrumented run tree): use it as is.
        return Path(tree) / "verl" / "trainer" / "config"
    stage = stage_patch_files(tree, workdir, "patchapply")
    proc = subprocess.run(["patch", "-p1", "--fuzz=0", "-i", str(POLL_PATCH)],
                          cwd=str(stage), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"poll patch did not apply: {proc.stderr.strip()[:300]}")
    dst = Path(workdir) / "config-poll"
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(Path(tree) / "verl" / "trainer" / "config", dst)
    shutil.copy2(stage / "verl/trainer/config/ppo_trainer.yaml", dst / "ppo_trainer.yaml")
    return dst


def run_arm(tree, workdir, overrides, label):
    res = {"label": label, "composed": False}
    config_dir = (Path(tree) / "verl" / "trainer" / "config" if label == "base"
                  else patched_config_dir(tree, workdir))
    try:
        cfg = compose(config_dir, overrides)
        res["composed"] = True
        res["config_dir"] = str(config_dir)
        res["readback"] = readback(cfg, list(EXPECTED) + REPORT)
    except Exception as exc:
        res["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
    return res


def check_expected(arm, expected=None):
    expected = EXPECTED if expected is None else expected
    if not arm.get("composed"):
        return [f"composition failed: {arm.get('error')}"]
    got = arm["readback"]
    return [f"{k}: {got.get(k)!r} != {want!r}" for k, want in expected.items() if got.get(k) != want]


def negative_control(tree, overrides):
    try:
        compose(Path(tree) / "verl" / "trainer" / "config", overrides + [NEGATIVE_CONTROL])
        return {"fired": False, "note": "invented key composed -> the check is blind"}
    except Exception as exc:
        return {"fired": True, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}


# ------------------------------------------------------------------ selftest


def counter_patch_status(tree, workdir):
    """Does the poll-count patch still apply to this tree, unchanged?"""
    stage = Path(workdir) / "countercheck"
    shutil.rmtree(stage, ignore_errors=True)
    for rel in COUNTER_FILES:
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(tree) / rel, dst)
    already = instrumentation(tree)["present"]
    already_applied = already["counter_in_buffer"] and already["counter_aggregation"]
    proc = subprocess.run(["patch", "-p1", "--dry-run", "--fuzz=0", "-i", str(COUNTER_PATCH)],
                          cwd=str(stage), capture_output=True, text=True)
    return {"applies": proc.returncode == 0, "returncode": proc.returncode,
            "already_applied": already_applied, "usable": proc.returncode == 0 or already_applied,
            "files": list(COUNTER_FILES), "stderr": proc.stderr.strip().splitlines()}


def instrumentation(tree):
    """Markers of the two patches on the tree that will actually run."""
    root = Path(tree)

    def has(rel, needle):
        path = root / rel
        return path.exists() and needle in path.read_text()

    present = {
        "poll_interval_default": has("verl/trainer/config/ppo_trainer.yaml", "poll_interval: 2.0"),
        "poll_interval_wiring": has("verl/trainer/ppo/v1/trainer_base.py", "poll_interval=sampler_config.get"),
        "counter_in_buffer": has("verl/trainer/ppo/v1/replay_buffer.py", "sampler/blocking_polls"),
        "counter_aggregation": has("verl/trainer/ppo/v1/utils.py", '"training/sampler/blocking_polls"'),
    }
    return {"present": present, "instrumented": all(present.values())}


def pick_sim_python(explicit=None):
    """An interpreter that can exec the buffer source: `int | float` needs Python 3.10+."""
    if explicit:
        return explicit
    local = Path("/tmp/m5-trf-venv/bin/python3")
    return str(local) if local.exists() else sys.executable


def sim_reading(tree, python=sys.executable):
    """Run the counter harness on the exact tree that will run; it must be 12/12."""
    proc = subprocess.run([str(python), str(SIM_LOOP), "--tree", str(tree), "--selftest"],
                          capture_output=True, text=True)
    lines = [line for line in proc.stdout.strip().splitlines() if line]
    return {"python": str(python), "returncode": proc.returncode,
            "tail": lines[-1] if lines else "",
            "stderr": proc.stderr.strip().splitlines()[-3:]}


def wrapper_guard(tree, expected):
    """The wrapper must refuse an un-instrumented tree before touching GPUs."""
    env = dict(os.environ, SRC=str(tree), CHECK_ONLY="1", MODE="guard")
    proc = subprocess.run(["bash", str(WRAPPER)], env=env, capture_output=True, text=True)
    return {"tree": str(tree), "expected": expected, "returncode": proc.returncode,
            "ok": proc.returncode == expected,
            "stdout": proc.stdout.strip().splitlines()[-2:],
            "stderr": proc.stderr.strip().splitlines()[-2:]}


def guard_control():
    """Minimal tree with all four markers, then with the counter stripped: 0 then 65."""
    tmp = Path(tempfile.mkdtemp(prefix="guard-control-"))
    (tmp / "verl/trainer/config").mkdir(parents=True)
    (tmp / "verl/trainer/ppo/v1").mkdir(parents=True)
    (tmp / "verl/trainer/config/ppo_trainer.yaml").write_text("poll_interval: 2.0\n")
    (tmp / "verl/trainer/ppo/v1/trainer_base.py").write_text("poll_interval=sampler_config.get\n")
    (tmp / "verl/trainer/ppo/v1/replay_buffer.py").write_text("sampler/blocking_polls\n")
    (tmp / "verl/trainer/ppo/v1/utils.py").write_text('"training/sampler/blocking_polls"\n')
    with_markers = wrapper_guard(tmp, 0)
    (tmp / "verl/trainer/ppo/v1/replay_buffer.py").write_text("counter removed\n")
    stripped = wrapper_guard(tmp, 65)
    shutil.rmtree(tmp, ignore_errors=True)
    return {"with_markers": with_markers, "stripped": stripped,
            "ok": with_markers["ok"] and stripped["ok"]}


def selftest():
    """Prove the detectors fire, not just that the happy path prints zeros."""
    try:
        import hydra  # noqa: F401
    except ImportError:
        print("selftest needs an interpreter with hydra (e.g. /tmp/hydra-probe-venv); "
              "the compose half cannot run without it", file=sys.stderr)
        return 3
    results = []

    intact = [f"{k}={v}" for k, v in REQUIRED.items()]
    results.append(("intact_capture_clean", check_capture(intact)["problems"] == []))

    # A recipe that stopped pinning chunked prefill must surface, not pass quietly.
    dropped = [o for o in intact if not o.startswith("actor_rollout_ref.rollout.enable_chunked_prefill")]
    results.append(("dropped_pin_detected", check_capture(dropped)["problems"] != []))

    # `set -xe pipefail` eats positional args; a capture missing its tail must fail.
    eaten = [o for o in intact if not o.startswith("trainer.") and "+" not in o]
    results.append(("eaten_tail_detected", check_capture(eaten)["problems"] != []))

    # The '+' bookkeeping: a new key passed without '+' would be rejected by Hydra.
    noplus = [o.replace("+actor_rollout_ref.model", "actor_rollout_ref.model") for o in intact]
    results.append(("missing_plus_detected", check_capture(noplus)["problems"] != []))

    # Compose-level check fires on a wrong value, on a key that is really checked.
    fake = {"composed": True, "readback": dict(EXPECTED)}
    fake["readback"]["actor_rollout_ref.rollout.n"] = 5
    results.append(("wrong_value_detected", check_expected(fake) != []))

    # Negative control: a clean override composes, an invented key does not.
    tmp = Path(tempfile.mkdtemp())
    (tmp / "ppo_trainer.yaml").write_text("algorithm:\n  filter_groups:\n    enable: False\n")
    try:
        cfg = compose(tmp, ["algorithm.filter_groups.enable=True"])
        clean = cfg.algorithm.filter_groups.enable is True
    except Exception:
        clean = False
    results.append(("clean_override_composes", clean))
    results.append(("negative_control_fires", negative_control(tmp, [])["fired"]))

    # Instrumentation markers: all four -> instrumented, one missing -> not.
    probe = Path(tempfile.mkdtemp())
    (probe / "verl/trainer/config").mkdir(parents=True)
    (probe / "verl/trainer/ppo/v1").mkdir(parents=True)
    (probe / "verl/trainer/config/ppo_trainer.yaml").write_text("poll_interval: 2.0\n")
    (probe / "verl/trainer/ppo/v1/trainer_base.py").write_text("poll_interval=sampler_config.get\n")
    (probe / "verl/trainer/ppo/v1/replay_buffer.py").write_text("sampler/blocking_polls\n")
    (probe / "verl/trainer/ppo/v1/utils.py").write_text('"training/sampler/blocking_polls"\n')
    results.append(("instrumentation_all_markers", instrumentation(probe)["instrumented"]))
    (probe / "verl/trainer/ppo/v1/utils.py").write_text('"something_else"\n')
    results.append(("instrumentation_missing_marker", not instrumentation(probe)["instrumented"]))

    # The wrapper guard: an instrumented tree passes, the same tree without one marker is
    # refused before any GPU work. Restore the marker the previous check removed.
    (probe / "verl/trainer/ppo/v1/utils.py").write_text('"training/sampler/blocking_polls"\n')
    results.append(("guard_passes_instrumented", wrapper_guard(probe, 0)["ok"]))
    (probe / "verl/trainer/ppo/v1/replay_buffer.py").write_text("no counter here\n")
    refused = wrapper_guard(probe, 65)
    results.append(("guard_blocks_uninstrumented", refused["ok"]))
    shutil.rmtree(probe, ignore_errors=True)

    print(json.dumps({"selftest": [{"check": n, "ok": bool(v)} for n, v in results],
                      "ok": all(v for _, v in results)}, indent=2))
    return 0 if all(v for _, v in results) else 1


# ---------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", help="verl checkout to check (e.g. /tmp/verl-31cfccb4)")
    ap.add_argument("--run-tree",
                    help="the instrumented copy the window will actually run (prep-tree.sh); "
                         "must carry both patches and is checked with the sim harness")
    ap.add_argument("--sim-python", help="interpreter for sim-loop.py (needs Python 3.10+)")
    ap.add_argument("--compare-script",
                    help="the same base recipe from the tree the archive was produced on; "
                         "the two captured override lists are diffed")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.tree:
        ap.error("--tree is required unless --selftest")
    if not args.run_tree:
        ap.error("--run-tree is required: the window must run on the instrumented copy "
                 "(bash prep-tree.sh <clean-tree> <dst-tree>)")

    import hydra
    import omegaconf

    workdir = Path(tempfile.mkdtemp(prefix="preflight-g4-"))
    tree = Path(args.tree).resolve()

    cap = capture_from_wrapper(tree, workdir)
    overrides = cap.pop("overrides", [])
    cap["check"] = check_capture(overrides)

    patch = poll_patch_status(tree, workdir)
    counter = counter_patch_status(tree, workdir)
    arm_base = run_arm(tree, workdir, overrides, "base")
    arm_poll = run_arm(tree, workdir, overrides + ["trainer.v1.sampler.poll_interval=0.1"], "poll")
    for arm in (arm_base, arm_poll):
        arm["problems"] = check_expected(arm)

    result = {
        "env": {"python": sys.version.split()[0], "hydra": hydra.__version__,
                "omegaconf": omegaconf.__version__, "bash_major": bash_major()},
        "tree": str(tree),
        "arm": WRAPPER_ENV,
        "capture": cap,
        "poll_patch": patch,
        "counter_patch": counter,
        "arms": {"base": arm_base, "poll": arm_poll},
        "negative_control": negative_control(tree, overrides),
    }

    if args.compare_script:
        old = capture_from_wrapper(tree, workdir, name="compare", base_src=args.compare_script)
        result["compare"] = {
            "base_script": str(args.compare_script),
            "returncode": old.get("returncode"),
            "n_overrides": len(old.get("overrides", [])),
            "drift": diff_overrides(old.get("overrides", []), overrides),
        }

    # Effective baseline interval must be 2.0 either way: an unpatched tree has no key and
    # ReplayBuffer keeps its constructor default, an instrumented tree carries 2.0 in the
    # yaml. The poll arm must resolve to the value we pass.
    base_poll = arm_base.get("readback", {}).get("trainer.v1.sampler.poll_interval")
    poll_poll = arm_poll.get("readback", {}).get("trainer.v1.sampler.poll_interval")
    base_absent = isinstance(base_poll, str) and base_poll.startswith("<missing")
    base_default = base_absent or base_poll == 2.0
    result["poll_interval"] = {"base": base_poll, "poll": poll_poll,
                               "base_absent_as_expected": base_absent,
                               "base_effective_2s": base_default}
    result["poll_interval_reaches_trainer"] = bool(base_default and poll_poll == 0.1)

    # The run tree: the arm reads `training/sampler/blocking_polls`, so an un-instrumented
    # tree wastes the window silently. Check its markers, run the counter harness on it,
    # and prove the wrapper guard refuses a tree that lost one marker.
    run_tree = Path(args.run_tree).resolve()
    inst = instrumentation(run_tree)
    sim = sim_reading(run_tree, pick_sim_python(args.sim_python))
    guard = {"instrumented": wrapper_guard(run_tree, 0), "control": guard_control()}
    result["run_tree"] = {"path": str(run_tree), "instrumentation": inst, "sim": sim, "guard": guard}

    ok = (cap["returncode"] == 0 and not cap["check"]["problems"]
          and patch["usable"] and counter["usable"]
          and not arm_base["problems"] and not arm_poll["problems"]
          and result["negative_control"]["fired"]
          and result["poll_interval_reaches_trainer"]
          and inst["instrumented"] and sim["returncode"] == 0
          and guard["instrumented"]["ok"] and guard["control"]["ok"])
    if "compare" in result:
        ok = ok and result["compare"]["drift"]["same"]
    result["ok"] = ok

    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    print(text)
    shutil.rmtree(workdir, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
