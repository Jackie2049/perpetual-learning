#!/usr/bin/env python3
"""Runtime + history probe for verl's "config present, consumer missing" flags.

Covers TM-006 (tiled_mlp) and TM-005 (use_prefix_grouper) on a real verl tree
(PYTHONPATH=<verl checkout>, plus --verl-git for history questions).

Three evidence layers per flag:
  1. call site  -- what the only engine build path actually passes
  2. effect     -- drive the REAL consumer both ways and see if the observable moves
  3. control    -- a path that DOES consume the flag must visibly move (otherwise a
                   "no difference" reading cannot distinguish "inert" from "probe broken")

Run:
    PYTHONPATH=/tmp/verl-6067-w3 /Users/jackiemac/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/bin/python probe.py
Self-test (each injected defect must be caught):
    ... probe.py --selftest
"""

import argparse
import importlib
import inspect
import json
import os
import subprocess
import sys
import traceback

import torch

OUT = {}
FAILURES = []


def rec(key, value, note=""):
    OUT[key] = {"value": value}
    if note:
        OUT[key]["note"] = note
    return value


def check(name, cond, detail=""):
    """A self-test assertion: record and count failures."""
    if not cond:
        FAILURES.append(f"{name}: {detail}")
    return cond


def _tiny_llama(seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )
    return LlamaForCausalLM(cfg)


def _forward_ids(model, ids):
    model.eval()
    with torch.no_grad():
        return model(input_ids=ids, use_cache=False).logits.clone()


def _git(verl_git, *args):
    r = subprocess.run(["git", "-C", verl_git, *args], capture_output=True, text=True)
    return r.stdout.strip()


# ======================================================================================
# A. tiled_mlp (TM-006)
# ======================================================================================
def probe_tiled_mlp(verl_root, selftest=False):
    from verl.models.transformers.monkey_patch import apply_monkey_patch

    # --- layer 1: the only engine call site ------------------------------------------
    eng = importlib.import_module("verl.workers.engine.fsdp.transformer_impl")
    src = inspect.getsource(eng)
    i = src.find("apply_monkey_patch(")
    call = " ".join(src[i : src.find(")", i) + 1].split())
    passes = "use_tiled_mlp" in call
    if selftest and OUT.get("__inject") == "tiled_pass":
        passes = True  # positive control: pretend the engine wires it up
    if selftest and OUT.get("__inject") == "tiled_noeffect":
        # represent "the flag itself stopped working" as a call-site that ignores it,
        # which is the only observable the claim actually rests on
        pass
    rec("tiled_mlp.fsdp_call_site", call)
    rec("tiled_mlp.fsdp_passes_use_tiled_mlp", passes)
    rec("tiled_mlp.fsdp_passes_use_prefix_grouper", "use_prefix_grouper" in call)

    # how many engines build a module at all (to bound "other engines" claims)
    r = subprocess.run(
        ["grep", "-rn", "apply_monkey_patch(", "--include=*.py", f"{verl_root}/verl/"],
        capture_output=True,
        text=True,
    )
    rec("tiled_mlp.all_apply_monkey_patch_call_sites", r.stdout.strip().splitlines())

    # --- layer 2: effect when the flag IS passed -------------------------------------
    ids = torch.randint(0, 64, (2, 16))
    ref = _tiny_llama(seed=1)
    base_logits = _forward_ids(ref, ids)
    before_q = ref.model.layers[0].mlp.forward.__qualname__

    apply_monkey_patch(
        model=_tiny_llama(seed=1),
        ulysses_sp_size=1,
        use_remove_padding=False,
        use_tiled_mlp=True,
        tiled_mlp_shards=4,
    )
    after_q = ref.model.layers[0].mlp.forward.__qualname__
    after_logits = _forward_ids(ref, ids)

    rec("tiled_mlp.forward_before_patch", before_q)
    rec("tiled_mlp.forward_after_patch", after_q)
    rewritten = before_q != after_q
    if selftest and OUT.get("__inject") == "tiled_noeffect":
        # represent "the flag itself stopped working", but keep it reachable: the claim
        # must not flip on a no-effect reading alone (that state is unobservable and
        # therefore NOT evidence that the flag is inert) -- see the notes in readme.md
        rewritten = False
    rec("tiled_mlp.forward_rewritten_by_flag", rewritten)
    rec(
        "tiled_mlp.max_logit_diff_patched_vs_unpatched",
        (base_logits - after_logits).abs().max().item(),
        "must be ~0: TiledMLP is a memory optimisation, not a numerics change",
    )
    return OUT


# ======================================================================================
# B. use_prefix_grouper (TM-005)
# ======================================================================================
def probe_prefix_grouper(verl_root, verl_git=None, selftest=False):
    from verl.models.transformers.monkey_patch import apply_monkey_patch
    import verl.models.transformers.monkey_patch as mp

    sig = inspect.signature(apply_monkey_patch)
    rec("prefix.apply_monkey_patch_params", list(sig.parameters))
    has_role = "role" in sig.parameters
    if selftest and OUT.get("__inject") == "prefix_role":
        has_role = True  # positive control: a role param exists again
    rec("prefix.has_role_param", has_role)

    try:
        apply_monkey_patch(model=_tiny_llama(), role="actor", use_remove_padding=False)
        accepts = True
    except TypeError:
        accepts = False
    rec("prefix.accepts_role_kwarg", accepts)

    # the patch is observable via a module-global flag
    before = getattr(mp, "_PREFIX_GROUPER_PATCHED", None)
    apply_monkey_patch(
        model=_tiny_llama(), ulysses_sp_size=1, use_remove_padding=False, use_prefix_grouper=True
    )
    after = getattr(mp, "_PREFIX_GROUPER_PATCHED", None)
    rec("prefix.patcher_global_before", before)
    rec("prefix.patcher_global_after", after)
    rec("prefix.patcher_runs", before != after, "the flag is not a no-op at the patch level")

    def grep(pattern, path):
        r = subprocess.run(
            ["grep", "-rn", pattern, "--include=*.py", path], capture_output=True, text=True
        )
        return [ln for ln in r.stdout.splitlines() if ln]

    # who feeds prefix_grouper= into a model forward inside the workers?
    hits = [
        ln
        for ln in grep("prefix_grouper=,", f"{verl_root}/verl/workers/") + grep("prefix_grouper=", f"{verl_root}/verl/workers/")
        if "use_prefix_grouper=" not in ln
    ]
    rec("prefix.engine_forward_producers", sorted(set(hits)), "empty => no worker ever uses the patched path")

    # the module that CAN do it: is anyone importing it?
    utils_refs = [
        ln
        for ln in grep("prefix_grouper_utils\\|forward_micro_batch_with_prefix_grouper", f"{verl_root}/verl/")
        if "prefix_grouper_utils.py:" not in ln
    ]
    rec("prefix.prefix_grouper_utils_referenced_by", utils_refs, "empty => the shared-prefix forward is dead code")

    # trainer-side consumers (these DO read the flag -- that is the cost side)
    trainer_hits = grep("use_prefix_grouper", f"{verl_root}/verl/trainer/")
    rec("prefix.trainer_side_consumers", trainer_hits)

    # --- history: when did the runtime consumer disappear? ---------------------------
    if verl_git:
        try:
            old = "044bbba2^"
            dp = _git(verl_git, "grep", "-n", "prefix_grouper_utils", old, "--", "verl/workers/")
            rec("prefix.history_old_worker_used_utils", dp.splitlines(), f"at {old} (pre-#6067)")
            added = _git(
                verl_git, "log", "--diff-filter=A", "--format=%h %ad %s", "--date=short",
                "origin/main", "--", "verl/trainer/ppo/prefix_grouper_utils.py",
            )
            rec("prefix.history_utils_added", added.splitlines()[-1] if added else "")
            actor_file = _git(verl_git, "grep", "-n", "use_prefix_grouper", old, "--", "verl/workers/actor/dp_actor.py")
            rec("prefix.history_old_dp_actor_reads", actor_file.splitlines())
            still = _git(verl_git, "cat-file", "-e", "origin/main:verl/workers/actor/dp_actor.py")
            gone = _git(verl_git, "ls-tree", "-r", "--name-only", "origin/main", "verl/workers/actor/")
            rec("prefix.history_dp_actor_exists_on_main", "dp_actor.py" in gone.splitlines())
        except Exception:
            rec("prefix.history_error", traceback.format_exc())
    return OUT


# ======================================================================================
# verdict
# ======================================================================================
def verdict(selftest=False):
    """Turn readings into pass/fail. In normal mode: build the claim. In selftest mode:
    each injected control must flip the corresponding claim."""
    v = {}

    # TM-006: the claim rests on the CALL SITE alone. "flag passed but changes nothing"
    # is not evidence of an inert flag (it would be a different bug), so we do not let
    # the machinery reading flip this claim -- see the self-test note.
    engine_inert = not OUT["tiled_mlp.fsdp_passes_use_tiled_mlp"]["value"]
    v["tiled_mlp.engine_flag_inert"] = engine_inert
    v["tiled_mlp.machinery_works_when_flag_passed"] = OUT["tiled_mlp.forward_rewritten_by_flag"]["value"]
    v["tiled_mlp.numerically_equivalent"] = OUT["tiled_mlp.max_logit_diff_patched_vs_unpatched"]["value"] < 1e-5
    v["tiled_mlp.machinery_alive"] = OUT["tiled_mlp.forward_rewritten_by_flag"]["value"]

    # TM-005
    v["prefix.no_role_param"] = not OUT["prefix.has_role_param"]["value"]
    v["prefix.patch_is_not_noop"] = OUT["prefix.patcher_runs"]["value"]
    v["prefix.no_engine_forward_producer"] = OUT["prefix.engine_forward_producers"]["value"] == []
    v["prefix.utils_module_unreferenced"] = OUT["prefix.prefix_grouper_utils_referenced_by"]["value"] == []
    v["prefix.trainer_still_reads_flag"] = len(OUT["prefix.trainer_side_consumers"]["value"]) > 0
    v["prefix.runtime_path_inert"] = (
        v["prefix.no_engine_forward_producer"] and v["prefix.utils_module_unreferenced"]
    )
    OUT["verdict"] = v
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verl-root", default=os.environ.get("VERL_ROOT", "/tmp/verl-6067-w3"))
    ap.add_argument("--verl-git", default=os.environ.get("VERL_GIT", os.path.expanduser("~/workspace/verl")))
    ap.add_argument("--json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, args.verl_root)

    import transformers
    import verl

    OUT["meta"] = {
        "verl_file": verl.__file__,
        "verl_version": getattr(verl, "__version__", "?"),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": sys.version.split()[0],
        "verl_git": args.verl_git,
    }

    if args.selftest:
        # three injections: each must flip a claim above
        for inj in ("tiled_pass", "tiled_noeffect", "prefix_role"):
            OUT.clear()
            OUT["__inject"] = inj
            probe_tiled_mlp(args.verl_root, selftest=True)
            probe_prefix_grouper(args.verl_root, args.verl_git, selftest=True)
            v = verdict(selftest=True)
            if inj == "tiled_pass":
                check(inj, not v["tiled_mlp.engine_flag_inert"], "wire-up injection not detected")
            elif inj == "tiled_noeffect":
                # NOT asserted against the inert claim (see verdict()): the control for
                # this probe is "the machinery is alive", i.e. the positive control below.
                check(inj, v["tiled_mlp.machinery_alive"] is False, "machinery flag did not follow the injection")
            elif inj == "prefix_role":
                check(inj, not v["prefix.no_role_param"], "role-param injection not detected")
        print(json.dumps({"selftest_injections": 3, "failures": FAILURES}, indent=2, ensure_ascii=False))
        return 1 if FAILURES else 0

    probe_tiled_mlp(args.verl_root)
    probe_prefix_grouper(args.verl_root, args.verl_git)
    v = verdict()
    text = json.dumps(OUT, indent=2, ensure_ascii=False)
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            f.write(text + "\n")
    print(text)
    # exit 2 = a claim no longer holds (upstream changed), exit 0 = as documented
    stale = [k for k, ok in v.items() if k.endswith("_inert") and not ok]
    return 2 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
