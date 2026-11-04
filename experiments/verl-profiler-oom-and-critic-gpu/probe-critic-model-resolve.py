#!/usr/bin/env python3
"""Does `PrecisionDebuggerProfiler` resolve a model for the critic stages on the NEW worker?

The static finding says `critic_update` / `compute_values` have no trigger: no
`@DistProfiler.annotate(role=...)` exists for them after #6067. That is only half of the
story. `PrecisionDebuggerProfiler.start()` also needs `model`, obtained by
`_resolve_model(self_instance, stage)`, which walks `_MODEL_ATTRS_BY_ROLE[role]` on the
**worker object** and falls back to `self.module`.

After #6067 the critic is a standalone `TrainingWorker` (trainer_base.py: `Role.Critic ->
TrainingWorker`) that stores its model at `self.engine.module` and has neither `self.critic`
nor `self.module`. So the critic's candidate chains cannot resolve: adding the decorator
alone would still collect nothing (one "model not resolved" warning per step, zero dump).

This probe runs the real `_resolve_model` / `start` source (extracted with AST, no torch, no
msprobe, no GPU) against stand-ins whose attribute names are **read from the real worker
classes in the same tree**. It also runs the pre-#6067 tree as a historical control: there
`self.critic_module` existed and the same table resolved.

Exit codes
----------
0  conclusions hold for the requested preset
2  upstream changed (a critic-stage decorator appeared, or resolution now succeeds the naive way)
3  tree or file missing
4  --selftest failure

Usage
-----
    python3 probe-critic-model-resolve.py --tree /tmp/verl-31cfccb4 --preset main --json out.json
    python3 probe-critic-model-resolve.py --tree /tmp/verl-pre6067 --preset pre6067
    python3 probe-critic-model-resolve.py --selftest
"""

import argparse
import ast
import hashlib
import json
import logging
import os
import re
import sys
import types
from pathlib import Path
from typing import Optional

PROFILER_REL = "verl/utils/profiler/precision_debugger_profile.py"
WORKERS_REL = "verl/workers/engine_workers.py"
OLD_WORKERS_REL = "verl/workers/fsdp_workers.py"
TRAINER_REL = "verl/trainer/ppo/v1/trainer_base.py"
CRITIC_STAGES = ("compute_values", "critic_update", "compute_rm_score")
METHODS = (
    "_normalize_stage",
    "_resolve_attr",
    "_is_valid_model",
    "_get_candidate_attrs",
    "_resolve_model",
    "_should_collect",
    "start",
)


# --------------------------------------------------------------------------- extraction
def extract_module_source(path):
    """Return (dicts, method sources) taken verbatim from the real profiler module."""
    src = Path(path).read_text()
    mod = ast.parse(src)
    dicts, funcs, statics = {}, {}, set()
    for node in mod.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("_STAGE_TO_ROLE", "_MODEL_ATTRS_BY_ROLE", "_SKIP_STAGES"):
                dicts[name] = ast.literal_eval(node.value)
    for node in mod.body:
        if isinstance(node, ast.ClassDef) and node.name == "PrecisionDebuggerProfiler":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in METHODS:
                    funcs[item.name] = ast.get_source_segment(src, item)
                    if any(getattr(d, "id", None) == "staticmethod" for d in item.decorator_list):
                        statics.add(item.name)
    missing = sorted(set(METHODS) - set(funcs))
    return dicts, funcs, statics, missing, src


def class_attr_names(path, class_name):
    """Every `self.<attr> =` target that appears anywhere in the real class body.

    Returns (names, how). Local Python is 3.9 and pre-#6067 `fsdp_workers.py` uses `match`, so
    the AST path can fail; the regex fallback reads the class body line range instead. `how` is
    recorded in the readings so a fallback parse is never mistaken for a full one.
    """
    src = Path(path).read_text()
    try:
        mod = ast.parse(src)
    except SyntaxError:
        return _class_attr_names_regex(src, class_name), "regex-fallback(py39)"
    for node in ast.walk(mod):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            names = set()
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign):
                    for tgt in sub.targets:
                        if isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) \
                                and tgt.value.id == "self":
                            names.add(tgt.attr)
                elif isinstance(sub, ast.AnnAssign):
                    tgt = sub.target
                    if isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) \
                            and tgt.value.id == "self":
                        names.add(tgt.attr)
            return names, "ast"
    return None, "ast"


def _class_attr_names_regex(src, class_name):
    lines = src.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith(f"class {class_name}(") or line.startswith(f"class {class_name}:"):
            start = i
            break
    if start is None:
        return None
    names = set()
    pat_eq = re.compile(r"(?<![=!<>+\-*/])=(?!=)")
    pat_self = re.compile(r"self\.(\w+)")
    for line in lines[start + 1:]:
        if line and not line[0].isspace() and not line.startswith(("class ", "def ", "@", "#")):
            break
        body = line.strip()
        m = pat_eq.search(body)
        if not body.startswith("self.") or not m:
            continue
        names.update(pat_self.findall(body[:m.start()]))
    return names


def count_decorators(root):
    """Literal `role="<critic stage>"` occurrences anywhere under verl/**/*.py."""
    hits = {}
    for py in Path(root, "verl").rglob("*.py"):
        text = py.read_text(errors="replace")
        for stage in CRITIC_STAGES:
            n = text.count(f'role="{stage}"')
            if n:
                hits.setdefault(stage, []).append(str(py.relative_to(root)))
    return hits


# --------------------------------------------------------------------------- probe object
class Sentinel:
    """Stands in for a non-model attribute the real class also assigns."""


class FakeModule:
    def forward(self, *a, **k):  # only needs to be callable for _is_valid_model
        return None


class EngineLike:
    def __init__(self):
        self.module = FakeModule()


PY39_SHIM = ("list | tuple", "(list, tuple)")


def py39_shims(funcs):
    """Count the only Python>=3.10 runtime syntax in the extracted methods.

    `_resolve_model` contains `isinstance(value, list | tuple)`; PEP 604 at runtime is 3.10+.
    The local interpreter is 3.9, so that one expression is rewritten for execution. The count
    is recorded and asserted, so an upstream rewrite cannot silently pass through the shim.
    """
    counts = {name: src.count(PY39_SHIM[0]) for name, src in funcs.items()}
    return counts, sum(counts.values())


def build_probe(dicts, funcs, statics=()):
    ns = {"os": os, "logging": logging, "Optional": Optional}
    ns.update(dicts)
    ns["logger"] = logging.getLogger("probe")
    for name, src in funcs.items():
        src = src.replace(*PY39_SHIM)
        exec(compile(src, f"<real:{name}>", "exec"), ns)
    cls = type("Probe", (), {})
    for name in METHODS:
        setattr(cls, name, staticmethod(ns[name]) if name in statics else ns[name])
    obj = cls()
    return obj, ns


def make_worker(attr_names, containers):
    """A stand-in carrying exactly the attribute names the real class assigns."""
    w = types.SimpleNamespace()
    for name in attr_names:
        setattr(w, name, Sentinel())
    for name, obj in containers.items():
        setattr(w, name, obj)
    return w


# --------------------------------------------------------------------------- cases
def run_preset(tree, preset):
    tree = Path(tree)
    prof = tree / PROFILER_REL
    if not prof.is_file():
        return {"error": f"missing {prof}"}, 3, None
    dicts, funcs, statics, missing, prof_src = extract_module_source(prof)
    if missing:
        return {"error": f"methods not found in {PROFILER_REL}: {missing}"}, 3, None
    shim_counts, shim_total = py39_shims(funcs)
    probe, ns = build_probe(dicts, funcs, statics)
    readings = {
        "tree": str(tree),
        "preset": preset,
        "sha256": {str(prof.relative_to(tree)): hashlib.sha256(prof_src.encode()).hexdigest()},
        "py39_shims": {"counts": shim_counts, "total": shim_total},
        "staticmethods": sorted(statics),
        "tables": {k: {kk: list(vv) if isinstance(vv, tuple) else vv for kk, vv in v.items()}
                   for k, v in dicts.items() if k.endswith("_BY_ROLE") or k == "_STAGE_TO_ROLE"},
        "cases": [],
    }

    def case(name, obj, stage, expect, table=None):
        if table is not None:
            old = ns["_MODEL_ATTRS_BY_ROLE"]
            ns["_MODEL_ATTRS_BY_ROLE"] = table
        try:
            got = probe._resolve_model(obj, stage)
        finally:
            if table is not None:
                ns["_MODEL_ATTRS_BY_ROLE"] = old
        ok = (got is None and expect == "none") or (isinstance(got, FakeModule) and expect == "module")
        readings["cases"].append({"case": name, "stage": stage, "expected": expect,
                                  "got": "none" if got is None else type(got).__name__, "ok": ok})
        return ok

    if preset == "main":
        workers_file = tree / WORKERS_REL
        if not workers_file.is_file():
            return {"error": f"missing {workers_file}"}, 3, None
        readings["sha256"][WORKERS_REL] = hashlib.sha256(workers_file.read_bytes()).hexdigest()
        trainer = tree / TRAINER_REL
        if trainer.is_file():
            readings["sha256"][TRAINER_REL] = hashlib.sha256(trainer.read_bytes()).hexdigest()
        tw_attrs, tw_how = class_attr_names(workers_file, "TrainingWorker")
        aruw_attrs, ar_how = class_attr_names(workers_file, "ActorRolloutRefWorker")
        if tw_attrs is None or aruw_attrs is None:
            return {"error": "worker classes not found"}, 2, None
        readings["worker_attrs"] = {"TrainingWorker": sorted(tw_attrs),
                                    "ActorRolloutRefWorker": sorted(aruw_attrs)}
        readings["parse_mode"] = {"TrainingWorker": tw_how, "ActorRolloutRefWorker": ar_how}
        readings["critic_stage_decorators"] = count_decorators(tree)
        # Guards: the layout claim itself must still hold, otherwise the stand-in is fake.
        for must in ("engine",):
            if must not in tw_attrs:
                readings["error"] = f"TrainingWorker no longer assigns self.{must}"
                return readings, 2, None
        for must in ("module", "critic"):
            if must in tw_attrs:
                readings["error"] = f"TrainingWorker now assigns self.{must}: layout changed"
                return readings, 2, None

        critic_worker = make_worker(tw_attrs, {"engine": EngineLike()})
        actor_worker = make_worker(
            aruw_attrs,
            {"actor": make_worker(tw_attrs, {"engine": EngineLike()}),
             "ref": make_worker(tw_attrs, {"engine": EngineLike()})},
        )
        ok = []
        ok.append(case("critic_update on TrainingWorker", critic_worker, "critic_update", "none"))
        ok.append(case("compute_values on TrainingWorker", critic_worker, "compute_values", "none"))
        ok.append(case("actor_update on ActorRolloutRefWorker", actor_worker, "actor_update", "module"))
        ok.append(case("actor_compute_log_prob", actor_worker, "actor_compute_log_prob", "module"))
        ok.append(case("ref_compute_log_prob", actor_worker, "ref_compute_log_prob", "module"))

        # Candidate fix: one generic entry appended to every role's chain. Actor/ref must not move.
        fixed = {k: tuple(v) + ("engine.module",) for k, v in ns["_MODEL_ATTRS_BY_ROLE"].items()}
        ok.append(case("FIX: critic_update with engine.module fallback", critic_worker,
                       "critic_update", "module", table=fixed))
        ok.append(case("FIX: compute_values with engine.module fallback", critic_worker,
                       "compute_values", "module", table=fixed))
        ok.append(case("FIX: actor_update unchanged", actor_worker, "actor_update", "module",
                       table=fixed))
        ok.append(case("FIX: ref_compute_log_prob unchanged", actor_worker,
                       "ref_compute_log_prob", "module", table=fixed))

        # Observable consequence: start() returns False, warns once, writes no directory.
        prof_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "m5-probe-resolve-dump"
        if prof_dir.exists():
            import shutil
            shutil.rmtree(prof_dir)
        fake = types.SimpleNamespace(
            precision_cfg=types.SimpleNamespace(config_path="cfg.json", strict=False),
            _available=True, _dump_root=str(prof_dir), _debugger=None,
            _current_global_step=None, _stages={"critic_update"},
        )
        for name in ("_normalize_stage", "_should_collect", "_is_valid_model", "start"):
            setattr(fake, name, ns[name] if name in statics else types.MethodType(ns[name], fake))
        records = []

        class Capture(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = Capture()
        ns["logger"].addHandler(handler)
        ns["logger"].setLevel(logging.DEBUG)
        try:
            started = fake.start(stage="critic_update", global_step=1, model=None)
        finally:
            ns["logger"].removeHandler(handler)
        readings["start_with_unresolved_model"] = {
            "returned": bool(started),
            "dump_root_created": prof_dir.exists(),
            "warnings": records,
        }
        ok.append(readings["start_with_unresolved_model"]["returned"] is False
                  and not readings["start_with_unresolved_model"]["dump_root_created"]
                  and any("not resolved" in r for r in records))

        decorators_absent = not readings["critic_stage_decorators"]
        readings["verdict"] = {
            "critic_decorators_absent": decorators_absent,
            "resolution_none_on_training_worker": all(c["ok"] for c in readings["cases"][:2]),
            "actor_positive_controls_ok": all(c["ok"] for c in readings["cases"][2:5]),
            "candidate_fix_works": all(c["ok"] for c in readings["cases"][5:9]),
            "start_path_confirms": ok[-1],
        }
        good = decorators_absent and all(readings["verdict"].values())
        return readings, (0 if good else 2), ok

    # pre-#6067 historical control
    old_file = tree / OLD_WORKERS_REL
    if not old_file.is_file():
        return {"error": f"missing {old_file}"}, 3, None
    readings["sha256"][OLD_WORKERS_REL] = hashlib.sha256(old_file.read_bytes()).hexdigest()
    attrs, how = class_attr_names(old_file, "CriticWorker")
    if attrs is None:
        attrs, how = class_attr_names(old_file, "ActorRolloutRefWorker")
    if attrs is None:
        return {"error": "old critic worker class not found"}, 2, None
    readings["worker_attrs"] = {"CriticWorker": sorted(attrs)}
    readings["parse_mode"] = {"CriticWorker": how}
    readings["critic_stage_decorators"] = count_decorators(tree)
    if "critic_module" not in attrs:
        readings["error"] = "old worker no longer assigns self.critic_module"
        return readings, 2, None
    old_worker = make_worker(attrs, {"critic_module": FakeModule()})
    ok = [case("critic_update on old composite worker", old_worker, "critic_update", "module"),
          case("compute_values on old composite worker", old_worker, "compute_values", "module")]
    has = {s: bool(readings["critic_stage_decorators"].get(s)) for s in ("critic_update", "compute_values")}
    readings["verdict"] = {
        "decorators_present": has,
        "resolution_ok_on_old_layout": all(ok),
    }
    good = all(has.values()) and all(ok)
    return readings, (0 if good else 2), ok


# --------------------------------------------------------------------------- selftest
def selftest():
    results = []

    def check(name, cond):
        results.append((name, bool(cond)))

    tree = Path(os.environ.get("M5_MAIN_TREE", "/tmp/verl-31cfccb4"))
    if not (tree / PROFILER_REL).is_file():
        print(f"[selftest] SKIP: no tree at {tree}")
        return 4
    dicts, funcs, statics, missing, _ = extract_module_source(tree / PROFILER_REL)
    check("extraction complete", not missing)
    shim_counts, shim_total = py39_shims(funcs)
    check("py39 shim count is exactly 1 (list | tuple)", shim_total == 1)
    probe, ns = build_probe(dicts, funcs, statics)

    tw, _how = class_attr_names(tree / WORKERS_REL, "TrainingWorker")
    tw = tw or set()
    critic_worker = make_worker(tw, {"engine": EngineLike()})
    check("baseline: critic_update unresolved", probe._resolve_model(critic_worker, "critic_update") is None)

    # Perturbation 1: give the stand-in a `critic` container -> resolution must flip.
    mutated = make_worker(tw, {"engine": EngineLike()})
    mutated.critic = make_worker(tw, {"engine": EngineLike()})
    check("perturbation: critic container flips resolution",
          isinstance(probe._resolve_model(mutated, "critic_update"), FakeModule))

    # Perturbation 2: give it `module` -> the documented fallback fires.
    mutated2 = make_worker(tw, {"engine": EngineLike()})
    mutated2.module = FakeModule()
    check("perturbation: module fallback flips resolution",
          isinstance(probe._resolve_model(mutated2, "critic_update"), FakeModule))

    # Perturbation 3: table gains engine.module -> flips without touching the worker.
    fixed = {k: tuple(v) + ("engine.module",) for k, v in ns["_MODEL_ATTRS_BY_ROLE"].items()}
    ns["_MODEL_ATTRS_BY_ROLE"] = fixed
    check("perturbation: table fix flips resolution",
          isinstance(probe._resolve_model(critic_worker, "critic_update"), FakeModule))
    ns["_MODEL_ATTRS_BY_ROLE"] = dicts["_MODEL_ATTRS_BY_ROLE"]
    check("restore: original table back", probe._resolve_model(critic_worker, "critic_update") is None)

    # Perturbation 4: unknown stage still rejected (the probe must not accept anything).
    check("guard: unknown stage unresolved", probe._resolve_model(critic_worker, "train_batch") is None)

    failed = [n for n, ok in results if not ok]
    for n, ok in results:
        print(f"[selftest] {'PASS' if ok else 'FAIL'} {n}")
    print(f"[selftest] {len(results) - len(failed)}/{len(results)} passed")
    return 0 if not failed else 4


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tree", default="/tmp/verl-31cfccb4")
    ap.add_argument("--preset", choices=("main", "pre6067"), default=None)
    ap.add_argument("--json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    preset = args.preset
    if preset is None:
        preset = "main" if Path(args.tree, WORKERS_REL).is_file() else "pre6067"
    readings, rc, _ = run_preset(args.tree, preset)
    text = json.dumps(readings, indent=1, sort_keys=True)
    if args.json:
        Path(args.json).write_text(text + "\n")
    for c in readings.get("cases", []):
        print(f"[{'ok' if c['ok'] else 'FAIL'}] {c['case']}: got={c['got']} want={c['expected']}")
    if "verdict" in readings:
        print(json.dumps(readings["verdict"], indent=1, sort_keys=True))
    if "error" in readings:
        print(f"[error] {readings['error']}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
