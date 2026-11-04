#!/usr/bin/env python3
"""Decide which branch of verl's `load_valuehead_model` builds the critic model.

Why this exists
---------------
`run-critic-pdbg.sh` treats `/tmp/m5-trl-0.27.0` as a hard dependency and aborts the
whole window when it is missing. But `verl/utils/model.py:load_valuehead_model` tries
`AutoModelForTokenClassification.from_pretrained(...)` FIRST and only falls back to trl's
`AutoModelForCausalLMWithValueHead` if that raises. If the critic config class is in the
token-classification mapping, branch 1 wins and trl is never imported -- so a missing trl
target must not abort the run, and a *needed* trl target must.

How the branch is decided
-------------------------
`transformers/models/auto/auto_factory.py:_get_model_class` resolves
`model_mapping[type(config)]`, i.e. `config.__class__.__name__ -> model_type -> class`.
A causal-LM checkpoint does NOT make that lookup fail: `config.architectures` is only used
to disambiguate a list of candidate classes, so a Qwen2 LM checkpoint silently yields
`Qwen2ForTokenClassification` with a fresh `score` head. That is why branch 1 wins for Qwen2
and why this is a config-class question, not a weights question.

Reads config.json only: no weights, no GPU, no training.

Exit codes (the wrapper branches on these)
------------------------------------------
0   the winning branch is usable on this box (branch 1, or branch 2 with trl importable)
65  branch 2 wins and trl is not importable -> the run would die in minutes, abort now
2   cannot decide (transformers/mapping unreadable) -> caller must apply the strict rule
4   --selftest failure

Usage
-----
    python3 probe-valuehead-branch.py --model /home/zxw/models/Qwen2.5-0.5B-Instruct \
        --json /tmp/m5-pdbg-critic/valuehead-branch.json
    python3 probe-valuehead-branch.py --selftest
"""

import argparse
import json
import sys

EXIT_USABLE, EXIT_ABORT, EXIT_UNDECIDABLE, EXIT_SELFTEST = 0, 65, 2, 4


def decide(mapped, trl_available):
    """Pure verdict: `mapped` = critic config class is in the token-classification mapping."""
    if mapped:
        return {"branch": "token_classification", "trl_required": False, "usable": True}
    return {"branch": "trl_valuehead", "trl_required": True, "usable": bool(trl_available)}


def trl_import_path():
    """Which import the verl fallback would take (`trl.experimental.ppo` first, then `trl`)."""
    for mod, attr in (("trl.experimental.ppo", "AutoModelForCausalLMWithValueHead"),
                      ("trl", "AutoModelForCausalLMWithValueHead")):
        try:
            m = __import__(mod, fromlist=[attr])
        except Exception:
            continue
        if hasattr(m, attr):
            return mod
    return None


def lookup(mapping, cls):
    """Same two calls the probe makes on transformers' mapping object; split out to be testable.

    `_LazyAutoMapping` (auto_factory.py) provides `__contains__` and `get(key, default)`, and
    `_get_model_class` itself resolves `model_mapping[type(config)]` -- so membership is exactly
    the question "would branch 1 find a class".
    """
    return bool(cls in mapping), mapping.get(cls, None)


def probe(model_path):
    """Return (report, exit_code). Reads config.json only."""
    try:
        import transformers
        from transformers import AutoConfig, AutoModelForTokenClassification
    except Exception as exc:  # no transformers on PYTHONPATH -> cannot decide here
        return {"decided": False, "error": f"{type(exc).__name__}: {exc}"}, EXIT_UNDECIDABLE

    report = {"model_path": model_path, "transformers_version": transformers.__version__}
    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    except Exception as exc:
        return {**report, "decided": False, "error": f"AutoConfig: {type(exc).__name__}: {exc}"}, EXIT_UNDECIDABLE

    # The engine mutates these before the call (verl/workers/engine/fsdp/transformer_impl.py:291-300);
    # num_labels is what makes the fresh head emit one value per token.
    cfg.num_labels = 1
    for attr, value in (("classifier_dropout", 0.0), ("hidden_dropout", "0"), ("summary_dropout_prob", 0.0)):
        setattr(cfg, attr, value)

    mapping = getattr(AutoModelForTokenClassification, "_model_mapping", None)
    if mapping is None:
        return {**report, "decided": False, "error": "AutoModelForTokenClassification has no _model_mapping"}, EXIT_UNDECIDABLE

    try:
        mapped, resolved = lookup(mapping, type(cfg))
    except Exception as exc:
        return {**report, "decided": False, "error": f"mapping lookup: {type(exc).__name__}: {exc}"}, EXIT_UNDECIDABLE

    path = trl_import_path()
    verdict = decide(mapped, path is not None)
    report.update({
        "decided": True,
        "config_class": type(cfg).__name__,
        "config_model_type": getattr(cfg, "model_type", None),
        "token_classification_mapping_hit": bool(mapped),
        "resolved_class": getattr(resolved, "__name__", None),
        "trl_import_path": path,
        **verdict,
    })
    return report, EXIT_USABLE if verdict["usable"] else EXIT_ABORT


def selftest():
    results = []

    def case(name, got, want):
        results.append((name, got == want, got, want))

    # The three truth-table rows that matter: "mapped but no trl" is the state the wrapper
    # currently aborts on and must not; "unmapped and no trl" is the only real abort.
    case("mapped_no_trl_is_usable", decide(True, False),
         {"branch": "token_classification", "trl_required": False, "usable": True})
    case("unmapped_with_trl_is_usable", decide(False, True),
         {"branch": "trl_valuehead", "trl_required": True, "usable": True})
    case("unmapped_no_trl_aborts", decide(False, False),
         {"branch": "trl_valuehead", "trl_required": True, "usable": False})

    # The lookup code path against a stand-in mirroring `_LazyAutoMapping`'s contract (membership
    # by config class + `get(key, default)`). The real mapping object cannot be built on this box
    # (no transformers under python 3.9), so the contract is what is tested here; on the server
    # any exception degrades to "undecidable", which keeps the caller's strict rule.
    class _Mapping:
        def __init__(self, table):
            self._t = table

        def __contains__(self, key):
            return key in self._t

        def get(self, key, default=None):
            return self._t.get(key, default)

    class _MappedCfg:
        pass

    class _UnmappedCfg:
        pass

    class _ResolvedCls:
        pass

    table = _Mapping({_MappedCfg: _ResolvedCls})
    case("lookup_mapped", lookup(table, _MappedCfg), (True, _ResolvedCls))
    case("lookup_unmapped", lookup(table, _UnmappedCfg), (False, None))

    # Real-mapping cases only when transformers is importable; skipped otherwise (reported).
    # They are the actual measurement: qwen2 is mapped (branch 1), a bare PretrainedConfig is not.
    real, skipped, info = [], [], []
    try:
        import transformers
        from transformers import AutoModelForTokenClassification, Qwen2Config as _Qwen2Config

        m = AutoModelForTokenClassification._model_mapping
        real.append(("real_qwen2_mapped", type(_Qwen2Config()) in m, True))
        real.append(("real_qwen2_resolves",
                     getattr(m.get(type(_Qwen2Config()), None), "__name__", None),
                     "Qwen2ForTokenClassification"))
        real.append(("real_unmapped_config", type(transformers.PretrainedConfig()) in m, False))
        info.append(("real_trl_import_path", trl_import_path()))
    except Exception as exc:
        skipped.append(f"transformers real-mapping cases skipped: {type(exc).__name__}: {exc}")

    results.extend(real)

    failed = [r for r in results if not r[1]]
    for name, ok, got, want in results:
        print(f"[selftest] {'PASS' if ok else 'FAIL'} {name}: got={got} want={want}")
    for line in skipped:
        print(f"[selftest] SKIP {line}")
    for name, got in info:
        print(f"[selftest] INFO {name}: {got}")
    print(f"[selftest] {len(results) - len(failed)}/{len(results)} passed")
    return EXIT_SELFTEST if failed else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", help="critic model path (config.json is read, nothing else)")
    ap.add_argument("--json", help="write the report here")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if not args.model:
        ap.error("--model is required unless --selftest")

    report, rc = probe(args.model)
    text = json.dumps(report, indent=1, sort_keys=True)
    print(text)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(text + "\n")
    if not report.get("decided"):
        print(f"[verdict] UNDECIDABLE (rc={rc}): {report.get('error')}")
    elif report["token_classification_mapping_hit"]:
        print(f"[verdict] branch 1: {report['resolved_class']} (num_labels=1 head); "
              f"trl not on this path (importable={report['trl_import_path'] is not None})")
    else:
        print(f"[verdict] branch 2: trl {report['trl_import_path'] or 'MISSING'} "
              f"-> trl_required={report['trl_required']} usable={report['usable']}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
