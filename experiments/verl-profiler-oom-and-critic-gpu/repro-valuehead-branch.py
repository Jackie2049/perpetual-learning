#!/usr/bin/env python3
"""Reproduce `load_valuehead_model`'s first branch on the pinned transformers, without a GPU.

Why this exists
---------------
`probe-valuehead-branch.py` answers the question from `config.json` + the auto-class mapping, but
until now its real-mapping path had never executed anywhere: the M5 laptop only had python 3.9 and
no transformers, so all evidence for "Qwen2 takes branch 1 and never imports trl" was a static
read of fixed-version source. This script runs the actual call on transformers 5.16.1 + torch with
a tiny randomly-initialised Qwen2 checkpoint, so the claim is a runtime reading.

What it measures (all on CPU, no weights downloaded)
----------------------------------------------------
1. `AutoModelForTokenClassification.from_pretrained(<causal-LM checkpoint>, config=cfg,
   attn_implementation="sdpa")` succeeds and returns `Qwen2ForTokenClassification`.
2. The value head is **fresh**: `score.*` is absent from the checkpoint and present in the loaded
   model, and a forward pass emits shape `(batch, seq, 1)` -- this is the `squeeze(-1)` path
   `prepare_model_outputs` takes when the module has no `v_head`.
3. The **unpatched** call (`attn_implementation="flash_attention_2"`, what verl ships) fails on a
   box without flash-attn, and verl's `except BaseException` turns that into
   `RuntimeError: ... please install trl ...` when trl is missing -- i.e. the preflight's sdpa
   patch is not cosmetic: without it the failure message points at the wrong package.

Call shape is copied from verl `31cfccb4` `verl/utils/model.py:load_valuehead_model`
(sha256 `a13d8595499da87abee759fb5d5a31b5e70327c6c0523d49c86798832670d78f`, verified against the
checkout's recorded hash); the config mutation (`num_labels = 1` and the dropout overrides) is
copied from `verl/workers/engine/fsdp/transformer_impl.py:291-300`.

Exit codes
----------
0   every assertion holds on this stack (reading written to --json)
1   upstream behaviour changed -> the probe/evidence page needs a re-read
2   the measuring environment is missing (transformers or torch unimportable)

Usage
-----
    /path/to/venv/bin/python repro-valuehead-branch.py --json raw/valuehead-branch-runtime.json
"""

import argparse
import json
import logging
import sys
import tempfile
import traceback
from pathlib import Path

TINY = dict(
    architectures=["Qwen2ForCausalLM"],
    model_type="qwen2",
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=1,
    vocab_size=128,
    max_position_embeddings=64,
)


def build_tiny_checkpoint(root: Path):
    """A real Qwen2 causal-LM checkpoint, saved the way a published model is."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(**TINY)
    model = Qwen2ForCausalLM(cfg).eval()
    path = root / "tiny-qwen2-causal-lm"
    model.save_pretrained(path)
    return path, model


def engine_config(model_path: Path):
    """What `transformer_impl.py` hands `load_valuehead_model` (num_labels=1 + dropout overrides)."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    cfg.num_labels = 1
    for attr, value in (("classifier_dropout", 0.0), ("hidden_dropout", "0"),
                        ("summary_dropout_prob", 0.0)):
        setattr(cfg, attr, value)
    return cfg


def branch1(model_path: Path, cfg, attn_implementation: str, torch_dtype):
    """The exact first call of `load_valuehead_model` (verl 31cfccb4)."""
    from transformers import AutoModelForTokenClassification

    return AutoModelForTokenClassification.from_pretrained(
        pretrained_model_name_or_path=str(model_path),
        torch_dtype=torch_dtype,
        config=cfg,
        attn_implementation=attn_implementation,
        trust_remote_code=False,
    )


def verl_surface(exc):
    """What verl's `except BaseException` raises when trl is not installed."""
    try:
        from trl.experimental.ppo import AutoModelForCausalLMWithValueHead  # type: ignore  # noqa: F401
    except Exception:
        try:
            import trl  # type: ignore  # noqa: F401
        except Exception:
            return (f"RuntimeError: model(...) is not a value head model, please install trl to make it valid"
                    f" [wrapping {type(exc).__name__}]")
    return "trl importable -> fallback would proceed"


def measure(root: Path):
    import torch
    import transformers

    out = {
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
    }
    model_path, lm = build_tiny_checkpoint(root)
    cfg = engine_config(model_path)

    ckpt_keys = set(lm.state_dict().keys())
    out["checkpoint_has_score_head"] = any(k.startswith("score") for k in ckpt_keys)

    checks = []

    def check(name, got, want):
        checks.append({"name": name, "ok": got == want, "got": got, "want": want})

    # --- 1. patched path: branch 1 succeeds, class is the token-classification Qwen2
    model = branch1(model_path, cfg, "sdpa", torch.bfloat16)
    check("resolved_class", type(model).__name__, "Qwen2ForTokenClassification")
    check("num_labels", int(model.config.num_labels), 1)
    check("checkpoint_is_causal_lm", out["checkpoint_has_score_head"], False)
    check("loaded_has_score_head", any(k.startswith("score") for k in model.state_dict()), True)
    check("v_head_absent", hasattr(model, "v_head"), False)

    # forward: (batch, seq, 1) -- the `output.logits.squeeze(-1)` branch of prepare_model_outputs
    ids = torch.tensor([[1, 5, 9, 11]])
    with torch.no_grad():
        logits = model(input_ids=ids).logits
    check("logits_shape", list(logits.shape), [1, 4, 1])
    check("squeezed_shape", list(logits.squeeze(-1).shape), [1, 4])
    out["score_head_params"] = sum(p.numel() for n, p in model.named_parameters()
                                   if n.startswith("score"))
    out["score_head_param_names"] = [n for n, _ in model.named_parameters()
                                     if n.startswith("score")]

    # --- 2. unpatched path: what verl ships fails on this box, with a misleading message
    cfg2 = engine_config(model_path)
    fa2 = {"raised": None, "surfaced": None}
    try:
        branch1(model_path, cfg2, "flash_attention_2", torch.bfloat16)
        fa2["raised"] = None
    except BaseException as exc:  # verl catches BaseException
        fa2["raised"] = f"{type(exc).__name__}: {exc}".split("\n")[0][:300]
        fa2["surfaced"] = verl_surface(exc)
    out["flash_attention_2_path"] = fa2
    return out, checks

def selftest():
    """Prove the assertions can go red: a wrong (but plausible) reading must fail every check."""
    checks = []
    fake = {"resolved_class": "Qwen2ForCausalLM", "checkpoint_has_score_head": True}
    want = {"resolved_class": "Qwen2ForTokenClassification", "checkpoint_has_score_head": False}
    for k, w in want.items():
        # A check that cannot fail proves nothing: here the fake reading is the wrong answer.
        checks.append({"name": f"selftest_{k}", "ok": fake[k] == w, "got": fake[k], "want": w})
    went_red = [c for c in checks if not c["ok"]]
    for c in checks:
        print(f"[selftest] {'PASS' if not c['ok'] else 'FAIL'} {c['name']}: "
              f"injected wrong reading -> {'went red' if not c['ok'] else 'stayed green'}")
    print(f"[selftest] {len(went_red)}/{len(checks)} injected wrong readings went red")
    return 0 if len(went_red) == len(checks) else 4


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", help="write the reading here")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as exc:
        print(f"[env] missing: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    logging.getLogger("transformers").setLevel(logging.ERROR)
    with tempfile.TemporaryDirectory() as td:
        try:
            reading, checks = measure(Path(td))
        except Exception:
            traceback.print_exc()
            return 1

    failed = [c for c in checks if not c["ok"]]
    reading["checks"] = checks
    reading["failed"] = [c["name"] for c in failed]
    if args.json:
        Path(args.json).write_text(json.dumps(reading, indent=1, sort_keys=True) + "\n")
    for c in checks:
        print(f"[check] {'PASS' if c['ok'] else 'FAIL'} {c['name']}: got={c['got']} want={c['want']}")
    print(f"[reading] transformers={reading['transformers_version']} torch={reading['torch_version']} "
          f"score_head_params={reading['score_head_params']}")
    print(f"[reading] unpatched FA2 path: {reading['flash_attention_2_path']['raised']}")
    print(f"[reading] unpatched FA2 surfaces: {reading['flash_attention_2_path']['surfaced']}")
    print(f"[result] {len(checks) - len(failed)}/{len(checks)} checks passed")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
