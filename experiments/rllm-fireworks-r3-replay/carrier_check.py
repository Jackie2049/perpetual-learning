"""Checks whether `tinker.ModelInput` can carry the routing matrices that rLLM
attaches to it under router_replay=R3, and whether the rLLM verl path (the
known-good consumer) expects a different shape.

  (1) decl     : is `routing_matrices` a declared field of tinker.ModelInput?
  (2) ctor     : does `ModelInput.from_ints(..., routing_matrices=...)` exist?
                 (this is how fireworks-training-cookbook builds R3 datums)
  (3) assign   : can the attribute be set on an existing instance?
  (4) serial   : does an assigned attribute survive model_dump / model_validate
                 (the trip to the training service)?
  (5) shape    : which element type does each path produce?
                 rLLM verl path -> (tokens, num_layers, topk) tensor
                 rLLM tinker path -> str
                 cookbook      -> List[str]

Exit 0 = the carrier gap reproduces.
Exit 1 = ModelInput now carries routing matrices (rLLM's assignment is fine).
"""
from __future__ import annotations

import inspect
import sys

import tinker


def main() -> int:
    from importlib.metadata import version

    print(f"[env] tinker {version('tinker')}")

    fields = list(getattr(tinker.ModelInput, "model_fields", {}))
    declared = "routing_matrices" in fields
    print(f"[1] ModelInput declared fields = {fields}; carries routing_matrices = {declared}")

    try:
        sig = inspect.signature(tinker.ModelInput.from_ints)
        ctor = "routing_matrices" in sig.parameters
        print(f"[2] ModelInput.from_ints{sig}; accepts routing_matrices = {ctor}")
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        ctor = False
        print(f"[2] from_ints signature unavailable: {exc}")

    mi = tinker.ModelInput(chunks=[tinker.EncodedTextChunk(tokens=[11, 12, 13])])
    try:
        mi.routing_matrices = ["m0", "m1", "m2"]
        assigned = True
        print("[3] attribute assignment on a live instance: accepted")
    except Exception as exc:  # frozen pydantic models reject unknown fields
        assigned = False
        print(f"[3] attribute assignment rejected: {type(exc).__name__}")

    if assigned:
        roundtrip = tinker.ModelInput.model_validate(mi.model_dump())
        survives = getattr(roundtrip, "routing_matrices", None) == ["m0", "m1", "m2"]
        print(f"[4] survives model_dump/model_validate = {survives}")
    else:
        survives = False
        print("[4] nothing to serialize")

    print("[5] element type in rLLM's tinker path = str (transform.py passes the strings through)")
    print("    element type in rLLM's verl path    = torch tensor (len, num_layers, topk):")
    print("    rllm/trainer/verl/dataclass.py:23 and rllm/engine/rollout/verl_engine.py:144-148")

    defect = not (declared and ctor and assigned and survives)
    print()
    print(f"carrier gap reproduces = {defect}")
    return 0 if defect else 1


if __name__ == "__main__":
    sys.exit(main())
