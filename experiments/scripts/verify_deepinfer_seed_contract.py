"""Execute isolated DeepInfer seed/engine-type definitions without GPU imports.

Usage: python3 experiments/scripts/verify_deepinfer_seed_contract.py /path/to/DeepInfer
Baseline: 8b16149741a4935e1dec8cea27853f273b374c5b.
Only SamplingParams._make_seed and EngineType are compiled from the supplied
source AST. A seeded test Random instance supplies getrandbits for None inputs.
This is not a test of the sampler kernel, complete package, or GPU determinism.
"""
import argparse
import ast
import enum
import hashlib
import random
from pathlib import Path


def load_definition(path, class_name, method_name=None):
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == class_name)
    if method_name is not None:
        cls.body = [next(node for node in cls.body
                         if isinstance(node, ast.FunctionDef)
                         and node.name == method_name)]
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__",
                       names=[ast.alias(name="annotations")], level=0),
        cls,
    ], type_ignores=[])
    namespace = {"random": random.Random(2049), "enum": enum}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    print(f"{path.name} sha256={hashlib.sha256(source.encode()).hexdigest()}")
    return namespace[class_name], namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    root = parser.parse_args().source_root
    params, namespace = load_definition(
        root / "deepinfer/sampling_params.py", "SamplingParams", "_make_seed"
    )
    mask = (1 << 64) - 1
    cases = [(0, 0), (1, 1), (mask, mask), (-1, mask), (1 << 64, 0),
             ((1 << 64) + 7, 7), (-(1 << 64) - 1, mask),
             (True, 1), (False, 0)]
    checks = 0
    for value, expected in cases:
        assert params._make_seed(value) == expected
        checks += 1
    for invalid in (1.0, "1", b"1"):
        try:
            params._make_seed(invalid)
        except TypeError:
            checks += 1
        else:
            raise AssertionError(f"Expected TypeError: {invalid!r}")
    oracle = random.Random(2049)
    for _ in range(2):
        assert params._make_seed(None) == oracle.getrandbits(64)
        checks += 1
    namespace["random"] = random.Random(2049)
    assert params._make_seed(None) == random.Random(2049).getrandbits(64)
    checks += 1
    engine_type, _ = load_definition(
        root / "deepinfer/engine/engine_type.py", "EngineType"
    )
    for name, expected in (("NORMAL", True), ("PROMPT", False),
                           ("COMPLETION", True)):
        assert getattr(engine_type, name).default_zero_overhead_scheduling is expected
        checks += 1
    print(f"PASS: {checks} source-definition checks.")
    print("Explicit int seeds wrap modulo 2**64; None consumes the test PRNG.")
    print("Default async scheduling: NORMAL/COMPLETION on; PROMPT off.")
    print("No torch, DeepInfer package, model, or accelerator was imported.")


if __name__ == "__main__":
    main()
