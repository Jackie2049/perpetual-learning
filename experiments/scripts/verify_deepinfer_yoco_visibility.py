#!/usr/bin/env python3
"""Check the original YoCo C++ visibility helper, not full metadata or attention.

Reuses upstream L1 case shapes/reference via AST, with an explicit valid_start
extension for zero-state recompute. --cp-reference-only checks composition of
the upstream pure-Python CP index references, without C++ compilation.
No torch, extension build, model or device.
"""

import __future__
import argparse
import ast
from bisect import bisect_right
from collections import Counter
import hashlib
from itertools import accumulate, zip_longest
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import types
import sys


PRELUDE = r"""
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
struct Message { template<class T> Message &operator<<(const T &) { return *this; } };
Message require(bool ok) { if (!ok) throw std::runtime_error("visibility TORCH_CHECK failed"); return {}; }
#define TORCH_CHECK_GE(a, b) require((a) >= (b))
#define TORCH_CHECK_LE(a, b) require((a) <= (b))
"""
MAIN = r"""
int main() {
    int pos, q_start, j, window, rect_lo, valid_start;
    while (std::cin >> pos >> q_start >> j >> window >> rect_lo >> valid_start) {
        auto v = yoco_swa_visibility(pos, q_start, j, window, rect_lo, valid_start);
        std::cout << v.start_slot << ' ' << v.sec1_len << ' ' << v.k_lo << ' ' << v.new_len << '\n';
    }
}
"""


def load_upstream_reference(path):
    """Only definitions needed for L1 shapes/oracle; do not execute the CLI."""
    selected = {"SeqSpec", "MaskCase", "build_mask_cases", "_reference_visible"}
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in selected]
    if len(nodes) != len(selected):
        raise ValueError("upstream L1 definitions changed; inspect source")
    module = types.ModuleType("_deepinfer_yoco_l1_fixture")
    sys.modules[module.__name__] = module  # dataclass annotation handling
    exec("from __future__ import annotations\nfrom dataclasses import dataclass", module.__dict__)
    compiled = compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec", flags=__future__.annotations.compiler_flag)
    exec(compiled, module.__dict__)
    return module


def check_cp_references(source):
    path = source / "deepinfer/model_executor/layers/tilelang_ops/prepare_cp_mirror.py"
    tree = ast.parse(path.read_text())
    names = {"prepare_cp_mirror_ref", "yoco_cp_boundary_fused_ref", "_yoco_contrib_pad_len"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names), "CP reference definitions changed"

    # Python 3.9 host compatibility: preserve the original bodies and enforce
    # zip(strict=True) semantics in this isolated namespace.
    def checked_zip(*iterables, strict=False):
        sentinel = object()
        for values in zip_longest(*iterables, fillvalue=sentinel) if strict else zip(*iterables):
            if any(v is sentinel for v in values):
                raise ValueError("zip() arguments have different lengths")
            yield values

    namespace = {"accumulate": accumulate, "zip": checked_zip}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    mirror = namespace["prepare_cp_mirror_ref"]
    boundary = namespace["yoco_cp_boundary_fused_ref"]
    scalar = namespace["_yoco_contrib_pad_len"]
    shapes = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_YOCO_CONTRIB_SEQS" for t in node.targets):
            # Upstream includes list repetition ([2] * 2000), not a literal.
            allowed = (ast.List, ast.Tuple, ast.Constant, ast.BinOp, ast.Mult, ast.Load)
            assert all(isinstance(n, allowed) for n in ast.walk(node.value)), "inspect changed shape expression"
            shapes = eval(compile(ast.Expression(node.value), str(path), "eval"), {"__builtins__": {}})
    assert shapes is not None, "upstream CP shape matrix changed"
    shapes = list(shapes) + [([17] * 513, [1 if i % 2 else 17 for i in range(513)])]
    rng = random.Random(20261020)
    for _ in range(100):
        sq = [rng.randint(1, 131) for _ in range(rng.randint(1, 12))]
        shapes.append((sq, [rng.randint(1, n) for n in sq]))

    def gather(values, indices):
        return [values[i] if 0 <= i < len(values) else 0 for i in indices]

    cases = moved = padded = nonzero_pad = 0
    ratios = (1, 2, 4, 128)
    for sq, keep in shapes:
        sk = [n + rng.randrange(0, 257) for n in sq]
        cu_full, cu_c = [0, *accumulate(sq)], [0, *accumulate(keep)]
        full_ids = list(range(1, sum(sq) + 1))  # zero reserved for gather OOB
        compact_ids = [cu_full[i] + j + 1 for i in range(len(sq)) for j in range(sq[i] - keep[i], sq[i])]
        k_bases = {ratio: [0, *accumulate(max(1, n // ratio) for n in sk)] for ratio in ratios}
        for cp in (1, 2, 4, 8):
            contribs, reverse, pad_len = boundary(sq, keep, cp)
            assert sum(map(len, contribs)) == sum(keep)
            assert pad_len == scalar(sq, keep, cp) == max(map(len, contribs))
            full_mirrors = [mirror(sq, sk, cp, r, ratios)[0] for r in range(cp)]
            payloads = [gather(full_ids, indices) for indices in full_mirrors]
            old_owner = {full_idx + 1: r for r, indices in enumerate(full_mirrors) for full_idx in indices if full_idx >= 0}
            gathered = [v for r in range(cp) for v in gather(payloads[r], contribs[r] + [-1] * (pad_len - len(contribs[r])))]
            assert Counter(v for v in gathered if v != 0) == Counter(compact_ids), "contrib loses/duplicates valid tokens"
            assert gather(gathered, reverse) == compact_ids, "rank-major gather does not reconstruct compact global order"
            for r in range(cp):
                indices, reverse_mirror, ks, ke = mirror(keep, sk, cp, r, ratios)
                out = gather(gathered, gather(reverse, indices))
                assert len(out) == sum(2 * ((n + 2 * cp - 1) // (2 * cp)) for n in keep)
                for local_row, compact_idx in enumerate(indices):
                    if compact_idx < 0:
                        padded += 1
                        nonzero_pad += out[local_row] != 0
                        continue  # padding contents intentionally not an output invariant
                    token_id = compact_ids[compact_idx]
                    assert out[local_row] == token_id, "boundary changes valid token identity"
                    moved += old_owner[token_id] != r
                    assert reverse_mirror[compact_idx] == r * len(indices) + local_row
                    seq = bisect_right(cu_c, compact_idx) - 1
                    offset = compact_idx - cu_c[seq]
                    absolute_pos = sk[seq] - keep[seq] + offset
                    for ri, ratio in enumerate(ratios):
                        base = k_bases[ratio][seq]
                        assert ks[ri][local_row] == base
                        assert ke[ri][local_row] == base + (absolute_pos + 1) // ratio, "compact query lost absolute key position"
            cases += 1
    assert moved > 0, "matrix must require cross-rank migration"
    assert padded > 0 and nonzero_pad > 0, "matrix must expose nonzero padding after the second gather"
    print(f"prepare_cp_mirror.py SHA256 {hashlib.sha256(path.read_bytes()).hexdigest()}")
    print(f"PASS CP reference composition: {cases} shape/CP cases; {moved} migrated valid rows; {padded} pad rows ({nonzero_pad} nonzero)")
    print("Scope: original Python reference bodies + fixture gather/all-gather; not TileLang kernels, real communication, or model execution.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--cxx", default="clang++")
    parser.add_argument("--cxxflag", action="append", default=[])
    parser.add_argument("--cp-reference-only", action="store_true", help="run pure-Python CP composition checks only")
    args = parser.parse_args()
    if args.cp_reference_only:
        check_cp_references(args.source)
        return
    compiler = shutil.which(args.cxx)
    if not compiler:
        parser.error(f"compiler not found: {args.cxx}")
    cpp_path = args.source / "csrc/engine/input_metadata/prepare_tensor_input_v4.cpp"
    ref_path = args.source / "examples/yoco/yoco_equiv.py"
    source = cpp_path.read_text()
    start = "struct YocoSwaVisibility {"
    end = "/// CP > 1 时"
    if source.count(start) != 1 or source.count(end) != 1:
        parser.error("helper extraction boundaries changed; inspect source")
    helper = source.split(start, 1)[1].split(end, 1)[0]
    helper = start + helper
    upstream = load_upstream_reference(ref_path)
    cases = []
    for case in upstream.build_mask_cases(140, 128, 1024):
        for seq in case.seqs:
            cases.append((case.name, seq, 140, 128))
    rng = random.Random(20261020)
    for i in range(200):
        q_start = rng.randrange(0, 300)
        q_len = rng.randrange(1, 65)
        n = q_start + q_len + rng.randrange(0, 200)
        window = rng.choice([1, 4, 8, 128])
        keep = window + rng.choice([0, 1, 6, 12])
        cases.append((f"boundary-{i}", upstream.SeqSpec(n, q_start, q_len), keep, window))

    rows, fixtures = [], []
    for name, spec, keep, window in cases:
        for valid_start in sorted({0, spec.q_start, min(spec.q_start, max(0, spec.N - keep))}):
            clip = max(0, min(spec.N - keep - spec.q_start, spec.q_len - 1))
            assert spec.q_len - clip >= 1
            for j in range(spec.q_len):
                pos = spec.q_start + j
                rows.append(f"{pos} {spec.q_start} {j} {window} {spec.N - keep} {valid_start}")
                fixtures.append((name, spec, keep, window, valid_start, clip, j))

    for path in (cpp_path, ref_path):
        print(f"{path.name} SHA256 {hashlib.sha256(path.read_bytes()).hexdigest()}", flush=True)
    print(subprocess.check_output([compiler, "--version"], text=True).splitlines()[0], flush=True)
    with tempfile.TemporaryDirectory(prefix="deepinfer-yoco-cpu-") as scratch:
        cpp, binary = Path(scratch) / "visibility.cpp", Path(scratch) / "visibility"
        cpp.write_text(PRELUDE + helper + MAIN)
        subprocess.run([compiler, "-std=c++17", "-O2", *args.cxxflag, str(cpp), "-o", str(binary)], check=True, timeout=60)
        output = subprocess.run([str(binary)], input="\n".join(rows) + "\n", capture_output=True, text=True, check=True, timeout=30)
    results = output.stdout.splitlines()
    assert len(results) == len(fixtures)
    empty_rows = retained_rows = 0
    for result, (name, spec, keep, window, valid_start, clip, j) in zip(results, fixtures):
        start_slot, hist_len, k_lo, new_len = map(int, result.split())
        got = list(range(spec.q_start - window + start_slot, spec.q_start - window + start_slot + hist_len))
        got += list(range(spec.q_start + k_lo, spec.q_start + k_lo + new_len))
        expected = [p for p in upstream._reference_visible(spec, spec.q_start + j, keep, window) if p >= valid_start]
        assert got == expected, (name, spec, valid_start, j, got, expected)
        empty_rows += not got
        if j >= clip:
            retained_rows += 1
            # Compact section2 moves the base to q_start+clip, preserving absolute keys.
            if new_len:
                assert k_lo >= clip, (name, "visible key precedes retained rows")
                compact_keys = list(range(spec.q_start + clip + (k_lo - clip), spec.q_start + clip + (k_lo - clip) + new_len))
                assert compact_keys == list(range(spec.q_start + k_lo, spec.q_start + k_lo + new_len))
            if spec.q_start + j < spec.N - keep:
                assert j == spec.q_len - 1 and not got, (name, "placeholder must have empty visibility")
    print(f"PASS {len(fixtures)} query/valid_start cases; {retained_rows} retained rows; {empty_rows} empty rows")
    print("Scope: original visibility helper + upstream oracle only; clip/compact rebasing is fixture arithmetic, not full metadata execution.")


if __name__ == "__main__":
    main()
