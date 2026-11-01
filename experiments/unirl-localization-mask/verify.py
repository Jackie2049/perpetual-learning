"""#354 original storage-key masks on native TensorRefs and own metadata handles.

python verify.py /cached/UniRL-git
No handle.local(), transport, Worker.call, Ray, gradients or GPU/NPU.
"""
import __future__
import ast
from collections import Counter
import hashlib
import json
import os
import platform
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import torch

HEAD = "cc8c3f01d1c1497f473e42f256346fecf57c3397"
BASE = "c8af5d699c81fb02a0086e34b342b1efacc7d431"


def main(cache):
    hashes = {}
    env = dict(os.environ, GIT_NO_LAZY_FETCH="1")

    def read(path):
        raw = subprocess.check_output(["git", "-C", cache, "show", f"{HEAD}:{path}"], env=env)
        hashes[path] = hashlib.sha256(raw).hexdigest()
        compile(raw, path, "exec")
        return ast.parse(raw)

    def named(tree, name):
        return next(node for node in tree.body if getattr(node, "name", None) == name)

    ns = ModuleType("cpu_original_localization_mask")
    sys.modules[ns.__name__] = ns
    ns.Counter = Counter

    def execute(nodes):
        tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
        exec(compile(tree, "original_ast", "exec", flags=__future__.annotations.compiler_flag), ns.__dict__)

    ancestor = subprocess.check_output(["git", "-C", cache, "merge-base", HEAD, BASE], env=env).decode().strip()
    assert ancestor == BASE
    execute(read("unirl/distributed/tensor/batch.py").body)
    refs = read("unirl/distributed/tensor/ref.py")
    execute([node for node in refs.body if not (isinstance(node, ast.ImportFrom)
             and node.module == "unirl.distributed.tensor.batch")])
    execute([named(read("unirl/distributed/utils.py"), "collect_leaves")])
    dispatch = read("unirl/distributed/group/dispatch.py")
    execute([named(dispatch, name) for name in ("required_store_keys", "remap_required_store_keys")])
    for path in ("unirl/distributed/group/handle.py", "unirl/distributed/group/worker.py"):
        read(path)  # Static wiring / compile only.

    def forbidden_local():
        raise AssertionError("Metadata controls must not materialize any handle")

    def ref(key=None, object_ref=None):
        handle = SimpleNamespace(store_key=key, object_ref=object_ref, shape=(4, 2),
                                 dtype=torch.float32, device="cpu", local=forbidden_local)
        return ns.TensorRef.from_handles([handle])

    a, b = ref("source:A"), ref("source:B")
    view = a.slice(1, 3)
    assert view is not a and view.spans[0].handle is a.spans[0].handle
    assert ns.ref_store_keys(view) == {"source:A"}

    def mask(args, *, reads=None, skips=None):
        return ns.required_store_keys({"reads": reads, "skips": skips}, args, {})

    cases = {}

    def check(name, actual, expected):
        assert actual == expected, (name, actual, expected)
        cases[name] = sorted(repr(key) for key in actual) if isinstance(actual, set) else actual

    check("reads_view_uses_storage_identity", mask((a, b), reads=lambda *args: view), {"source:A"})
    check("skip_exact_ref", mask((a, b), skips=lambda *args: a), {"source:B"})
    check("skip_new_view_matches_no_occurrence", mask((a, b), skips=lambda *args: view), {"source:A", "source:B"})
    check("same_object_alias_one_occurrence_skipped", mask((a, a, b), skips=lambda *args: [a]),
          {"source:A", "source:B"})
    check("same_object_alias_all_occurrences_skipped", mask((a, a, b), skips=lambda *args: [a, a]), {"source:B"})
    check("different_refs_share_key_outside_skip", mask((a, view, b), skips=lambda *args: a),
          {"source:A", "source:B"})
    check("empty_arguments", mask((), reads=lambda: a), set())
    check("keyed_ref_empty_mask_not_required", ns.ref_is_required(a, set()), False)
    check("none_mask_requires_keyed_ref", ns.ref_is_required(a, None), True)
    unknown = ref()
    check("keyless_ref_empty_mask_still_required", ns.ref_is_required(unknown, set()), True)
    multi = ns.TensorRef.concat([a, b])
    check("multi_span_any_key_requires_whole_ref", ns.ref_is_required(multi, {"source:A"}), True)
    object_key = ref(object_ref=SimpleNamespace(binary=lambda: b"own-object-C"))
    check("object_ref_binary_fallback", ns.ref_store_keys(object_key), {("object_ref", b"own-object-C")})

    target_a, target_b = ref("target:A"), ref("target:B")
    before = ({"a": a, "b": b},)
    after = ({"a": target_a, "b": target_b},)
    check("positional_remap", ns.remap_required_store_keys({"source:A"}, before, {}, after, {}), {"target:A"})
    # Deliberately violate the helper's structural correspondence precondition.
    # Same leaf count is not a semantic validation of a real localizer's output.
    swapped = ({"a": target_b, "b": target_a},)
    check("same_count_swapped_structure_is_not_rejected",
          ns.remap_required_store_keys({"source:A"}, before, {}, swapped, {}), {"target:B"})
    try:
        ns.remap_required_store_keys({"source:A"}, before, {}, (target_a,), {})
    except RuntimeError as exc:
        structure_error = str(exc)
    else:
        raise AssertionError("Different TensorRef counts must be rejected")
    assert "tree structure" in structure_error
    assert "ray" not in sys.modules
    return {"head": HEAD, "actual_merge_base": ancestor, "source_sha256": hashes,
            "python": platform.python_version(), "torch": torch.__version__,
            "original_helper_controls": cases, "different_leaf_count_error": structure_error,
            "scope": "14 own metadata controls plus leaf-count rejection; original Batch/ref module, "
                     "collect_leaves and four key/mask helpers. Own handle metadata/binary seam, "
                     "native TensorRef views; no materialization, native transport/localizer, Worker.call, "
                     "Handle launch/restore/finalizer, Ray/NCCL, autograd, scorer, bandwidth or training. "
                     "Handle/worker sources compiled and read only."}


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1]), indent=2))
