"""#8420 original method/recursive conversion with synthetic row controls.

Usage: python -B verify.py /path/to/verl-git
Original sources: verl-project/verl (Apache-2.0). AST isolation skips imports;
processor is only a presence fixture; process_image/process_video are recorders.
No parquet/tokenizer/actual processor/media decode/model or hardware execution.
The local or-fallback contrast is not a proposed or upstream implementation.
"""

import __future__, ast, hashlib, json, os, pickle, platform, re, subprocess, sys
from types import SimpleNamespace
import numpy as np
repo = sys.argv[1]
base = "2ab35292039a51175a4b3d3b3b2d76e951a0d3e4"
head = "b11c863ab8d5a2b0fc568c2539779eb82c2455b9"
hashes, results = {}, {}
def read(rev, path):
    raw = subprocess.check_output(["git", "show", rev + ":" + path], cwd=repo,
        env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
    hashes[rev + ":" + path] = hashlib.sha256(raw).hexdigest()
    return raw
def load(node, ns):
    exec(compile(ast.Module(body=[node], type_ignores=[]), "original-definition", "exec",
        flags=__future__.annotations.compiler_flag), ns)
helper_path = "verl/utils/py_functional.py"
helper = read(head, helper_path)
assert helper == read(base, helper_path)
for label, rev in [("base", base), ("head", head)]:
    calls = []
    def image(value, **kw):
        calls.append(["image", value, kw["image_patch_size"]])
        return "decoded-image:" + str(value)
    def video(value, **kw):
        calls.append(["video", value, kw["image_patch_size"]])
        return "decoded-video:" + str(value)
    ns = dict(np=np, re=re, process_image=image, process_video=video)
    load(next(n for n in ast.parse(helper).body if isinstance(n, ast.FunctionDef)
        and n.name == "convert_nested_value_to_list_recursive"), ns)
    raw = read(rev, "verl/utils/dataset/multiturn_sft_dataset.py")
    compile(raw, "original-module", "exec")
    cls = next(n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef)
        and n.name == "MultiTurnSFTDataset")
    load(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_build_messages"), ns)
    cases = [("missing_no_processor", {}, False, "hi", "ok"),
        ("missing_with_processor", {}, True, "hi", "ok"),
        ("null_with_processor", {"images": None, "videos": None}, True, "hi", "ok"),
        ("empty_with_processor", {"images": [], "videos": []}, True, "hi", "ok"),
        ("null_no_processor", {"images": None, "videos": None}, False, "hi", "AssertionError"),
        ("empty_no_processor", {"images": [], "videos": []}, False, "hi", "AssertionError"),
        ("placeholder_null", {"images": None}, True, "<image>hi", "IndexError"),
        ("unused_image", {"images": ["a"]}, True, "hi", "AssertionError"),
        ("array_images", {"images": np.array(["a", "b"], dtype=object)}, True, "<image>A<image>B", "ok"),
        ("array_videos", {"videos": np.array(["v0", "v1"], dtype=object)}, True, "<video>A<video>B", "ok")]
    rows = {}
    for name, media, has_processor, content, expected in cases:
        calls.clear()
        row = {"messages": np.array([{"role": "user", "content": content}], dtype=object), **media}
        before = pickle.dumps(row)
        owner = SimpleNamespace(messages_key="messages", image_key="images", video_key="videos",
            image_patch_size=14, processor=object() if has_processor else None)
        try:
            output = ns["_build_messages"](owner, row)
            status, detail = "ok", output
        except (AssertionError, TypeError, IndexError) as exc:
            status, detail = type(exc).__name__, str(exc)
        assert status == ("TypeError" if label == "base" and name in
            ("null_with_processor", "placeholder_null") else expected), (label, name, status)
        assert pickle.dumps(row) == before
        rows[name] = {"status": status, "output_or_error": detail, "decoder_recorder_calls": list(calls)}
    assert rows["array_images"]["decoder_recorder_calls"] == [["image", "a", 14], ["image", "b", 14]]
    assert rows["array_videos"]["decoder_recorder_calls"] == [["video", "v0", 14], ["video", "v1", 14]]
    results[label] = rows
try:
    np.array(["a", "b"], dtype=object) or []
except ValueError:
    pass
else:
    raise AssertionError("the local or-fallback contrast must reject ambiguous array truth")
print(json.dumps({"python": platform.python_version(), "numpy": np.__version__, "source_sha256": hashes,
    "results": results, "local_or_fallback_array_contrast": "ValueError",
    "scope": "Original AST-isolated methods; synthetic rows, processor presence fixture and decoder recorders; no parquet/tokenizer/processor/model execution"},
    ensure_ascii=False, sort_keys=True))
