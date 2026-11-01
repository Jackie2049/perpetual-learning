"""Eight own CPU boundary controls, using original prompt predicates/fixtures.

Standard library only. No Datasets, Transformers, original pytest suite, export,
cache hash, media decode or worker process is executed.
"""

import argparse
import ast
import contextlib
import copy
import __future__
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import platform
import ssl
import subprocess
import sys
import traceback
from types import ModuleType, SimpleNamespace

HEAD = "cbf31a27075f5f112241e716e480daf4401b3699"
BASE = "2ab35292039a51175a4b3d3b3b2d76e951a0d3e4"
SOURCE = "verl/utils/dataset/rl_dataset.py"
TEST = "tests/utils/dataset/test_rl_dataset_on_cpu.py"
WIRING = "verl/utils/tokenizer/tokenizer.py"
NAMES = {"_get_processor_path", "_get_tokenizer_path", "_build_messages_for_filter",
         "_filter_prompt_length_with_processor", "_filter_prompt_length_with_tokenizer"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    args = parser.parse_args()
    sources, hashes = {}, {}
    for path in (SOURCE, TEST, WIRING):
        raw = subprocess.check_output(["git", "show", f"{HEAD}:{path}"], cwd=args.repo,
                                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        compile(ast.parse(raw), path, "exec")
        sources[path] = raw
        hashes[path] = hashlib.sha256(raw).hexdigest()
    module = ModuleType("isolated_prompt_predicates")
    sys.modules[module.__name__] = module  # pickle must resolve the actual fixture class first
    ns = module.__dict__
    ns.update(copy=copy, os=os, re=__import__("re"), traceback=traceback, ssl=ssl)
    nodes = [n for n in ast.parse(sources[SOURCE]).body
             if isinstance(n, ast.FunctionDef) and n.name in NAMES]
    assert {n.name for n in nodes} == NAMES
    # Annotation-only dependencies are postponed; no production body is rewritten.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), SOURCE, "exec",
                 flags=__future__.annotations.compiler_flag), ns)
    fixture_names = {"_FakeTokenizer", "_FakeProcessor", "_UnpickleableProcessor"}
    fixtures = [n for n in ast.parse(sources[TEST]).body
                if isinstance(n, ast.ClassDef) and n.name in fixture_names]
    assert {n.name for n in fixtures} == fixture_names
    exec(compile(ast.Module(body=fixtures, type_ignores=[]), TEST, "exec"), ns)
    # Original fixtures format/count whitespace, not real model tokenizer behavior.
    live = ns["_UnpickleableProcessor"]()
    reports = []
    try:
        pickle.dumps(live)
    except TypeError as exc:
        assert "SSLContext" in str(exc)
        reports.append(dict(case="live_processor_pickle", error=type(exc).__name__, detail=str(exc)))
    else:
        raise AssertionError("live SSLContext unexpectedly pickled")
    processor_predicate = ns["_filter_prompt_length_with_processor"]
    common = dict(max_prompt_length=2, prompt_key="prompt", image_key="images", video_key="videos",
                  audio_key="audios", image_patch_size=14, apply_chat_template_kwargs={},
                  mm_processor_kwargs={}, tool_schemas=None,
                  process_multi_modal_info_fn=lambda *args: (None, None, None))
    for name, content, expected in (("short_row", "short", True),
                                    ("overlong_row", "one two three", False),
                                    ("malformed_row", None, False)):
        row = {"prompt": [{"role": "user", "content": content}]}
        before = copy.deepcopy(row)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            actual = processor_predicate(row, processor=live, **common)
        assert actual is expected and row == before
        assert bool(err.getvalue()) == (name == "malformed_row")
        reports.append(dict(case=name, kept=actual, stdout=out.getvalue(), stderr=err.getvalue(),
                            input_unchanged=True))

    def raises(label, function, expected, match):
        out, err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                function()
        except expected as exc:
            assert match in str(exc)
            assert out.getvalue() == err.getvalue() == ""
            reports.append(dict(case=label, error=type(exc).__name__, detail=str(exc),
                                no_row_skip_diagnostic=True))
        else:
            raise AssertionError(label)

    row = {"prompt": [{"role": "user", "content": "short"}]}
    raises("missing_processor_source", lambda: processor_predicate(row, **common), ValueError,
           "processor_path is required")

    def failing_loader(*args, **kwargs):
        raise ValueError("controlled worker initialization failure")

    # Loader functions are explicit failure seams, not real cached/HF loading.
    ns["_get_filter_processor"] = failing_loader
    ns["_get_filter_tokenizer"] = failing_loader
    raises("processor_load_failure", lambda: processor_predicate(row, processor_path="fixture", **common),
           ValueError, "controlled worker initialization failure")
    raises("tokenizer_load_failure", lambda: ns["_filter_prompt_length_with_tokenizer"](
        row, tokenizer_path="fixture", max_prompt_length=2, prompt_key="prompt",
        apply_chat_template_kwargs={}, tool_schemas=None), ValueError,
        "controlled worker initialization failure")
    assert ns["_get_processor_path"](SimpleNamespace(name_or_path="", tokenizer=SimpleNamespace(name_or_path="inner"))) == "inner"
    assert ns["_get_processor_path"](SimpleNamespace(name_or_path="outer", tokenizer=SimpleNamespace(name_or_path="inner"))) == "outer"
    assert ns["_get_processor_path"](SimpleNamespace(name_or_path="", tokenizer=SimpleNamespace(name_or_path=""))) is None
    assert ns["_get_tokenizer_path"](SimpleNamespace(name_or_path="")) is None
    reports.append(dict(case="reload_source_selection", outer_preferred=True, inner_fallback=True,
                        empty_is_unavailable=True))
    assert len(reports) == 8
    print(json.dumps(dict(head=HEAD, base=BASE, python=platform.python_version(), source_sha256=hashes,
                         controls=reports, scope="Five complete original function definitions; annotations "
                         "postponed, bodies unchanged. Three original fake formatter classes; no original "
                         "test body. Eight own controls, real SSLContext/pickle and CPU text-list handling. "
                         "Media hook returns no media; two loaders deliberately raise. No cache/export/hash "
                         "execution, actual tokenizer/processor, Dataset.filter, subprocess workers/spawn, "
                         "Arrow fingerprint/cache, original pytest suite, GPU/NPU, model or training. "
                         "Three complete inputs compile."), indent=2))


if __name__ == "__main__":
    main()
