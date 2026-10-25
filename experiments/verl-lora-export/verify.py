"""Run eight original #7341 CPU cases that do not load PEFT or write models.

Sources: verl-project/verl, Apache-2.0, ByteDance contributors.
AST loading bypasses package initialization, not tested function bodies.
The five other upstream cases are deliberately not executed.
"""

import __future__
from abc import ABC, abstractmethod
import argparse
import ast
from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType
from typing import Optional
from unittest.mock import MagicMock, call, patch
import warnings

import pytest
import torch

HEAD = "fb304fc116fbacc2ccfef6c01eefc8acae76bf65"
VERSIONS = {"head": HEAD, "ancestor": "b75f0d06a72b74e471cd4d3402bcbb0ceaa2e0c8",
            "main": "3c6d9b43b67c74ec343d77874cd8fa6e841523da"}
SOURCE = "verl/model_merger/base_model_merger.py"
TEST = "tests/model_merger/test_fuse_lora.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    args = parser.parse_args()
    hashes = {}
    trees = {}

    def source(revision, path):
        raw = subprocess.check_output(["git", "show", revision + ":" + path], cwd=args.source_repo,
                                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        hashes[revision + ":" + path] = hashlib.sha256(raw).hexdigest()
        tree = ast.parse(raw, filename=path)
        compile(tree, path, "exec")
        return tree

    def execute(tree, names, namespace):
        nodes = [node for node in tree.body if getattr(node, "name", None) in names]
        assert len(nodes) == len(names)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "original_definitions", "exec",
                     flags=__future__.annotations.compiler_flag), namespace)

    assert "verl" not in sys.modules
    for name in ("verl", "verl.model_merger", "verl.model_merger.base_model_merger"):
        module = ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(sys.modules[parent], child, module)
    namespace = module.__dict__
    namespace.update(ABC=ABC, abstractmethod=abstractmethod, argparse=argparse, dataclass=dataclass,
                     field=field, os=os, warnings=warnings, torch=torch, Optional=Optional)
    for label, revision in VERSIONS.items():
        trees[label] = source(revision, SOURCE)
    execute(trees["head"], {"parse_args", "ModelMergerConfig", "_default_hf_model_config_path",
                            "generate_config_from_args", "BaseModelMerger"}, namespace)

    tests = {"__name__": "_original_lora_tests", "argparse": argparse, "os": os, "shutil": shutil,
             "tempfile": tempfile, "OrderedDict": OrderedDict, "MagicMock": MagicMock,
             "patch": patch, "call": call, "pytest": pytest, "torch": torch}
    tests.update({name: namespace[name] for name in ("parse_args", "ModelMergerConfig", "generate_config_from_args")})
    test_tree = source(HEAD, TEST)
    classes = {"TestFuseLoraConfig", "TestSaveLoraAdapterSplitsKeys", "TestFuseLoraIntoModel"}
    execute(test_tree, classes, tests)
    selected = [(node.name, method.name) for node in test_tree.body
                if getattr(node, "name", None) == "TestFuseLoraConfig"
                for method in node.body if isinstance(method, ast.FunctionDef) and method.name.startswith("test_")]
    selected += [(name, "test_no_lora_keys_returns_none")
                 for name in ("TestSaveLoraAdapterSplitsKeys", "TestFuseLoraIntoModel")]
    passed = []
    for cls, method in selected:
        getattr(tests[cls](), method)()
        passed.append(cls + "." + method)
    all_tests = [node.name + "." + method.name for node in test_tree.body if isinstance(node, ast.ClassDef)
                 for method in node.body if isinstance(method, ast.FunctionDef) and method.name.startswith("test_")]
    assert len(passed) == 8 and len(all_tests) == 13
    # Static comparison of the split path; neither extraction nor fusion is run.
    adapter_ast = {}
    for label, tree in trees.items():
        base = next(node for node in tree.body if getattr(node, "name", None) == "BaseModelMerger")
        method = next(node for node in base.body if getattr(node, "name", None) == "save_lora_adapter")
        adapter_ast[label] = ast.dump(method, include_attributes=False)
    print(json.dumps({"versions": VERSIONS, "source_sha256": hashes,
                      "passed_original_cases": passed, "not_run": sorted(set(all_tests) - set(passed)),
                      "split_method_ast_matches_ancestor": adapter_ast["head"] == adapter_ast["ancestor"],
                      "split_method_ast_matches_main": adapter_ast["head"] == adapter_ast["main"],
                      "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                      "pytest": pytest.__version__, "system": platform.system(),
                                      "machine": platform.machine()},
                      "scope": "original config/no-adapter cases; no PEFT import, fusion, model saving, Ray or GPU"},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
