"""Run existing PR test bodies against extracted original CPU-safe classes."""
import ast
import functools
import hashlib
import json
import subprocess
import sys
from typing import Any, Optional

import torch

torch.set_default_device('cpu')

REPO = sys.argv[1]
BASE = '4b26c788d41ce932ce7706c5ee8e249ab52f4c51'
FIRST = 'fea49f196d99d641b896d32e30dcb4a5234f6373'
HEAD = '8c227d6abceb21da69b49a62262709a3ca158056'
MAIN = '3c6d9b43b67c74ec343d77874cd8fa6e841523da'
SOURCE = 'verl/utils/activation_offload.py'
TEST = 'tests/utils/test_activation_offload_structured_outputs_on_cpu.py'
CLASSES = {'FSDPParameterFilter', 'CpuOffloadHookWithOffloadHandler', 'OffloadHandler',
           'GroupCommitFunction', 'ActivationHandler'}


def source(ref, path):
    return subprocess.check_output(['git', '-C', REPO, 'show', f'{ref}:{path}']).decode()


test_text = source(HEAD, TEST)
test_tree = ast.parse(test_text)
# Only the original framework import is replaced by the extracted namespace.
test_tree.body = [node for node in test_tree.body
                  if not (isinstance(node, ast.ImportFrom) and node.module == 'verl.utils.activation_offload')]
test_names = [node.name for node in test_tree.body
              if isinstance(node, ast.FunctionDef) and node.name.startswith('test_')]
assert len(test_names) == 5
compiled = {}
for ref in (BASE, FIRST, HEAD, MAIN):
    text = source(ref, SOURCE)
    compile(text, f'{ref}:{SOURCE}', 'exec')
    compiled[ref] = hashlib.sha256(text.encode()).hexdigest()
compile(test_text, f'{HEAD}:{TEST}', 'exec')

results = {}
for ref in (BASE, FIRST, HEAD):
    original = ast.parse(source(ref, SOURCE))
    nodes = [node for node in original.body
             if isinstance(node, ast.ImportFrom) and node.module == '__future__'
             or isinstance(node, ast.ClassDef) and node.name in CLASSES
             or isinstance(node, ast.Assign) and any(
                 isinstance(target, ast.Name) and target.id == 'group_prefetch_offload_commit'
                 for target in node.targets)]
    assert {node.name for node in nodes if isinstance(node, ast.ClassDef)} == CLASSES
    ns = {'__name__': f'offload_{ref}', 'torch': torch, 'functools': functools,
          'Any': Any, 'Optional': Optional}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), f'{ref}:{SOURCE}:extracted', 'exec'), ns)
    if ref == BASE:
        original_helper = ns['group_prefetch_offload_commit']

        def old_helper_arguments(handler, *tensors):
            assert len(tensors) == 1
            return original_helper(tensors[0], handler)

        ns['group_prefetch_offload_commit'] = old_helper_arguments
    exec(compile(test_tree, f'{HEAD}:{TEST}:original-bodies', 'exec'), ns)
    names = test_names[:1] if ref == BASE else test_names
    observed = {}
    for name in names:
        try:
            ns[name]()
        except (AssertionError, RuntimeError) as exc:
            observed[name] = f'{type(exc).__name__}: {exc}'
        else:
            observed[name] = 'PASS'
    results[ref] = observed

assert 'saved activation popped before the group was reloaded' in results[BASE][test_names[0]]
assert sum(value == 'PASS' for value in results[FIRST].values()) == 4
assert results[FIRST]['test_detached_first_output_stays_detached_and_secondary_gradient_matches'] != 'PASS'
assert all(value == 'PASS' for value in results[HEAD].values())
assert compiled[BASE] == compiled[MAIN], 'Do not extend the baseline result to a different main module'
print(json.dumps({'torch': torch.__version__, 'results': results,
                  'original_test_sha256': hashlib.sha256(test_text.encode()).hexdigest(),
                  'compiled_production_sha256': compiled,
                  'production_classes': sorted(CLASSES),
                  'baseline_adapter': 'old helper argument order only; secondary-output regression only',
                  'scope': 'real CPU autograd and original recording handler; no async streams, transfers, FSDP or checkpointing'},
                 indent=2))
