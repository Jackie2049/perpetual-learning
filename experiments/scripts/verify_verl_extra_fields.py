"""#8341 original test bodies against isolated original client classes, CPU only.

Usage: python -B verify_verl_extra_fields.py /path/to/verl --deps /path/to/pydantic-target
Python3.12+, numpy, pytest, Pydantic2; local fixed Git objects, no implicit fetch.
AST-loads both full client classes (only tracing decorators removed), original
TokenOutput and normalize_token_ids. Package shells bypass native verl imports.
Calls the two unmodified original async test bodies with real pytest.MonkeyPatch;
the base-generate/sleep substitutes are the author's own fixtures. No Ray, server,
agent loop, device or complete 49-test/native-import validation.
"""
import __future__
import argparse
import ast
import asyncio
import hashlib
import logging
import os
from pathlib import Path
import subprocess
import sys
import types
from typing import Any, Optional

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('repo', type=Path)
parser.add_argument('--deps', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.deps.resolve()))
import numpy as np
import pydantic
import pytest

BASE = 'dc709afb7a74fd2645640a4e5f2fdf40166b4fbb'
HEAD = '0cede26bf2c0d7433af0cba5c2c38ce278e6b07e'
TEST = 'tests/workers/rollout/test_llm_server_extra_fields_on_cpu.py'
env = dict(os.environ, GIT_NO_LAZY_FETCH='1')


def source(rev, path):
    raw = subprocess.check_output(['git', 'show', f'{rev}:{path}'], cwd=args.repo, env=env)
    compile(raw, path, 'exec')
    print(rev[:8], path, 'sha256=' + hashlib.sha256(raw).hexdigest())
    return raw


def execute(nodes, namespace):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'original selected definitions', 'exec',
                 flags=__future__.annotations.compiler_flag), namespace)


for name in ('verl', 'verl.workers', 'verl.workers.rollout'):
    package = types.ModuleType(name)
    package.__path__ = []
    sys.modules[name] = package
replica = types.ModuleType('verl.workers.rollout.replica')
sys.modules[replica.__name__] = replica
replica.__dict__.update(BaseModel=pydantic.BaseModel, Any=Any, Optional=Optional)
token_tree = ast.parse(source(HEAD, 'verl/workers/rollout/replica.py'))
execute([next(n for n in token_tree.body if isinstance(n, ast.ClassDef) and n.name == 'TokenOutput')], replica.__dict__)
tokenizer_tree = ast.parse(source(HEAD, 'verl/utils/tokenizer/tokenizer.py'))
normalizer = next(n for n in tokenizer_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'normalize_token_ids')
test_source = source(HEAD, TEST)

for rev in (BASE, HEAD):
    client = types.ModuleType('verl.workers.rollout.llm_server')
    sys.modules[client.__name__] = client
    sys.modules['verl.workers.rollout'].llm_server = client
    client.__dict__.update(asyncio=asyncio, np=np, TokenOutput=replica.TokenOutput,
                           logger=logging.getLogger('isolated-extra-fields'))
    execute([normalizer], client.__dict__)
    tree = ast.parse(source(rev, 'verl/workers/rollout/llm_server.py'))
    nodes = [n for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == '_MERGED_EXTRA_FIELDS' for t in n.targets)]
    for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name in {'LLMServerClient', 'FullyAsyncLLMServerClient'}]:
        for method in cls.body:
            if isinstance(method, ast.AsyncFunctionDef) and method.name == 'generate':
                assert len(method.decorator_list) == 1
                assert ast.unparse(method.decorator_list[0]) == 'rollout_trace_op'
                method.decorator_list = []
        nodes.append(cls)
    execute(nodes, client.__dict__)
    test = types.ModuleType('_original_extra_fields_tests')
    exec(compile(test_source, TEST, 'exec'), test.__dict__)
    for name in ('test_single_attempt_keeps_backend_extra_fields', 'test_resume_keeps_latest_extra_fields_and_merged_keys'):
        with pytest.MonkeyPatch.context() as patch:
            try:
                asyncio.run(getattr(test, name)(patch))
            except KeyError as exc:
                assert rev == BASE and exc.args == ('text',), (rev, exc)
                print('EXPECTED BASE FAIL', name, "KeyError('text')")
            else:
                assert rev == HEAD
                print('PASS original test body', name)
    if rev == HEAD:
        # Supplement the original tests with last-PRESENT, shallow-value and invocation-scope controls.
        payload = np.arange(3)
        first = replica.TokenOutput(token_ids=[1], stop_reason='aborted',
                                    extra_fields={'text': 'first', 'only_first': 1, 'nested': {'a': 1}, 'payload': payload})
        second = replica.TokenOutput(token_ids=[2], stop_reason='aborted',
                                     extra_fields={'text': None, 'nested': {'b': 2}})
        third = replica.TokenOutput(token_ids=[3], stop_reason='completed')
        shared_client = test._client()
        with pytest.MonkeyPatch.context() as patch:
            test._install_segments(patch, [first, second, third])
            output = asyncio.run(shared_client.generate(request_id='supplement', prompt_ids=[9], sampling_params={}))
        assert output.extra_fields['text'] is None and output.extra_fields['only_first'] == 1
        assert output.extra_fields['nested'] == {'b': 2} and output.extra_fields['payload'] is payload
        with pytest.MonkeyPatch.context() as patch:
            test._install_segments(patch, [replica.TokenOutput(token_ids=[4], stop_reason='completed')])
            separate = asyncio.run(shared_client.generate(request_id='new-request', prompt_ids=[9], sampling_params={}))
        assert not {'text', 'only_first', 'nested', 'payload'} & separate.extra_fields.keys()
        print('PASS 3 supplemental controls: last-present/explicit None, shallow values, no next-call carry')

assert 'ray' not in sys.modules and 'torch' not in sys.modules
print('Python', sys.version.split()[0], 'numpy', np.__version__, 'pytest', pytest.__version__, 'Pydantic', pydantic.__version__)
print('2 original test bodies: base both expected-fail/head both pass; no native pytest collection or backend validation.')
