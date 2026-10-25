"""Run #6850's original five download tests, without cloud or framework startup.

Python 3.12+, fsspec, filelock, pytest; fixed objects in an existing local verl Git repo.
MemoryFileSystem is real fsspec, not an S3 service or checkpoint durability test.
"""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from unittest.mock import patch

import filelock
import fsspec
import pytest

REPO = sys.argv[1]
BASE = '2390a3f5cff9d80d4844f8b4616ef8c2616276de'
HEAD = 'c87b60ceace43f33eee027c95cbab32d05f8bc74'
MAIN = '3c6d9b43b67c74ec343d77874cd8fa6e841523da'
TEST = 'tests/utils/test_fs_on_cpu.py'
NAMES = [
    'test_is_fsspec_path', 'test_copy_to_local_fetches_fsspec_file',
    'test_copy_to_local_local_path_passthrough', 'test_copy_to_local_rejects_fsspec_glob',
    'test_copy_to_local_fsspec_dir_trailing_slash',
]


def source(ref, path):
    return subprocess.check_output(['git', '-C', REPO, 'show', f'{ref}:{path}']).decode()


test_text = source(HEAD, TEST)
tree = ast.parse(test_text)
tree.body = [node for node in tree.body if not (
    isinstance(node, ast.Import) and any(alias.name == 'verl.utils.fs' for alias in node.names))]
assert set(NAMES).issubset({n.name for n in tree.body if isinstance(n, ast.FunctionDef)})
results, hashes = {}, {}
for ref in (BASE, HEAD, MAIN):
    hdfs = types.ModuleType('hdfs_io')
    hdfs.__file__ = f'{ref}:verl/utils/hdfs_io.py'
    hdfs_text = source(ref, 'verl/utils/hdfs_io.py')
    exec(compile(hdfs_text, f'{ref}:hdfs_io.py', 'exec'), hdfs.__dict__)
    fs = types.ModuleType('isolated_fs')
    fs.__file__ = f'{ref}:verl/utils/fs.py'
    fs_text = source(ref, 'verl/utils/fs.py')
    with patch.dict(sys.modules, {'hdfs_io': hdfs}):
        exec(compile(fs_text, f'{ref}:fs.py', 'exec'), fs.__dict__)
    hashes[ref] = {'fs.py': hashlib.sha256(fs_text.encode()).hexdigest(),
                   'hdfs_io.py': hashlib.sha256(hdfs_text.encode()).hexdigest()}
    ns = {'fs': fs}
    exec(compile(tree, f'{HEAD}:{TEST}:original-bodies', 'exec'), ns)
    observed = {}
    for name in NAMES:
        with tempfile.TemporaryDirectory(prefix='verl-object-read-') as directory:
            try:
                fn = ns[name]
                fn(Path(directory)) if inspect.signature(fn).parameters else fn()
            except (AssertionError, AttributeError, pytest.fail.Exception) as exc:
                observed[name] = f'{type(exc).__name__}: {exc}'
            else:
                observed[name] = 'PASS'
    if ref == MAIN:
        uri = 's3://not-contacted/example/checkpoint'
        assert not fs.is_non_local(uri)
        assert fs.copy_to_local(uri) == uri  # use_shm defaults False; no S3 backend call.
    results[ref] = observed

assert all(v == 'PASS' for v in results[HEAD].values())
for ref in (BASE, MAIN):
    assert results[ref]['test_copy_to_local_local_path_passthrough'] == 'PASS'
    assert sum(v == 'PASS' for v in results[ref].values()) == 1
compile(test_text, f'{HEAD}:{TEST}', 'exec')
static = {}
for path in ('verl/utils/checkpoint/fsdp_checkpoint_manager.py',
             'verl/utils/checkpoint/checkpoint_handler.py', 'verl/trainer/ppo/ray_trainer.py'):
    text = source(MAIN, path)
    compile(text, f'{MAIN}:{path}', 'exec')
    static[path] = hashlib.sha256(text.encode()).hexdigest()
print(json.dumps({'python': sys.version.split()[0], 'fsspec': fsspec.__version__,
                  'filelock': filelock.__version__, 'pytest': pytest.__version__,
                  'results': results, 'source_sha256': hashes,
                  'original_test_sha256': hashlib.sha256(test_text.encode()).hexdigest(),
                  'static_only_sha256': static,
                  'main_s3_uri_passthrough': True,
                  'scope': 'original five download bodies; real memory backend and temporary cache, no S3/GCS/HDFS I/O or checkpoint manager execution'},
                 indent=2))
