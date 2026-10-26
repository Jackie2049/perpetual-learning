"""Original factory with source-derived API shapes and fake loaders; no models.

Run with Python 3.12, packaging and authenticated gh:
    python verify.py /absolute/path/to/verl
Requires the fixed Git objects below locally; prints JSON without saving files.
"""
import ast
import base64
import hashlib
import json
import os
import subprocess
import sys
import types
import warnings
from unittest.mock import patch

from packaging import version

VERL = sys.argv[1]
BASE = '3c6d9b43b67c74ec343d77874cd8fa6e841523da'
HEAD = '240f7f0ae09286429fbd81d21a2a403e323665fb'
TF = {'5.5.3': 'c6c8503869367af938666810e01a71866ca4fe93',
      '5.12.1': 'ddb849abe009d1089e6c691bfc897f27211c663c',
      '5.15.0': '5eddc12edfaf8cafde8c9bae4ccb12f8a139b4f9'}


def git(ref, path):
    return subprocess.check_output(['git', '-C', VERL, 'show', f'{ref}:{path}'],
                                   env={**os.environ, 'GIT_NO_LAZY_FETCH': '1'})


def functions(raw, names):
    tree = ast.parse(raw)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    return nodes


manifest = {}
methods = {}
for ver, ref in TF.items():
    raw = base64.b64decode(json.loads(subprocess.check_output([
        'gh', 'api', f'repos/huggingface/transformers/contents/src/transformers/processing_utils.py?ref={ref}'
    ]))['content'])
    compile(raw, ref, 'exec')
    cls = next(n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef) and n.name == 'ProcessorMixin')
    method = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_check_special_mm_tokens'), None)
    methods[ver] = method
    manifest[ver] = {'ref': ref, 'sha256': hashlib.sha256(raw).hexdigest(),
                     'has_direct_check': method is not None, 'bases': [ast.unparse(n) for n in cls.bases]}
assert methods['5.5.3'] is not None
assert methods['5.12.1'] is not None and methods['5.15.0'] is not None

compat_raw = git(BASE, 'verl/utils/transformers_compat.py')
compat_ns = {'lru_cache': __import__('functools').lru_cache, 'Optional': __import__('typing').Optional,
             'version': version, 'importlib': __import__('importlib.metadata')}
exec(compile(ast.Module(body=functions(compat_raw, ['is_transformers_version_in_range']), type_ignores=[]), 'compat', 'exec'), compat_ns)
gate = compat_ns['is_transformers_version_in_range']
compat_mod = types.ModuleType('verl.utils.transformers_compat')
compat_mod.is_transformers_version_in_range = gate
results = []
for ref in (BASE, HEAD):
    raw = git(ref, 'verl/utils/tokenizer/tokenizer.py')
    compile(raw, ref, 'exec')
    manifest[ref] = hashlib.sha256(raw).hexdigest()
    names = ['hf_processor'] + (['_check_special_mm_tokens_on_lists'] if ref == HEAD else [])
    ns = {'types': types, 'warnings': warnings}
    exec(compile(ast.Module(body=functions(raw, names), type_ignores=[]), ref, 'exec'), ns)
    for ver, overridden in [('5.5.3', False), ('5.12.1', False), ('5.12.1', True), ('5.15.0', False)]:
        # Class shells represent source method availability; loaders do not run HF.
        attrs = {}
        if methods[ver] is not None:
            check_ns = {}
            future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
            exec(compile(ast.fix_missing_locations(ast.Module(body=[future, methods[ver]], type_ignores=[])), ver, 'exec'), check_ns)
            attrs['_check_special_mm_tokens'] = check_ns['_check_special_mm_tokens']
        mixin = type('ProcessorMixin', (), attrs)
        proc_cls = type('MllamaProcessor', (mixin,),
                        {'_check_special_mm_tokens': lambda *args: None} if overridden else {})
        processor = proc_cls()
        tf_mod = types.ModuleType('transformers')
        tf_mod.ProcessorMixin = mixin
        tf_mod.PreTrainedTokenizerBase = type('PreTrainedTokenizerBase', (), {})
        tf_mod.AutoProcessor = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: processor)
        tf_mod.AutoConfig = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: types.SimpleNamespace())
        gate.cache_clear()
        with patch.dict(sys.modules, {'transformers': tf_mod, 'verl.utils.transformers_compat': compat_mod}), \
             patch('importlib.metadata.version', return_value=ver), warnings.catch_warnings(record=True) as seen:
            out = ns['hf_processor']('fake-model')
        bound = out is not None and getattr(getattr(out, '_check_special_mm_tokens', None), '__func__', None) is ns.get('_check_special_mm_tokens_on_lists', object())
        expected_none = False
        expected_bound = ref == HEAD and ver in ('5.5.3', '5.12.1') and not overridden
        assert (out is None) == expected_none
        assert bound == expected_bound
        assert out is None or out is processor
        assert len(seen) == int(expected_none)
        results.append({'ref': ref[:8], 'version': ver, 'override': overridden,
                        'returned_none': out is None, 'bound_wrapper': bound,
                        'warnings': [str(w.message) for w in seen]})
print(json.dumps({'manifest': manifest, 'factory_calls': results,
                  'scope': '8 original-factory calls with fake loaders and class shells; no actual Transformers installation or processor/tokenizer execution'}, indent=2))
