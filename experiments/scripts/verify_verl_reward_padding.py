"""Fixed #8101 CPU tensor-geometry controls, not a worker/manager/pytest run.

Usage: python verify_verl_reward_padding.py /path/to/verl-git
Requires Python3.12+, torch, transformers and tokenizers; downloads no models.
Selects original padding helper and tensor construction/consumer AST statements.
SimpleNamespace supplies fields only; no Ray/TQ/DataProto, inference or explicit device queries.
"""
import ast
import hashlib
import json
import os
import subprocess
import sys
from types import MethodType, SimpleNamespace

import torch
import transformers
import tokenizers
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

BASE = '8718ca30a3f002f93b7c4fd99b9b2506718681bc'
HEAD = 'a58b1d2b9bbc465d89910de583f8a30aa7dc917e'
AGENT = 'verl/experimental/agent_loop/agent_loop.py'
DAPO = 'verl/experimental/reward_loop/reward_manager/dapo.py'
repo = sys.argv[1]
env = dict(os.environ, GIT_NO_LAZY_FETCH='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
hashes = {}


def source(rev, path):
    raw = subprocess.check_output(['git', '-C', repo, 'show', f'{rev}:{path}'], env=env)
    hashes[f'{rev}:{path}'] = hashlib.sha256(raw).hexdigest()
    compile(raw, f'{rev}:{path}', 'exec')
    return ast.parse(raw)


def nodes(tree, kind, name):
    found = [n for n in ast.walk(tree) if isinstance(n, kind) and n.name == name]
    assert len(found) == 1
    return found[0]


def execute(body, scope):
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                 'selected_original_AST', 'exec'), scope)


tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=Tokenizer(WordLevel({'LEAK': 0, 'prompt': 1, 'answer': 2, '<pad>': 3, '<eos>': 4})),
    pad_token='<pad>', eos_token='<eos>')

consumer_tree = source(HEAD, DAPO)
consumer = nodes(consumer_tree, ast.AsyncFunctionDef, 'run_single')
# The four original assignments after selecting the final DataProto row.
length_nodes = consumer.body[2:6]
assert [n.targets[0].id for n in length_nodes] == [
    'response_ids', 'response_length', 'valid_response_length', 'valid_response_ids']
penalty = next(n for n in consumer.body if isinstance(n, ast.If)
               and 'self.overlong_buffer_cfg is not None' in ast.unparse(n.test))
results = []

for rev in (BASE, HEAD):
    tree = source(rev, AGENT)
    pad = nodes(tree, ast.FunctionDef, '_pad_token_ids')
    scope = {'torch': torch}
    execute([pad], scope)
    worker = SimpleNamespace(tokenizer=tokenizer)
    worker._pad_token_ids = MethodType(scope['_pad_token_ids'], worker)
    method = nodes(tree, ast.AsyncFunctionDef, '_compute_score')
    timer = next(n for n in ast.walk(method) if isinstance(n, ast.With))
    loop = next(n for n in timer.body if isinstance(n, ast.For))
    prefix = timer.body[:timer.body.index(loop)]
    stop = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'multi_modal_inputs')
    appends = [n for n in loop.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
               and isinstance(n.value.func, ast.Attribute) and isinstance(n.value.func.value, ast.Name)
               and n.value.func.value.id in ('all_prompts', 'all_responses', 'all_input_ids', 'all_attention_mask')]
    assert len(appends) == 4
    selected_loop = ast.For(target=loop.target, iter=loop.iter, body=loop.body[:stop] + appends, orelse=[])
    assignment = next(n for n in timer.body if isinstance(n, ast.Assign)
                      and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'batch')
    original_dict = assignment.value.args[0]
    pairs = [(k, v) for k, v in zip(original_dict.keys, original_dict.values, strict=True)
             if k.value != 'position_ids']
    batch_assignment = ast.Assign(targets=[ast.Name(id='batch', ctx=ast.Store())],
                                  value=ast.Dict(keys=[k for k, _ in pairs], values=[v for _, v in pairs]))
    for lengths in ([(3, 5), (9, 2)], [(10, 2), (1, 6)], [(3, 2), (3, 2)]):
        outputs = [SimpleNamespace(prompt_ids=[1] * p, response_ids=[2] * r) for p, r in lengths]
        scope = {'torch': torch, 'self': worker, 'outputs': outputs}
        execute(prefix + [selected_loop, batch_assignment], scope)
        batch = scope['batch']
        consumer_scope = dict(torch=torch, data_item=SimpleNamespace(batch={k: t[-1] for k, t in batch.items()}))
        execute(length_nodes, consumer_scope)
        length = int(consumer_scope['valid_response_length'])
        decoded = tokenizer.decode(consumer_scope['valid_response_ids'], skip_special_tokens=True)
        consumer_scope.update(self=SimpleNamespace(max_resp_len=7, overlong_buffer_cfg=SimpleNamespace(
            enable=True, len=4, penalty_factor=1., log=True)), reward=1., reward_extra_info={})
        execute([penalty], consumer_scope)
        expected = lengths[-1][1]
        assert length == (expected if rev == HEAD or lengths[0] == lengths[1] else (5 if expected == 2 else 1))
        if rev == HEAD:
            assert decoded == tokenizer.decode(outputs[-1].response_ids)
            assert torch.equal(batch['input_ids'], torch.cat([batch['prompts'], batch['responses']], dim=1))
        results.append(dict(version='head' if rev == HEAD else 'base', lengths=lengths,
                            inferred_response_length=length, decoded=decoded,
                            input_width=batch['input_ids'].size(1),
                            penalty=float(consumer_scope['reward_extra_info']['overlong_reward'])))
print(json.dumps(dict(python=sys.version.split()[0], torch=torch.__version__, transformers=transformers.__version__,
                     tokenizers=tokenizers.__version__, results=results, sha256=hashes), indent=2))
