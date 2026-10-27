"""Check fixed historical source provenance, AST structure and research ledger.

No framework import or source-function execution; these are static checks.
"""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import re
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', required=True, type=Path)
args = parser.parse_args()
root = args.source.resolve()
oid = '53bb5d21372a667954afb7109a2ef1df485e3137'
initial = '30911f133aa300ae9d8e341dba8e63192335705e'

def git(*argv):
    return subprocess.check_output(['git', '-C', str(root), *argv], text=True).strip()

assert git('rev-parse', 'HEAD') == oid
assert git('rev-parse', 'v0.1rc') == oid
assert git('rev-list', '--max-parents=0', 'HEAD') == initial
assert git('rev-list', '--count', 'HEAD') == '4'
assert not git('status', '--porcelain')

paths = [
    'verl/trainer/ppo/hybrid_engine/megatron_vllm.py',
    'verl/trainer/ppo/workers/megatron_workers.py',
    'verl/trainer/ppo/rollout/vllm_rollout/vllm_rollout.py',
    'verl/trainer/ppo/ray_trainer.py',
    'single_controller/base/decorator.py',
    'verl/third_party/vllm/__init__.py',
] + ['verl/third_party/vllm/vllm_v_0_5_4/' + p for p in [
    'parallel_state.py', 'llm.py', 'llm_engine_sp.py',
    'spmd_gpu_executor.py', 'worker.py', 'megatron_weight_loaders.py',
]]
trees = {}
for path in paths:
    source = (root / path).read_text()
    blob = subprocess.check_output(['git', '-C', str(root), 'show', f'{oid}:{path}'], text=True)
    assert source == blob, path
    trees[path] = ast.parse(source, filename=path)

def method(path, owner, name):
    cls = next(n for n in trees[path].body if isinstance(n, ast.ClassDef) and n.name == owner)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)

def calls(node):
    found = sorted((n for n in ast.walk(node) if isinstance(n, ast.Call)), key=lambda n: (n.lineno, n.col_offset))
    return [(n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id)
            for n in found if isinstance(n.func, (ast.Attribute, ast.Name))]

manager = paths[0]
enter = calls(method(manager, 'MegatronVLLMShardingManager', '__enter__'))
expected_enter = ['load_params_to_cuda', 'allgather_params', 'get_all_params',
                  'normalize_pp_vpp_params', '_post_process_params', 'sync_model_weights']
assert enter == expected_enter
exit_calls = calls(method(manager, 'MegatronVLLMShardingManager', '__exit__'))
assert exit_calls == ['offload_params_to_cpu', 'get_micro_data_parallel_world_size',
                      'keys', 'offload_model_weights', 'train', 'empty_cache']
worker = method(paths[1], 'ActorRolloutRefWorker', 'generate_sequences')
with_node = next(n for n in worker.body if isinstance(n, ast.With))
recompute = next(n for n in ast.walk(worker) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == 'compute_log_prob')
assert recompute.lineno > with_node.end_lineno

ledger_path = Path(__file__).resolve().parents[2] / 'notebook/rollout-performance-research.md'
ledger = ledger_path.read_text().split('### 5.1 ', 1)[1].split('### 5.6 ', 1)[0]
rows = [row for row in ledger.splitlines() if row.startswith('| [')]
tiers, groups = Counter(), Counter()
ids = []
for row in rows:
    cells = [v.strip() for v in row.strip('|').split('|')]
    assert len(cells) in (5, 7)
    tier = re.findall(r'\[(T[123])\]', cells[-1])
    assert len(tier) <= 1
    tiers[tier[0] if tier else 'unmarked'] += 1
    groups['serving' if len(cells) == 5 else 'RL-related'] += 1
    match = re.search(r'arxiv.org/abs/(\d{4}\.\d+)', cells[0])
    if match:
        ids.append(match[1])
assert len(rows) == 126 and len(ids) == len(set(ids)) == 120
assert tiers == {'T1': 20, 'T2': 84, 'T3': 21, 'unmarked': 1}
assert groups == {'RL-related': 106, 'serving': 20}
print(json.dumps({
    'result': 'pass', 'source_commit': oid, 'tag': 'v0.1rc',
    'commit_date': git('show', '-s', '--format=%cI', oid),
    'initial_public_chain_commit': initial,
    'initial_commit_date': git('show', '-s', '--format=%cI', initial),
    'history_commits': 4, 'source_asts': len(trees),
    'enter_calls': enter, 'actor_recompute_after_sharding_context': True,
    'ledger': {'identities': len(rows), 'unique_arxiv': len(set(ids)),
               'depth': dict(tiers), 'groups': dict(groups)},
    'limits': 'Static provenance, blob, AST and ledger checks only. No source-function, framework, tensor, collective, device or performance execution; not the paper experiment commit.'
}, ensure_ascii=False, indent=2))
