"""UniRL #486 full original stdlib planner: synthetic ledger/objective controls.

Usage: python3 -B verify_unirl_offload_plan.py /path/to/UniRL-git
Python 3.9+, fixed local Git blob; no implicit fetch, torch or device discovery.
Profiles below are explicitly synthetic, not calibration measurements. This
executes the original planner, NOT an executor, FSDP runtime or timing benchmark.
"""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

HEAD = '57cec429b5978d7da4bfd5f751764c0fd5da4c06'
path = 'unirl/train/backend/fsdp/offload_plan.py'
raw = subprocess.check_output(['git', 'show', f'{HEAD}:{path}'], cwd=Path(sys.argv[1]),
                              env=dict(os.environ, GIT_NO_LAZY_FETCH='1'))
print(path, 'sha256=' + hashlib.sha256(raw).hexdigest())
m = types.ModuleType('_original_offload_plan')
sys.modules[m.__name__] = m
exec(compile(raw, path, 'exec'), m.__dict__)
blocks = tuple(m.BlockSpec(f'b{i}', 'synthetic-owner', i, 10 * (i + 1), 20 * (i + 1)) for i in range(4))
inventory = m.BlockInventory(blocks, root_bytes=7)


def profile(gpu=1000, ranks=1, host=1000, baseline=13):
    return m.PhaseProfile(
        fingerprint={'source': 'synthetic-not-measured'},
        per_rank={str(rank): {phase: m.PhaseCost(
            compute_ms={b.block_id: 10. for b in blocks},
            transfer_ms={b.block_id: 3. for b in blocks},
            resident_gpu_bytes=baseline, host_bytes=2,
            measured=tuple(b.block_id for b in blocks)) for phase in m.PHASES} for rank in range(ranks)},
        node_of_rank={str(rank): 'node' for rank in range(ranks)},
        gpu_budget_bytes={str(rank): gpu for rank in range(ranks)}, host_budget_bytes={'node': host})


# Independent arithmetic oracle for candidate GPU bytes and exposed first-copy delay.
comparisons = 0
for bound in (None, 1, 2):
    capability = m.ExecutorCapability('native_fsdp' if bound is None else 'two_slot_streaming', bound)
    for headroom in (0, 5):
        for budget in (200, 270, 1000):
            for exposure in (None, 0.):
                plan = m.plan_placement(inventory, profile(gpu=budget), capability,
                                        exposure_limit_ms=exposure, headroom_bytes_by_rank={'0': headroom})
                expected_feasible = []
                for k, candidate in enumerate(plan.candidates):
                    resident = sum(30 * (i + 1) for i in range(k))
                    remaining = sorted([30 * (i + 1) for i in range(k, 4)], reverse=True)
                    live = sum(remaining if bound is None else remaining[:bound])
                    peak = 13 + 7 + headroom + resident + live
                    stall = 3. if k == 0 else 0.
                    assert candidate['predicted_peak_gpu_bytes']['0'] == peak
                    assert candidate['predicted_peak_host_bytes']['node'] == 2 + sum(10 * (i + 1) for i in range(k, 4))
                    assert candidate['predicted_exposed_stall_ms'] == stall
                    assert candidate['predicted_total_ms'] == 40. + stall  # Max phase, NOT sum of four phases.
                    feasible = peak <= budget and (exposure is None or stall <= exposure)
                    assert candidate['feasible'] == feasible
                    if feasible:
                        expected_feasible.append(k)
                    comparisons += 1
                assert (plan.status == 'FEASIBLE') == bool(expected_feasible)
                if expected_feasible:
                    assert len(plan.resident_block_ids) == expected_feasible[0]
                else:
                    assert not plan.resident_block_ids and not plan.streamed_block_ids
print('PASS:', comparisons, 'candidate arithmetic controls across 36 synthetic profile/capability settings')

native = m.ExecutorCapability('native_fsdp', None)
bounded = m.ExecutorCapability('two_slot_streaming', 2)
flat = m.plan_placement(inventory, profile(), native, headroom_bytes_by_rank={'0': 5})
assert [r['predicted_peak_gpu_bytes']['0'] for r in flat.candidates] == [325] * 5
limited = m.plan_placement(inventory, profile(gpu=250), bounded, headroom_bytes_by_rank={'0': 5})
assert limited.status == 'FEASIBLE' and len(limited.resident_block_ids) == 0
assert limited.latency_optimal_k == 1 and not limited.candidates[1]['feasible']
assert limited.transfer_busy_ms == 43. and limited.no_compute_transfer_ms == 3.
print('native peaks=[325]*5; declared bound2 peaks=', [r['predicted_peak_gpu_bytes']['0'] for r in limited.candidates])
print('chosen k=0; latency_optimal_k=1 is infeasible; transfer_busy_ms=43 vs copy durations sum=12')

partial = m.plan_placement(inventory, profile(gpu=270), bounded,
                           headroom_bytes_by_rank={'0': 5}, exposure_limit_ms=0.)
assert partial.resident_block_ids == ('b0',)
node = m.plan_placement(inventory, profile(ranks=2, host=203), native)
assert node.resident_block_ids == ('b0',) and node.candidates[0]['predicted_peak_host_bytes']['node'] == 204
double = m.plan_placement(inventory, profile(ranks=2, host=203), replace(native, double_buffer=True))
assert double.resident_block_ids == ('b0', 'b1', 'b2')
tight = replace(profile(ranks=2), gpu_budget_bytes={'0': 1000, '1': 319})
assert m.plan_placement(inventory, tight, native).status == 'INFEASIBLE'

missing = replace(profile(), per_rank={'0': {p: c for p, c in profile().per_rank['0'].items() if p != 'transition'}})
refused = m.plan_placement(inventory, missing, native)
assert refused.status == 'INFEASIBLE' and refused.missing_evidence == ('transition',)
assert all(not row['feasible'] for row in refused.candidates)
fixed_inventory = replace(inventory, blocks=(*blocks, m.BlockSpec('fixed', 'root', 4, 5, 5, fixed=True)))
fixed = m.plan_placement(fixed_inventory, profile(), native)
assert fixed.fixed_block_ids == ('fixed',) and 'fixed' not in fixed.streamed_block_ids
assert fixed.candidates[0]['predicted_peak_gpu_bytes']['0'] == 330
alias_inventory = replace(inventory, blocks=(replace(blocks[0], alias_group='tie'),
                                             replace(blocks[0], block_id='b1', order=1, alias_group='tie'), *blocks[2:]))
alias = m.plan_placement(alias_inventory, profile(), native)
assert any('alias group split' in reason for reason in alias.candidates[1]['reasons'])
assert json.loads(partial.to_json())['status'] == 'FEASIBLE'
assert partial.to_json() == m.plan_placement(inventory, profile(gpu=270), bounded,
    headroom_bytes_by_rank={'0': 5}, exposure_limit_ms=0.).to_json()
assert not any(n.split('.')[0] in {'torch', 'ray', 'vllm', 'sglang'} for n in sys.modules)
print('PASS: exposure-selected partial, node sum/double buffer, tight rank, missing phase, fixed/alias, deterministic JSON')
print('Python', sys.version.split()[0], '; original pure planner only, no calibrated prediction/executor/device claim.')
