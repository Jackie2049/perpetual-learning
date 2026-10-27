"""Fixed-source AST/extracted-body controls, not engine or allocator tests.

Run with Python 3.12 and two read-only checkouts at the exact commits below.
No framework imports, dependencies, network calls or source edits.
"""
import argparse
import ast
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace as NS

class RequestStub(NS):
    __hash__ = object.__hash__

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--vllm', required=True, type=Path)
parser.add_argument('--sglang', required=True, type=Path)
args = parser.parse_args()
V, S = args.vllm.resolve(), args.sglang.resolve()
commits = {V: 'eaf1627c01f553ea9a26b3d77f3cff8495dbf404',
           S: 'cb55d77d37c5f982e75edfd7f4a016f6a6ad35a1'}
for root, expected in commits.items():
    actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != expected:
        raise ValueError(f'Wrong source version at {root}: {actual}, expected {expected}')
vp = 'vllm/v1/core/sched/scheduler.py'
sp = 'python/sglang/srt/managers/schedule_policy.py'
bp = 'python/sglang/srt/managers/schedule_batch.py'
mem = 'python/sglang/srt/mem_cache/prefill_budget.py'
parsed = {}
for root, files in [(V, [vp, 'vllm/v1/core/sched/async_scheduler.py',
                        'vllm/v1/core/kv_cache_manager.py', 'vllm/config/scheduler.py']),
                    (S, [sp, bp, mem, 'python/sglang/srt/managers/scheduler.py',
                        'python/sglang/srt/managers/scheduler_components/new_token_ratio_tracker.py'])]:
    for path in files:
        source = (root / path).read_text()
        blob = subprocess.check_output(['git', '-C', str(root), 'show', f'HEAD:{path}'], text=True)
        if source != blob:
            raise ValueError(f'Selected source is modified: {root / path}')
        parsed[root / path] = (source, ast.parse(source))

def load(root, path, owner, name, scope=None):
    source, tree = parsed[root / path]
    container = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    node = next(n for n in container.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = {} if scope is None else scope
    exec(compile('from __future__ import annotations\n' + textwrap.dedent(ast.get_source_segment(source, node)),
                 str(root / path) + ':' + str(node.lineno), 'exec'), scope)
    return scope[name]

# 1. Original MTP chunk-boundary helper: finish, leave full lookahead, or zero.
reserve = load(V, vp, 'Scheduler', '_reserve_prefill_lookahead')
boundary_cases = 0
for total in range(1, 25):
    for done in range(total):
        for take in range(total - done + 1):
            for look in (1, 2, 4):
                got = reserve(NS(num_prefill_lookahead=look), NS(num_tokens=total), done, take)
                assert 0 <= got <= take
                if got:
                    remaining = total - done - got
                    assert remaining == 0 or remaining >= look
                boundary_cases += 1

# 2. Original preempt body, with counters/queue/cache APIs represented by fakes.
status = NS(RUNNING='running', PREEMPTED='preempted')
preempt = load(V, vp, 'Scheduler', '_preempt_request', {'RequestStatus': status})
for drop, prior_drop, prior_stale in [(False, False, 0), (True, False, 0),
                                     (False, True, 2), (False, True, 0)]:
    calls, queue = [], []
    req = RequestStub(request_id='r', status='running', num_computed_tokens=14,
             spec_token_ids=[4, 5], num_stale_output_tokens=prior_stale,
             num_in_flight_tokens=6, num_output_placeholders=3,
             num_preemptions=0, drop_stale_output=prior_drop,
             output_token_ids=[7, 8])
    ctl = NS(aux_output_connector=None, _free_request_blocks=lambda r: calls.append('kv'),
             encoder_cache_manager=NS(free=lambda r: calls.append('encoder')),
             _pending_encoder_input_ids={'r': [0]}, _inflight_prefills=set(),
             log_stats=False, waiting=NS(prepend_request=queue.append),
             reset_preempted_req_ids=set())
    preempt(ctl, req, 0, drop)
    assert req.status == 'preempted' and req.num_computed_tokens == 0
    assert req.num_stale_output_tokens == 6 and req.num_in_flight_tokens == 6
    assert req.num_output_placeholders == 0 and req.spec_token_ids == []
    assert req.output_token_ids == [7, 8] and req.num_preemptions == 1
    assert req.drop_stale_output == (drop or (prior_drop and prior_stale > 0))
    assert queue == [req] and calls == ['kv', 'encoder']

# 3. Original optimistic bookkeeping, not evidence that GPU work has finished.
after = load(V, vp, 'Scheduler', '_update_after_schedule')
r = RequestStub(num_computed_tokens=0, num_in_flight_tokens=0, num_tokens=5,
       num_output_placeholders=0, use_structured_output=False)
ctl = NS(requests={'r': r}, defer_block_free=False, _inflight_prefills=set(),
         finished_req_ids={'old'}, reset_preempted_req_ids={'old'})
for take, expected in [(3, True), (2, False)]:
    out = NS(num_scheduled_tokens={'r': take}, has_structured_output_requests=False)
    after(ctl, out)
    assert r.is_prefill_chunk == expected
assert r.num_computed_tokens == r.num_in_flight_tokens == 5
assert ctl.finished_req_ids == ctl.reset_preempted_req_ids == set()

# 4. Original free-fence predicate with no actual blocks or device operations.
can_free = load(V, vp, 'Scheduler', '_request_blocks_can_be_freed')
for deferred, last, processed, expected in [(False, 7, 0, True), (True, 7, 6, False),
                                           (True, 7, 7, True), (True, 6, 7, True)]:
    assert can_free(NS(defer_block_free=deferred, processed_step_seq=processed),
                    NS(last_sched_seq=last)) == expected

# 5. Original SGLang retraction order: pop the least preferred index.
cfg = NS(retraction_policy='length', schedule_low_priority_values_first=True)
order = load(S, bp, 'ScheduleBatch', '_get_decode_retraction_order',
             {'get_schedule': lambda: cfg, 'sys': sys})
reqs = [NS(output_ids=[0] * o, origin_input_ids=[0] * p, priority=pr)
        for o, p, pr in [(5, 20, 1), (2, 4, 3), (2, 10, 2)]]
assert order(reqs) == [0, 1, 2]  # equal short output: longer prompt retracts first
cfg.retraction_policy = 'priority'
assert order(reqs)[-1] == 1      # lower numeric priority wins
cfg.schedule_low_priority_values_first = False
assert order(reqs)[-1] == 0

# 6. Original single-pool PrefillBudget class; only capacity provider is fake.
source, tree = parsed[S / mem]
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PrefillBudget')
ns = {}
exec(compile('from __future__ import annotations\n' + ast.get_source_segment(source, node), str(S / mem), 'exec'), ns)
budget = ns['PrefillBudget'](NS(page_size=4, available_size=lambda: 40),
                            NS(supports_mamba=lambda: False, evictable_size=lambda: 8))
check = dict(extend_input_len=4, total_tokens=48, max_new_tokens=1,
             input_tokens=4, swa_host_hit_length=0, chunk_limit=4)
assert budget.check_prefill(**check) == (False, None)  # strict total < remaining
check['total_tokens'] = 47
assert budget.check_prefill(**check) == (True, 4)
budget.reserve(5, 7)
assert budget.remaining_current == 36 and budget.remaining_total == 29
assert budget.can_allocate_prefill(paged_input=29, extend_input_len=5, max_new_tokens=7, chunk_limit=4)
assert not budget.can_allocate_prefill(paged_input=30, extend_input_len=5, max_new_tokens=7, chunk_limit=4)

# 7. Original shortest-uncached-prefill order, not predicted output length.
work = load(S, sp, 'SchedulePolicy', '_shortest_prefill_work')
sort = load(S, sp, 'SchedulePolicy', '_sort_by_shortest_prefill',
            {'SchedulePolicy': NS(_shortest_prefill_work=work)})
reqs = [NS(rid=k, origin_input_ids=[0] * n, output_ids=[], num_matched_prefix_tokens=h,
           time_stats=NS(wait_queue_entry_time=t))
        for k, n, h, t in [('a', 20, 18, 0), ('b', 5, 0, 1), ('c', 1, 1, 2)]]
sort(reqs, {'c'})
assert [r.rid for r in reqs] == ['a', 'b', 'c']
assert work(reqs[-1]) == 1

# 8. Original ratio estimator; environment reader represented by a constant.
ratio = load(S, 'python/sglang/srt/managers/scheduler_components/new_token_ratio_tracker.py',
             'NewTokenRatioTracker', 'estimate_new_token_ratio_after_retract',
             {'envs': NS(SGLANG_RETRACT_DECODE_STEPS=NS(get=lambda: 4))})
assert ratio([]) == 0
assert ratio([NS(output_ids=[0] * 2, sampling_params=NS(max_new_tokens=10))]) == 6 / 11
assert ratio([NS(output_ids=[0] * 9, sampling_params=NS(max_new_tokens=10))]) == 1
print(json.dumps({
    'result': 'pass', 'python': sys.version.split()[0],
    'vllm_commit': commits[V], 'sglang_commit': commits[S],
    'source_asts': len(parsed), 'control_groups': 8,
    'lookahead_cases': boundary_cases,
    'groups': ['vllm_mtp_chunk_boundary', 'vllm_preempt_bookkeeping',
               'vllm_optimistic_schedule_counters', 'vllm_free_fence_predicate',
               'sglang_retraction_order', 'sglang_single_pool_prefill_budget',
               'sglang_uncached_prefill_order', 'sglang_retraction_ratio_estimator'],
    'limits': 'Fake request/queue/cache/capacity APIs. No framework import, native pytest, kernels, real allocation, concurrency, distributed execution or timing.'
}, indent=2))
