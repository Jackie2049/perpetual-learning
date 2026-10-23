"""Run full original stdlib UniRL #487 schema/converters and author's fork guard fix.

Usage: python -B verify_unirl_trace_contract.py /path/to/UniRL-git
Python 3.9+, local immutable Git blobs; no implicit fetch or external dependency.
Only parent package namespaces are shells; both implementation modules execute
unaltered. Synthetic requests are contract fixtures, not a real capture/profile.
No simulator, runner, model, timing prediction, GPU/NPU or training validation.
"""
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import types

UPSTREAM = 'e8779559d2ceb441c062f603f90e7fd31370e69b'
FOLLOWUP = '4ad73720989c8b7c9229da4ecabd9ee654e67dc4'
repo = Path(sys.argv[1])
env = dict(os.environ, GIT_NO_LAZY_FETCH='1')
for name in ('experimental', 'experimental.aisim'):
    package = types.ModuleType(name)
    package.__path__ = []
    sys.modules[name] = package


def load(rev, name):
    path = f'experimental/aisim/{name}.py'
    raw = subprocess.check_output(['git', 'show', f'{rev}:{path}'], cwd=repo, env=env)
    print(rev[:8], path, 'sha256=' + hashlib.sha256(raw).hexdigest())
    module = types.ModuleType('experimental.aisim.' + name)
    module.__package__ = 'experimental.aisim'
    sys.modules[module.__name__] = module
    exec(compile(raw, path, 'exec'), module.__dict__)
    return module


def rejects(call):
    try:
        call()
    except (ValueError, TypeError):
        return True
    return False


valid_json = []
for rev in (UPSTREAM, FOLLOWUP):
    s, c = load(rev, 'schema'), load(rev, 'convert')
    manifest = s.Manifest(
        run_id='synthetic-contract-control', capture_status='complete', unirl_commit=rev,
        simulator_commit_or_version='not-run', replay_contract='dependency_arrival',
        backend_name='synthetic', model_revision='synthetic', model_dtype='not-modelled',
        sampling_config_digest='synthetic', hardware_model='not-profiled', topology_digest='synthetic',
        engine_count=1, tp_size=1, policy_identity='synthetic-policy', measurement_boundary='synthetic',
        timestamp_unit='ms', privacy_mode='no-content', redaction_policy='synthetic',
        capture_overhead_method='not-measured')

    def request(name, arrival, turn=0):
        return s.WorkloadRequest(name, 'batch', 'group', 'trajectory', turn, 'engine', 'worker',
                                 'synthetic-policy', 8, 4, arrival)

    observed = [request(name, s.Arrival('observed', offset_ms=time))
                for name, time in (('r2', 100.), ('r1', 100.), ('r3', 130.))]
    open_manifest = replace(manifest, replay_contract='open_loop_observed_arrival')
    opened = c.to_open_loop(open_manifest, observed)
    assert [r['arrival_ms'] for r in opened.requests] == [0., 0., 30.]
    assert opened.sidecar['arrival_origin_ms'] == 100.
    assert [r['request_id'] for r in opened.requests] == ['r2', 'r1', 'r3']
    assert s.assertion_tie_break(observed) == ['r2', 'r1', 'r3']
    assert all(set(row) == {'request_id', 'input_tokens', 'output_tokens', 'arrival_ms'}
               for row in opened.requests)
    assert rejects(lambda: c.to_open_loop(manifest, observed))
    assert rejects(lambda: c.to_open_loop(replace(open_manifest, capture_status='incomplete'), observed))

    dependencies = [request('first', s.Arrival('after_dependencies', offset_ms=7.)),
                    request('second', s.Arrival('after_dependencies', wait_for=('first',),
                                               external_delay_ms=0.), turn=1)]
    converted = c.to_dependency(manifest, dependencies)
    assert converted.requests[0]['arrival_ms'] == 7.
    assert converted.requests[1]['wait_for'] == ['first']
    assert converted.requests[1]['tool_wait_ms'] == 0.
    assert 'arrival_ms' not in converted.requests[1]
    assert converted.sidecar['slot_semantics'] == converted.sidecar['barrier_semantics'] == 'not_modelled'
    valid_json.append(converted.to_json())
    assert rejects(lambda: c.to_open_loop(open_manifest, dependencies))
    assert rejects(lambda: c.to_dependency(manifest, observed))

    for label, invalid in (
            ('observed manifest for dependency rows', open_manifest),
            ('incomplete dependency capture', replace(manifest, capture_status='incomplete'))):
        rejected = rejects(lambda: c.to_dependency(invalid, dependencies))
        assert rejected == (rev == FOLLOWUP)
        print(label + ': ' + ('REJECTED' if rejected else 'ACCEPTED (known upstream gap)'))

    unknown_delay = replace(dependencies[1], arrival=s.Arrival('after_dependencies', wait_for=('first',)))
    assert rejects(lambda: s.validate_workload([dependencies[0], unknown_delay]))
    unknown_id = replace(dependencies[1], arrival=s.Arrival('after_dependencies', wait_for=('absent',),
                                                           external_delay_ms=0.))
    assert rejects(lambda: s.validate_workload([dependencies[0], unknown_id]))
    cyclic_first = replace(dependencies[0], arrival=s.Arrival('after_dependencies', wait_for=('second',),
                                                            external_delay_ms=0.))
    assert rejects(lambda: s.validate_workload([cyclic_first, dependencies[1]]))
    text = s.workload_to_jsonl(dependencies)
    decoded = s.workload_from_jsonl(text)
    s.validate_workload(decoded)
    assert s.workload_to_jsonl(decoded) == text

    results = [s.MeasuredResult(r.request_id, 'synthetic-clock', 'submit-complete', 10., 20., 'completed')
               for r in dependencies]
    s.validate_measured(results, dependencies)
    assert rejects(lambda: s.validate_measured(results[:1], dependencies))
    assert rejects(lambda: s.validate_measured([*results, results[0]], dependencies))
    assert rejects(lambda: s.validate_measured([replace(results[0], request_id='unknown'), results[1]], dependencies))
    c.assert_no_outcome_leakage(converted)
    leaked = replace(converted, requests=({**converted.requests[0], 'service_time_ms': 10.},))
    assert rejects(lambda: c.assert_no_outcome_leakage(leaked))
    assert c.result_sidecar([{'complete_ms': 20.}])['purpose'] == 'report_only'
    print('PASS: original open/dependency conversion, identity/order, explicit wait, graph rejection, JSONL, outcome controls')

assert valid_json[0] == valid_json[1], 'valid dependency output changed'
assert not any(name.split('.')[0] in {'torch', 'ray', 'vllm', 'sglang', 'aisimulate'} for name in sys.modules)
print('PASS: valid converted JSON identical; no heavyweight/simulator imports; Python', sys.version.split()[0])
print('Synthetic stdlib contract execution only; NOT simulator, capture, timing or fidelity validation.')
