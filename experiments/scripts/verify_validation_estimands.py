#!/usr/bin/env python3
"""Read pinned verl Bayes@N helper and compare to exact binary posterior moments.

Python >=3.10 and NumPy. No full verl import, trainer, TQ/Ray, model or GPU.
pass@k expectation checks use a separate explicit IID Bernoulli model, not rLLM.
--check-identity adds original config guard and manifest/key parsing fragments;
configuration objects are doubles, no original full validation pipeline is run.
"""
import argparse
import ast
from collections import defaultdict
from copy import deepcopy
from fractions import Fraction as F
import hashlib
from itertools import product
import math
from numbers import Integral, Real
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

REVISION = '8971a9ff7666dd2584484d3c8f042d3da4162457'
SOURCE = 'verl/trainer/ppo/metric_utils.py'


def load(root):
    data = subprocess.check_output(['git', 'show', f'{REVISION}:{SOURCE}'], cwd=root)
    print(REVISION, SOURCE, 'sha256=' + hashlib.sha256(data).hexdigest())
    fn = next(n for n in ast.parse(data.decode()).body
              if isinstance(n, ast.FunctionDef) and n.name == '_bayes_at_n_metrics')
    assert not fn.decorator_list
    namespace = {'math': math, 'np': np, 'Integral': Integral, 'Real': Real, 'defaultdict': defaultdict}
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, fn], type_ignores=[])), '<original Bayes helper>', 'exec'), namespace)
    return namespace[fn.name]


def cohort(rows):
    m, n = len(rows), len(rows[0])
    return dict(data_sources=['s']*(m*n), sample_uids=[f'q{i}' for i in range(m) for _ in range(n)],
                infos_dict={'acc': [x for row in rows for x in row]},
                expected_counts={('s', f'q{i}'): n for i in range(m)}, sample_trial_ids=list(range(n))*m)


def exact(rows):
    m, n = len(rows), len(rows[0])
    means, variances = [], []
    for row in rows:
        a, b = sum(row)+1, n-sum(row)+1
        means.append(F(a, a+b))
        variances.append(F(a*b, (a+b)**2*(a+b+1)))
    return sum(means)/m, sum(variances)/(m*m)


def pass_k(n, c, k):
    assert 0 <= c <= n and 1 <= k <= n
    return 1-F(math.comb(n-c, k), math.comb(n, k))


def check_identity(root, helper):
    """Execute original guard and two original validation fragments, not _validate.

    Config is an attribute-access dict double. UID creation, worker scheduling,
    final-output deduplication, TQ, grading, cleanup and logging are not run.
    """
    path = 'verl/trainer/ppo/v1/trainer_base.py'
    data = subprocess.check_output(['git', 'show', f'{REVISION}:{path}'], cwd=root)
    print(path, 'sha256=' + hashlib.sha256(data).hexdigest())
    cls = next(n for n in ast.parse(data.decode()).body if isinstance(n, ast.ClassDef) and n.name == 'PPOTrainer')
    guard = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_check_bayes_at_n_config')
    validate = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_validate')
    batch_loop = next(n for n in validate.body if isinstance(n, ast.For))
    stop = next(i for i,n in enumerate(batch_loop.body) if isinstance(n, ast.Assign)
                and any(isinstance(t,ast.Name) and t.id == 'batch' for t in n.targets))
    assert isinstance(batch_loop.body[0], ast.Assign) and 'uuid.uuid4' in ast.unparse(batch_loop.body[0])
    record = ast.parse('def record(self, batch_dict, expected_acc_counts, trial_count): pass').body[0]
    record.body = batch_loop.body[1:stop]  # exact manifest block, after caller-supplied UID creation
    parse = ast.parse('def parse_ids(final_keys, batch_uids, sample_trial_ids): pass').body[0]
    parse.body = [next(n for n in batch_loop.body if isinstance(n,ast.If)
                       and any(isinstance(k,ast.Call) and isinstance(k.func,ast.Attribute)
                               and k.func.attr == 'append' and isinstance(k.func.value,ast.Name)
                               and k.func.value.id == 'sample_trial_ids' for k in ast.walk(n)))]
    namespace = {'math':math, 'np':np, 'Integral':Integral, 'bayes_at_n':True}
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future,guard,record,parse], type_ignores=[])),
                 '<original validation guard/fragments>', 'exec'), namespace)

    class Config(dict):
        __getattr__ = dict.__getitem__

    def trainer():
        return SimpleNamespace(trainer_mode='sync', config=Config(
            trainer=Config(val_bayes_at_n=True),
            actor_rollout_ref=Config(rollout=Config(full_determinism=False,
                val_kwargs=Config(n=2, do_sample=True, temperature=1., top_p=1., top_k=-1),
                agent=Config(default_agent_loop='single_turn_agent'))),
            reward=Config(reward_manager=Config(source='register',name='naive'),reward_model=Config(enable=False))))

    count = 0
    assert namespace[guard.name](trainer()) == 2
    disabled = SimpleNamespace(config=Config(trainer=Config(val_bayes_at_n=False)))
    assert namespace[guard.name](disabled) is None  # no access to missing rollout fields
    count += 2
    mutations = [lambda t: setattr(t,'trainer_mode','separate_async'),
                 lambda t: t.config.actor_rollout_ref.rollout.val_kwargs.update(top_k=1),
                 lambda t: t.config.actor_rollout_ref.rollout.val_kwargs.update(temperature=float('nan')),
                 lambda t: t.config.actor_rollout_ref.rollout.val_kwargs.update(top_p=0),
                 lambda t: t.config.actor_rollout_ref.rollout.val_kwargs.update(n=True),
                 lambda t: t.config.actor_rollout_ref.rollout.agent.update(agent_loop_config_path='custom.yaml'),
                 lambda t: t.config.reward.reward_model.update(enable=True)]
    for mutation in mutations:
        t = trainer(); mutation(t)
        try: namespace[guard.name](t)
        except ValueError: count += 1
        else: raise AssertionError('guard accepted unsupported configuration')

    t, expected = trainer(), {}
    batch = {'uid':['q_with_under','q1'], 'data_source':['s','s']}
    namespace['record'](t,batch,expected,2)
    assert expected == {('s','q_with_under'):2, ('s','q1'):2}
    count += 1
    invalid_batches = [batch, {'uid':['new'], 'data_source':['']},
                       {'uid':['new'], '__rollout_n__':[1]}, {'uid':['new'], 'agent_name':['tool_agent']}]
    for bad in invalid_batches:
        try: namespace['record'](t,bad,deepcopy(expected),2)
        except ValueError: count += 1
        else: raise AssertionError('manifest fragment accepted invalid batch')

    keys = ['q1_1_0','q_with_under_0_0','q1_0_0','q_with_under_1_0']
    uids = ['q1','q_with_under','q1','q_with_under']
    ids = []; namespace['parse_ids'](keys,uids,ids)
    assert ids == [1,0,0,1]
    helper(['s']*4,uids,{'acc':[1,0,0,1]},expected,ids)
    count += 1
    invalid_keys = [(['other_0_0'],['q1']), (['q1_0_1'],['q1']),
                    (['q1_minus_0'],['q1']), (['q1_0_0'],[])]
    for keys,uids in invalid_keys:
        try: namespace['parse_ids'](keys,uids,[])
        except ValueError: count += 1
        else: raise AssertionError('key fragment accepted invalid identity')

    # String-distinct session keys parse to the same integer trial; helper rejects.
    ids = []; namespace['parse_ids'](['q_01_0','q_1_0'],['q','q'],ids)
    try: helper(['s','s'],['q','q'],{'acc':[0,1]},{('s','q'):2},ids)
    except ValueError: count += 1
    else: raise AssertionError('duplicate numeric session ID accepted')
    ids = []; namespace['parse_ids'](['q_with_under_0_0','q_with_under_1_0'],['q_with_under']*2,ids)
    try: helper(['s']*2,['q_with_under']*2,{'acc':[0,1]},expected,ids)
    except ValueError: count += 1
    else: raise AssertionError('entire missing question accepted')
    assert count == 21, count
    print('PASS 21 original config-guard/manifest/key/helper controls; selected fragments, not full validation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('verl_root', type=Path)
    parser.add_argument('--check-identity', action='store_true')
    args = parser.parse_args()
    helper = load(args.verl_root)
    if args.check_identity:
        check_identity(args.verl_root, helper)
    cases = 0
    for m, n in product(range(1,4), repeat=2):
        for flat in product((0,1), repeat=m*n):
            rows = [flat[i*n:(i+1)*n] for i in range(m)]
            inputs = cohort(rows)
            before = deepcopy(inputs)
            result = helper(**inputs)
            mean, variance = exact(rows)
            assert math.isclose(result['s'][f'bayes@{n}/mean'], float(mean), rel_tol=0, abs_tol=1e-14)
            assert math.isclose(result['s'][f'bayes@{n}/std'], math.sqrt(float(variance)), rel_tol=0, abs_tol=1e-14)
            assert mean == (n*F(sum(flat),m*n)+1)/(n+2)
            assert inputs == before
            # Reverse completion arrival order while preserving each native trial ID.
            shuffled = deepcopy(inputs)
            for key in ('data_sources', 'sample_uids', 'sample_trial_ids'):
                shuffled[key].reverse()
            shuffled['infos_dict']['acc'].reverse()
            assert helper(**shuffled) == result
            cases += 1
    assert cases == 682
    print('PASS 682 binary matrices against exact Fraction moments, plus 682 identity-preserving reorder controls.')

    invalid = [dict(expected_counts=None), dict(expected_counts={}), dict(sample_trial_ids=None),
               dict(sample_trial_ids=[0,0]), dict(sample_trial_ids=[0,2]),
               dict(sample_trial_ids=[False,1]), dict(infos_dict={'acc':[0,0.5]}),
               dict(infos_dict={'acc':[0,float('nan')]}), dict(infos_dict={'acc':[0]}),
               dict(expected_counts={('s','q0'):2,('s','missing'):2})]
    for changes in invalid:
        inputs = cohort([[0,1]])
        inputs.update(changes)
        try:
            helper(**inputs)
        except ValueError:
            continue
        raise AssertionError(changes)
    print('PASS 10 incomplete/nonbinary/duplicate-ID input rejections by original helper.')

    separated, mixed = [[0,0],[1,1]], [[0,1],[0,1]]
    assert exact(separated) == (F(1,2), F(3,160))
    assert exact(mixed) == (F(1,2), F(1,40))
    print('same mean 1/2, posterior variance separated=3/160 vs mixed=1/40; not pooled-Beta variance 1/28.')
    # Equal-N ranking equivalence is conditional: unequal N can reverse it.
    assert F(3,5) > F(29,50) and F(4,7) < F(30,52)
    assert pass_k(8,2,1) == F(1,4) and pass_k(8,2,2) == F(13,28)
    checks = 0
    for n in range(1,9):
        for k in range(1,n+1):
            for p in (F(0),F(1,5),F(1,2),F(4,5),F(1)):
                expected = sum(F(math.comb(n,c))*p**c*(1-p)**(n-c)*pass_k(n,c,k) for c in range(n+1))
                assert expected == 1-(1-p)**k
                checks += 1
    assert checks == 180
    print('PASS 180 exact IID-Bernoulli expectation controls for the combinatorial pass@k estimator.')
    print('No full trainer/session/grader/outer-padding pipeline; no paper empirical ranking or real IID validation.')


if __name__ == '__main__':
    main()
