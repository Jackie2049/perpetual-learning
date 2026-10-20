#!/usr/bin/env python3
"""Read pinned verl Bayes@N helper and compare to exact binary posterior moments.

Python >=3.10 and NumPy. No full verl import, trainer, TQ/Ray, model or GPU.
pass@k expectation checks use a separate explicit IID Bernoulli model, not rLLM.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('verl_root', type=Path)
    args = parser.parse_args()
    helper = load(args.verl_root)
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
    print('No original trainer/guard/session/grader/outer padding tests; no paper empirical ranking or real IID validation.')


if __name__ == '__main__':
    main()
