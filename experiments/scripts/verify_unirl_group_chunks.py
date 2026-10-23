"""UniRL #485 original stdlib group-boundary helper and source-only engine checks.

Usage: python3 -B verify_unirl_group_chunks.py /path/to/UniRL-git
Python 3.9+, fixed local Git objects, no implicit fetch or dependencies.
The complete original plan.py executes; engine/NoiseRecipe only compile, never
import. No Sample/Part, model, noise, engine parity, memory or device validation.
"""
from collections import Counter
import hashlib
from itertools import product
import os
from pathlib import Path
import subprocess
import sys

HEAD = '9e36cf99eea03e9e2f6035976ac97c1a70759f9e'
BASE = '772fd7c50ec8221a032179aeec8320df935b3abf'
repo = Path(sys.argv[1])
env = dict(os.environ, GIT_NO_LAZY_FETCH='1')


def source(rev, path):
    raw = subprocess.check_output(['git', 'show', f'{rev}:{path}'], cwd=repo, env=env)
    print(rev[:8], path, 'sha256=' + hashlib.sha256(raw).hexdigest())
    compile(raw, path, 'exec')
    return raw


ns = {}
exec(compile(source(HEAD, 'unirl/rollout/engine/trainside/plan.py'), 'original plan.py', 'exec'), ns)
validate = ns['validate_group_chunks']
for rev in (BASE, HEAD):
    source(rev, 'unirl/rollout/engine/trainside/engine.py')
source(HEAD, 'unirl/types/noise_recipe.py')


def accepted(ids, capacity):
    before = list(ids)
    try:
        result = validate(ids, capacity)
    except ValueError:
        assert ids == before
        return False
    assert result is None and ids == before
    return True


def oracle(ids, capacity):
    # Independent model: enumerate concrete fixed-size chunks, not group start modulo math.
    chunks = [Counter(ids[i:i + capacity]) for i in range(0, len(ids), capacity)]
    for label in set(ids):
        positions = [i for i, value in enumerate(ids) if value == label]
        if positions != list(range(positions[0], positions[-1] + 1)):
            return False
        occupied = [chunk[label] for chunk in chunks if chunk[label]]
        if len(occupied) > 1 and any(count != capacity for count in occupied):
            return False
    return True


checks = positives = 0
for count in range(8):
    for layout in product(('a', 'b', 'c'), repeat=count):
        for capacity in range(1, 6):
            ids = list(layout)
            actual = accepted(ids, capacity)
            assert actual == oracle(ids, capacity), (layout, capacity)
            positives += int(actual)
            checks += 1
assert checks == 16400
print('PASS:', checks, 'bounded layouts (0..7 rows, 3 labels, FBS1..5); accepted=', positives)

for group, capacity, expected in ((16, 4, True), (16, 8, True), (16, 1, True),
                                  (24, 1, True), (16, 6, False), (16, 64, True)):
    assert accepted(['a'] * group, capacity) == expected
assert not accepted(['a'] * 2 + ['b'] * 3, 4)  # Smaller b group straddles row4.
assert not accepted(['a', 'b', 'a'], 64)  # Helper rejects, but engine skips helper for bs <= FBS.
assert accepted(['a'] * 2 + ['b'] * 2 + ['c'] * 4, 4)
assert not any(n.split('.')[0] in {'torch', 'ray', 'vllm', 'sglang'} for n in sys.modules)
print('PASS: 6 large geometries + small-straddle/interleaving/mixed-group controls; no input mutation')
print('Python', sys.version.split()[0], '; original helper only, not engine invocation or RNG/model parity.')
