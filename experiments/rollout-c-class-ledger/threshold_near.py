import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
p4 = json.loads((ROOT/'rollout-p4-c-mechanisms/summary.json').read_text())
p3 = json.loads((ROOT/'rollout-p3-dapo-group-filter/summary.json').read_text())

def cross(c, t):
    for i, v in enumerate(c):
        if v >= t:
            return i
    return None

for tag, d, key, base in (('p4', p4, 'arms', 'base'), ('p3', p3, 'modes', 'R_nofilter')):
    arms = d[key]
    B = {s['seed']: s['acc_curve'] for s in arms[base]['per_seed']}
    print('==', tag)
    for name, v in arms.items():
        if name == base:
            continue
        stats = {}
        for delta in (0.001, 0.005, 0.01):
            same = 0
            for s in v['per_seed']:
                c = s['acc_curve']; bc = B[s['seed']]
                lo = cross(c, 0.9 - delta); hi = cross(c, 0.9 + delta)
                blo = cross(bc, 0.9 - delta); bhi = cross(bc, 0.9 + delta)
                if lo == hi == blo == bhi:
                    same += 1
            stats[delta] = same
        print(f"  {name:5s} both-arms-unmoved at thr±0.001/±0.005/±0.01: "
              f"{stats[0.001]}/30 {stats[0.005]}/30 {stats[0.01]}/30")
