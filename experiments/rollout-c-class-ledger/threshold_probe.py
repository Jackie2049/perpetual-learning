import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def load(tag):
    if tag == 'p4':
        d = json.loads((ROOT/'rollout-p4-c-mechanisms/summary.json').read_text())
        return {k: [s['acc_curve'] for s in v['per_seed']] for k, v in d['arms'].items()}, 'base'
    d = json.loads((ROOT/'rollout-p3-dapo-group-filter/summary.json').read_text())
    return {k: [s['acc_curve'] for s in v['per_seed']] for k, v in d['modes'].items()}, 'R_nofilter'

def first(c, thr):
    for i, v in enumerate(c):
        if v >= thr:
            return i
    return None

THRS = (0.85, 0.87, 0.9, 0.93, 0.95)
for tag in ('p4', 'p3'):
    arms, base = load(tag)
    b = arms[base]
    for name, a in arms.items():
        if name == base:
            continue
        rows = []
        for thr in THRS:
            num = sum(first(c, thr) or 0 for c in a)
            den = sum(first(c, thr) or 0 for c in b)
            rows.append(num/den if den else None)
        ok = [r for r in rows if r is not None]
        span = round(100*(max(ok)-min(ok)), 1) if ok else None
        print(f"{tag}:{name}", [None if r is None else round(r, 3) for r in rows],
              'span_pp=', span, 'sign_flip=', len({r > 1 for r in ok}) > 1,
              'zero_margin_arm=', sum(1 for c in a if first(c, 0.9) is not None))
