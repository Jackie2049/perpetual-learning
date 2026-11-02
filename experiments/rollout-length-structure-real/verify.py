#!/usr/bin/env python3
"""复现校验：从 raw/ 重算全部统计量，与归档 JSON 逐字段比对（CPU，M5）。

只比数值字段（summary 里的 path/label 与重算无关，跳过）。
用法：python3 verify.py    # exit 0 = 全部一致
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import analyze_lengths as A  # noqa: E402
import reanalysis_length_regime as R  # noqa: E402

# (arm 标签, raw 文件名)
ARMS = [
    ('qwen15b', 'qwen15b_gsm8k_n8'),
    ('qwen15b-seed43', 'qwen15b_gsm8k_n8_seed43'),
    ('qwen15b-cap1024', 'qwen15b_gsm8k_n8_cap1024'),
    ('qwen05b', 'qwen05b_gsm8k_n8'),
    ('qwen05b-cap1024', 'qwen05b_gsm8k_n8_cap1024'),
]

FIELDS = ('cv_group', 'cv_in_pooled', 'cv_in_rms_rel', 'icc1', 'icc1_log',
          'grand_mean', 'cv_total', 'var_between', 'var_within_pooled')


def close(a, b, tol=1e-9):
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is b
        return abs(a - b) <= tol * max(1.0, abs(a), abs(b))
    return a == b


def main():
    failures = []
    for arm, rawstem in ARMS:
        raw = os.path.join(HERE, 'raw', rawstem + '.jsonl.gz')
        summ = os.path.join(HERE, 'summary-%s.json' % arm)
        with open(summ, encoding='utf-8') as f:
            arch = json.load(f)
        groups = A.load(raw)
        lengths = [g['lengths'] for g in groups.values()]
        for key, k in (('structure_n8', 8), ('structure_n4', 4)):
            got = A.structure(lengths, k)
            for fld in FIELDS:
                if not close(got[fld], arch[key][fld]):
                    failures.append('%s %s %s: %.12g != %.12g' % (arm, key, fld, got[fld], arch[key][fld]))
        trunc = sum(g['trunc'] for g in groups.values())
        total = sum(len(g['lengths']) for g in groups.values())
        if trunc != arch['truncated_samples'] or total != arch['n_samples']:
            failures.append('%s truncated/n mismatch' % arm)

        # 回代分析：确定性（种子固定），逐臂重算均值
        rpath = os.path.join(HERE, 'reanalysis-%s.json' % arm)
        if not os.path.exists(rpath):
            continue
        with open(rpath, encoding='utf-8') as f:
            rarch = json.load(f)
        real_groups = R.load_groups(raw)
        rows = []
        for s in R.SEEDS:
            import random
            rng = random.Random(s * 7919 + 13)
            picked = rng.sample(real_groups, R.N_GROUPS)
            rows.append(R.one_seed([x for g in picked for x in g], s))
        got = R.summarize(rows, 'real')
        for fld in ('ldm_over_grouped', 'interleaved_over_ldm'):
            if not close(got[fld]['mean'], rarch['arms']['real_resampled'][fld]['mean']):
                failures.append('%s %s: %.12g != %.12g' % (
                    arm, fld, got[fld]['mean'], rarch['arms']['real_resampled'][fld]['mean']))
    if failures:
        print('\n'.join(failures))
        print('FAIL %d field(s)' % len(failures))
        return 1
    print('OK %d arms, summary + reanalysis 逐字段一致' % len(ARMS))
    return 0


if __name__ == '__main__':
    sys.exit(main())
