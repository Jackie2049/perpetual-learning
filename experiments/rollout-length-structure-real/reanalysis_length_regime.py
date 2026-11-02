#!/usr/bin/env python3
"""真实长度结构代入 §13.38 的两种块组成读数（CPU，M5）。

§13.38 的两个主读数（`ldm_over_grouped`、`interleaved_over_ldm`）依赖长度结构参数
`(cv_group, cv_in)`，而该参数是构造的（`realistic=(0.60, 0.08)`）。本脚本用
sample_lengths.py 采到的真实响应长度替换它，并做三档对照：

  A. `constructed`  §13.38 的构造档（自检：必须复现 1.0462 / 1.0045 附近）
  B. `synthetic`    用**实测参数**（cv_group／cv_in／均值）重新抽样
  C. `real`         从真实组里重抽样 32 个组，长度原样使用

A 对 B／C 的作用是分离两件事：如果 B≈C，说明差异来自参数；如果 B≠C，说明构造分布
的形状（高斯 vs 真实右偏、截断）本身也有影响。

用法：
    python3 reanalysis_length_regime.py raw/qwen15b_gsm8k_n8.jsonl.gz --out reanalysis-qwen15b.json
"""

import argparse
import gzip
import json
import math
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
for p in (os.path.join(EXP, 'rollout-minibatch-baseline-bias'),
          os.path.join(EXP, 'rollout-minibatch-composition'),
          os.path.join(EXP, 'rollout-group-split-tradeoff')):
    if p not in sys.path:
        sys.path.insert(0, p)

import minibatch_baseline_bias as M  # noqa: E402
import train_side_composition as T  # noqa: E402

N_GROUPS, GROUP_SIZE, K = 32, 8, 4
SEEDS = list(range(30))


def load_groups(path):
    groups = {}
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            groups.setdefault(r['prompt_idx'], []).append(r['response_tokens'])
    full = [g for _, g in sorted(groups.items()) if len(g) == GROUP_SIZE]
    return full


def synthetic_lengths(seed, cv_group, cv_in, mean, n_groups=N_GROUPS, n_per=GROUP_SIZE):
    rng = random.Random(seed * 7919 + 13)
    lengths = []
    for _ in range(n_groups):
        gm = max(1.0, rng.gauss(mean, mean * cv_group))
        for _ in range(n_per):
            lengths.append(int(round(max(1.0, gm * (1.0 + rng.gauss(0.0, cv_in))))))
    return lengths


def one_seed(lengths, seed):
    uids, grads = T._grad_model(random.Random(seed), M.RHO_GROUP, None)
    order, _parts = M.ldm_order(lengths, K)
    ldm_grads = [grads[i] for i in order]
    shapes = M.contiguous_shapes(len(ldm_grads), K)
    ldm = M._noise_ratio(None, ldm_grads, K, shapes)
    grouped = M._noise_ratio(None, grads, K, M.contiguous_shapes(len(grads), K))
    inter = M._noise_ratio_interleaved(None, grads, K)
    purity = M.block_group_purity([uids[i] for i in order], shapes)
    return {'ldm': ldm, 'grouped': grouped, 'interleaved': inter,
            'ldm_over_grouped': ldm / grouped,
            'interleaved_over_ldm': inter / ldm,
            'purity': purity}


def summarize(rows, label):
    lor = [r['ldm_over_grouped'] for r in rows]
    iol = [r['interleaved_over_ldm'] for r in rows]
    pur = [r['purity'] for r in rows]

    def ms(xs):
        return {'mean': statistics.mean(xs),
                'stderr': statistics.pstdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0,
                'min': min(xs), 'max': max(xs)}

    return {
        'label': label,
        'n_seeds': len(rows),
        'ldm_over_grouped': ms(lor),
        'interleaved_over_ldm': ms(iol),
        'interleaved_over_ldm_ge1_fraction': sum(1 for x in iol if x >= 1.0) / len(iol),
        'ldm_groups_per_block': {k: statistics.mean(p[k] for p in pur)
                                 for k in ('mean_groups_per_block', 'intact_groups',
                                           'total_groups', 'mean_blocks_per_group')},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    real_groups = load_groups(args.input)
    all_lengths = [x for g in real_groups for x in g]
    gmeans = [statistics.mean(g) for g in real_groups]
    cv_group = statistics.stdev(gmeans) / statistics.mean(all_lengths)
    pooled = statistics.mean([statistics.variance(g) for g in real_groups])
    cv_in = math.sqrt(pooled) / statistics.mean(all_lengths)
    mean_len = statistics.mean(all_lengths)

    out = {
        'input': args.input,
        'measured_params': {'cv_group': cv_group, 'cv_in': cv_in, 'mean_length': mean_len,
                            'n_groups_pool': len(real_groups)},
        'arms': {},
    }

    # A. 构造档自检：与 §13.38 的 summary.json 逐项对照
    rows_a = []
    for s in SEEDS:
        rows_a.append(one_seed(M.build_lengths(s, 'realistic'), s))
    out['arms']['constructed_realistic'] = summarize(rows_a, 'constructed realistic (0.60, 0.08)')
    out['arms']['constructed_iid'] = summarize(
        [one_seed(M.build_lengths(s, 'iid'), s) for s in SEEDS], 'constructed iid (0.45, 0.45)')

    # B. 实测参数重新抽样（高斯形状不变）
    rows_b = []
    for s in SEEDS:
        rows_b.append(one_seed(synthetic_lengths(s, cv_group, cv_in, mean_len), s))
    out['arms']['synthetic_measured_params'] = summarize(rows_b, 'synthetic @ measured params')

    # C. 真实组重抽样（形状/截断原样）
    rows_c = []
    for s in SEEDS:
        rng = random.Random(s * 7919 + 13)
        picked = rng.sample(real_groups, N_GROUPS)
        lengths = [x for g in picked for x in g]
        rows_c.append(one_seed(lengths, s))
    out['arms']['real_resampled'] = summarize(rows_c, 'real groups resampled')

    # 与 §13.38 归档值并排
    try:
        with open(os.path.join(EXP, 'rollout-minibatch-baseline-bias', 'summary.json'),
                  encoding='utf-8') as f:
            ref = json.load(f)
        out['reference_13_38'] = {
            'realistic_ldm_over_grouped': ref.get('main_regime', {}).get('ldm_over_grouped_mean'),
            'realistic_interleaved_over_ldm': ref.get('main_regime', {}).get('interleaved_over_ldm_mean'),
        }
    except Exception as e:  # 归档缺失时不影响主结果
        out['reference_13_38'] = {'error': repr(e)}

    outp = args.out or (args.input.rsplit('/', 1)[-1] + '.reanalysis.json')
    with open(outp, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    for name, arm in out['arms'].items():
        print('%-28s ldm/grouped=%.4f±%.4f  inter/ldm=%.4f (>=1: %.0f%%)' % (
            name, arm['ldm_over_grouped']['mean'], arm['ldm_over_grouped']['stderr'],
            arm['interleaved_over_ldm']['mean'], 100 * arm['interleaved_over_ldm_ge1_fraction']))
    print('measured: cv_group=%.3f cv_in=%.3f mean=%.1f -> %s' % (cv_group, cv_in, mean_len, outp))


if __name__ == '__main__':
    main()
