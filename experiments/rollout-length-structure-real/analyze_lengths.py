#!/usr/bin/env python3
"""把真实 rollout 的响应长度结构折算成 §13.38 的参数并报统计量（CPU，M5）。

输入是 sample_lengths.py 产出的 JSONL（每条 = 一个 prompt 的一次采样）。
输出 summary.json，核心三组读数：

  1. 长度结构：`cv_group`（组均值长度离散）、`cv_in`（组内长度离散）、组内相关系数 ICC。
     §13.38 的构造档是 `realistic=(0.60, 0.08)`；本脚本给出真实模型上的对应值。
  2. 截断与分布：max_tokens=256 下的截断率、长度分位数（截断会压掉分布右尾）。
  3. prompt 长度与响应长度的相关（真实模型上的直接读数）。

用法：
    python3 analyze_lengths.py raw/qwen15b_gsm8k_n8.jsonl.gz --out summary-qwen15b.json
"""

import argparse
import gzip
import json
import math
import random
import statistics


def load(path):
    groups = {}
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            g = groups.setdefault(r['prompt_idx'], {'prompt_tokens': r['prompt_tokens'],
                                                    'lengths': [], 'trunc': 0, 'finish': {}})
            g['lengths'].append(r['response_tokens'])
            g['trunc'] += int(bool(r.get('truncated')))
            fr = r.get('finish_reason')
            g['finish'][fr] = g['finish'].get(fr, 0) + 1
    # 保持样本顺序：sample_idx 升序（文件顺序即 generate 顺序）
    return groups


def pearson(xs, ys):
    n = len(xs)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else None


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(ranks(xs), ranks(ys))


def structure(lengths_per_group, k):
    """k = 每组取前 k 条（用于 n=4 与 n=8 两档对照）。"""
    groups = [v[:k] for v in lengths_per_group if len(v) >= k]
    allv = [x for g in groups for x in g]
    grand = statistics.mean(allv)
    gmeans = [statistics.mean(g) for g in groups]
    n_g, k_g = len(groups), k
    # 组间：组均值的样本方差
    var_between = statistics.variance(gmeans)
    # 组内：合并方差（等组容量）
    within_vars = [statistics.variance(g) for g in groups]
    var_within = statistics.mean(within_vars)
    # 单因素随机效应 ICC(1)
    msb = k_g * var_between
    msw = var_within
    icc = (msb - msw) / (msb + (k_g - 1) * msw) if (msb + (k_g - 1) * msw) > 0 else None
    # §13.38 的生成参数化：group_mean ~ N(L, L*cv_group)，样本 = group_mean*(1+N(0,cv_in))
    cv_group = statistics.stdev(gmeans) / grand
    cv_in = math.sqrt(var_within) / grand
    rel_dev = [(x - statistics.mean(g)) / statistics.mean(g) for g in groups for x in g]
    cv_in_rel = math.sqrt(statistics.mean([d * d for d in rel_dev]))
    ranges = [max(g) - min(g) for g in groups]
    # 对数尺度 ICC：长度右偏时原始尺度 ANOVA 可能被长尾组主导，作为稳健性对照
    log_groups = [[math.log(x) for x in g] for g in groups]
    log_means = [statistics.mean(g) for g in log_groups]
    log_within = [statistics.variance(g) for g in log_groups]
    log_msb = k_g * statistics.variance(log_means)
    log_msw = statistics.mean(log_within)
    icc_log = ((log_msb - log_msw) / (log_msb + (k_g - 1) * log_msw)
               if (log_msb + (k_g - 1) * log_msw) > 0 else None)
    return {
        'n_groups': n_g,
        'n_per_group': k_g,
        'n_samples': len(allv),
        'grand_mean': grand,
        'sd_total': statistics.stdev(allv),
        'cv_total': statistics.stdev(allv) / grand,
        'median': statistics.median(allv),
        'p10': sorted(allv)[max(0, int(0.10 * len(allv)) - 1)],
        'p90': sorted(allv)[min(len(allv) - 1, int(0.90 * len(allv)))],
        'min': min(allv),
        'max': max(allv),
        'cv_group': cv_group,
        'cv_in_pooled': cv_in,
        'cv_in_rms_rel': cv_in_rel,
        'var_between': var_between,
        'var_within_pooled': var_within,
        'icc1': icc,
        'icc1_log': icc_log,
        'between_var_share': var_between / (var_between + var_within) if (var_between + var_within) else None,
        'within_range_mean': statistics.mean(ranges),
        'within_range_median': statistics.median(ranges),
        'group_mean_min': min(gmeans),
        'group_mean_max': max(gmeans),
    }


def bootstrap_ci(groups, k, field_fn, n_boot=400, seed=0):
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample = [groups[rng.randrange(len(groups))] for _ in range(len(groups))]
        st = structure(sample, k)
        vals.append(field_fn(st))
    vals.sort()
    return {'lo': vals[int(0.025 * len(vals))], 'hi': vals[int(0.975 * len(vals)) - 1],
            'sd': statistics.stdev(vals), 'n_boot': n_boot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input')
    ap.add_argument('--out', default=None)
    ap.add_argument('--label', default=None)
    ap.add_argument('--cap', type=int, default=256, help='max_tokens used in sampling')
    args = ap.parse_args()

    groups = load(args.input)
    lengths_per_group = [g['lengths'] for g in groups.values()]
    trunc = sum(g['trunc'] for g in groups.values())
    total = sum(len(g['lengths']) for g in groups.values())
    # 截断样本单独看：若把撞上限的样本剔除，长度结构会怎么变
    no_trunc = [[x for x in g['lengths'] if x < args.cap] for g in groups.values()]
    payload = {
        'input': args.input,
        'label': args.label,
        'n_prompts': len(groups),
        'n_samples': total,
        'truncated_samples': trunc,
        'truncated_fraction': trunc / total if total else None,
        'finish_reasons': {},
        'structure_n8': structure(lengths_per_group, 8),
        'structure_n4': structure(lengths_per_group, 4),
        'structure_n8_clean_groups': None,
        'prompt_len_vs_response': {},
    }
    for g in groups.values():
        for k, v in g['finish'].items():
            payload['finish_reasons'][k] = payload['finish_reasons'].get(k, 0) + v

    # 只保留"整组 8 条都没撞上限"的组：看截断本身把长度结构改变了多少
    clean = [g for g in no_trunc if len(g) == 8]
    if len(clean) >= 20:
        payload['structure_n8_clean_groups'] = structure(clean, 8)
        payload['structure_n8_clean_groups']['n_groups_total'] = len(lengths_per_group)

    # prompt 长度 ↔ 响应长度
    pxs, rxs = [], []
    for g in groups.values():
        pxs.append(g['prompt_tokens'])
        rxs.append(statistics.mean(g['lengths']))
    payload['prompt_len_vs_response'] = {
        'pearson_prompt_vs_group_mean': pearson(pxs, rxs),
        'spearman_prompt_vs_group_mean': spearman(pxs, rxs),
        'n_groups': len(pxs),
        'prompt_tokens_min': min(pxs),
        'prompt_tokens_max': max(pxs),
        'prompt_tokens_mean': statistics.mean(pxs),
    }

    # bootstrap：主要参数的抽样误差（重抽样 prompt 组）
    gl = [v for v in lengths_per_group if len(v) >= 8]
    payload['bootstrap_n8'] = {
        'cv_group': bootstrap_ci(gl, 8, lambda s: s['cv_group']),
        'cv_in_pooled': bootstrap_ci(gl, 8, lambda s: s['cv_in_pooled']),
        'icc1': bootstrap_ci(gl, 8, lambda s: s['icc1']),
    }

    # 与 §13.38 构造档并排
    payload['reference_regimes'] = {
        'iid': (0.45, 0.45),
        'realistic': (0.60, 0.08),
        'strong_corr': (1.00, 0.05),
    }
    out = args.out or (args.input.rsplit('/', 1)[-1] + '.summary.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: payload['structure_n8'][k] for k in
                      ('n_groups', 'grand_mean', 'cv_group', 'cv_in_pooled', 'icc1')},
                     ensure_ascii=False))
    print('truncated_fraction=%.4f -> %s' % (payload['truncated_fraction'], out))


if __name__ == '__main__':
    main()
