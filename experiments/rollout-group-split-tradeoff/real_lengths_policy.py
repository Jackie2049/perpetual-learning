#!/usr/bin/env python3
"""把 §13.30 第三批的「`R` 随调度策略变化」搬到**真实 rollout 长度**上（第四批，2026-11-03 M1）。

支撑 notebook/rollout-performance-research.md §13.30 第四批、补 §11.4／§11.5 的语义面缺口。

**为什么需要这一步**：§13.30 前三批的负载都是 `sample_group_lens` 生成的——组内每条
长度**独立**抖动（组内相关 ≈ 0）。而 §13.50（M5 第五十四批）用真实模型量出来：同一 prompt 的
8 条响应长度**高度相关**（ICC 0.364–0.522，不截断档 cv_group 0.34–0.36、cv_in 0.32–0.38）。
也就是说前三批量的 `R` 是在"组内几乎无相关"的负载上得到的，而真实负载的组内相关相当高。

这不是无关紧要的负载细节：**组原子的成本是 `max(g)`，逐条的成本是全部 `g`**。组内相关越高，
`max(g)` 越接近组均值、组臂被最慢那条绑住的程度越低——分组税的**大小**必然对 ICC 敏感。
第三批没有回答"真实负载上 `R` 是多少"，更没回答"更强调度会不会把 `R` 抬高"在真实负载上是否仍成立。

**做法**：不改排程器（直接复用 `_makespan_by_policy`），只换负载来源——
  臂 A  构造负载（与前三批口径一致：cv=0.6、组内独立），用作用量级对照；
  臂 B  真实长度（`../rollout-length-structure-real/raw/*.jsonl.gz` 的 response_tokens），
        按 §13.39 的打乱方式切批。

**必须报的两个口径**（否则读数不可比）：
  ① 真实长度是**整数 token**，同组内会有完全相同的值（截断档尤其多，45% 样本卡在 256）。
     排程器会按长度排序组，大量并列会让 LOF 的"长度序"退化成"到达序"——这是真实的，
     但会让两档策略的差被稀释，须单独诊断并列率；
  ② 真实档的组数是 480（1.5B cap256），远多于构造档的 16；批内组数变化本身改变 `R`
     （§13.30 第一批已示组数是敏感参数），所以**同组数复核**要并列报。

用法：
    python3 real_lengths_policy.py            # 重算并写 real-lengths-summary.json
    python3 real_lengths_policy.py --verify   # 与归档逐字段比对
"""

import gzip
import json
import os
import random
import statistics
import sys

from group_split_tradeoff import (POLICIES, POLICY_FIFO, POLICY_LOF, POLICY_LOF_FLAT,
                                  _makespan_by_policy, sample_group_lens)

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_DIR = os.path.join(HERE, '..', 'rollout-length-structure-real', 'raw')
SUMMARY_PATH = os.path.join(HERE, 'real-lengths-summary.json')

SEEDS = list(range(30))
N_GROUPS_CONSTRUCT = 16
N_PER = 8
CV_CONSTRUCT = 0.6
BACKLOGS = (1, 4, 16)

# 真实长度档：主档是 RL 实际配置（max_response_length=256），另有不截断对照
REAL_ARMS = (
    ('qwen15b_cap256', 'qwen15b_gsm8k_n8.jsonl.gz'),
    ('qwen15b_cap1024', 'qwen15b_gsm8k_n8_cap1024.jsonl.gz'),
    ('qwen05b_cap256', 'qwen05b_gsm8k_n8.jsonl.gz'),
    ('qwen05b_cap1024', 'qwen05b_gsm8k_n8_cap1024.jsonl.gz'),
)


def arm_cap(label):
    """该档的 max_response：截断率必须按它判定，不能一律用 256。"""
    return 1024.0 if 'cap1024' in label else 256.0


def _sample_icc_group_lens(rng, icc, n_groups=16, n_per=8, cv=0.6, mean_len=50.0):
    """组内长度 = 组均值 + 组内抖动，组间/组内方差按目标 ICC 配比。

    ICC = var_between / (var_between + var_within)。总 cv 固定为 `cv`，
    于是组内 cv 随 ICC 上升而下降（组内更齐），这正是真实负载"高 ICC"的形态。
    """
    cv_between = cv * (icc ** 0.5)
    cv_within = cv * ((1.0 - icc) ** 0.5)
    out = []
    for _ in range(n_groups):
        mu = rng.gauss(mean_len, mean_len * cv_between)
        out.append([max(1.0, rng.gauss(mu, mean_len * cv_within)) for _ in range(n_per)])
    return out


def arm_gains(groups, n_groups, backlog, policy, seeds=SEEDS):
    """把策略的收益拆到两臂上：组臂快了百分之几、轨迹臂快了百分之几。

    `R = (T_g − T_t)/T_g`。若 LOF 让**组臂**降得多、轨迹臂降得少，比值上升；
    反过来就下降。这是第三批"更强调度抬税"与第四批"反号"的共同机制，
    所以必须把它作为读数报出来，而不是只报比值。
    """
    rng = random.Random(0)
    g_rel, t_rel = [], []
    for _ in range(len(seeds)):
        pick = rng.sample(groups, n_groups)
        base = _makespan_by_policy(pick, backlog, POLICY_FIFO)
        alt = _makespan_by_policy(pick, backlog, policy)
        g_rel.append((alt['group_makespan'] - base['group_makespan']) / base['group_makespan'])
        t_rel.append((alt['traj_makespan'] - base['traj_makespan']) / base['traj_makespan'])
    return {
        'group_arm_rel_mean': statistics.mean(g_rel),
        'group_arm_n_faster': sum(1 for x in g_rel if x < 0),
        'traj_arm_rel_mean': statistics.mean(t_rel),
        'traj_arm_n_faster': sum(1 for x in t_rel if x < 0),
        'n_total': len(g_rel),
    }


def load_real_groups(path):
    """把逐样本记录按 prompt 聚成组，返回 [[len, ...], ...]（按 prompt_idx 排序）。"""
    groups = {}
    with gzip.open(path, 'rt') as f:
        for line in f:
            d = json.loads(line)
            groups.setdefault(d['prompt_idx'], []).append(float(d['response_tokens']))
    return [groups[k] for k in sorted(groups)]


def describe(groups, cap=256.0):
    """负载结构描述：与 §13.50 同口径（组间/组内方差、ICC1），便于与那批读数对齐。

    cap 是该档的 max_response。截断按"撞到本档上限"判定——写死 256 会把 cap1024 档
    读成 60%／45% 截断，而它其实几乎不截断（0.65%／0.03%）。
    """
    flat_per_group = [statistics.mean(g) for g in groups]
    grand = statistics.mean(flat_per_group)
    var_between = statistics.pvariance(flat_per_group)
    within = [v for g in groups for v in g]
    # 组内池化方差
    var_within = statistics.mean([statistics.pvariance(g) for g in groups])
    icc = var_between / (var_between + var_within) if (var_between + var_within) else 0.0
    ties = sum(1 for g in groups if len(set(g)) < len(g)) / float(len(groups))
    return {
        'n_groups': len(groups),
        'n_per_group': len(groups[0]),
        'grand_mean': grand,
        'cv_between': statistics.pstdev(flat_per_group) / grand if grand else 0.0,
        'cv_within_pooled': (var_within ** 0.5) / grand if grand else 0.0,
        'icc1': icc,
        'max_response': cap,
        'truncated_fraction': sum(1 for g in groups for v in g if v >= cap) / float(
            sum(len(g) for g in groups)),
        'groups_with_ties_fraction': ties,
    }


def tax_table(groups, backlogs=BACKLOGS, policies=(POLICY_FIFO, POLICY_LOF, POLICY_LOF_FLAT)):
    """给定一批真实组，报各策略下的 R（分组税）。用该批的全部组当一批。"""
    out = {}
    for b in backlogs:
        out['backlog%d' % b] = {
            p: _makespan_by_policy(groups, b, p)['tax'] for p in policies
        }
    return out


def seed_resampled_tax(groups, n_groups, backlogs, policy, seeds=SEEDS, rng_seed=0):
    """按种子从真实组里无放回抽 n_groups 组，算该策略下的 R 分布。

    这里**不重抽组内长度**——真实组的长度结构原样保留，只换抽到哪些 prompt。
    这正是 §13.50 "真实组重抽" 的口径。
    """
    rng = random.Random(rng_seed)
    vals = []
    for _ in range(len(seeds)):
        pick = rng.sample(groups, n_groups)
        vals.append({b: _makespan_by_policy(pick, b, policy)['tax'] for b in backlogs})
    return vals


def paired_policy_delta(groups, n_groups, backlog, policy, seeds=SEEDS):
    """逐种子配对差：同一批抽到的组，只换排队顺序，看 R 怎么动。

    与 §13.30 第三批的 `policy_invariance` 同口径（那里是逐种子重生成负载），
    差别只在负载来源。
    """
    rng = random.Random(0)
    deltas = []
    for _ in range(len(seeds)):
        pick = rng.sample(groups, n_groups)
        base = _makespan_by_policy(pick, backlog, POLICY_FIFO)['tax']
        alt = _makespan_by_policy(pick, backlog, policy)['tax']
        deltas.append(alt - base)
    return {
        'mean': statistics.mean(deltas),
        'min': min(deltas),
        'max': max(deltas),
        'n_negative': sum(1 for d in deltas if d < 0),
        'n_positive': sum(1 for d in deltas if d > 0),
        'n_total': len(deltas),
    }


def build_real_arm_structure():
    """四个真实档的负载结构（只读 jsonl 并描述，不排程）。

    单独成函数是为了能被测试直接调用：截断率按本档上限判定的接线要能用
    便宜的方式核，而不是靠跑完整 build（约半小时）。
    """
    out = {}
    for label, fname in REAL_ARMS:
        groups = load_real_groups(os.path.join(REAL_DIR, fname))
        out[label] = describe(groups, cap=arm_cap(label))
    return out


def build_summary():
    out = {
        'experiment': '§13.30 第四批：R 在真实 rollout 长度上的读数',
        'doc': 'notebook/rollout-performance-research.md §13.30 第四批',
        'note': ('真实负载来自 ../rollout-length-structure-real/raw/（§13.50，M5 实测）。'
                 '本脚本只读那些 jsonl.gz 的 response_tokens 字段，不重新生成任何响应。'),
        'construct_baseline': {},
        'real_arms': {},
        'policy_shift_on_real': {},
    }

    # ---- 臂 A：构造负载（与前三批同口径），用来确认本脚本的排程器复现出 34.0% / 36.8% ----
    for b in BACKLOGS:
        vals = [
            _makespan_by_policy(
                sample_group_lens(random.Random(s), n_groups=N_GROUPS_CONSTRUCT,
                                  n_per=N_PER, cv=CV_CONSTRUCT), b, POLICY_FIFO)['tax']
            for s in SEEDS
        ]
        lof = [
            _makespan_by_policy(
                sample_group_lens(random.Random(s), n_groups=N_GROUPS_CONSTRUCT,
                                  n_per=N_PER, cv=CV_CONSTRUCT), b, POLICY_LOF)['tax']
            for s in SEEDS
        ]
        out['construct_baseline']['backlog%d' % b] = {
            'fifo_mean': statistics.mean(vals),
            'lof_mean': statistics.mean(lof),
            'delta_pp': 100.0 * (statistics.mean(lof) - statistics.mean(vals)),
        }

    # ---- 臂 A2：组内相关对照（把负载结构拆成两个独立参数） ----
    # 真实档唯一被改变的东西是"组内相关"，而它同时改变组间/组内方差的分配。
    # 这一组用构造负载单独扫 ICC：组内长度 = 组均值 + 独立抖动，组均值与抖动按
    # 目标 ICC 配比总方差，组内 cv 固定 0.6（与前三批同一离散度）。
    # 若"策略位移随 ICC 变号"在这里复现，则真实档的异常不是数据问题而是结构问题。
    out['icc_controlled'] = {}
    for icc in (0.0, 0.25, 0.5, 0.75):
        row = {}
        for b in (4, 16):
            fifo, lof = [], []
            for s in SEEDS:
                lens = _sample_icc_group_lens(random.Random(s), icc,
                                              n_groups=N_GROUPS_CONSTRUCT,
                                              n_per=N_PER, cv=CV_CONSTRUCT)
                fifo.append(_makespan_by_policy(lens, b, POLICY_FIFO)['tax'])
                lof.append(_makespan_by_policy(lens, b, POLICY_LOF)['tax'])
            row['backlog%d' % b] = {
                'fifo_mean': statistics.mean(fifo),
                'lof_mean': statistics.mean(lof),
                'delta_pp': 100.0 * (statistics.mean(lof) - statistics.mean(fifo)),
            }
        out['icc_controlled']['icc%.2f' % icc] = row

    # ---- 臂 A2b：组内离散度扫描（第三批结论的适用边界在这里） ----
    # 构造档（cv=0.6）给出"更强调度抬税"；真实档（组内 cv≈0.155）给出相反符号。
    # ICC 扫描证明"组内相关"不是控制项，并列诊断也排除截断并列。所以把组内**离散度**
    # 单独扫一遍：若符号只在 cv 的某一侧翻转，第三批的结论就要带上这个条件。
    out['cv_sweep'] = {}
    for cv in (0.10, 0.155, 0.25, 0.40, 0.60, 1.00):
        row = {}
        for b in (4, 16):
            fifo, lof = [], []
            for s in SEEDS:
                rng = random.Random(s)
                lens = [[max(1.0, rng.gauss(50.0, 50.0 * cv)) for _ in range(N_PER)]
                        for _ in range(N_GROUPS_CONSTRUCT)]
                fifo.append(_makespan_by_policy(lens, b, POLICY_FIFO)['tax'])
                lof.append(_makespan_by_policy(lens, b, POLICY_LOF)['tax'])
            row['backlog%d' % b] = {
                'fifo_mean': statistics.mean(fifo),
                'lof_mean': statistics.mean(lof),
                'delta_pp': 100.0 * (statistics.mean(lof) - statistics.mean(fifo)),
                'n_negative': sum(1 for x, y in zip(fifo, lof) if y < x),
            }
        out['cv_sweep']['cv%.3f' % cv] = row

    # ---- 臂 A3：并列率对照（截断是真实负载独有的结构，须与 ICC 分开） ----
    # 真实 cap256 档 78% 的组内存在长度完全相同的成员（45% 样本卡在 256 上限）。
    # 排程器按长度排序时，并列会让 LOF 的"长度序"退化成"到达序"——这一项若本身
    # 就能改变策略位移的符号，那它和 ICC 是两个独立的负载结构参数，不能混为一谈。
    # 做法：在 ICC=0（独立抖动）的构造负载上，把每条长度**向上取整到 R 的倍数**，
    # 人为制造并列；R 越小并列越多。
    out['tie_controlled'] = {}
    for res in (1.0, 5.0, 20.0):
        row = {}
        for b in (4, 16):
            fifo, lof = [], []
            for s in SEEDS:
                lens = _sample_icc_group_lens(random.Random(s), 0.0,
                                              n_groups=N_GROUPS_CONSTRUCT,
                                              n_per=N_PER, cv=CV_CONSTRUCT)
                if res > 1.0:
                    lens = [[float(max(1, int(round(v / res)) * int(res))) for v in g]
                            for g in lens]
                tie_frac = sum(1 for g in lens if len(set(g)) < len(g)) / float(len(lens))
                fifo.append(_makespan_by_policy(lens, b, POLICY_FIFO)['tax'])
                lof.append(_makespan_by_policy(lens, b, POLICY_LOF)['tax'])
            row['backlog%d' % b] = {
                'resolution': res,
                'fifo_mean': statistics.mean(fifo),
                'lof_mean': statistics.mean(lof),
                'delta_pp': 100.0 * (statistics.mean(lof) - statistics.mean(fifo)),
            }
        out['tie_controlled']['res%.0f' % res] = row

    # ---- 臂 B：真实长度 ----
    structure = build_real_arm_structure()
    for label, fname in REAL_ARMS:
        path = os.path.join(REAL_DIR, fname)
        groups = load_real_groups(path)
        desc = structure[label]
        # 全量：把 480 组当作一批，直接排程
        full = tax_table(groups)
        # 同组数（16 组）重抽，与构造档可比
        sub_seeds = {b: {p: seed_resampled_tax(groups, N_GROUPS_CONSTRUCT, [b], p)
                         for p in (POLICY_FIFO, POLICY_LOF, POLICY_LOF_FLAT)}
                     for b in BACKLOGS}
        sub = {
            'backlog%d' % b: {
                p: {
                    'mean': statistics.mean(v[b] for v in sub_seeds[b][p]),
                    'min': min(v[b] for v in sub_seeds[b][p]),
                    'max': max(v[b] for v in sub_seeds[b][p]),
                }
                for p in (POLICY_FIFO, POLICY_LOF, POLICY_LOF_FLAT)
            }
            for b in BACKLOGS
        }
        out['real_arms'][label] = {
            'source': os.path.relpath(path, HERE),
            'structure': desc,
            'full_batch': full,
            'n16_resampled': sub,
        }
        # 策略位移：取主档（cap256）做逐种子配对差
        if label.endswith('cap256'):
            shift = {}
            for b in (4, 16):
                shift['backlog%d' % b] = {
                    'lof_vs_fifo_n16': paired_policy_delta(groups, N_GROUPS_CONSTRUCT, b,
                                                           POLICY_LOF),
                    'lof_flat_vs_fifo_n16': paired_policy_delta(
                        groups, N_GROUPS_CONSTRUCT, b, POLICY_LOF_FLAT),
                    'lof_arm_gains_n16': arm_gains(groups, N_GROUPS_CONSTRUCT, b, POLICY_LOF),
                }
                # 构造档同口径的臂拆分，用于与真实档并排对照
                fifo_g, fifo_t, lof_g, lof_t = [], [], [], []
                for s in SEEDS:
                    lens = sample_group_lens(random.Random(s), n_groups=N_GROUPS_CONSTRUCT,
                                             n_per=N_PER, cv=CV_CONSTRUCT)
                    a = _makespan_by_policy(lens, b, POLICY_FIFO)
                    c = _makespan_by_policy(lens, b, POLICY_LOF)
                    fifo_g.append(a['group_makespan'])
                    fifo_t.append(a['traj_makespan'])
                    lof_g.append(c['group_makespan'])
                    lof_t.append(c['traj_makespan'])
                shift['backlog%d' % b]['construct_arm_gains'] = {
                    'group_arm_rel_mean': statistics.mean(
                        (y - x) / x for x, y in zip(fifo_g, lof_g)),
                    'traj_arm_rel_mean': statistics.mean(
                        (y - x) / x for x, y in zip(fifo_t, lof_t)),
                }
            out['policy_shift_on_real'][label] = shift

    out['reading'] = (
        '§13.30 第四批读数：① 真实档 R 小于构造档——1.5B cap256 4 批 2.7%／16 批 11.4%，'
        '构造档同口径 34.0%／42.4%；cap1024 两档 21.8–23.9%／29.5–33.4%（截断率 0.03%／0.65%，'
        'cap256 为 45.3%／60.8%）。② LOF−FIFO 在低 cv 深积压上反号：1.5B cap256 16 批 '
        '−0.95pp（27/30 为负）、0.5B −0.39pp（18/30），4 批两档仍为正（+1.99pp／+0.44pp）。'
        '③ 符号由组内离散度决定、不由 ICC 决定：cv 扫描 4 批 −2.77pp（cv0.10）→'
        '+6.07pp（cv1.00），翻转点在 cv 0.25–0.40；ICC 扫描（cv 固定 0.6）符号恒正。'
        '④ 两臂收益要逐档读：1.5B 4 批组臂 −0.45%／轨迹臂 −2.5%，16 批组臂 −1.60%／'
        '轨迹臂 −0.55%——积压加深后受益方换人。')
    return out


def main(argv):
    out = build_summary()
    text = json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True)
    if '--verify' in argv:
        with open(SUMMARY_PATH) as f:
            ref = json.load(f)
        ok = json.dumps(ref, ensure_ascii=False, indent=1, sort_keys=True) == text
        print('VERIFY OK: 逐字段相同' if ok else 'VERIFY FAIL: 与归档不一致')
        return 0 if ok else 1
    with open(SUMMARY_PATH, 'w') as f:
        f.write(text + '\n')
    print('== 构造负载（应复现 §13.30 第三批：4 批 FIFO 34.0% / LOF 36.8%）==')
    for k, v in out['construct_baseline'].items():
        print('  %-11s fifo %.4f  lof %.4f  delta %+.2fpp'
              % (k, v['fifo_mean'], v['lof_mean'], v['delta_pp']))
    print('== 组内离散度扫描（符号在哪里翻）==')
    for k, row in out['cv_sweep'].items():
        print('  %-9s  ' % k + '  '.join(
            '%s fifo %.4f lof %.4f (%+.2fpp, neg %d/30)'
            % (b, v['fifo_mean'], v['lof_mean'], v['delta_pp'], v['n_negative'])
            for b, v in row.items()))
    print('== ICC 可控对照（组内 cv 固定 0.6，只扫组内相关）==')
    for k, row in out['icc_controlled'].items():
        print('  %-8s  ' % k + '  '.join(
            '%s fifo %.4f lof %.4f (%+.2fpp)' % (b, v['fifo_mean'], v['lof_mean'], v['delta_pp'])
            for b, v in row.items()))
    for label, arm in out['real_arms'].items():
        s = arm['structure']
        print('== %s == n_groups=%d icc=%.3f cv_between=%.3f cv_within=%.3f ties=%.2f trunc=%.3f'
              % (label, s['n_groups'], s['icc1'], s['cv_between'], s['cv_within_pooled'],
                 s['groups_with_ties_fraction'], s['truncated_fraction']))
        for k, v in arm['full_batch'].items():
            print('   全量 %-11s fifo %.4f  lof %.4f  lof_flat %.4f'
                  % (k, v[POLICY_FIFO], v[POLICY_LOF], v[POLICY_LOF_FLAT]))
        for k, v in arm['n16_resampled'].items():
            print('   n16  %-11s fifo %.4f  lof %.4f (delta %+.2fpp)'
                  % (k, v[POLICY_FIFO]['mean'], v[POLICY_LOF]['mean'],
                     100 * (v[POLICY_LOF]['mean'] - v[POLICY_FIFO]['mean'])))
    print('wrote', SUMMARY_PATH)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
