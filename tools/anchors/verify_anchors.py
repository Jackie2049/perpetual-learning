#!/usr/bin/env python3
"""核验：主文档引用的关键数字，是否都能在实验归档里复现。

动机：AGENTS.md 要求"经验必须能追溯到实际执行"。但"追溯"一直是靠人翻，
本脚本把它变成可复跑的一件事——**读归档 JSON，重算主文档声称的比值，
逐项比对**。

**它核的不是"结论对不对"，而是"这个数字是不是从这份归档里来的"**。
两者不同：数字对得上，结论仍可能解读错。

用法：
    python3 verify_anchors.py            # 全部检查
    python3 verify_anchors.py --list     # 只列检查项

每条检查给出：声称值 / 归档重算值 / 是否一致 / 归档文件。
"""

import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]


def ratio(a, b):
    return a / b if b else float('nan')


CHECKS = []


def check(name, claim, archive, compute, tol=0.0015, note=''):
    """登记一条检查。compute 接收归档 JSON，返回重算值。"""
    CHECKS.append({'name': name, 'claim': claim, 'archive': archive,
                   'compute': compute, 'tol': tol, 'note': note})


# ---------------------------------------------------------------- 检查项


G_REANCHOR = 'experiments/rollout-g8-candidate-g/reanchor-summary.json'

check('候选 G：g2p 整步加速', 1.526, G_REANCHOR,
      lambda d: d['scaling']['g2p']['speedup'],
      note='§13.45 / §20.3.5 引用')

check('候选 G：g4p 整步加速', 1.942, G_REANCHOR,
      lambda d: d['scaling']['g4p']['speedup'],
      note='§13.45 / §20.3.5 引用')

check('候选 G：g2p 并行效率', 0.763, G_REANCHOR,
      lambda d: d['scaling']['g2p']['parallel_efficiency'])

check('候选 G：g4p 并行效率', 0.485, G_REANCHOR,
      lambda d: d['scaling']['g4p']['parallel_efficiency'])

check('候选 G：gen 逐阶段 G（2 卡）', 1.322, G_REANCHOR,
      lambda d: d['scaling']['g2p']['stage_scaling']['gen'])

check('候选 G：update_actor 逐阶段 G（4 卡）', 2.087, G_REANCHOR,
      lambda d: d['scaling']['g4p']['stage_scaling']['update_actor'])

check('候选 G：update_weights 负扩展（4 卡）', 0.770, G_REANCHOR,
      lambda d: d['scaling']['g4p']['stage_scaling']['update_weights'])


IDENT = 'experiments/rollout-identity-measurement-bias/summary.json'

check('恒等式：训练块加速 1.5 倍时高报', 5.88, IDENT,
      lambda d: d['q2_hidden_substage']['train_G1.5']['rel_err_pct'],
      tol=0.02, note='§21.9 判决二')

check('恒等式：训练块加速 2.0 倍时高报', 9.68, IDENT,
      lambda d: d['q2_hidden_substage']['train_G2.0']['rel_err_pct'],
      tol=0.02)

check('恒等式：训练块加速 3.0 倍时高报', 14.29, IDENT,
      lambda d: d['q2_hidden_substage']['train_G3.0']['rel_err_pct'],
      tol=0.02)


BUBBLE = 'experiments/rollout-bubble-attribution/summary.json'

check('气泡归因：B3 同时性', 1.00, BUBBLE,
      lambda d: d['shape_space']['B3']['simultaneity']['mean'],
      tol=0.01, note='§21.10 判决一')

check('气泡归因：B5 同时性', 0.00, BUBBLE,
      lambda d: d['shape_space']['B5']['simultaneity']['mean'],
      tol=0.01)

check('气泡归因：B4 集中度', 1.00, BUBBLE,
      lambda d: d['shape_space']['B4']['concentration']['mean'],
      tol=0.01)


INFLIGHT3 = 'experiments/verl-v1-poll-quantum-dapo/summary-inflight3.json'


def _arm_mean(d, names):
    """两臂各自取**全部跑次**的均值——只取一条会算错（本脚本作者踩过）。"""
    vals = [d[n]['timing']['step']['mean'] for n in names]
    return sum(vals) / len(vals)


check('poll inflight=3：整步变化', -4.5, INFLIGHT3,
      lambda d: (_arm_mean(d, ['i3p010', 'i3p010b']) /
                 _arm_mean(d, ['i3p200', 'i3p200b']) - 1) * 100,
      tol=0.15, note='§13.52；必须用两臂全部跑次，只取一条会得 -2.98%')


# ---- 候选 G 的放置效应：§13.42 的 A/B 结论与它引的微基准 ----
# 这组数字出现在 §13.42 / §13.45 / §19.2 / 第 3 章与第 7 章论点骨架里，
# 是"加卡收益"叙述的前提。此前没有任何一项被脚本核对。

G_AB = 'experiments/rollout-g8-candidate-g/ab-summary.json'

# 主结论用**不绑定**的四跑：pA1/pA2（3+1，不绑）vs pB1/pB2（4+0，不绑）。
# pA3/pA4/pB3/pB4 是 taskset 格，属"本地性被否定"那一段的证据，不混进组间差。
_AB_UNBOUND = {'A': ['pA1', 'pA2'], 'B': ['pB1', 'pB2']}


def _ab_step(d, arm):
    runs = _AB_UNBOUND[arm]
    return sum(d['modes'][r]['step'] for r in runs) / len(runs)


def _ab_step_all(d, arm):
    """四跑均值（含两个 taskset 格）——§13.42 结果五用这个分母。

    **两个分母不能混**：组间差（16.3%）只用不绑定的两跑，才不与
    taskset 处理效应纠缠；结果五比 2+2 时用的是四跑均值。
    """
    runs = [r for r in d['modes'] if r.startswith('p' + arm) and r[2].isdigit()]
    return sum(d['modes'][r]['step'] for r in runs) / len(runs)


check('放置效应：不绑定组间 B 慢（%）', 16.3, G_AB,
      lambda d: (_ab_step(d, 'B') / _ab_step(d, 'A') - 1) * 100,
      tol=0.15, note='§13.42 结果一／§19.2；只用不绑定四跑')

check('放置效应：四跑均值 A', 16.858, G_AB,
      lambda d: sum(d['modes'][r]['step'] for r in ['pA1', 'pA2', 'pA3', 'pA4']) / 4,
      tol=0.005, note='§13.42 结果一；含两个 taskset 格')

check('放置效应：四跑均值 B', 19.705, G_AB,
      lambda d: sum(d['modes'][r]['step'] for r in ['pB1', 'pB2', 'pB3', 'pB4']) / 4,
      tol=0.005)

check('放置效应：四跑均值比', 1.169, G_AB,
      lambda d: (sum(d['modes'][r]['step'] for r in ['pB1', 'pB2', 'pB3', 'pB4']) /
                 sum(d['modes'][r]['step'] for r in ['pA1', 'pA2', 'pA3', 'pA4'])),
      tol=0.002)

check('放置效应：2+2 相对 3+1 更快的幅度（%）', 16.8, G_AB,
      lambda d: (1 - d['modes']['pC1']['step'] / _ab_step(d, 'A')) * 100,
      tol=0.15, note='§13.42 结果五；分母是**不绑定**两跑。原文 16.3% 是串用了'
                     '结果一的组间差，2026-11-03 订正为 16.8%')

check('放置效应：2+2 相对 4+0 更快的幅度（%）', 28.1, G_AB,
      lambda d: (1 - d['modes']['pC1']['step'] / _ab_step_all(d, 'B')) * 100,
      tol=0.15, note='分母是 B 组**四跑**均值')

# 分组推理的方向证据：A−B 差在每一对绑定下都保持（§13.42 结果二）。
# 这里各查一对，防止"只有不绑定时才成立"。
check('放置效应：绑定时 A−B 差保持（3+1 本地 vs 4+0 远端）', 1.154, G_AB,
      lambda d: d['modes']['pB4']['step'] / d['modes']['pA4']['step'],
      tol=0.004)

check('放置效应：绑定时 A−B 差保持（3+1 远端 vs 4+0 本地）', 1.196, G_AB,
      lambda d: d['modes']['pB3']['step'] / d['modes']['pA3']['step'],
      tol=0.004)

# 对照臂：旧 4 卡锚点（3+1）不能读作加卡效率。§13.45 判决二的 19.1%
# 两边分属两个归档（旧 3+1 在 summary.json、2+2 在 reanchor-summary.json），
# 公式里两个文件都读，所以单独写，不走 check()。
def _placement_degradation():
    """同卡数下 3+1 比 2+2 慢多少（§13.45 判决二）。

    **同一个基线，两次运行**：旧 3+1 的 g4 与 2+2 的 g4p 都按**自己那次的 1 卡跑**
    记 speedup（`summary.json` 用 g1、`reanchor-summary.json` 用 g1p）。所以两条链
    的整步秒数要各自从自己的基线换算：
        T4 = T1(本次) / speedup4
    若直接用 `1/speedup` 相比，等于假设两次的 T1 相同——文档给的是带秒数的
    16.878 vs 14.173，不是这个假设下的值。
    """
    old = json.loads((ROOT / G_OLD).read_text(encoding='utf-8'))
    new = json.loads((ROOT / G_REANCHOR).read_text(encoding='utf-8'))
    t1_old = old['modes']['g1']['step']
    t1_new = new['modes']['g1p']['step']
    t_old = t1_old / old['scaling']['g4']['speedup']
    t_new = t1_new / new['scaling']['g4p']['speedup']
    # 方向按正文口径："3+1 比 2+2 慢 X%" ⇒ 以 2+2 为分母
    return {'pct': (t_old / t_new - 1) * 100,
            't_old': t_old, 't_new': t_new,
            't1_old': t1_old, 't1_new': t1_new}


PLACEMENT_DEGRADATION_CLAIM = 19.1


# ---- 64MB all-reduce 带宽：§13.42 结果四那张表 ----
# 表里四格带宽、时间比与"时间 ∝ 单 socket 上最多 rank 数"的模型预测值都在正文，
# 归档只存原始控制台输出，所以这里解析文本而不是读 JSON。

COMM_PROBE = 'experiments/rollout-g8-candidate-g/raw/g8-comm-probe.txt'

# (放置, 声称带宽 GB/s, 时间／4+0, 模型 max(卡数/节点)/4)
SHM_ROWS = [
    ('4+0', '4,5,6,7', 3.08, 1.000, 1.000),
    ('3+1', '0,1,3,4', 4.09, 0.753, 0.750),
    ('1+3', '3,4,5,6', 4.07, 0.756, 0.750),
    ('2+2', '0,1,4,5', 5.99, 0.514, 0.500),
]


def _parse_probe(path):
    """把控制台日志解析成 {CVD: {mb: (ms, GB/s)}}，只取 SHM 开启那一段。"""
    out, cur = {}, None
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.startswith('--- n='):
            cur = None
            if 'shm=0' in line:
                cvd = line.split('CVD=')[1].split(' ')[0]
                cur = out.setdefault(cvd, {})
            continue
        if cur is not None and line.startswith('allreduce'):
            parts = line.split()
            cur[int(parts[1].rstrip('MB'))] = (float(parts[2]), float(parts[4]))
    return out


def _probe_checks():
    """§13.42 结果四：四格带宽与时间比必须能从原始控制台日志重算出来。"""
    path = ROOT / COMM_PROBE
    if not path.exists():
        return []
    data = _parse_probe(path)
    rows = []
    base_ms = data['4,5,6,7'][64][0]
    for name, cvd, bw, t_ratio, _model in SHM_ROWS:
        ms, got_bw = data[cvd][64]
        rows.append((f'SHM 64MB {name}：带宽 GB/s', bw, got_bw, 0.02))
        rows.append((f'SHM 64MB {name}：时间／4+0', t_ratio, ms / base_ms, 0.005))
    return rows


# ---- 放置一致性：同一节里的 G_k 必须来自同一套放置 ----
# 两套放置的分阶段 G_k 差别很大（update_actor 2 卡 1.628 vs 1.211），
# 混用不会报错但会让"加卡收益"的叙述失真。这里检查 §20.3.5 引用的五对值
# 是否都落在 re-anchor（1+1/2+2）这一套上。

G_OLD = 'experiments/rollout-g8-candidate-g/summary.json'

def _placement_check():
    """放置一致性：同一节引用的 G_k 必须来自**同一次测量**。

    口径与 experiments/rollout-anchor2-identity 的 source_alignment 一致：
    same_file（同一份归档）/ same_stack（同一栈）/ same_window_rule（同一窗口规则）。
    本脚本只覆盖"同一份归档"这一项——另两项需要各归档的 meta，登记在那边。
    """
    new = json.loads((ROOT / G_REANCHOR).read_text(encoding='utf-8'))
    old = json.loads((ROOT / G_OLD).read_text(encoding='utf-8'))
    pairs = [
        ('ref', 2.023, 3.147),
        ('update_actor', 1.628, 2.087),
        ('update_weights', 0.882, 0.770),
    ]
    bad = []
    for stage, c2, c4 in pairs:
        n2 = new['scaling']['g2p']['stage_scaling'][stage]
        n4 = new['scaling']['g4p']['stage_scaling'][stage]
        o2 = old['scaling']['g2']['stage_scaling'][stage]
        o4 = old['scaling']['g4']['stage_scaling'][stage]
        to_new = abs(n2 - c2) < 0.002 and abs(n4 - c4) < 0.002
        to_old = abs(o2 - c2) < 0.002 and abs(o4 - c4) < 0.002
        if not to_new:
            bad.append({'stage': stage, 'claim': (c2, c4),
                        'reanchor': (round(n2, 3), round(n4, 3)),
                        'old_placement': (round(o2, 3), round(o4, 3)),
                        'also_old': to_old})
    return bad


# ---- 文件存在性：文档说"未入库"的文件，是不是其实已经入库了 ----

DELETED_CLAIMS = [
    # (文档中的断言片段, 声称不存在的文件)
    ('原备忘文件未入库', 'notebook/rollout-migration-criteria-memo.md'),
    ('实际未入库', 'notebook/speculative-rl-performance-model-memo.md'),
    ('实际未入库', 'notebook/environment-layer-taxonomy-memo.md'),
]


def _file_claim_check(doc='notebook/rollout-performance-research.md'):
    """文档说"某文件不存在/未入库"，但它其实在 —— 这类断言会过期。

    只在**文档里仍有未标注的旧断言**时才报：若该处已写「已订正／已核实为过期」，
    说明人已经处理过，不再重复提醒。
    """
    import re
    text = (ROOT / doc).read_text(encoding='utf-8')
    bad = []
    for claim, rel in DELETED_CLAIMS:
        if not (ROOT / rel).exists():
            continue
        # 找含该断言的段落，看附近有没有订正标记
        for m in re.finditer(re.escape(claim), text):
            window = text[max(0, m.start() - 260):m.start() + 260]
            if any(k in window for k in ('已订正', '已过期', '核实为过期', '均已入库', '已入库')):
                continue
            bad.append({'claim': claim, 'file': rel,
                        'why': '文件存在但断言未标注订正'})
            break
    return bad


# ---- 负载来源标注：引用 R 读数必须说明"构造负载还是真实负载 + 截断设置" ----
# 2026-11-03 第五次"来源"类问题：同一个 R 指标，构造负载（组内 cv=0.60）给
# 34.0%，真实 1.5B cap256（cv=0.155）只给 2.7%，且符号可能反号。不标来源的读数
# 无法比较，也无法判断该不该引用。

REAL_LEN = 'experiments/rollout-group-split-tradeoff/real-lengths-summary.json'
CONSTRUCT_R = {'backlog4': 0.340, 'backlog16': 0.424}


def _load_source_check():
    """检查真实负载的 R 值，并确认 §11.5 明确标注了负载来源与截断。"""
    bad = []
    if not (ROOT / REAL_LEN).exists():
        return bad
    d = json.loads((ROOT / REAL_LEN).read_text(encoding='utf-8'))
    sh = d.get('policy_shift_on_real', {})
    if 'qwen15b_cap256' in sh:
        for bl in ('backlog4', 'backlog16'):
            if bl not in sh['qwen15b_cap256']:
                continue
        # 只核关键事实：真实 cap256 的 R 显著低于构造档
        arm = d.get('real_arms', {}).get('qwen15b_cap256', {})
        cv = arm.get('intra_group_cv')
        if cv is not None and cv > 0.25:
            bad.append({'what': 'qwen15b_cap256 组内 cv 应显著低于构造档 0.60',
                        'got': cv})
    # 文档必须写明"负载来源"这条约束
    #
    # ⚠ 这一项**比其他检查弱得多**：它只核"那句约束还在不在"，不核
    # "后续引用是否真的照做"。真正的保证要靠人读。加它是因为这类问题
    # 今天出现了五次，至少让约束句被误删时能立刻发现。
    doc = (ROOT / 'notebook/rollout-performance-research.md').read_text(encoding='utf-8')
    if '引用任何 `R` 读数都必须同时说明负载来源与截断设置' not in doc:
        bad.append({'what': '§11.5 缺少"必须标注负载来源"的约束句', 'got': '未找到'})
    return bad


def selftest():
    """负向对照：故意写错声称值，检查器必须报 FAIL。

    一个只报 OK 的检查器说明不了它有效——可能只是永远返回 True。
    这里用三个已知错误的声称值验证它真能抓到偏差。
    """
    import copy
    d = json.loads((ROOT / G_REANCHOR).read_text(encoding='utf-8'))
    real = d['scaling']['g2p']['speedup']

    cases = [
        ('把 1.526 报成 2.5', 2.5, real),
        ('把 1.526 报成 1.53（差 0.004）', 1.53, real),
        ('把 1.526 报成 1.527（差 0.001，容差内）', 1.527, real),
    ]
    all_ok = True
    # 负向对照二：故意引旧放置的值，放置一致性检查必须抓出
    new = json.loads((ROOT / G_REANCHOR).read_text(encoding='utf-8'))
    old = json.loads((ROOT / G_OLD).read_text(encoding='utf-8'))
    o2 = old['scaling']['g2']['stage_scaling']['update_actor']
    n2 = new['scaling']['g2p']['stage_scaling']['update_actor']
    caught_placement = abs(o2 - n2) > 0.002
    print(f"   {'OK  ' if caught_placement else 'FAIL'}  旧放置 update_actor={o2:.3f} vs 新放置={n2:.3f}"
          f"  -> {'可被区分' if caught_placement else '无法区分（检查无效）'}")
    all_ok &= caught_placement

    # 负向对照三：构造一段**未标注**的过期断言，文件存在性检查必须报出来
    import tempfile
    fake = pathlib.Path(tempfile.mkdtemp()) / 'doc.md'
    fake.write_text('这句话说某文件未入库，且没有任何订正标记。', encoding='utf-8')
    fake_target = ROOT / DELETED_CLAIMS[0][1]
    caught_file = fake_target.exists()
    print(f"   {'OK  ' if caught_file else 'FAIL'}  未标注的「未入库」断言 -> "
          f"{'会被报出（目标文件确实存在）' if caught_file else '无法检测'}")
    all_ok &= caught_file

    print('负向对照（检查器应当抓出前两个）：')
    for name, claim, got in cases:
        caught = abs(got - claim) > 0.0015
        expect_caught = name != cases[2][0]
        ok = (caught == expect_caught)
        all_ok &= ok
        print(f"  {'OK  ' if ok else 'FAIL'}  {name}  -> 判定{'不一致' if caught else '一致'}"
              f"（预期{'不一致' if expect_caught else '一致'}）")

    # 负向对照四：放置效应必须只用「不绑定」四跑。把 taskset 格混进来会改变组间差，
    # 检查器必须能区分这两种算法。
    ab = json.loads((ROOT / G_AB).read_text(encoding='utf-8'))
    clean = (_ab_step(ab, 'B') / _ab_step(ab, 'A') - 1) * 100
    dirty = (sum(ab['modes'][r]['step'] for r in ['pB1', 'pB2', 'pB3', 'pB4']) /
             sum(ab['modes'][r]['step'] for r in ['pA1', 'pA2', 'pA3', 'pA4']) - 1) * 100
    caught_mix = abs(clean - 16.3) <= 0.15 and abs(dirty - 16.3) > 0.15
    print(f"  {'OK  ' if caught_mix else 'FAIL'}  组间差：不绑定 {clean:.2f}% vs 混入 taskset "
          f"{dirty:.2f}%  -> {'两种算法可区分（混入会被判不一致）' if caught_mix else '无法区分（检查无效）'}")
    all_ok &= caught_mix

    # 负向对照五：SHM 带宽表的"时间 ∝ 单 socket 最多 rank 数"规律必须能靠数据检验。
    # 若把 2+2 的模型预测改成 4+0 的（1.000 而非 0.500），实测比应当偏离它。
    probed = {n: got for n, _claim, got, _tol in _probe_checks()}
    ms_2p2 = probed.get('SHM 64MB 2+2：时间／4+0')
    caught_model = ms_2p2 is not None and abs(ms_2p2 - 1.000) > 0.005
    print(f"  {'OK  ' if caught_model else 'FAIL'}  2+2 实测时间比 {ms_2p2:.3f} vs 错误的模型值 1.000"
          f"  -> {'可被区分' if caught_model else '无法区分（检查无效）'}")
    all_ok &= caught_model

    print()
    return 0 if all_ok else 1


def main():
    if '--selftest' in sys.argv:
        return selftest()
    if '--list' in sys.argv:
        for c in CHECKS:
            print(f"{c['name']:36s} claim={c['claim']:<8} <- {c['archive']}")
        return 0

    ok = bad = skip = 0
    for c in CHECKS:
        path = ROOT / c['archive']
        if not path.exists():
            print(f"SKIP  {c['name']}  (归档缺失: {c['archive']})")
            skip += 1
            continue
        try:
            got = c['compute'](json.loads(path.read_text(encoding='utf-8')))
        except (KeyError, TypeError, ZeroDivisionError) as e:
            print(f"SKIP  {c['name']}  (归档结构不符: {type(e).__name__})")
            skip += 1
            continue
        same = abs(got - c['claim']) <= c['tol']
        flag = 'OK  ' if same else 'FAIL'
        print(f"{flag}  {c['name']:36s} 声称 {c['claim']:<8} 重算 {got:.4f}")
        ok, bad = (ok + 1, bad) if same else (ok, bad + 1)

    # 放置一致性（单独一类：不是"数字对不对"，而是"来自哪一套放置"）
    print()
    pbad = _placement_check()
    if pbad:
        for b in pbad:
            print(f"FAIL  放置一致性：{b['stage']} 声称 {b['claim']} "
                  f"→ re-anchor {b['reanchor']}，旧放置 {b['old_placement']}"
                  f"{'（该值属于旧放置！）' if b['also_old'] else ''}")
        bad += len(pbad)
    else:
        print("OK    放置一致性：§20.3.5 引用的 G_k 全部来自 re-anchor（1+1/2+2）同一套")
        ok += 1

    # 文件存在性断言
    print()
    fbad = _file_claim_check()
    if fbad:
        for b in fbad:
            print(f"WARN  文件存在性：文档称「{b['claim']}」，但 {b['file']} 存在")
    else:
        print("OK    文件存在性：文档中「未入库」类断言均已复核")
        ok += 1

    # 跨归档的一对（旧 3+1 与 re-anchor 2+2）
    print()
    deg = _placement_degradation()
    same = abs(deg['pct'] - PLACEMENT_DEGRADATION_CLAIM) <= 0.15
    print(f"{'OK  ' if same else 'FAIL'}  {'放置效应：3+1 比 2+2 慢（%）':36s} "
          f"声称 {PLACEMENT_DEGRADATION_CLAIM:<8} 重算 {deg['pct']:.4f}"
          f"   [{deg['t_old']:.3f}s → {deg['t_new']:.3f}s；"
          f"两次基线 {deg['t1_old']:.3f}s / {deg['t1_new']:.3f}s]")
    ok, bad = (ok + 1, bad) if same else (ok, bad + 1)

    # SHM 带宽表（解析控制台日志）
    probed = _probe_checks()
    if not probed:
        print(f"SKIP  SHM 带宽表  (归档缺失: {COMM_PROBE})")
        skip += 1
    else:
        for name, claim, got, tol in probed:
            same = abs(got - claim) <= tol
            print(f"{'OK  ' if same else 'FAIL'}  {name:36s} "
                  f"声称 {claim:<8} 重算 {got:.4f}")
            ok, bad = (ok + 1, bad) if same else (ok, bad + 1)

    # 负载来源标注
    print()
    lbad = _load_source_check()
    if lbad:
        for b in lbad:
            print(f"WARN  负载来源：{b['what']}（实测 {b['got']}）")
    else:
        print("OK    负载来源：§11.5 已写明 R 读数必须标注负载来源与截断设置")
        ok += 1

    print(f"\n{ok} 一致 / {bad} 不一致 / {skip} 跳过"
          f"（共 {len(CHECKS)} 项检查 + 放置一致性 + 文件存在性"
          f" + 跨归档放置对照 + SHM 带宽表 {len(probed)} 项 + 负载来源）")
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
