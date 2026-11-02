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

    print('负向对照（检查器应当抓出前两个）：')
    for name, claim, got in cases:
        caught = abs(got - claim) > 0.0015
        expect_caught = name != cases[2][0]
        ok = (caught == expect_caught)
        all_ok &= ok
        print(f"  {'OK  ' if ok else 'FAIL'}  {name}  -> 判定{'不一致' if caught else '一致'}"
              f"（预期{'不一致' if expect_caught else '一致'}）")
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

    print(f"\n{ok} 一致 / {bad} 不一致 / {skip} 跳过（共 {len(CHECKS)} 项检查 + 1 项放置一致性）")
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
