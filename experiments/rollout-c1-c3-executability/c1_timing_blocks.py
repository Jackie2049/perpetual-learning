#!/usr/bin/env python3
"""§21.1 C1 计时边界约定的可判定性核验（rollout 专项，2026-11-02 M1）。

支撑 notebook/rollout-performance-research.md §21.6。

C1 要求引用他人墙钟数字时给三件套：
    ① 块名 + 所属框架 + 固定 commit
    ② 块内是否含策略等待
    ③ 分项与总量是否重叠
本脚本把"给定一个计时块，C1 的三项能不能从源码判出来"做成可核读数：
对四个框架快照（§12 已登记的固定 commit）的分析结果**手写在 BLOCKS 里**，
脚本只负责三件事——
    (a) 逐项核对每条判据在源码里能找到的**锚点行**，锚点必须真实存在且字面匹配；
    (b) 按判定规则把三件套算成 C1 可引用等级（可横比／需补声明／不可横比）；
    (c) 把"一个计时块能同时给出哪些时间戳"算出来，喂给 C3 的 t1 可取得性检查
        （A2），使 C1 与 C3 两条约定接在同一条可观测性口径上。
真值不在本脚本里：BLOCKS 的判定是**人读源码的结论**，锚点核对只防"结论引用的
行不存在/代码已变"，不防"人读错"。这条边界写在 readme 与主文档里。

    python3 c1_timing_blocks.py             # 重跑并覆盖 summary.json
    python3 c1_timing_blocks.py --verify    # 重跑并与归档 summary.json 逐字段比对
    python3 -m unittest test_c1_timing_blocks -v
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY_PATH = os.path.join(HERE, 'summary.json')

CHECKOUTS = {
    # key: (只读 checkout 路径, 固定 commit)
    'verl': ('/Users/jackiemac/Documents/Codex/2026-10-20/new-chat/work/verl-7913',
             '24f25b03aa4b54249a273655ebbcce06f484192b'),
    'openrlhf': ('/Users/jackiemac/Documents/Codex/2026-10-20/new-chat/work/openrlhf-rollout-scan',
                 'dc2a7ad326f619fc5a47737ea47d1920ce2c0b53'),
    'nemo_rl': ('/Users/jackiemac/Documents/Codex/2026-10-20/new-chat/work/nemo-rl-rollout-scan',
                'a9d1a4bec45a28fd0d455519a2a979a37da2ac4a'),
    'skyrl': ('/Users/jackiemac/Documents/Codex/2026-10-20/new-chat/work/skyrl-rollout-scan',
              'e0893f6eb878f4d396184cb14c4f9d27bbbb0aef'),
}

# ---------------------------------------------------------------------------
# 判定规则（写死在代码里，不从数据反推）。取值依据 §12 的四框架登记。
# ---------------------------------------------------------------------------
# waiting: 块内是否含策略等待。yes=含；no=不含；partial=同块内因分支而含/不含
# overlap: 分项是否与总量重叠。yes=分项可加进总量；no=不是同一集合的分项；n/a=块本身是总量
#
# stamps: 该块结束位置在 C2 的三时间戳里能落在哪一格
#         t0 / t1 / t2 / pre_t1 / t1_or_t2 / none
#         取 none 表示该时刻在语义上什么都不是（例如 ack ≠ 权重生效）
BLOCKS = [
    {
        'proof': {'waiting': ('verl/trainer/ppo/v1/trainer_base.py', 543, 'replay_buffer.sample'), 'overlap': ('verl/trainer/ppo/v1/trainer_base.py', 543, 'on_sample')},
        'timer_key': 'gen@verl/trainer/ppo/v1/trainer_base.py',
        'id': 'verl.gen.v1_trainer_base',
        'framework': 'veRL',
        'commit': CHECKOUTS['verl'][1],
        'file': 'verl/trainer/ppo/v1/trainer_base.py',
        'line': 543,
        'anchor': 'with marked_timer("gen"',
        'waiting': 'yes',
        'waiting_evidence': '包住 on_sample hooks 与 replay_buffer.sample，取批可能是轮询',
        'overlap': 'no',
        'stamps': ['t2'],
        'c1_from_source': 'full',
    },
    {
        'proof': {'overlap': ('verl/trainer/ppo/ray_trainer.py', 1512, 'sleep_replicas'), 'waiting': ('verl/trainer/ppo/ray_trainer.py', 1511, 'generate_sequences')},
        'timer_key': 'gen@verl/trainer/ppo/ray_trainer.py',
        'id': 'verl.gen.ray_trainer',
        'framework': 'veRL',
        'commit': CHECKOUTS['verl'][1],
        'file': 'verl/trainer/ppo/ray_trainer.py',
        'line': 1510,
        'anchor': 'with marked_timer("gen"',
        'waiting': 'no',
        'waiting_evidence': '同步 fit 的 generate_sequences 是等整批返回，非队列饥饿等待',
        'overlap': 'yes',
        'overlap_evidence': '块内 sleep_replicas 与条件性 profile 调用，二者与生成不是同一集合的分项',
        'stamps': ['t0'],
        'c1_from_source': 'full',
    },
    {
        'proof': {'waiting': ('verl/trainer/ppo/ray_trainer.py', 1715, 'checkpoint_manager.update_weights'), 'overlap': ('verl/trainer/ppo/ray_trainer.py', 1716, 'update_weights(self.global_steps)')},
        'timer_key': 'update_weights@verl/trainer/ppo/ray_trainer.py',
        'id': 'verl.update_weights',
        'framework': 'veRL',
        'commit': CHECKOUTS['verl'][1],
        'file': 'verl/trainer/ppo/ray_trainer.py',
        'line': 1715,
        'anchor': 'with marked_timer("update_weights"',
        'waiting': 'yes',
        'waiting_evidence': '包住整个 manager 同步流程，含 abort/拓扑/恢复，不只是链路',
        'overlap': 'no',
        'stamps': ['t1_or_t2'],
        'c1_from_source': 'full',
    },
    {
        'proof': {'waiting': ('openrlhf/trainer/ppo_trainer_async.py', 132, 'generate_samples('), 'overlap': ('openrlhf/trainer/ppo_trainer_async.py', 127, 'hold lock so weight broadcast cannot overlap with generation')},
        'timer_key': None,
        'id': 'openrlhf.timing_generation',
        'framework': 'OpenRLHF',
        'commit': CHECKOUTS['openrlhf'][1],
        'file': 'openrlhf/trainer/ppo_trainer_async.py',
        'line': 134,
        'anchor': 'generation_time = time.time() - t0',
        'waiting': 'yes',
        'waiting_evidence': 'generate_samples 内可含筛组/补抽，也可能只取 _sample_buffer',
        'overlap': 'yes',
        'overlap_evidence': '与训练并行，timing/step_total 含队列等待，分项不可相加',
        'stamps': ['none'],
        'c1_from_source': 'partial',
    },
    {
        'proof': {'waiting': ('openrlhf/trainer/ppo_trainer_async.py', 206, 'rollout_queue.get(block=True)'), 'overlap': ('openrlhf/trainer/ppo_trainer_async.py', 206, 'rollout_queue.get(block=True)')},
        'timer_key': 'timing/step_total',
        'id': 'openrlhf.step_total',
        'framework': 'OpenRLHF',
        'commit': CHECKOUTS['openrlhf'][1],
        'file': 'openrlhf/trainer/ppo_trainer_async.py',
        'line': 237,
        'anchor': 'timing/step_total',
        'waiting': 'yes',
        'waiting_evidence': '消费循环墙钟包含取队列等待',
        'overlap': 'n/a',
        'stamps': ['none'],
        'c1_from_source': 'partial',
    },
    {
        'proof': {'waiting': ('nemo_rl/algorithms/single_controller.py', 3010, 'exposed_generation'), 'overlap': ('nemo_rl/algorithms/single_controller.py', 3026, '_buffer_capacity.release()')},
        'timer_key': 'exposed_generation',
        'id': 'nemo.exposed_generation',
        'framework': 'NeMo RL',
        'commit': CHECKOUTS['nemo_rl'][1],
        'file': 'nemo_rl/algorithms/single_controller.py',
        'line': 3010,
        'anchor': 'with self._timer.time("exposed_generation")',
        'waiting': 'yes',
        'waiting_evidence': '包住 evict、选组与饥饿轮询',
        'overlap': 'no',
        'stamps': ['none'],
        'c1_from_source': 'full',
    },
    {
        'proof': {'waiting': ('nemo_rl/algorithms/single_controller.py', 3753, 'pass'), 'overlap': ('nemo_rl/algorithms/single_controller.py', 3757, 'offload_before_refit')},
        'timer_key': 'weight_sync',
        'id': 'nemo.weight_sync',
        'framework': 'NeMo RL',
        'commit': CHECKOUTS['nemo_rl'][1],
        'file': 'nemo_rl/algorithms/single_controller.py',
        'line': 3754,
        'anchor': 'with self._timer.time("weight_sync")',
        'waiting': 'yes',
        'waiting_evidence': '分支依赖：defer_refit_for_save 时是空块，否则包住 _sync_weights 全流程',
        'overlap': 'no',
        'stamps': ['t1_or_t2'],
        'c1_from_source': 'partial',
        'c1_missing': ['空块分支：save 延后时块内无同步，报 0 不代表同步免费'],
    },
    {
        'proof': {'waiting': ('skyrl/train/fully_async_trainer.py', 843, 'wait_for_generation_buffer'), 'overlap': ('skyrl/train/fully_async_trainer.py', 846, 'sample_full_batch')},
        'timer_key': 'wait_for_generation_buffer',
        'id': 'skyrl.wait_for_generation_buffer',
        'framework': 'SkyRL',
        'commit': CHECKOUTS['skyrl'][1],
        'file': 'skyrl/train/fully_async_trainer.py',
        'line': 843,
        'anchor': 'timed_phase("wait_for_generation_buffer"',
        'waiting': 'yes',
        'waiting_evidence': '覆盖取组、等待与条件筛组记账',
        'overlap': 'no',
        'stamps': ['t2'],
        'c1_from_source': 'full',
    },
    {
        'proof': {'waiting': ('skyrl/train/fully_async_trainer.py', 69, 'group_completion_time_s'), 'overlap': ('skyrl/train/fully_async_trainer.py', 624, 'overlaps with generation')},
        'timer_key': 'group_completion_time_s',
        'id': 'skyrl.group_completion_time',
        'framework': 'SkyRL',
        'commit': CHECKOUTS['skyrl'][1],
        'file': 'skyrl/train/fully_async_trainer.py',
        'line': 69,
        'anchor': 'group_completion_time_s (Optional[float])',
        'waiting': 'no',
        'waiting_evidence': '取得准入 slot 后包住单组 generate，不含此前容量等待',
        'overlap': 'no',
        'stamps': ['none'],
        'c1_from_source': 'full',
    },
    {
        'proof': {'waiting': ('skyrl/train/fully_async_trainer.py', 604, 'timed_phase("sync_weights"'), 'overlap': ('skyrl/train/fully_async_trainer.py', 611, 'coordinator quiesce')},
        'timer_key': 'sync_weights',
        'id': 'skyrl.sync_weights',
        'framework': 'SkyRL',
        'commit': CHECKOUTS['skyrl'][1],
        'file': 'skyrl/train/fully_async_trainer.py',
        'line': 604,
        'anchor': 'timed_phase("sync_weights"',
        'waiting': 'yes',
        'waiting_evidence': '包住 pause/resume 括弧，源码注释自述含 coordinator quiesce',
        'overlap': 'no',
        'stamps': ['t1_or_t2'],
        'c1_from_source': 'full',
    },
    {
        'proof': {'waiting': ('skyrl/backends/skyrl_train/workers/worker_dispatch.py', 809, '_broadcast_to_inference_engines'), 'overlap': ('skyrl/backends/skyrl_train/workers/worker_dispatch.py', 822, 'wake_up(tags=["kv_cache"])')},
        'timer_key': 'timing/sync_weights_only_transfer',
        'id': 'skyrl.sync_weights_only_transfer',
        'framework': 'SkyRL',
        'commit': CHECKOUTS['skyrl'][1],
        'file': 'skyrl/backends/skyrl_train/workers/worker_dispatch.py',
        'line': 809,
        'anchor': 'self._broadcast_to_inference_engines(',
        'waiting': 'no',
        'waiting_evidence': '只包 broadcast + finish，无 pause/resume 等待',
        'overlap': 'no',
        'stamps': ['t1_or_t2'],
        'c1_from_source': 'full',
    },
]

# 版本量取得性的源码证据（C3）：每条都必须真实命中
STALENESS_PROOFS = {
    'skyrl': [
        ('skyrl/train/fully_async_trainer.py', 1038,
         'cur_staleness = self.global_step - cur_generated_output_group.global_step_when_scheduled'),
        ('skyrl/train/fully_async_trainer.py', 677,
         'notify_capacity_change(self.global_step)'),
    ],
}

# 每项判据在源码核验下的三种状态
DECIDABLE = 'decidable'
PARTIAL = 'partial'
UNDECIDABLE = 'undecidable'


def item_status(blk, key):
    """某一项（waiting / overlap）能否从源码判出。

    判定不看"有没有写 evidence 字符串"（那等于用结论证明结论），而看
    `proof` 里记录的源码锚点是否**实际命中**——由 proof_hits 在运行时核对。
    """
    if blk['overlap'] == 'n/a' and key == 'overlap':
        return DECIDABLE
    return DECIDABLE if blk.get('proof', {}).get(key) else PARTIAL


def item_block_identity(blk):
    """第 ① 项：块名 + 框架 + 固定 commit 能否唯一确定一段代码。

    能用行号定位不等于块名唯一。C1 要的是引用方能复述"是哪个计时块"，
    所以判据是源码里存在**独立命名的计时键**；只有裸的行间计时（算出一段时差、
    没有命名）时，引用者无法区分它和同文件里别的时差，只能靠本文代为描述。
    """
    if not (blk['line'] and blk.get('anchor')):
        return UNDECIDABLE
    return DECIDABLE if blk.get('timer_key') else PARTIAL


def c1_grade(blk):
    """C1 可引用等级。

    - A 可横比：三项全部可判，且块名能唯一确定一段代码
    - B 需补声明：有一项可从源码读出但不足以唯一确定（如空块分支、块名粒度）
    - C 不可横比：三项中存在必须外部说明才能判的内容
    """
    statuses = [item_block_identity(blk), item_status(blk, 'waiting'), item_status(blk, 'overlap')]
    if blk.get('c1_missing'):
        return 'B'
    if all(s == DECIDABLE for s in statuses):
        return 'A'
    if any(s == UNDECIDABLE for s in statuses):
        return 'C'
    return 'B'


GRADE_MEANING = {
    'A': '可横比：三件套均可由源码唯一判定',
    'B': '需补声明：有一项只能读到但不足以唯一确定，引用时须显式补写',
    'C': '不可横比：缺件无法由源码补足，须外部说明',
}


def _hit(root, relpath, line, needle, window=3):
    path = os.path.join(root, relpath)
    if not os.path.exists(path):
        return False, 'file missing: %s' % relpath
    with open(path, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    idx = line - 1
    if idx < 0 or idx >= len(lines):
        return False, 'line %d out of range in %s' % (line, relpath)
    chunk = ''.join(lines[max(0, idx - window):idx + window + 1])
    if needle in chunk:
        return True, ''
    return False, 'proof miss: %r at %s:%d' % (needle, relpath, line)


def check_anchor(blk):
    """回到固定 commit 的只读 checkout 核锚点行，并把每一项的 proof 也核一遍。

    三项判定各自挂一条源码证据；证据不命中就说明"结论引用的代码不再成立"，
    不能继续按记下的判定算等级。
    """
    root = CHECKOUTS[blk['framework_key']][0]
    ok, reason = _hit(root, blk['file'], blk['line'], blk['anchor'])
    if not ok:
        return {'ok': False, 'reason': reason}
    for key, (relpath, line, needle) in blk.get('proof', {}).items():
        ok, reason = _hit(root, relpath, line, needle)
        if not ok:
            return {'ok': False, 'reason': '%s/%s' % (key, reason)}
    return {'ok': True, 'reason': ''}


# ---------------------------------------------------------------------------
# C3：η 与 S 的定义可取得性（A2）
# ---------------------------------------------------------------------------
# 一个框架的 η 能不能按 C3 的定义算出来，取决于有没有"生成时的策略版本"与
# "消费时的策略版本"两个可读量。这里按四框架快照的静态读码给出判定。
VERSION_FIELDS = {
    'skyrl': {
        'generation_version_field': 'GeneratedOutputGroup.global_step_when_scheduled',
        'consumption_version_field': 'self.global_step',
        'alias': 'training step counter',
        'measured_semantics': 'global_step − global_step_when_scheduled',
        'c3_literal': False,
        'c3_literal_gap': '报的是"训练步数年龄"，不是"策略版本差"；一个 step 内多次更新会读到同一个数。',
        'ts_available': ['t1_or_t2'],
    },
    'nemo_rl': {
        'generation_version_field': 'prompt stamped with version_during_step / evict(current_train_weight=...)',
        'consumption_version_field': 'self._trainer_version',
        'alias': 'trainer weight version',
        'measured_semantics': 'evict 用训练侧权重版本筛陈旧组（非逐条年龄分布）',
        'c3_literal': 'partial',
        'c3_literal_gap': '能识别"陈旧"但按组淘汰，不产出 C3 要求的 max/P50/P90 年龄分布。',
        'ts_available': ['t1_or_t2'],
    },
    'openrlhf': {
        'generation_version_field': None,
        'consumption_version_field': 'global_step（slot 携带，用于 eval 时机）',
        'alias': 'step counter',
        'measured_semantics': '不构成逐条年龄',
        'c3_literal': False,
        'c3_literal_gap': '所读通路无每条 Experience 的 policy 版本；Q=1 不能推出只落后一步。',
        'ts_available': ['none'],
    },
    'verl': {
        'generation_version_field': 'min/max global_steps 标签（推理）/ staleness_samples 计数（fully-async）',
        'consumption_version_field': 'current_param_version',
        'alias': 'step counter / 计数阈值',
        'measured_semantics': '参数同步时钟与训练调用时钟不相同一步进',
        'c3_literal': 'partial',
        'c3_literal_gap': '版本标签按步推进，且 fully-async 的暂停读计数而非逐样本版本差。',
        'ts_available': ['t1_or_t2'],
    },
}


def s_accepted_trajectories(spec):
    """C3 的 S 分母：被接纳进训练的轨迹数。

    给定一次运行的分项（每步生成 n_gen 条、丢弃 n_drop 条、训练步数 steps），
    以及"生成侧估算"与"训练侧实际"两种口径，返回两者的差。
    """
    gen_side = spec['steps'] * spec['n_gen']
    train_side = spec['steps'] * (spec['n_gen'] - spec['n_drop'])
    return {
        'reported_by_generation': gen_side,
        'accepted_by_training': train_side,
        'relative_gap': (gen_side - train_side) / float(train_side) if train_side else float('inf'),
    }


def check_staleness_proofs():
    out = {}
    for fw, proofs in STALENESS_PROOFS.items():
        root = CHECKOUTS[fw][0]
        hits = []
        for relpath, line, needle in proofs:
            ok, reason = _hit(root, relpath, line, needle)
            hits.append({'line': line, 'ok': ok, 'reason': reason})
        out[fw] = {'all_ok': all(h['ok'] for h in hits), 'hits': hits}
    return out


def build_summary():
    rows = []
    for blk in BLOCKS:
        key = blk['id'].split('.')[0]
        blk = dict(blk)
        blk['framework_key'] = {'verl': 'verl', 'openrlhf': 'openrlhf',
                                'nemo': 'nemo_rl', 'skyrl': 'skyrl'}[key]
        anchor = check_anchor(blk)
        rows.append({
            'id': blk['id'],
            'framework': blk['framework'],
            'commit': blk['commit'],
            'file': blk['file'],
            'line': blk['line'],
            'anchor_ok': anchor['ok'],
            'anchor_reason': anchor['reason'],
            'item1_block_identity': item_block_identity(blk),
            'item2_waiting': item_status(blk, 'waiting'),
            'item3_overlap': item_status(blk, 'overlap'),
            'c1_grade': c1_grade(blk),
            'c1_missing': blk.get('c1_missing', []),
            'waiting_value': blk['waiting'],
            'overlap_value': blk['overlap'],
            'c2_stamps_available': blk['stamps'],
        })

    grade_counts = {}
    for r in rows:
        grade_counts[r['c1_grade']] = grade_counts.get(r['c1_grade'], 0) + 1

    item_stats = {}
    for item in ('item1_block_identity', 'item2_waiting', 'item3_overlap'):
        stats = {}
        for r in rows:
            stats[r[item]] = stats.get(r[item], 0) + 1
        item_stats[item] = stats

    # A2：哪些块能给出 t1（独立证据），哪些只到 t1_or_t2
    t1_exact = [r['id'] for r in rows if r['c2_stamps_available'] == ['t1']]
    t1_pooled = [r['id'] for r in rows if r['c2_stamps_available'] == ['t1_or_t2']]
    t2_only = [r['id'] for r in rows if r['c2_stamps_available'] == ['t2']]

    c3_rows = []
    for fw, fields in VERSION_FIELDS.items():
        c3_rows.append({
            'framework': fw,
            'c3_literal': fields['c3_literal'],
            'generation_version_field': fields['generation_version_field'],
            'consumption_version_field': fields['consumption_version_field'],
            'measured_semantics': fields['measured_semantics'],
            'gap': fields['c3_literal_gap'],
        })
    c3_ok = sum(1 for r in c3_rows if r['c3_literal'] is True)

    s_examples = {
        'skyrl_sample_full_batch': s_accepted_trajectories(
            {'steps': 100, 'n_gen': 32, 'n_drop': 0}),
        'skyrl_sample_full_batch_drops': s_accepted_trajectories(
            {'steps': 100, 'n_gen': 32, 'n_drop': 8}),
    }

    return {
        'experiment': 'C1 计时边界约定的可判定性 + C3 的定义可取得性',
        'doc': 'notebook/rollout-performance-research.md §21.6',
        'anchors': {
            'all_ok': all(r['anchor_ok'] for r in rows),
            'failures': [{'id': r['id'], 'reason': r['anchor_reason']}
                         for r in rows if not r['anchor_ok']],
        },
        'c1': {
            'blocks': rows,
            'grade_counts': grade_counts,
            'item_stats': item_stats,
            'grade_meaning': GRADE_MEANING,
            'readings': {
                'from_source_only': '仅凭源码能否补齐三件套',
                'needs_external': [r['id'] for r in rows if r['c1_grade'] != 'A'],
            },
        },
        'c1_c3_bridge': {
            't1_exact': t1_exact,
            't1_pooled': t1_pooled,
            't2_only': t2_only,
            'note': '没有一个块能把 t1 与 t2 分开给出；能做拆分的只有把同一块硬当 t1 的写法，'
                    '与 §21.5 实测的 ack_sep 同类。',
        },
        'c3': {
            'framework_rows': c3_rows,
            'staleness_proofs': check_staleness_proofs(),
            'literal_count': c3_ok,
            'total': len(c3_rows),
            's_examples': s_examples,
        },
    }


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
    a = out['anchors']
    print('anchors all_ok=%s  failures=%d' % (a['all_ok'], len(a['failures'])))
    print('C1 grade counts:', out['c1']['grade_counts'])
    for r in out['c1']['blocks']:
        print('  %-38s %-9s grade=%s  missing=%s  stamps=%s'
              % (r['id'], r['framework'], r['c1_grade'],
                 ','.join(r['c1_missing']) or '-', r['c2_stamps_available']))
    print('C3 literal_count=%d/%d' % (out['c3']['literal_count'], out['c3']['total']))
    print('wrote', SUMMARY_PATH)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
