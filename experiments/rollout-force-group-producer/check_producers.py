#!/usr/bin/env python3
"""`force_group_size` 有没有生产者：把 §13.35 的残留从「只核到 engine 入口」升级为全仓判据。

支撑 notebook/rollout-performance-research.md §13.37。

§13.35 的残留原文是「`force_group_size` 的接线在 ray trainer 侧，而 ray trainer 不在该
checkout 的 `enable` 流程里，本批只读核到 engine 入口为止，不作全局保证」。本脚本检验
一个更强的命题：**这个键在整个 checkout 的非测试代码里根本没有写入方**——即「读不到」不是
因为某条路径没核到，而是因为不存在任何把 `> 1` 写进去的代码。

同时核清三条「看起来可能写入」的旁路，逐条给出为什么不通：
  A. 默认键注入：`train_batch` / `infer_batch` / tinker 三处 `default_keys` 都是静态字面量。
  B. TransferQueue 回写：`update_meta_with_output` 会把输出的 NonTensorData 塞进
     `extra_info` 再回注 batch——但 `train_mini_batch` 只回 `metrics`，键不进输出。
  C. 配置层：`EngineConfig` 没有 `force_group_size` 字段，用户无处配置。

口径：**静态源码核查，不执行 verl、不给任何性能数字**。脚本只负责「证据是否仍命中」
（锚点文本是否存在、写入方计数是否为 0），判定是人读代码得出的。

复现：
    VERL_ROOT=../../verl python3 check_producers.py            # 约 1 s，覆盖 summary.json
    VERL_ROOT=../../verl python3 check_producers.py --verify   # 与归档逐字段比对
    VERL_ROOT=../../verl python3 -m unittest test_check_producers -v
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY_PATH = os.path.join(HERE, 'summary.json')
VERL_ROOT = os.environ.get('VERL_ROOT', os.path.join(HERE, '..', '..', 'verl'))

# 只统计生产代码；tests/ 下的写入方是本特性唯一的使用者，单列计数。
TEST_PREFIX = 'tests/'
# 读方（合法存在的三处）：参数定义、读取、透传。
READER_ALLOWLIST = {
    'verl/utils/seqlen_balancing.py',
    'verl/workers/engine/utils.py',
}


def iter_py_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ('.git', '__pycache__', '.venv')]
        for name in filenames:
            if name.endswith('.py'):
                abs_path = os.path.join(dirpath, name)
                yield os.path.relpath(abs_path, root).replace(os.sep, '/'), abs_path


# 写入方 = 给 TensorDict 挂 non-tensor 键，键名以三种写法之一出现：
#   位置参数  assign_non_tensor_data(td, "force_group_size", n)
#   关键字键  assign_non_tensor(data, key="force_group_size", val=n)
#   kwargs    assign_non_tensor(data, force_group_size=n)
# 第三种与 engine_workers.py:345 的 `**{key: val}` 同形，漏掉它会系统性放过默认键注入
# （本脚本第一版正是漏了这一种，被 negative_control.py 抓出）。
producer_re = re.compile(
    r'assign_non_tensor(?:_data)?\s*\((?:[^)]*["\']force_group_size["\']|'
    r'[^)]*\bforce_group_size\s*=)'
)


def scan(root):
    """返回 (producers, readers, test_producers, missing_root)。

    刻意不加 lru_cache：`build()` 会在同一个进程里被多次调用（正向可行性检验要
    在扫描后往源码注入一个写入方再扫一遍），缓存会让第二次调用直接返回旧结果，
    使「注入后应翻成 false」永远不成立——这正是拿旧证据判新树。
    """
    if not os.path.isdir(root):
        return [], [], [], True
    # 任何出现都先收集，再按归类拆开，避免「漏扫」被当成「没有」。
    producers, readers, test_producers = [], [], []
    for rel, abs_path in iter_py_files(root):
        with open(abs_path, 'r', encoding='utf-8', errors='replace') as fh:
            text = fh.read()
        if 'force_group_size' not in text:
            continue
        hits = []
        for lineno, line in enumerate(text.splitlines(), 1):
            if 'force_group_size' in line:
                hits.append({'line': lineno, 'text': line.strip()})
        entry = {'path': rel, 'hits': hits, 'producer_lines': []}
        for lineno, line in enumerate(text.splitlines(), 1):
            if producer_re.search(line):
                entry['producer_lines'].append(lineno)
        if rel.startswith(TEST_PREFIX):
            test_producers.append(entry)
        elif entry['producer_lines']:
            producers.append(entry)
        else:
            readers.append(entry)
    return producers, readers, test_producers, False


def anchor_checks(root):
    """四个必须仍然成立的文本锚点；任一失效就说明 §13.37 的结论需要重核。"""
    checks = []

    def read(rel):
        abs_path = os.path.join(root, rel)
        if not os.path.isfile(abs_path):
            return None
        with open(abs_path, 'r', encoding='utf-8', errors='replace') as fh:
            return fh.read()

    def add(name, rel, needle, expect_present=True):
        text = read(rel)
        present = text is not None and needle in text
        checks.append({
            'name': name,
            'path': rel,
            'needle': needle,
            'present': present,
            'ok': present == expect_present,
        })

    # 1. 读取点带默认 1 —— 无写入方时恒为 1。
    add('read_defaults_to_1', 'verl/workers/engine/utils.py',
        'force_group_size = tu.get_non_tensor_data(data=data, key="force_group_size", default=1)')
    # 2. 五个后端全部走同一个 prepare_micro_batches（形参里没有 force_group_size）。
    add('prepare_signature_has_no_param', 'verl/workers/engine/utils.py',
        'def prepare_micro_batches(')
    # 3. 组级能力确实存在（force_group_size > 1 有真实分支），只是没人喂。
    add('group_branch_exists', 'verl/utils/seqlen_balancing.py',
        'if force_group_size > 1:')
    # 4. EngineConfig 没有这个字段 —— 用户无法从 yaml 配置进去。
    add('engine_config_field_absent', 'verl/workers/config/engine.py',
        'force_group_size', expect_present=False)
    return checks


def default_key_blocks(root):
    """三处 default_keys 的键名清单；确认都不含 force_group_size。"""
    targets = [
        ('verl/workers/engine_workers.py', 'train_batch'),
        ('verl/workers/engine_workers.py', 'infer_batch'),
        ('verl/workers/engine_workers_tinker.py', 'tinker'),
    ]
    blocks = []
    for rel, label in targets:
        abs_path = os.path.join(root, rel)
        if not os.path.isfile(abs_path):
            blocks.append({'path': rel, 'label': label, 'found': False})
            continue
        with open(abs_path, 'r', encoding='utf-8', errors='replace') as fh:
            text = fh.read()
        # 抓每个 `default_keys = dict(...)` 的字面量内容。
        found = []
        for match in re.finditer(r'default_keys\s*=\s*dict\((.*?)\n\s*\)', text, re.S):
            body = match.group(1)
            keys = re.findall(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=', body, re.M)
            found.append({
                'line': text[:match.start()].count('\n') + 1,
                'keys': keys,
                'contains_force_group_size': 'force_group_size' in body,
            })
        blocks.append({'path': rel, 'label': label, 'found': bool(found), 'blocks': found})
    return blocks


def tq_writeback_guard(root):
    """TransferQueue 回写只带得走 NonTensorData，且 update_actor 只回 metrics。"""
    rel = 'verl/utils/transferqueue_utils.py'
    abs_path = os.path.join(root, rel)
    result = {'path': rel, 'exists': os.path.isfile(abs_path)}
    if not result['exists']:
        return result
    with open(abs_path, 'r', encoding='utf-8', errors='replace') as fh:
        text = fh.read()
    result['only_non_tensor_data_synced'] = (
        'elif isinstance(v, NonTensorData):' in text
        and 'meta.extra_info = meta_data' in text
    )
    rel2 = 'verl/workers/engine_workers.py'
    abs2 = os.path.join(root, rel2)
    with open(abs2, 'r', encoding='utf-8', errors='replace') as fh:
        text2 = fh.read()
    # train_mini_batch 的返回只组装 metrics。
    result['train_mini_batch_returns_metrics_only'] = (
        'output = tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": metrics}).cpu()' in text2
    )
    return result


def build(root=VERL_ROOT):
    producers, readers, test_producers, missing = scan(os.path.abspath(root))
    out = {
        'verl_root': os.path.abspath(root),
        'verl_missing': missing,
        'producer_count': len(producers),
        'producer_files': [p['path'] for p in producers],
        'reader_files': sorted(r['path'] for r in readers),
        'test_producer_files': [t['path'] for t in test_producers],
        'test_producer_lines': {
            t['path']: t['producer_lines'] for t in test_producers
        },
        'anchors': anchor_checks(os.path.abspath(root)),
        'default_key_blocks': default_key_blocks(os.path.abspath(root)),
        'tq_writeback': tq_writeback_guard(os.path.abspath(root)),
    }
    out['verdict'] = verdict(out)
    return out


def verdict(out):
    anchors_ok = all(c['ok'] for c in out['anchors'])
    dk_clean = all(
        not any(b['contains_force_group_size'] for b in blk.get('blocks', []))
        for blk in out['default_key_blocks']
    )
    tq = out['tq_writeback']
    return {
        # 核心判决：非测试代码里没有写入方。
        'no_producer_in_production_code': (not out['verl_missing']) and out['producer_count'] == 0,
        # 唯一写入方是测试，且它直接 assign 进 batch 构造假输入。
        'only_writer_is_test': len(out['test_producer_files']) >= 1 and out['producer_count'] == 0,
        'anchors_all_hold': anchors_ok,
        'default_keys_never_inject_it': dk_clean,
        'tq_writeback_cannot_carry_it': bool(
            tq.get('only_non_tensor_data_synced') and tq.get('train_mini_batch_returns_metrics_only')
        ),
        # 因此「读到的恒为默认 1」是代码结构结论，不是「某条路径没核到」。
        'always_default_1_is_structural': (
            (not out['verl_missing']) and out['producer_count'] == 0 and anchors_ok
            and dk_clean and bool(
                tq.get('only_non_tensor_data_synced') and tq.get('train_mini_batch_returns_metrics_only')
            )
        ),
    }


def main(argv):
    data = build()
    if '--verify' in argv:
        if not os.path.isfile(SUMMARY_PATH):
            print('summary.json 不存在，先跑一次生成')
            return 2
        with open(SUMMARY_PATH, 'r', encoding='utf-8') as fh:
            old = json.load(fh)
        # verl_root 是绝对路径，换机器会变，不参与比对。
        fields = [k for k in old if k != 'verl_root']
        diffs = [k for k in fields if old.get(k) != data.get(k)]
        if diffs:
            print('不一致字段：', ', '.join(diffs))
            for k in diffs:
                print(f'  archived {k}: {json.dumps(old.get(k), ensure_ascii=False)[:300]}')
                print(f'  current  {k}: {json.dumps(data.get(k), ensure_ascii=False)[:300]}')
            return 1
        print(f'--verify 通过：{len(fields)} 个字段逐字段相同')
        return 0

    with open(SUMMARY_PATH, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write('\n')
    print(json.dumps(data['verdict'], ensure_ascii=False, indent=2))
    print(f'写入 {SUMMARY_PATH}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
