#!/usr/bin/env python3
"""llama.cpp 那 71 条 `unwritten` 候选的归因档位（逐条绑到可 grep 的事实）。

`scan_cpp_flags.py` 只回答"这个字段在本快照里有没有任何写入/传参"。答案是"没有"
不等于缺陷——多数是结构原因。本脚本把每条候选分到一档，并给出一句可以人工复核
的理由；**不预先认定任何一条是缺陷**。

    python3 classify.py <llama.cpp 根> [--json out.json]

退出码：0 = 每条候选都有档位；1 = 有候选没被分到档位（说明快照变了，档位表过期）。
"""

import json
import pathlib
import re
import subprocess
import sys

SCAN_NAME = 'scan-cpp-llama.json'
SNAP_NAME = 'scan_cpp_flags.py'

# 逐条规则：(档位, 判定, 依据)。命中第一条即定档。
# 依据里写的是"为什么这不是缺陷"，取证时按行号复核。
RULES = [
    ('vendor', '第三方 vendored 代码（httplib 等）', '不属于 llama.cpp 自己的配置面'),
    ('backend_kernel', 'ggml 后端/gen kernel 参数结构体',
     '字段由 kernel 生成器或后端 host 代码填写，取值来自设备能力探测而非用户配置'),
    ('test', '测试夹具', '只在 tests/ 下使用，无用户面'),
    ('logic_not_config', '运行期状态量而非配置开关',
     '名字描述的是某次计算的事实（如某个 buffer 是否被分配），不是给用户设的'),
]

# 明确查过的几条：给出具体事实而不是档位套话。
KNOWN = {
    'reasoning_control': (
        'WIRED',
        'server 每请求下发（server-common.cpp:1530）、schema 绑定引用（server-schema.cpp:351）、'
        '守卫真的读它（sampling.cpp:315）⇒ 已接线，扫描器漏判是因为写入发生在 '
        'std::string → json → 解析回 field_bool 的引用赋值，不是直接的 `x.field = `'),
    'show_key_map_stats': (
        'DEAD_BRANCH_UPSTREAM',
        '唯一的写入机会是构造函数，而 common_ngram_map 的构造函数只接 4 个形参、'
        '不含该字段（ngram-map.h:72-75）；全仓 0 处赋值 ⇒ `if (map.show_key_map_stats)` '
        '恒假。字段与分支由 #19253（b4d05a3d2f，2026-02-02）同一次提交引入，'
        '是开发者调试用开关，未接命令行'),
    'filter_allowed_keys': (
        'REMOVED_BY_DESIGN',
        '配套白名单在 #24739（552258c535，2026-06-18 router rework）里被有意移除，'
        '残留的 `if (filter_allowed_keys && ...)` 永不进入；上游 #27929 正在讨论是否恢复'
        '远程 preset 白名单 ⇒ 属于已知未决事项，不是新发现'),
    'in_place': (
        'POSITIONAL_AGGREGATE',
        '写入是位置实参的第四项：drama3-gen.cpp:12/15/16 `push_back({"tower_pos", 1, 1, true})`；'
        '扫描器只认带注释名的聚合初始化与 `.field =`，位置形式看不到'),
    'shares_model': (
        'DESIGNATED_INIT',
        '写入在 common.cpp:1284 `/*.shares_model =*/ !has_draft`（注释名带点号），'
        '扫描器的注释名正则没覆盖 `.` 前缀'),
    'req_count': (
        'INCREMENT_ONLY',
        '写入是 `mapping[name].req_count++`（server-models.cpp:1699）与 `--`（:1731），'
        '扫描器只认 `=` 赋值'),
    'n_attempts': (
        'INCREMENT_ONLY',
        '写入是 `meta.memory.n_attempts++`（server-models-sched.cpp:434）'),
    'topk_blocks': (
        'DESIGNATED_INIT',
        '写入在 minimax-m3.cpp:29 `/*.topk_blocks =*/ (int) hparams.indexer_top_k`（注释名带点号）'),
    'enabled_': (
        'CTOR_MEMBER_INIT',
        '写入在构造函数初始化列表 `enabled_(enabled)`（mtmd-helper.cpp:99）'),
    '_dotall': (
        'CTOR_MEMBER_INIT',
        '写入在构造函数初始化列表 `_dotall(dotall)`（json-schema-to-grammar.cpp:843）'),
    'blocks_found_in_stage': (
        'LOCAL_VAR',
        '同名局部变量（clip.cpp:3271 `int blocks_found_in_stage = 0;`），'
        '扫描器把它误当成某结构体的字段'),
    'is_dir': (
        'LOCAL_VAR',
        '同名局部变量（server-tools.cpp:179/506），结构体 list_entry 的那份由调用方按位置传入'),
    'paren_depth': (
        'LOCAL_VAR',
        '函数内局部变量（json-schema-to-grammar.cpp:439），与字段同名不同作用域'),
    'a_layer': (
        'LOCAL_VAR',
        '比较器里的局部变量（llama-model-loader.h:54），由 sscanf 写'),
    'is_copy': (
        'STRUCT_CTOR_ARG',
        'tensor_transformation 由构造实参初始化（export-lora.cpp 的转换表），'
        '扫描器没跟上它的构造点'),
    'timing_per_token': (
        'DEAD_FIELD',
        '只在 common.h:260 声明，全仓 0 处读取、0 处赋值'),
    'defer_loading': (
        'JSON_PARSED',
        '写入在 chat.cpp:673-684（从请求 JSON 解析），扫描器的正则没识别 json 取值路径'),
    'is_inference': (
        'JSON_OR_CTOR',
        '由解析模板时的上下文决定，consumer 在 common/parsers/* 里按它分支'),
}


def read(root, rel):
    return (pathlib.Path(root) / rel).read_text(encoding='utf-8', errors='ignore')


def repo_grep(root, name):
    """返回 [(file, line_no, text)]，跳过 .git。"""
    out = []
    for p in pathlib.Path(root).rglob('*'):
        if not p.is_file() or p.suffix not in {'.h', '.hpp', '.cpp', '.c', '.cu', '.cuh'}:
            continue
        if '.git' in p.parts:
            continue
        try:
            txt = p.read_text(encoding='utf-8', errors='ignore')
        except OSError:
            continue
        if name not in txt:
            continue
        for i, line in enumerate(txt.splitlines(), 1):
            if re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', line):
                out.append((p.relative_to(root).as_posix(), i, line.strip()))
    return out


def classify(root, fields):
    report = []
    for f in fields:
        name = f['field']
        hits = repo_grep(root, name)
        paths = sorted({h[0] for h in hits})
        entry = {'field': name, 'n_consumers': f['n_consumers'],
                 'n_writers': f['n_writers'], 'n_forwarders': f['n_forwarders'],
                 'decl_files': sorted({d['file'] for d in f['decls']}),
                 'mention_files': paths}
        if name in KNOWN:
            entry['bucket'], entry['why'] = KNOWN[name]
        elif all(p.startswith(('vendor/', 'ggml/src/ggml-httplib')) for p in entry['decl_files']):
            entry['bucket'], entry['why'] = RULES[0][1], RULES[0][2]
        elif all(p.startswith('ggml/src/') for p in entry['decl_files']):
            entry['bucket'], entry['why'] = RULES[1][1], RULES[1][2]
        elif all(p.startswith('tests/') for p in entry['decl_files']):
            entry['bucket'], entry['why'] = RULES[2][1], RULES[2][2]
        else:
            entry['bucket'], entry['why'] = 'UNCLASSIFIED', ''
        report.append(entry)
    return report


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith('--')]
    if not argv:
        print(__doc__)
        return 2
    root = argv[0]
    here = pathlib.Path(__file__).resolve().parent
    scan_path = here / 'raw' / SCAN_NAME
    if not scan_path.exists():
        scan_path = pathlib.Path('/tmp') / SCAN_NAME
    rep = json.load(open(scan_path))
    fields = [f for f in rep['fields'] if f['unwritten']]
    out = classify(root, fields)
    print('candidates=%d' % len(out))
    from collections import Counter
    c = Counter(e['bucket'] for e in out)
    for k, v in c.most_common():
        print('  %-32s %2d' % (k, v))
    unclassified = [e['field'] for e in out if e['bucket'] == 'UNCLASSIFIED']
    if unclassified:
        print('UNCLASSIFIED:', ', '.join(unclassified))
    here_out = here / 'raw' / 'classify.json'
    here_out.write_text(json.dumps({'root': root, 'buckets': dict(c), 'entries': out},
                                   indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    return 0 if not unclassified else 1


if __name__ == '__main__':
    sys.exit(main())
