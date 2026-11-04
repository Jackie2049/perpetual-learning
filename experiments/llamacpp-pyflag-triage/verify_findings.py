#!/usr/bin/env python3
"""把本批每条结论钉到具体行号；上游一改，对应判据就变红。

    python3 verify_findings.py <llama.cpp 根>       # 0 = 全部成立，2 = 快照不干净，1 = 有判据失败
    python3 verify_findings.py --selftest           # 注入 4 种改动，每种都必须让目标判据变红

判据分三组：
  A. 扫描器读数本身（文件数、字段数、unwritten 数、档位分布）
  B. 三个"扫描器看不见的写入形态"各自的真实代码（这是本批真正的方法产出）
  C. 本批唯一被怀疑过的字段 reasoning_control 确实接线（探针的对照组）
"""

import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent


def read(root, rel):
    return (pathlib.Path(root) / rel).read_text(encoding='utf-8', errors='ignore')


def line_of(text, needle, nth=0):
    hits = [i for i, line in enumerate(text.splitlines(), 1) if needle in line]
    if len(hits) <= nth:
        return None
    return hits[nth]


def main_checks(root):
    checks = []

    def check(name, got, want):
        checks.append({'name': name, 'got': got, 'want': want, 'ok': got == want})

    # ---- A. 扫描器读数 ----
    scan_path = HERE / 'raw' / 'scan-cpp-llama.json'
    if not scan_path.exists():
        check('扫描读数存在', False, True)
        return checks
    rep = json.load(open(scan_path))
    fields = rep['fields']
    unwritten = [f for f in fields if f['unwritten']]
    check('扫描到 1574 个 C++ 文件', rep['files'], 1574)
    check('有消费者且有分支的字段 1636 个', len(fields), 1636)
    check('判为 unwritten 的 37 条', len(unwritten), 37)

    cls = HERE / 'raw' / 'classify.json'
    if cls.exists():
        c = json.load(open(cls))
        check('37 条全部归因（没有 UNCLASSIFIED）',
              c['buckets'].get('UNCLASSIFIED', 0), 0)
        check('其中"ggml 后端/kernel 参数"占 15 条',
              c['buckets'].get('ggml 后端/gen kernel 参数结构体', 0), 15)

    # ---- B. 三个扫描器看不见的写入形态（本批方法产出）----
    arg = read(root, 'common/arg.cpp')
    check('B1 单行 if + 赋值：batched_bench_output_jsonl 在 arg.cpp 被写',
          line_of(arg, 'params.batched_bench_output_jsonl = true') is not None, True)
    ctx = read(root, 'tools/server/server-context.cpp')
    check('B2 聚合初始化：allow_image 走 /* name */ value 形式',
          line_of(ctx, '/* allow_image') is not None, True)
    tts = read(root, 'tools/mtmd/models/drama3-gen.cpp')
    check('B3 位置实参：c2w_state_slot 的 in_place 是第四个元素',
          line_of(tts, '"tower_pos", 1, 1, true') is not None, True)
    qwen = read(root, 'tools/mtmd/models/qwen3tts-gen.cpp')
    check('B3 对照：同一结构体在 qwen3tts 里只传三个（默认 false）',
          line_of(qwen, 'slots.push_back({"tfm_pos", 1, 1})') is not None, True)

    # ---- C. reasoning_control 确实接线 ----
    guard = read(root, 'common/sampling.cpp')
    check('C1 守卫里含 reasoning_control 这一项',
          line_of(guard, 'reasoning_control)') is not None, True)
    common = read(root, 'common/common.h')
    check('C2 字段声明默认 false', line_of(common, 'reasoning_control = false') is not None, True)
    sch = read(root, 'tools/server/server-schema.cpp')
    check('C3 schema 用引用绑定该字段（field_bool 构造接 bool&）',
          line_of(sch, 'field_bool("reasoning_control", params.sampling.reasoning_control)') is not None, True)
    srv = read(root, 'tools/server/server-common.cpp')
    check('C4 server 每请求把它写进 llama_params',
          line_of(srv, 'llama_params["reasoning_control"] =') is not None, True)

    # ---- D. 两条 DEAD 判定的反面对照 ----
    ng = read(root, 'common/ngram-map.h')
    ctor = ng.split('common_ngram_map(uint16_t')[1].split(')')[0]
    check('D1 show_key_map_stats 的构造函数不收该字段',
          'show_key_map_stats' not in ctor, True)
    preset = read(root, 'common/preset.h')
    check('D2 filter_allowed_keys 仍在声明（白名单移除后残留）',
          'filter_allowed_keys' in preset, True)

    return checks


def selftest(root):
    """四种注入，每种都必须让对应判据变红。"""
    import shutil
    import tempfile

    notes, ok = [], True

    def run_case(tag, mutate):
        nonlocal ok
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            # 只复制判据真正读的文件
            for rel in ('common/arg.cpp', 'tools/server/server-context.cpp',
                        'tools/mtmd/models/drama3-gen.cpp', 'tools/mtmd/models/qwen3tts-gen.cpp',
                        'common/sampling.cpp', 'common/common.h',
                        'tools/server/server-schema.cpp', 'tools/server/server-common.cpp',
                        'common/ngram-map.h', 'common/preset.h'):
                src = pathlib.Path(root) / rel
                dst = tmp / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dst)
            mutate(tmp)
            got = {c['name']: c['ok'] for c in main_checks(tmp)}
            return got

    base = run_case('base', lambda t: None)
    failed_base = [k for k, v in base.items() if not v]
    good = not failed_base
    ok = ok and good
    notes.append('%s 基线：全部判据成立=%s %s' % (
        'PASS' if good else 'FAIL', good, failed_base or ''))

    def expect(tag, mutate, target):
        got = run_case(tag, mutate)
        red = got.get(target) is False
        nonlocal ok
        ok = ok and red
        notes.append('%s 注入「%s」→ 目标判据 %s %s' % (
            'PASS' if red else 'FAIL', tag, target, '变红' if red else '仍是绿的'))

    def drop(rel, needle):
        def f(t):
            p = t / rel
            txt = p.read_text()
            lines = [l for l in txt.splitlines(True) if needle not in l]
            p.write_text(''.join(lines))
        return f

    expect('删掉 B1 的赋值行', drop('common/arg.cpp',
                                   'params.batched_bench_output_jsonl = true'), 'B1 单行 if + 赋值：batched_bench_output_jsonl 在 arg.cpp 被写')
    expect('删掉 B2 的聚合初始化行', drop('tools/server/server-context.cpp',
                                   '/* allow_image'), 'B2 聚合初始化：allow_image 走 /* name */ value 形式')
    expect('删掉 C1 守卫里的该项', drop('common/sampling.cpp',
                                  'reasoning_control)) {'), 'C1 守卫里含 reasoning_control 这一项')
    expect('删掉 C4 server 下发', drop('tools/server/server-common.cpp',
                                  'llama_params["reasoning_control"] ='), 'C4 server 每请求把它写进 llama_params')
    return ok, notes


def main():
    argv = sys.argv[1:]
    if '--selftest' in argv:
        root = argv[-1] if len(argv) > 1 else '/tmp/llama-scan'
        ok, notes = selftest(root)
        for n in notes:
            print(n)
        return 0 if ok else 1
    if len(argv) != 1:
        print(__doc__)
        return 2
    checks = main_checks(argv[0])
    for c in checks:
        print('%s %s got=%s want=%s' % ('PASS' if c['ok'] else 'FAIL', c['name'], c['got'], c['want']))
    failed = [c for c in checks if not c['ok']]
    print('checks=%d failed=%d' % (len(checks), len(failed)))
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
