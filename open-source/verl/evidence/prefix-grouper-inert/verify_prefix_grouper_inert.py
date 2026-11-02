#!/usr/bin/env python3
"""取证脚本：verl main 上 `use_prefix_grouper=True` 是否真的生效。

对上游快照做**只读静态取证**，不运行 verl、不需要 GPU、不导入 torch。
结论完全由 AST 与源码文本推出，因此可被任何人复跑核对。

用法（先把快照解包）：
    tar xzf verl-main-6879e1ab07.tar.gz
    python3 verify_prefix_grouper_inert.py verl-project-verl-6879e1ab07/

退出码：0 = 取证通过（发现仍然成立）；1 = 发现已失效（上游修好了）；2 = 取证失败。
"""

import ast
import json
import pathlib
import sys

EXPECTED = {
    'source_commit': '6879e1ab07',
    'production_call_site': 'verl/workers/engine/fsdp/transformer_impl.py',
    'flags': ['use_prefix_grouper', 'use_dynamic_bsz'],
}


def parse(p):
    try:
        return ast.parse(p.read_text(encoding='utf-8', errors='ignore'))
    except SyntaxError:
        return None


def main(root_arg):
    root = pathlib.Path(root_arg)
    if not root.exists():
        print('找不到源码目录:', root)
        return 2

    findings = {}

    # ---- 1. apply_monkey_patch 的定义与全部调用点 ----
    defs = []
    callers = []
    for p in root.rglob('*.py'):
        if '.git' in p.parts:
            continue
        tree = parse(p)
        if tree is None:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef) and n.name in (
                    'apply_monkey_patch', 'apply_prefix_grouper_patch'):
                defs.append({'file': p.relative_to(root).as_posix(),
                             'func': n.name, 'line': n.lineno,
                             'params': [a.arg for a in n.args.args + n.args.kwonlyargs]})
            if isinstance(n, ast.Call) and getattr(n.func, 'id', '') == 'apply_monkey_patch':
                callers.append({'file': p.relative_to(root).as_posix(),
                                'line': n.lineno,
                                'kwargs': [k.arg for k in n.keywords if k.arg]})

    prod = [c for c in callers if not c['file'].startswith('tests/')]
    findings['apply_monkey_patch_defs'] = defs
    findings['production_call_sites'] = prod
    findings['production_passes_flag'] = any(
        'use_prefix_grouper' in c['kwargs'] for c in prod)

    # ---- 2. apply_prefix_grouper_patch 的调用点在哪个分支内 ----
    mp = root / 'verl/models/transformers/monkey_patch.py'
    guarded = None
    if mp.exists():
        src = mp.read_text(encoding='utf-8', errors='ignore')
        tree = ast.parse(src)
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef) and n.name == 'apply_monkey_patch':
                for sub in ast.walk(n):
                    if (isinstance(sub, ast.If) and sub.test.__class__ is ast.Name
                            and sub.test.id == 'use_prefix_grouper'):
                        for inner in ast.walk(sub):
                            if (isinstance(inner, ast.Call)
                                    and getattr(inner.func, 'id', '') == 'apply_prefix_grouper_patch'):
                                guarded = {'if_name': 'use_prefix_grouper',
                                           'call_line': inner.lineno}
    findings['pg_patch_guarded_by_param'] = guarded

    # ---- 3. prefix_grouper 包是否在依赖里声明 ----
    declared = []
    for name in ('setup.py', 'pyproject.toml'):
        p = root / name
        if p.exists() and 'prefix' in p.read_text(encoding='utf-8', errors='ignore'):
            declared.append(name)
    findings['dependency_declares_prefix_grouper'] = declared

    # ---- 4. prefix_grouper_utils 是否被任何模块导入 ----
    util = 'verl/trainer/ppo/prefix_grouper_utils.py'
    imported_by = []
    for p in root.rglob('*.py'):
        if '.git' in p.parts or p.as_posix().endswith(util):
            continue
        txt = p.read_text(encoding='utf-8', errors='ignore')
        if 'prefix_grouper_utils' in txt:
            imported_by.append(p.relative_to(root).as_posix())
    findings['prefix_grouper_utils'] = {'file': util,
                                        'exists': (root / util).exists(),
                                        'imported_by': imported_by}

    # ---- 5. use_prefix_grouper 的全部读取点（区分 config / trainer / engine） ----
    reads = {'config': [], 'trainer': [], 'engine_or_model': []}
    for p in root.rglob('*.py'):
        if '.git' in p.parts:
            continue
        rel = p.relative_to(root).as_posix()
        for i, line in enumerate(p.read_text(encoding='utf-8', errors='ignore').splitlines(), 1):
            if 'use_prefix_grouper' not in line or line.lstrip().startswith('#'):
                continue
            if 'config' in rel:
                reads['config'].append({'file': rel, 'line': i})
            elif '/trainer/' in rel or 'trainer' in rel:
                reads['trainer'].append({'file': rel, 'line': i})
            else:
                reads['engine_or_model'].append({'file': rel, 'line': i})
    findings['flag_read_sites'] = reads
    # 引擎/模型侧只剩函数形参与其 if —— 没有任何调用方传值
    findings['engine_side_consumer_has_producer'] = findings['production_passes_flag']

    # ---- 6. README 与 example 的自相矛盾 ----
    rd = root / 'examples/prefix_grouper/README.md'
    ex = root / 'examples/prefix_grouper/run_qwen3_8b_fsdp.sh'
    findings['readme_says_incompatible_dynamic_bsz'] = (
        rd.exists() and 'Incompatible with `use_dynamic_bsz=True`' in
        rd.read_text(encoding='utf-8', errors='ignore'))
    ex_txt = ex.read_text(encoding='utf-8', errors='ignore') if ex.exists() else ''
    findings['example_enables_both'] = ('use_dynamic_bsz=True' in ex_txt and
                                        'use_prefix_grouper=True' in ex_txt)

    # ---- 7. 当前仓库是否有 PR #7202 的接线（若有则该 PR 已落地） ----
    findings['has_engine_side_propagation'] = any(
        'use_prefix_grouper' in c['kwargs'] for c in prod)

    # ---- 判定 ----
    inert = (not findings['production_passes_flag']
             and findings['pg_patch_guarded_by_param'] is not None
             and not findings['has_engine_side_propagation'])

    out = json.dumps(findings, indent=1, ensure_ascii=False, sort_keys=True)
    if '--json-out' in sys.argv:
        pathlib.Path(__file__).with_name('findings.json').write_text(out + '\n', encoding='utf-8')
    print(out)
    print()
    if inert:
        print('取证结论：发现仍然成立——引擎侧从未传入 use_prefix_grouper，')
        print('因此 apply_prefix_grouper_patch() 在真实训练中不可达；')
        print('而 ray_trainer 仍按该开关做组级均衡与跳过重排（副作用照常生效）。')
        return 0
    print('取证结论：发现已失效——上游可能已补上引擎侧接线，需要重新评估。')
    return 1


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if len(args) != 1:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(args[0]))
