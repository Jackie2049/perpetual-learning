#!/usr/bin/env python3
"""扫描"测试能不能失败"：断言计数、无断言文件、被吞掉的断言。

起因：在 rLLM 发现 `tests/tools/test_tools.py` **有 0 个 assert**——它只 `print`，
失败也只打印一行。于是上游一个真实的分派缺陷（`use_async` 自动探测失效）被
"测试通过"掩盖了很久（见 open-source/rllm/tool-use-async-autodetect.md）。

这不是单点问题，而是一类可机械检测的模式。本工具回答三个问题：

  1. 哪些测试文件**一个断言都没有**？
  2. 哪些文件里 `assert` 只出现在**注释或字符串**里（看着有、其实没有）？
  3. 哪些测试用 `print` 当判据（`if ... : print("failed")` 而不 raise/assert）？

只读静态分析，不导入、不运行目标仓库。**不能**判断断言写得对不对——
那仍需人读。

用法：
    python3 scan_test_assertions.py <repo_root>
    python3 scan_test_assertions.py <repo_root> --json
"""

import ast
import json
import pathlib
import sys


def iter_test_files(root):
    for p in root.rglob('test_*.py'):
        parts = set(p.parts)
        if '.git' in parts or 'node_modules' in parts:
            continue
        yield p
    for p in root.rglob('*_test.py'):
        parts = set(p.parts)
        if '.git' in parts or 'node_modules' in parts:
            continue
        yield p


def analyze(p, root):
    try:
        src = p.read_text(encoding='utf-8', errors='ignore')
        tree = ast.parse(src)
    except (SyntaxError, UnicodeDecodeError, OSError):
        return None

    asserts = 0
    test_funcs = 0
    pytest_raises = 0
    print_fail = 0          # if ...: print(...) 且该分支里没有 assert/raise

    for n in ast.walk(tree):
        if isinstance(n, ast.Assert):
            asserts += 1
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith('test'):
            test_funcs += 1
        # self.assertX(...) / pytest.raises(...) / pytest.warns(...) / pytest.fail(...)
        # 注意：pytest.raises 是**上下文管理器**（with 语句），不是普通 Call 判据，
        # 但 ast.walk 仍会遍历到那个 Call，这里统一计数即可。
        if isinstance(n, ast.Call):
            f = n.func
            name = getattr(f, 'attr', None) or getattr(f, 'id', None) or ''
            if name.startswith('assert') or name in ('fail', 'xfail'):
                asserts += 1
            if name in ('raises', 'warns', 'deprecated_call'):
                pytest_raises += 1
                asserts += 1        # 上下文管理器也是判据
        # with pytest.raises(...) as / with self.assertRaises(...)
        if isinstance(n, ast.With):
            for item in n.items:
                ctx = item.context_expr
                if isinstance(ctx, ast.Call):
                    nm = getattr(ctx.func, 'attr', None) or getattr(ctx.func, 'id', None) or ''
                    if nm in ('raises', 'warns', 'assertRaises', 'assertWarns', 'deprecated_call'):
                        asserts += 1
        # 只在有 elif/else 的 if 里 print，且该分支无 assert/raise（近似判断）
        if isinstance(n, ast.If):
            for branch in n.body:
                is_print = (isinstance(branch, ast.Expr)
                            and isinstance(branch.value, ast.Call)
                            and getattr(branch.value.func, 'id', None) == 'print')
                if is_print:
                    has_guard = any(isinstance(x, (ast.Assert, ast.Raise))
                                    for x in ast.walk(n))
                    # 打印内容带 fail 语义但没 assert/raise
                    txt = ast.dump(branch)
                    if not has_guard and ('fail' in txt.lower() or 'error' in txt.lower()):
                        print_fail += 1

    # 是否把校验"委托"出去（如 check_allclose=True / validate=True / verify=...）
    delegated = any(k in src for k in ('check_allclose', 'validate=', 'verify=',
                                       'assert_runs', 'compare', 'expect'))
    # assert 是否只出现在注释/字符串里（AST 计数为 0 但文本里有）
    text_has_assert = ('assert' in src)
    return {
        'file': p.relative_to(root).as_posix(),
        'asserts': asserts,
        'test_funcs': test_funcs,
        'pytest_raises': pytest_raises,
        'print_as_check': print_fail,
        'assert_in_text_only': (asserts == 0 and text_has_assert),
        'may_delegate_check': delegated,
        # 只有"零断言且看不出委托"才算候选；看见了委托的单独列出，不下判决
        'no_assert': (asserts == 0 and not delegated),
        'no_assert_delegate_hint': (asserts == 0 and delegated),
    }


def summarize(rows):
    total = len(rows)
    no_assert = [r for r in rows if r['no_assert']]
    text_only = [r for r in rows if r['assert_in_text_only']]
    print_check = [r for r in rows if r['print_as_check'] > 0]
    return {
        'files': total,
        'files_no_assert': len(no_assert),
        'files_no_assert_but_may_delegate': len(
            [r for r in rows if r['no_assert_delegate_hint']]),
        'files_assert_in_text_only': len(text_only),
        'files_print_as_check': len(print_check),
        'examples_no_assert': [r['file'] for r in sorted(
            no_assert, key=lambda x: -x['test_funcs'])[:5]],
        'examples_print_as_check': [r['file'] for r in print_check[:5]],
        'examples_text_only': [r['file'] for r in text_only[:5]],
    }


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if len(args) != 1:
        print(__doc__)
        return 2
    root = pathlib.Path(args[0])
    if not root.exists():
        print('找不到目录:', root)
        return 2

    rows = []
    for p in iter_test_files(root):
        r = analyze(p, root)
        if r is not None:
            rows.append(r)

    report = {'root': root.name, 'summary': summarize(rows),
              'files': sorted(rows, key=lambda x: (x['asserts'], x['file']))}
    text = json.dumps(report, indent=1, ensure_ascii=False)
    if '--json' in sys.argv:
        (pathlib.Path(__file__).parent / ('scan-%s.json' % root.name)).write_text(
            text + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], indent=1, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
