#!/usr/bin/env python3
"""验证 tiled_mlp 补丁：修复前后，引擎调用 apply_monkey_patch 时传了什么。

不跑 verl、不需要 GPU、不导入 torch。做法是**从上游源码里抽出
`_build_module` 方法体**，把 `apply_monkey_patch` 换成一个记录器，
再用一个只带 `model_config.tiled_mlp` 的假 self 调用它，看记录到什么。

这验证的是"补丁是否真的把配置传下去"，不验证 TiledMLP 本身的正确性或性能。

用法：
    python3 verify_patch.py <unpacked-verl-root>              # 打补丁前
    python3 verify_patch.py <patched-verl-root> --expect-wired  # 打补丁后
"""

import ast
import pathlib
import sys
import textwrap

CALL_SITE = 'verl/workers/engine/fsdp/transformer_impl.py'
METHOD = '_build_module'


def extract_method_source(root):
    """从上游文件里原样取出 `_build_module` 的源码。"""
    src = (pathlib.Path(root) / CALL_SITE).read_text(encoding='utf-8')
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == METHOD:
            body = textwrap.dedent(''.join(lines[node.lineno - 1:node.end_lineno]))
            return body
    raise RuntimeError('找不到 %s' % METHOD)


def collect_apply_kwargs(root):
    """静态读出 apply_monkey_patch 调用点的关键字实参名（不执行代码）。"""
    src = (pathlib.Path(root) / CALL_SITE).read_text(encoding='utf-8')
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and getattr(node.func, 'id', '') == 'apply_monkey_patch'):
            return [k.arg for k in node.keywords if k.arg], node.lineno
    raise RuntimeError('找不到 apply_monkey_patch 调用点')


class _Cfg(dict):
    """够用的配置替身：支持属性访问与 .get。"""
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def get(self, k, d=None):
        return dict.get(self, k, d)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    expect_wired = '--expect-wired' in sys.argv
    if len(args) != 1:
        print(__doc__)
        return 2
    root = pathlib.Path(args[0])

    kwargs, line = collect_apply_kwargs(root)
    wired = 'use_tiled_mlp' in kwargs
    print('调用点 %s:%d' % (CALL_SITE, line))
    print('传的关键字实参:', kwargs)
    print('是否包含 use_tiled_mlp:', wired)

    if expect_wired:
        ok = wired and 'tiled_mlp_shards' in kwargs
        print()
        print('判定:', 'PASS — 配置已接线' if ok else 'FAIL — 仍未接线')
        return 0 if ok else 1

    ok = not wired
    print()
    print('判定:', 'PASS — 确认修复前未接线' if ok else 'FAIL — 修复前就已接线，前提不成立')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
