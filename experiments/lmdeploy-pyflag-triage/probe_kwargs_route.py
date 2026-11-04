#!/usr/bin/env python3
"""正向对照：`disable_fastapi_docs` 这条链路到底通不通。

`verify_findings.py` 证明的是**静态链路**（CLI → serve() → api_server 分支）。
但「launch_server 的具名形参里没有这个键」这句很容易被读成「断在这里」——
本探针把 **launch_server 的 `**kwargs` 收口**、**`_run_server` 的转发**、
**api_server 里 `if disable_fastapi_docs:` 那 7 行的真实源码**分别取出来在
CPU 上跑，看 `serve()` 最终收到什么、FastAPI 三个 URL 是否真的被关掉。

跑法：python3 probe_kwargs_route.py [--repo /tmp/lmdeploy-main] [--selftest]
  exit 0 = 链路通（docs 真被关掉）；exit 2 = 有断点（上游改了形状要重判）。
不 import lmdeploy（它会拉 torch / turbomind）、不启动服务、不装依赖。
"""
import argparse
import ast
import json
import re
import sys
import textwrap
from pathlib import Path


def extract_call_block(src: str, start_marker: str, n_lines: int) -> str:
    i = src.index(start_marker)
    lines = src[i:].splitlines()
    out, depth = [], 0
    for line in lines[:200]:
        out.append(line)
        depth += line.count('(') - line.count(')')
        if depth <= 0 and len(out) >= 1:
            break
    return '\n'.join(out)


def get_launch_server_kwargs_collect(src: str) -> str:
    """取 launch_server 里「收集 kwargs → 交给 _run_server」那一段（真实源码）。"""
    i = src.index('cur_server_kwargs = dict()')
    j = src.index('proc = mp.Process(', i)
    # 取到行首缩进为止，保证整段是同一个缩进层级（textwrap.dedent 才剥得对）
    i = src.rfind('\n', 0, i) + 1
    j = src.rfind('\n', 0, j) + 1
    return src[i:j]


def get_run_server_body(src: str) -> str:
    """取 `_run_server` 里调用 serve 那一行（真实源码）。"""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '_run_server':
            seg = ast.get_source_segment(src, node)
            return '\n'.join(l for l in seg.splitlines() if 'serve(model_path' in l)
    return ''


def get_docs_branch(api_src: str) -> str:
    """取 api_server 里 `if disable_fastapi_docs:` 那一段（真实源码，含 else 分支）。"""
    i = api_src.index('    if disable_fastapi_docs:')
    j = api_src.index('router = create_openai_router', i)
    return textwrap.dedent(api_src[i:j])


class FakeFastAPI:
    """替身：只记下三个 URL 参数。"""
    def __init__(self, docs_url='docs', redoc_url='redoc', openapi_url='openapi_url', lifespan=None):
        self.docs_url, self.redoc_url, self.openapi_url = docs_url, redoc_url, openapi_url
        self.lifespan = lifespan


def run_pipeline(root: Path, cli_value: bool):
    """把三段真实源码串起来跑：kwargs 收集 → _run_server 转发 → api_server 分支。"""
    ls = (root / 'lmdeploy/serve/openai/launch_server.py').read_text(encoding='utf-8')
    api = (root / 'lmdeploy/serve/openai/api_server.py').read_text(encoding='utf-8')

    collect = get_launch_server_kwargs_collect(ls)
    forward = get_run_server_body(ls)
    docs = get_docs_branch(api)

    # CLI 侧真正传下去的东西（从 cli/serve.py 的 launch_server 调用里读实参名）
    cli = (root / 'lmdeploy/cli/serve.py').read_text(encoding='utf-8')
    kw_sent = {m.group(1) for m in
               re.finditer(r'(\w+)=args\.', extract_call_block(cli, 'launch_server(', 30))}

    # ① kwargs 收集段（真实源码）：把 CLI 传进来的值装进 cur_server_kwargs
    ns = {'kwargs': {'disable_fastapi_docs': cli_value, 'log_level': 'ERROR'},
          'server_name': 'h', 'server_port': 1, 'backend_config': 'cfg',
          'backend_config_dp': 'cfg-dp', 'proxy_url': None,
          'http_or_https': 'http', 'server_urls': [], 'gpu_ids_per_dp': [0],
          'logger': type('L', (), {'info': lambda *a, **k: None})()}
    exec(compile(textwrap.dedent(collect), '<collect>', 'exec'), ns)     # noqa: S102
    collected = ns['cur_server_kwargs']

    # ② 转发段（真实源码）：_run_server(gpu_ids, model_path, **kwargs) → serve(model_path, **kwargs)
    calls = []
    ns2 = {'serve': lambda model_path, **kw: calls.append((model_path, kw)),
           'os': __import__('os'), 'cuda_visible_devices': '', 'gpu_ids': [0],
           'model_path': 'MODEL', 'kwargs': dict(collected),
           'logger': type('L', (), {'exception': lambda *a, **k: None})()}
    srv_body = '\n'.join(l for l in forward.splitlines())
    exec(compile(textwrap.dedent(srv_body), '<forward>', 'exec'), ns2)   # noqa: S102
    received = calls[0][1] if calls else {}

    # ③ api_server 分支（真实源码）：看 FastAPI 三个 URL
    ns3 = {'FastAPI': FakeFastAPI, 'disable_fastapi_docs': received.get('disable_fastapi_docs', False),
           'lifespan': 'LIFESPAN'}
    exec(compile(docs, '<docs-branch>', 'exec'), ns3)                    # noqa: S102
    app = ns3['app']

    return {
        'cli_sends_key': 'disable_fastapi_docs' in kw_sent,
        'cli_value': cli_value,
        'collected_value': collected.get('disable_fastapi_docs', '<missing>'),
        'serve_received': received.get('disable_fastapi_docs', '<missing>'),
        'fastapi_urls': {'docs_url': app.docs_url, 'redoc_url': app.redoc_url,
                         'openapi_url': app.openapi_url},
        'docs_disabled': app.docs_url is None and app.redoc_url is None and app.openapi_url is None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='/tmp/lmdeploy-main')
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    root = Path(a.repo)

    on = run_pipeline(root, True)
    off = run_pipeline(root, False)

    checks = [
        ('P1.cli_sends_key', on['cli_sends_key'], 'CLI 的 launch_server 调用里有该实参'),
        ('P2.value_survives_collect', on['collected_value'] is True and off['collected_value'] is False,
         'kwargs 收集段原样保留该值'),
        ('P3.value_reaches_serve', on['serve_received'] is True and off['serve_received'] is False,
         '经 _run_server 转发后 serve() 收到 True / False'),
        ('P4.docs_off_when_true', on['docs_disabled'], f"True 时三处 URL {on['fastapi_urls']}"),
        ('P5.docs_on_when_false',
         off['fastapi_urls']['docs_url'] == '/', f"False 时 docs_url {off['fastapi_urls']['docs_url']!r}"),
    ]

    if a.selftest:
        # 负向对照①：把收集段的 `kwargs` 换成空 dict —— 值必须传不下去
        orig = get_launch_server_kwargs_collect
        globals()['get_launch_server_kwargs_collect'] = lambda s: 'cur_server_kwargs = dict()'
        broken = run_pipeline(root, True)
        globals()['get_launch_server_kwargs_collect'] = orig
        # 负向对照②：把 api 分支里的 disable 分支删掉 —— docs 必须关不掉
        orig2 = get_docs_branch
        globals()['get_docs_branch'] = lambda s: "app = FastAPI(docs_url='/', lifespan=lifespan)"
        nobranch = run_pipeline(root, True)
        globals()['get_docs_branch'] = orig2

        neg = [('N1.broken_collect_detected', broken['serve_received'] == '<missing>'),
               ('N2.no_branch_detected', not nobranch['docs_disabled'])]
        for n, ok in neg:
            checks.append((n, ok, '负向对照'))
        red = [n for n, ok, _ in checks if not ok]
        for n, ok, d in checks:
            print(f'[{"ok" if ok else "XX"}] {n}: {d}')
        print(f'selftest: {len(checks) - len(red)}/{len(checks)}（负向对照必须让对应判据变红）')
        return 1 if red else 0

    for n, ok, d in checks:
        print(f'[{"ok" if ok else "XX"}] {n}: {d}')
    red = [n for n, ok, _ in checks if not ok]
    print(f'\nverdict: {"WIRED" if not red else "BROKEN"}'
          f'（True → {on["fastapi_urls"]}；False → {off["fastapi_urls"]}）')
    if a.json:
        out = Path(__file__).with_name('kwargs-route.json')
        out.write_text(json.dumps({'on': on, 'off': off,
                                   'checks': [{'name': n, 'ok': ok, 'detail': d} for n, ok, d in checks]},
                                  ensure_ascii=False, indent=1), encoding='utf-8')
        print('wrote', out)
    return 0 if not red else 2


if __name__ == '__main__':
    sys.exit(main())
