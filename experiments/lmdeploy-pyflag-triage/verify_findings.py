#!/usr/bin/env python3
"""LMDeploy pyflag 5 条 silently_inert 的逐条取证。

用法：
    python3 verify_findings.py [--repo /tmp/lmdeploy-main] [--json] [--selftest]

三条纪律（前两批踩过，见 tools/pyflag/readme.md 第七类与「自检的第二个坑」）：
  1. **逐行分类**，不用 `text.count(name)` 判「某处有没有传」——同一个名字会出现在
     def 行、docstring、`if flag:` 分支和告警文案里。
  2. **每条判定由具名 check 得出**，check 失败只影响自己那条判定。
  3. **自检注入必须让指定那条 check 变红**；红的是别的 check 也算失败，并在输出里打印红的名单。
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

COMMIT = '3228395546dfb2f6427faf39ec0c0c285f4c378a'
COMMIT_DATE = '2026-11-03T08:30:15Z'


# ---------------------------------------------------------------- 源码工具

def read(root: Path, rel: str) -> str:
    return (root / rel).read_text(encoding='utf-8')


def strip_comment_and_docstring(src: str) -> str:
    """去掉三引号块与 # 注释（行号保持不变：块内按行数替换成空行）。

    注意顺序：**先剥注释再剥三引号**。反过来的话 docstring 末尾那行三引号里的井号
    被当成注释起点、截掉整行，三引号就配不上对了——第一版踩过。
    """
    out = []
    for line in src.splitlines():
        idx = line.find('#')
        out.append(line[:idx] if idx >= 0 else line)
    src = '\n'.join(out)

    def blank(m):
        return '\n' * m.group(0).count('\n')
    src = re.sub(r'"""[\s\S]*?"""', blank, src)
    src = re.sub(r"'''[\s\S]*?'''", blank, src)
    return src


def code_lines(src: str):
    return list(enumerate(strip_comment_and_docstring(src).splitlines(), 1))


def calls_to(src: str, name: str):
    """`name(...)` 的调用点行（跳过 `def name(` 与 `class name(`）。"""
    hits = []
    for i, line in code_lines(src):
        for m in re.finditer(r'(?<![\w.])' + re.escape(name) + r'\s*\(', line):
            head = line[:m.start()].rstrip()
            if not head.endswith('def') and not re.search(r'\bclass\s*$', head):
                hits.append((i, line))
                break
    return hits


def call_body(src: str, lineno: int, max_lines: int = 80) -> str:
    """从调用点行开始，按括号配平截出整段调用（含多行实参）。"""
    lines = src.splitlines()
    depth, body, started = 0, [], False
    for line in lines[lineno - 1: lineno - 1 + max_lines]:
        body.append(line)
        for ch in line:
            if ch in '([{':
                depth += 1
                started = True
            elif ch in ')]}':
                depth -= 1
        if started and depth <= 0:
            break
    return '\n'.join(body)


def keyword_args(src: str, hits):
    """调用点里的关键字实参名（**必须按括号配平取整段**：本项目这些调用是多行的，
    只看首行会读到空集——第一版就是这个错，A4/A5 当场红）。"""
    args = set()
    for lineno, _ in hits:
        body = call_body(src, lineno)
        args.update(m.group(1) for m in re.finditer(r'(?<![\w.])(\w+)\s*=(?!=)', body))
    return args


def assignment_lines(src: str, name: str):
    """`name = ...` 形式的**真赋值**行号。

    要同时排除两种长得一模一样的写法（第七类易错点）：
      * `def f(name=...)` / `g(name=...)` —— 关键字实参或形参默认值；
      * `name == ...` / `!=` / `<=` —— 比较。
    判据：名字前面那个非空白字符不能是 `(` 或 `,`（那就是实参/形参列表里），
    `=` 后面也不能紧跟 `=`。
    """
    hits = []
    for i, line in code_lines(src):
        for m in re.finditer(r'(?<![\w.=!<>])' + re.escape(name) + r'\s*=(?!=)', line):
            before = line[:m.start()].rstrip()
            # 实参/形参列表里的 `x=...` 三种形状，全部排除：
            #   ① 本行前面已有 `(`（`f(a, x=1)`）或 `,`（`f(\n  a,\n  x=1)`）；
            #   ② 只有缩进、上一个非空行以 `,`/`(` 结尾（多行调用续行，`:123` 这种）；
            # 顶层顶格赋值（`:15`）不受①②影响——第一版两条都写宽了，把 `:15` 一起误杀。
            prev = [l for l in src.splitlines()[:i - 1] if l.strip()]
            if '(' in before or ',' in before:
                continue
            if not before and prev and prev[-1].rstrip().endswith((',', '(')):
                continue
            hits.append(i)
    return hits


def aliases_in(src: str, modname: str, name: str):
    """`from ... import x as y` 形式的别名（调用点写的是别名，按原名找会漏）。"""
    pat = r'from\s+[\w.]*' + re.escape(modname) + r'[\w.]*\s+import\s+[\w, ]*\b' + re.escape(name) + r'\s+as\s+(\w+)'
    return re.findall(pat, src)


def grep(root: Path, pattern: str, include='*.py', fixed=False):
    cmd = ['grep', '-rn']
    if fixed:
        cmd.append('-F')
    cmd += [pattern]
    if include:
        cmd += ['--include=' + include]
    cmd += ['.']
    out = subprocess.run(cmd, cwd=root, capture_output=True, text=True).stdout
    return [l for l in out.splitlines() if l.strip()]


def files_of(lines):
    return {l.split(':', 1)[0][2:] for l in lines}


def signature(src: str, fname: str):
    """取 `def fname(` 到匹配右括号的签名文本（含形参）。"""
    m = re.search(r'^def ' + re.escape(fname) + r'\(', src, re.M)
    if not m:
        return ''
    i, depth = m.end() - 1, 0
    while i < len(src):
        if src[i] == '(':
            depth += 1
        elif src[i] == ')':
            depth -= 1
            if depth == 0:
                return src[m.end():i]
        i += 1
    return ''


def params_of(sig: str):
    """把签名按顶层逗号切开，取形参名（去掉默认值与注解）。"""
    parts, depth, cur = [], 0, ''
    for ch in sig:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(cur)
            cur = ''
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    names = []
    for p in parts:
        p = p.strip().lstrip('*')
        if not p:
            continue
        names.append(re.split(r'\s*[:=]', p, 1)[0].strip())
    return names


# ---------------------------------------------------------------- 判定框架

class Checker:
    def __init__(self):
        self.results = []

    def check(self, name, cond, detail=''):
        self.results.append((name, bool(cond), detail))
        return bool(cond)

    def failed(self):
        return [n for n, ok, _ in self.results if not ok]


JUDGEMENTS = {}


def judgement(flag):
    def deco(fn):
        JUDGEMENTS[flag] = fn
        return fn
    return deco


# ---------------------------------------------------------------- A. disable_fastapi_docs

@judgement('disable_fastapi_docs')
def j_disable_fastapi_docs(root: Path, c: Checker):
    """CLI `--disable-fastapi-docs` 的穿透链路：
       cli/serve.py:354/388 ──serve() / launch_server()──► launch_server:114 `**kwargs`
                                                          ──_run_server:66──► serve()  (route A)
       cli/serve.py:354 ─────────────────── serve() 直调 (:274 形参)                (route B)
    B 通、A 也通 ⇒ 「launch_server 的形参里没有它」写不成缺陷。"""
    cli = read(root, 'lmdeploy/cli/serve.py')
    api = read(root, 'lmdeploy/serve/openai/api_server.py')
    lsv = read(root, 'lmdeploy/serve/openai/launch_server.py')
    util = read(root, 'lmdeploy/cli/utils.py')

    api_sig = signature(api, 'serve')
    lsv_sig = signature(lsv, 'launch_server')
    api_params = params_of(api_sig)
    lsv_params = params_of(lsv_sig)

    # 消费者真的在（形参 + 按它分支）
    c.check('A1.api_param', 'disable_fastapi_docs' in api_params, 'api_server.serve() 形参表含该键')
    c.check('A2.api_consumes',
            re.search(r'if disable_fastapi_docs:\s*\n\s*app = FastAPI\(docs_url=None,\s*\n\s*redoc_url=None,\s*\n\s*openapi_url=None',
                      api) is not None,
            'api_server:389 True 分支把三处 URL 置 None')
    # 负向对照：False 分支必须**开着** docs（否则「能关」这件事本身不成立）
    c.check('A2c.negative_else_branch', "app = FastAPI(docs_url='/', lifespan=lifespan)" in api,
            '负向对照：False 分支 docs_url=\'/\'')

    # CLI 侧：flag 定义 + 两处调用点都传
    c.check('A3.cli_flag',
            re.search(r"add_argument\('--disable-fastapi-docs',\s*\n?\s*action='store_true'", util) is not None,
            'cli/utils.py 定义 store_true')
    # 非 DP 那条路是 `from lmdeploy.serve.openai.api_server import serve as run_api_server`
    # 的**别名调用**——按原名 serve 找会读到 0 处（工具第一版就是这个错，第五类易错点的别名形态）
    aliases = aliases_in(cli, 'api_server', 'serve')
    api_hits = [(n, l) for a in aliases for n, l in calls_to(cli, a)]
    launch_hits = calls_to(cli, 'launch_server')
    c.check('A4.alias_call_site',
            aliases == ['run_api_server'] and len(api_hits) == 1,
            f'别名 {aliases}，调用点 {len(api_hits)} 处')
    c.check('A5.launch_call_site', len(launch_hits) == 1, f'launch_server() {len(launch_hits)} 处')
    # 关键字必须按括号配平取整段（这些调用都是多行）。
    # 「传了」与「别名找得到」是两件事、分成两个名字分开的 check —— 第一版把两者写进
    # 同一个 check，注入改坏别名后它跟着变红，selftest 报「红错地方」。别名本身由 A4 的
    # `aliases == [...]` 单独钉住。
    c.check('A6.api_call_passes_keyword', 'disable_fastapi_docs' in keyword_args(cli, api_hits),
            '非 DP 路由（别名调用）写了 disable_fastapi_docs=args.disable_fastapi_docs')
    c.check('A6b.launch_call_passes_keyword', 'disable_fastapi_docs' in keyword_args(cli, launch_hits),
            'DP 路由（launch_server）写了 disable_fastapi_docs=args.disable_fastapi_docs')
    # 签名解析器自证：要能看见 launch_server 里确实有的键——**但绝不能要求看见
    # `disable_fastapi_docs` 自己**（检查器只证明它能看见「存在的形参」，不能要求它
    # 看见「本来就该不存在的形参」，否则 A7 与 A6c 互相矛盾——selftest 第二条注入当场
    # 就是被这个矛盾判红的）。所以这里只认「首参数字段名」，不认 `**kwargs`。
    c.check('A6c.sig_parse_sanity', {'num_nodes', 'model_path', 'backend_config'} <= set(lsv_params),
            f'同法解析 launch_server 形参得 {lsv_params} ⇒ 不是解析器瞎了')

    # 关键：launch_server 形参里没有它，但有 **kwargs —— 这是「没断」的原因
    c.check('A7.launch_server_lacks_named_param', 'disable_fastapi_docs' not in lsv_params,
            'launch_server 具名形参**不含**该键（期望 False）')
    c.check('A8.launch_server_has_kwargs', '**kwargs' in lsv_sig, 'launch_server 签名含 **kwargs')
    c.check('A9.kwargs_forwarded_to_serve',
            re.search(r'cur_server_kwargs\.update\(kwargs\)', lsv) is not None and
            re.search(r'def _run_server\([\s\S]{0,200}?\*\*kwargs\):[\s\S]{0,400}?serve\(model_path, \*\*kwargs\)', lsv) is not None,
            'kwargs → cur_server_kwargs → _run_server(**kwargs) → serve(model_path, **kwargs)')

    return ('REACHABLE_BOTH_ROUTES',
            'DP 模式下 launch_server 用 **kwargs 把 CLI 值转发回 serve()；'
            '非 DP 模式 CLI 直调 serve()（形参直收）⇒ 两条路由都通，不是缺陷')


# ---------------------------------------------------------------- B. use_fa3

@judgement('use_fa3')
def j_use_fa3(root: Path, c: Checker):
    """`use_fa3` 是**模块级全局**（import 时探测 SM/CUDA 决定），不是用户开关。
    pyflag 把 `require_fa3_for_speculative_decoding()` / `use_fa3_warning()` 读全局当成
    「消费者」，而消费者不在「形参」上，所以 producers 必空——这是工具的对象选错。"""
    att = 'lmdeploy/pytorch/backends/cuda/attention/__init__.py'
    src = read(root, att)

    c.check('B1.module_level_assign', re.search(r'^use_fa3 = False$', src, re.M) is not None,
            '模块级 use_fa3 = False')
    c.check('B2.set_by_probe',
            'use_fa3 = True' in src and 'get_device_capability' in src and 'torch.version.cuda' in src,
            '探测 SM80+ 且 CUDA>=12.3 时置 True')
    # 消费者是「无参函数读全局」——不是形参
    c.check('B3.consumers_take_global',
            re.search(r'def require_fa3_for_speculative_decoding\(\)', src) is not None and
            re.search(r'def use_fa3_warning\(\)', src) is not None,
            '两个消费者的形参表为空 ⇒ 只能读全局')
    # 形参版本的 use_fa3 在 FlashMLAImpl，调用点在本文件 :116/123/128 传的是同一个全局
    c.check('B4.param_version_produced',
            len(re.findall(r'use_fa3=use_fa3', src)) >= 3,
            f'FlashMLAImpl(use_fa3=use_fa3) 共 {len(re.findall(r"use_fa3=use_fa3", src))} 处（有生产者）')
    # 负向对照：没有命令行入口 / setter / 环境变量
    cli_hits = grep(root, 'use-fa3')
    c.check('B4c.negative_no_cli', not cli_hits, '负向对照：无 --use-fa3 命令行入口')
    # 赋值点只能是「模块初值 + 探测赋值」这两处（其它出现都是 `use_fa3=use_fa3` 传参，
    # 前面那个 `(?<![\w=])` 排除了关键字实参——`def f(use_fa3=...)` 的 `=` 属于调用/声明
    # 语法而不是赋值，第七类易错点在这里又冒了一次头）
    # 只看**模块顶层**的赋值（缩进为 0）：`:116/123/128` 那几处是 `use_fa3=use_fa3`
    # 关键字实参，长得像赋值但语义是传参——第七类易错点在这条上第二次冒头（第一版把
    # 缩进块里的探测赋值也要求成顶层，判据直接自相矛盾）。
    # 两条独立断言：① 模块顶层只有 :15 一处（探测那次在 try 缩进块内）；② 全文件除了
    # 这两处没有第三个真赋值点。`preview_use_fa3 = True` 这种改名会让②变红（真赋值点少一个）。
    module_level = [i for i in assignment_lines(src, 'use_fa3')
                    if not src.splitlines()[i - 1].startswith((' ', '\t'))]
    c.check('B5.module_level_only_initial', module_level == [15],
            f'模块顶层赋值行号 {module_level}（应为 :15）')
    c.check('B5b.exactly_two_assignments', assignment_lines(src, 'use_fa3') == [15, 22],
            f'全文件真赋值点 {assignment_lines(src, "use_fa3")}（应为 :15 初值 + :22 探测）')
    c.check('B5c.no_env_entry', 'environ' not in src, '无 os.environ 入口')
    return ('MODULE_GLOBAL_PROBE', 'import 期硬件探测结果，非用户开关；形参版本那处有生产者')


# ---------------------------------------------------------------- C. use_proxy

@judgement('use_proxy')
def j_use_proxy(root: Path, c: Checker):
    """唯一消费者与两个生产者都在 `autotest/`（扫描器按路径排除），lmdeploy/ 包内 0 处。"""
    ap = read(root, 'autotest/utils/pytest_layout_utils.py')
    c.check('C1.consumer_param', 'use_proxy: bool = False' in ap, '形参在 autotest/utils')
    c.check('C2.consumer_branches', 'if use_proxy:' in ap, '按它分支')

    hits = grep(root, 'use_proxy')
    c.check('C3.files', files_of(hits) <= {'autotest/utils/pytest_layout_utils.py',
                                           'autotest/evaluate/test_api_evaluate.py'},
            f'全仓只出现在 {sorted(files_of(hits))}')
    prod = [l for l in hits if 'use_proxy=True' in l]
    c.check('C4.producers_in_tests', len(prod) == 2 and all('autotest/' in l for l in prod),
            f'{len(prod)} 个生产者全在 autotest')
    # 负向对照：包内必须 0 处（同一条 grep 换个谓语，确认不是「哪都没扫到」）
    pkg = [l for l in hits if l.startswith('./lmdeploy/')]
    c.check('C4c.negative_package_empty', not pkg, '负向对照：lmdeploy/ 包内 0 处')
    return ('TEST_ONLY', '消费者与生产者都在 autotest，用户面为零')


# ---------------------------------------------------------------- D. use_tqdm

@judgement('use_tqdm')
def j_use_tqdm(root: Path, c: Checker):
    """`Pipeline.infer(use_tqdm=False)`：仓内生产者只有 benchmark/ 两处，
    但 `Pipeline.__call__` 把 **kwargs 透传给 infer ⇒ 用户 `pipe(prompts, use_tqdm=True)` 可达。"""
    pipe = read(root, 'lmdeploy/pipeline.py')
    c.check('D1.param_default_false', re.search(r'use_tqdm:\s*bool\s*=\s*False', pipe) is not None, '默认 False')
    c.check('D2.consumes_as_pbar',
            re.search(r'pbar = tqdm\.tqdm\(total=len\(prompts\)\) if use_tqdm else None', pipe) is not None,
            'True ⇒ 建 tqdm 进度条')
    c.check('D3.pbar_reaches_infer_loop', 'pbar=pbar' in pipe, 'pbar 真的进了 _infer')

    hits = grep(root, 'use_tqdm')
    c.check('D4.package_definition_only', files_of(hits) >= {'lmdeploy/pipeline.py'} and
            {f for f in files_of(hits) if f.startswith('lmdeploy/')} == {'lmdeploy/pipeline.py'},
            f'lmdeploy/ 内只有 pipeline.py：{sorted(f for f in files_of(hits) if f.startswith("lmdeploy/"))}')
    prod = [l for l in hits if 'use_tqdm=True' in l]
    c.check('D5.producers_in_benchmark', len(prod) == 2 and all('benchmark/' in l for l in prod),
            f'{len(prod)} 个仓内生产者在 benchmark/')
    # 关键：可达性来自 __call__ 的 kwargs 透传
    c.check('D6.dunder_call_forwards_kwargs',
            re.search(r'def __call__\(self,[\s\S]{0,200}?\*\*kwargs\):\s*\n\s*return self\.infer\([\s\S]{0,80}?\*\*kwargs\)',
                      pipe) is not None,
            'Pipeline.__call__ → self.infer(..., **kwargs)')
    # 负向对照：全文件只这一处形参声明（否则「唯一消费者」是假的）
    c.check('D6c.negative_single_param', pipe.count('use_tqdm: bool') == 1, '负向对照：形参声明全文件 1 处')
    c.check('D7.no_cli_entry', not grep(root, 'use-tqdm', include='*.py'),
            '没有 --use-tqdm 命令行入口（只能作为 Python API 关键字）')
    return ('LIBRARY_PARAM_REACHABLE_VIA_CALL',
            '仓内生产者只在 benchmark；但 Pipeline(...)(..., use_tqdm=True) 经 __call__ 透传可达')


# ---------------------------------------------------------------- E. with_version

@judgement('with_version')
def j_with_version(root: Path, c: Checker):
    """`setup.py` 的内部 helper；两个调用点都在同文件、走默认/位置参数。"""
    s = read(root, 'setup.py')
    c.check('E1.def_in_setup', 'def parse_requirements(fname=' in s, '定义在 setup.py')
    c.check('E2.consumes_branch', 'if with_version and' in s, '函数体按它分支')

    hits = grep(root, 'with_version', include=None)
    c.check('E3.only_setup_py', files_of(hits) == {'setup.py'}, f'全仓只出现在 {sorted(files_of(hits))}')
    # 负向对照：同一条 grep 在 include=None 下能扫出别的文件（证明不是只扫了 setup.py）
    c.check('E3c.negative_grep_alive', len(files_of(grep(root, 'parse_requirements', include=None))) >= 1,
            '负向对照：宽 grep 能找到 parse_requirements 的其他出现处')
    c.check('E4.docstring_default_false',
            re.search(r'with_version \(bool, default=False\)', s) is not None,
            'docstring 写 default=False')
    c.check('E5.code_default_true',
            "def parse_requirements(fname='requirements.txt', with_version=True)" in s,
            '真实默认 True ⇒ docstring 与代码不一致（但无外部面）')
    return ('SETUP_INTERNAL', 'setup.py 内部 helper，无用户面；docstring 与默认值不一致只是文档问题')


# ---------------------------------------------------------------- 主流程

ORDER = ['disable_fastapi_docs', 'use_fa3', 'use_proxy', 'use_tqdm', 'with_version']


def run_all(root: Path):
    out = {}
    for flag in ORDER:
        c = Checker()
        try:
            verdict, why = JUDGEMENTS[flag](root, c)
        except Exception as e:                                       # noqa: BLE001
            verdict, why = 'ERROR', repr(e)
        out[flag] = {'verdict': verdict, 'why': why,
                     'checks': [{'name': n, 'ok': ok, 'detail': d} for n, ok, d in c.results],
                     'failed': c.failed()}
    return out


# (flag, 文件, 原文, 替换, 期望变红的 check)
# 纪律：原文必须在被测树里**恰好命中 1 次**（多行实参块的锚点要带足够上下文），
# 且注入只准让「期望那条」变红；红错地方也算失败，输出里会打印红的名单。
SELFTEST_CASES = [
    # A. kwargs 转发断掉 → A9
    ('disable_fastapi_docs', 'lmdeploy/serve/openai/launch_server.py',
     'def _run_server', 'def _inert_server', 'A9.kwargs_forwarded_to_serve'),
    # A. kwargs 兜底拿掉 → A8
    ('disable_fastapi_docs', 'lmdeploy/serve/openai/launch_server.py',
     '                  **kwargs):\n    """Run multiple server processes in dp mode."""',
     '):\n    """Run multiple server processes in dp mode."""', 'A8.launch_server_has_kwargs'),
    # A. 别名改名 → A4。注意 A6 会**连带**变红：它按别名定位调用点，别名没了自然找不到。
    #    这不是「红错地方」，是同一个事实的两个切面，所以这里允许 A4+A6 一起红
    #    （other= 白名单，非白名的连带变色仍然算失败）。
    ('disable_fastapi_docs', 'lmdeploy/cli/serve.py',
     'import serve as run_api_server', 'import serve as run_api_server_X', 'A4.alias_call_site',
     ('A6.api_call_passes_keyword',)),
    # B. 探测赋值改名 → B5
    ('use_fa3', 'lmdeploy/pytorch/backends/cuda/attention/__init__.py',
     'use_fa3 = True', 'preview_use_fa3 = True', 'B5b.exactly_two_assignments'),
    # B. 写进 os.environ → B5c
    ('use_fa3', 'lmdeploy/pytorch/backends/cuda/attention/__init__.py',
     "logger.debug('For higher performance, please install FlashAttention-3 '",
     "os.environ['FA3'] = '1'\n    logger.debug('For higher performance, please install FlashAttention-3 '",
     'B5c.no_env_entry'),
    # C. 去掉一个测试侧生产者 → C4
    ('use_proxy', 'autotest/evaluate/test_api_evaluate.py',
     "eval_subpath='longtext',\n        test_types=('infer',),\n        use_proxy=True,",
     "eval_subpath='longtext',\n        test_types=('infer',),", 'C4.producers_in_tests'),
    # D. 去掉 __call__ 的 kwargs 透传 → D6
    ('use_tqdm', 'lmdeploy/pipeline.py',
     'return self.infer(prompts, gen_config=gen_config, **kwargs)',
     'return self.infer(prompts, gen_config=gen_config)', 'D6.dunder_call_forwards_kwargs'),
    # E. setup.py 默认值改掉 → E5
    ('with_version', 'setup.py',
     'with_version=True', 'with_version=False', 'E5.code_default_true'),
]


def selftest(root: Path):
    bad = []
    for case in SELFTEST_CASES:
        flag, rel, old, new, expect = case[:5]
        other = set(case[5]) if len(case) > 5 else set()
        tmp = Path(tempfile.mkdtemp())
        dst = tmp / 'repo'
        shutil.copytree(root, dst, ignore=shutil.ignore_patterns('.git'))
        f = dst / rel
        src = f.read_text(encoding='utf-8')
        if src.count(old) != 1:
            bad.append(f'{flag}: 注入锚点命中 {src.count(old)} 次（要求恰好 1）：{old[:50]!r}')
            shutil.rmtree(tmp)
            continue
        f.write_text(src.replace(old, new, 1), encoding='utf-8')
        red = run_all(dst)[flag]['failed']
        if expect not in red:
            bad.append(f'{flag}: 注入后 {expect} 没红，红的是 {red}')
        elif [r for r in red if r != expect and r not in other]:
            bad.append(f'{flag}: 注入后除 {expect} 外还红了 {[r for r in red if r != expect and r not in other]}')
        shutil.rmtree(tmp)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='/tmp/lmdeploy-main')
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    root = Path(a.repo)

    if a.selftest:
        bad = selftest(root)
        for b in bad:
            print('SELFTEST FAIL', b)
        okn = len(SELFTEST_CASES) - len([b for b in bad])
        print(f'selftest: {okn}/{len(SELFTEST_CASES)} 注入让目标判据变红（红错地方也算失败）')
        return 1 if bad else 0

    res = run_all(root)
    total = sum(len(v['checks']) for v in res.values())
    failed = [(f, n) for f, v in res.items() for n in v['failed']]
    print(f'commit {COMMIT} ({COMMIT_DATE})')
    for flag in ORDER:
        v = res[flag]
        n_ok = sum(1 for c in v['checks'] if c['ok'])
        print(f'\n## {flag}  →  {v["verdict"]}   ({n_ok}/{len(v["checks"])})')
        print(f'   {v["why"]}')
        for c in v['checks']:
            print(f'   [{"ok" if c["ok"] else "XX"}] {c["name"]}: {c["detail"]}')
    print(f'\ntotal {total - len(failed)}/{total} checks passed, {len(failed)} failed')
    if a.json:
        out = Path(__file__).with_name('findings.json')
        out.write_text(json.dumps({'commit': COMMIT, 'commit_date': COMMIT_DATE,
                                   'findings': res}, ensure_ascii=False, indent=1), encoding='utf-8')
        print('wrote', out)
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
