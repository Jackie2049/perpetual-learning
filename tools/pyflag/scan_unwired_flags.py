#!/usr/bin/env python3
"""通用扫描：配置项"有没有生产者"——即某个布尔开关是否真的能改变行为。

起因：在 verl 上发现 `use_prefix_grouper=True` 在 FSDP 引擎里**没有任何生产
调用方传值**，因而功能静默失效，但它仍会改变 trainer 侧的批划分（见
open-source/verl/evidence/prefix-grouper-inert/）。这不是单点问题，而是一类
可机械检测的模式，所以把方法抽出来，对多个框架跑一遍。

判定逻辑（分三步，每步给出可核对的行号）：

  1. **读取点**：哪些文件读取了这个开关（`get("x", ...)` / `getattr(..., "x", ...)`）。
  2. **消费者**：哪些函数把该开关当作**形参**接收，并在体内按它分支。
  3. **生产者**：哪些调用点真的把该开关**传进去**（关键字实参，或位置实参）。

若"消费者存在 > 0"但"生产者 == 0"，该开关在本快照里即为**静默失效**：
它的分支不可达，但读取点（尤其 trainer 侧）可能仍有其它副作用。

只读静态分析，不导入、不运行目标框架。

用法：
    python3 scan_unwired_flags.py <repo_root> [--json]

判定边界见文件末尾 main() 的输出说明。
"""

import ast
import json
import pathlib
import sys
from collections import defaultdict

# 方法调用外面套的一层包装（Ray remote 等），剥掉后才是真正的消费者方法名
RAY_WRAPPERS = {'remote', 'options', 'run_async', 'run', 'submit'}


TEST_DIRS = {'tests', 'test', 'testing', 'examples', 'example', 'docs', 'benchmarks'}


def is_test_path(rel):
    parts = rel.split('/')
    if any(x in TEST_DIRS for x in parts[:-1]):
        return True
    name = parts[-1]
    return name.startswith('test_') or name.endswith('_test.py') or name == 'conftest.py'


def iter_py(root):
    for p in root.rglob('*.py'):
        parts = set(p.parts)
        if '.git' in parts or 'node_modules' in parts:
            continue
        yield p


def parse(p):
    try:
        return ast.parse(p.read_text(encoding='utf-8', errors='ignore'))
    except (SyntaxError, UnicodeDecodeError):
        return None


def collect_flag_reads(root, flag):
    """第 1 步：哪些文件读取该开关。"""
    hits = []
    for p in iter_py(root):
        rel = p.relative_to(root).as_posix()
        try:
            lines = p.read_text(encoding='utf-8', errors='ignore').splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            if flag in s:
                hits.append({'file': rel, 'line': i})
    return hits


def collect_consumers(root, flag):
    """第 2 步：哪些函数把该开关当形参、并在体内按它分支。"""
    out = []
    for p in iter_py(root):
        tree = parse(p)
        if tree is None:
            continue
        rel = p.relative_to(root).as_posix()
        _tag_parents(tree)
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            params = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
            if flag not in params:
                continue
            # 体内是否真的按它分支（if / 三目 / 布尔运算）
            # 只认真正的**形参**读取：若函数体内给同名名字重新赋值（如
            # `use_ray = getattr(...)`），它就不再是这个形参了，不算分支。
            rebound = {t.id for n in ast.walk(fn) if isinstance(n, ast.Assign)
                       for t in n.targets if isinstance(t, ast.Name)}
            branches = 0
            if flag not in rebound:
                for node in ast.walk(fn):
                    if isinstance(node, (ast.If, ast.IfExp)):
                        for sub in ast.walk(node.test):
                            if isinstance(sub, ast.Name) and sub.id == flag:
                                branches += 1
            cls = _enclosing_class(fn)
            # 构造器的调用点写的是**类名**，不是 __init__，故用类名参与匹配
            call_name = cls if (fn.name == '__init__' and cls) else fn.name
            out.append({'file': rel, 'func': fn.name, 'call_name': call_name,
                        'line': fn.lineno, 'module': rel[:-3].replace('/', '.'),
                        'cls': cls, 'branches_on_flag': branches})
    return out


def _enclosing_class(node):
    """返回该函数所属类名（若有）。用于收紧消费者匹配。"""
    for parent in getattr(node, '_parents', []):
        if isinstance(parent, ast.ClassDef):
            return parent.name
    return None


def _tag_parents(tree):
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child._parents = getattr(parent, '_parents', []) + [parent]


def collect_producers(root, flag, consumers):
    """第 3 步：哪些调用点把该开关传给了消费者函数。

    支持两种形式：关键字实参 `f(..., flag=True)`，
    以及位置实参（需要消费者签名里该形参的位置）。
    """
    sig_by_name = defaultdict(list)     # func name -> [param list]
    for p in iter_py(root):
        tree = parse(p)
        if tree is None:
            continue
        _tag_parents(tree)
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                sig_by_name[fn.name].append(
                    [a.arg for a in fn.args.args + fn.args.kwonlyargs])

    consumer_names = {c.get('call_name') or c['func'] for c in consumers}
    out = []
    for p in iter_py(root):
        tree = parse(p)
        if tree is None:
            continue
        rel = p.relative_to(root).as_posix()
        mod = rel[:-3].replace('/', '.')
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, 'id', None) or getattr(node.func, 'attr', None)
            # Ray / 异步包装：engine.init_process_group.remote(...) -> 取被包装的方法名
            if isinstance(node.func, ast.Attribute) and name in RAY_WRAPPERS \
                    and isinstance(node.func.value, ast.Attribute):
                name = node.func.value.attr
            if name not in consumer_names:
                continue
            # 不做模块限定：跨模块导入（train_sft.py -> models/actor.py）是常态。
            # 匹配放宽为「任一同名函数的签名里含该形参」，宁可放过可疑项也不误报。
            kw = [k.arg for k in node.keywords if k.arg]
            how = None
            if flag in kw:
                how = 'keyword'
            else:
                # 位置实参：该形参是否落在调用方给出的位置参数范围内
                for sig in sig_by_name.get(name, []):
                    if flag in sig:
                        idx = sig.index(flag)
                        if idx < len(node.args):
                            how = 'positional'
                            break
            if how:
                out.append({'file': rel, 'line': node.lineno, 'how': how})
    return out


def scan(root, flag, exclude_tests=True):
    reads = collect_flag_reads(root, flag)
    consumers = collect_consumers(root, flag)
    producers = collect_producers(root, flag, consumers)

    if exclude_tests:
        reads_nontest = [r for r in reads if not r['file'].startswith('tests/')
                         and '/tests/' not in r['file']]
        prod_nontest = [p for p in producers if not p['file'].startswith('tests/')
                        and '/tests/' not in p['file']]
        cons_nontest = [c for c in consumers if not c['file'].startswith('tests/')
                        and '/tests/' not in c['file']]
    else:
        reads_nontest, prod_nontest, cons_nontest = reads, producers, consumers

    decides = [c for c in cons_nontest if c['branches_on_flag'] > 0]
    return {
        'flag': flag,
        'reads_total': len(reads),
        'reads_nontest': len(reads_nontest),
        'consumers_nontest': cons_nontest,
        'producers_total': len(producers),
        'producers_nontest': prod_nontest,
        'silently_inert': (len(decides) > 0 and len(prod_nontest) == 0),
        'deciding_consumers': decides,
    }


def discover_flags(root):
    """自动发现布尔开关：形参带 `= False/True` 默认值、且名字以 use_/enable_ 开头的。"""
    found = set()
    for p in iter_py(root):
        tree = parse(p)
        if tree is None:
            continue
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            pos = list(fn.args.args) + list(fn.args.kwonlyargs)
            defaults = [None] * (len(fn.args.args) - len(fn.args.defaults)) + list(fn.args.defaults)
            for i, a in enumerate(fn.args.args):
                if i >= len(defaults):
                    continue
                d = defaults[i]
                if isinstance(d, ast.Constant) and isinstance(d.value, bool):
                    if a.arg.startswith(('use_', 'enable_', 'disable_', 'no_', 'with_')):
                        found.add(a.arg)
            for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
                if isinstance(d, ast.Constant) and isinstance(d.value, bool):
                    if a.arg.startswith(('use_', 'enable_', 'disable_', 'no_', 'with_')):
                        found.add(a.arg)
    return found


_sentinel = object()


def scan_many(root, flags):
    """一次解析、多个 flag 共用 AST（大仓库上按 flag 重解析会 O(n) 次，太慢）。"""
    trees = []
    for p in iter_py(root):
        t = parse(p)
        if t is not None:
            trees.append((p.relative_to(root).as_posix(), t))
    text_by_file = {}
    for rel, _ in trees:
        try:
            text_by_file[rel] = (root / rel).read_text(
                encoding='utf-8', errors='ignore').splitlines()
        except OSError:
            text_by_file[rel] = []

    defs = []          # (rel, funcname, call_name, paramlist, branches_set, line)
    calls = []         # (rel, callee_name, kwargs, n_args, line)
    for rel, tree in trees:
        _tag_parents(tree)
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            params = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
            rebound = {t.id for n in ast.walk(fn) if isinstance(n, ast.Assign)
                       for t in n.targets if isinstance(t, ast.Name)}
            branched = set()
            for node in ast.walk(fn):
                if isinstance(node, (ast.If, ast.IfExp)):
                    for sub in ast.walk(node.test):
                        if isinstance(sub, ast.Name):
                            branched.add(sub.id)
            cls = _enclosing_class(fn)
            call_name = cls if (fn.name == '__init__' and cls) else fn.name
            # 记录该形参的默认值：默认 False ⇒ 不传就是不启用（静默失效）
            #                      默认 True  ⇒ 不传反而是启用（静默生效）
            # 记录**每个形参**的默认值（多 flag 共用一次解析，判定时再按键取）
            param_defaults = {}
            pos = list(fn.args.args)
            defaults = [None] * (len(pos) - len(fn.args.defaults)) + list(fn.args.defaults)
            for a, d in zip(pos, defaults):
                if isinstance(d, ast.Constant):
                    param_defaults[a.arg] = d.value
            for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
                if isinstance(d, ast.Constant):
                    param_defaults[a.arg] = d.value
            defs.append((rel, fn.name, call_name, params, branched - rebound, fn.lineno,
                         param_defaults))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, 'id', None) or getattr(node.func, 'attr', None)
            if isinstance(node.func, ast.Attribute) and name in RAY_WRAPPERS \
                    and isinstance(node.func.value, ast.Attribute):
                name = node.func.value.attr
            if name:
                calls.append((rel, name, [k.arg for k in node.keywords if k.arg],
                              len(node.args), node.lineno))

    def sig_by_name():
        d = defaultdict(list)
        for rel, _fn, call_name, params, _b, _l, _dflts in defs:
            d[call_name].append(params)
        return d

    sigs = sig_by_name()
    report = {'root': root.name, 'flags': []}
    flags = [f for f in sorted(flags)]
    for flag in flags:
        consumers = [{'file': rel, 'func': fn, 'call_name': cn, 'line': ln,
                      'branches_on_flag': 1,
                      'param_default': dflts.get(flag, 'unspecified')}
                     for (rel, fn, cn, params, branched, ln, dflts) in defs if flag in branched]
        names = {c['call_name'] for c in consumers}
        prods = []
        for rel, name, kw, nargs, line in calls:
            if name not in names:
                continue
            if flag in kw:
                prods.append({'file': rel, 'line': line, 'how': 'keyword'})
                continue
            for sig in sigs.get(name, []):
                if flag in sig and sig.index(flag) < nargs:
                    prods.append({'file': rel, 'line': line, 'how': 'positional'})
                    break
        reads = sum(1 for rel, lines in text_by_file.items()
                    for ln in lines if flag in ln and not ln.lstrip().startswith('#'))
        consumers = [c for c in consumers if not is_test_path(c['file'])]
        prods = [p for p in prods if not is_test_path(p['file'])]
        inert = (len(consumers) > 0 and len(prods) == 0)
        defaults = {c['param_default'] for c in consumers}
        # 默认 False 且无人传 ⇒ 该分支永不执行（静默失效）
        # 默认 True  且无人传 ⇒ 该分支总是执行（静默生效，属另一类问题）
        mode = None
        if inert:
            mode = ('never' if defaults == {False}
                    else 'always' if defaults == {True}
                    else 'mixed')
        report['flags'].append({
            'flag': flag, 'reads_nontest': reads,
            'consumers_nontest': consumers, 'deciding_consumers': consumers,
            'producers_total': len(prods), 'producers_nontest': prods,
            'silently_inert': inert, 'defaults': sorted(
                [str(d) for d in defaults]), 'mode': mode,
        })
    return report


def shadowed_arms(path, tree):
    """If/elif arms that can never fire because an earlier arm switches on a sibling
    constexpr that every call site pins to the literal True.

    Reported only when all of these hold:
      * the arm's condition reduces to exactly one name (the flag), possibly wrapped
        in `cutlass.const_expr(...)`;
      * an earlier arm in the same chain tests a *different* parameter of the same
        function;
      * some call site of that function passes the sibling the literal True, and no
        call site passes it anything that could be False (literal False, a bare
        name, an expression). `f(flag=flag)` counts as pure forwarding only when the
        body never reassigns that parameter — otherwise it is a computed value.

    One pass over the tree, no per-call rescan. Known false positive, measured on
    the TensorRT-LLM snapshot: when a function has several call sites and one of
    them omits the sibling while another pins it True, the sibling is reported as
    pinned although the omitting site may keep the arm alive. 9 hits, 1 such false
    positive (`has_bias` in m3_lowm_splitk.py). Treat output as a candidate list.

    See readme 第十一类.
    """
    import ast as _ast

    def sole_name(expr):
        while isinstance(expr, _ast.Call) and len(expr.args) == 1 and not expr.keywords:
            fn = expr.func
            if isinstance(fn, _ast.Attribute) and fn.attr == 'const_expr':
                expr = expr.args[0]
                continue
            break
        return expr.id if isinstance(expr, _ast.Name) else None

    params_of = {}
    defaults = {}        # func_name -> {param: literal value, when statically known}
    rebound = {}
    chains = []          # (func_name, [If, If, ...]) per if/elif chain
    calls = []           # (host_func_name, callee_name, {kw: kind})

    def visit(node, host):
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            a = node.args
            ps = {x.arg for x in a.args + a.kwonlyargs}
            params_of[node.name] = ps
            re_ = set()
            for st in _ast.walk(node):
                if isinstance(st, (_ast.Assign, _ast.AnnAssign, _ast.AugAssign)):
                    tgts = st.targets if isinstance(st, _ast.Assign) else [st.target]
                    for t in tgts:
                        if isinstance(t, _ast.Name) and t.id in ps:
                            re_.add(t.id)
            rebound[node.name] = re_
            d = {}
            for arg, df in zip(a.args[len(a.args) - len(a.defaults):], a.defaults):
                d[arg.arg] = df
            for arg, df in zip(a.kwonlyargs, a.kw_defaults):
                if df is not None:
                    d[arg.arg] = df
            defaults[node.name] = d
            host = node.name
        if isinstance(node, _ast.If):
            chain = [node]
            cur = node
            while len(cur.orelse) == 1 and isinstance(cur.orelse[0], _ast.If):
                cur = cur.orelse[0]
                chain.append(cur)
            if len(chain) > 1:
                chains.append((host, chain))
        if isinstance(node, _ast.Call):
            f = node.func
            callee = f.id if isinstance(f, _ast.Name) else getattr(f, 'attr', None)
            if callee:
                kinds = {}
                for kw in node.keywords:
                    if not kw.arg:
                        continue
                    v = kw.value
                    if isinstance(v, _ast.Constant) and v.value is True:
                        kinds[kw.arg] = 'true'
                    elif (isinstance(v, _ast.Name) and v.id == kw.arg
                          and kw.arg in params_of.get(host, ())
                          and kw.arg not in rebound.get(host, ())):
                        # Pure forwarding only applies when the name is really the
                        # *caller's* parameter. In `run_splitk_dense_swiglu` the
                        # argument `b_swiglu=True` is a decision, not a forward,
                        # even though the callee has a parameter of that name.
                        kinds[kw.arg] = 'forward'
                    else:
                        kinds[kw.arg] = 'maybe_false'
                calls.append((host, callee, kinds))
        for ch in _ast.iter_child_nodes(node):
            visit(ch, host)

    visit(tree, None)

    pinned = {}
    for host, callee, kinds in calls:
        if host == callee:
            # A recursive/self-named call is not an external decision about the
            # callee's parameters.
            continue
        st = pinned.setdefault(callee, {'true': set(), 'maybe_false': set(),
                                        'named': set()})
        for name, kind in kinds.items():
            st['named'].add(name)
            if kind == 'true':
                st['true'].add(name)
            elif kind == 'maybe_false':
                st['maybe_false'].add(name)

    # A parameter every call site omits runs on its default. A False default makes
    # the sibling False everywhere; a True default makes the arm dead by
    # construction (pyflag reports that as mode "always"), so only False is folded
    # in here. Omitting cannot override an explicit value.
    for callee, st in pinned.items():
        for name, df in defaults.get(callee, {}).items():
            if name in st['named']:
                continue
            if isinstance(df, _ast.Constant) and df.value is False:
                st['maybe_false'].add(name)

    out = []
    for host, chain in chains:
        ps = params_of.get(host, set())
        st = pinned.get(host, {'true': set(), 'maybe_false': set()})
        for i, arm in enumerate(chain[1:], 1):
            flag = sole_name(arm.test)
            if not flag or flag not in ps:
                continue
            for prev in chain[:i]:
                sib = sole_name(prev.test)
                if not sib or sib == flag or sib not in ps:
                    continue
                if sib in st['true'] and sib not in st['maybe_false']:
                    out.append({
                        'file': str(path),
                        'flag': flag,
                        'arm_lineno': arm.lineno,
                        'shadowed_by': sib,
                        'sibling_line': prev.lineno,
                        'func': host,
                    })
                    break
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if len(args) != 1:
        print(__doc__)
        return 2
    root = pathlib.Path(args[0])
    if not root.exists():
        print('找不到目录:', root)
        return 2

    report = scan_many(root, discover_flags(root))
    # 第十一类：同一 if/elif 链上前面的 arm 恒真 ⇒ 后面的 arm 是死的。
    # 只要该 flag 同时被判为 silently_inert，就把它升级成 shadowed_arms 候选。
    import ast as _ast
    shadow = []
    inert_names = {f['flag'] for f in report['flags'] if f['silently_inert']}
    for p in root.rglob('*.py'):
        if '.git' in p.parts:
            continue
        try:
            tree = _ast.parse(p.read_text(errors='replace'))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for rec in shadowed_arms(p, tree):
            if rec['flag'] in inert_names:
                rec['file'] = str(p.relative_to(root))
                shadow.append(rec)
    report['shadowed_arms'] = shadow

    text = json.dumps(report, indent=1, ensure_ascii=False, sort_keys=True)
    if '--json' in sys.argv:
        # `--json <path>` writes there; a bare `--json` writes next to the tool.
        # A bare flag used to name the file after root.name, which is '' when the
        # target is `.` — that produced `scan-.json` and silently overwrote it.
        i = sys.argv.index('--json')
        named = sys.argv[i + 1] if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('--') else None
        out = pathlib.Path(named) if named else (
            pathlib.Path(__file__).parent / ('scan-%s.json' % (root.name or 'repo')))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + '\n', encoding='utf-8')
    print(text)
    return 0


# 待扫的开关：从各框架配置里挑出的"布尔性能开关"，名字跨框架要一致才有意义。
# 这里只列显式出现在多个框架里的候选，避免对单一框架定制。
CANDIDATE_FLAGS = []
CANDIDATE_FLAGS_BY_NAME = {}


def _load_candidates(*names):
    global CANDIDATE_FLAGS, CANDIDATE_FLAGS_BY_NAME
    CANDIDATE_FLAGS = [{'flag': n} for n in names]
    CANDIDATE_FLAGS_BY_NAME = {n: '' for n in names}


_load_candidates(
    'use_dynamic_bsz',
    'use_remove_padding',
    'use_fused_kernels',
    'use_kl_loss',
    'use_torch_compile',
    'balance_batch',
    'colocate_all',
    'enable_gradient_checkpointing',
)

if __name__ == '__main__':
    sys.exit(main())
