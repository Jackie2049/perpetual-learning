#!/usr/bin/env python3
"""C++ 版 pyflag：扫描"配置项有没有生产者"（成员默认值的形式）。

Python 版（scan_unwired_flags.py）从**函数形参默认值**出发找开关。C++ 代码里
这类开关的主要形态不是函数形参，而是**结构体成员默认值**：

    struct common_params {
        bool no_extra_bufts = false;   // 声明
    };
    mparams.use_extra_bufts = !params.no_extra_bufts;   // 生产者（赋值）

    struct clip_context_params { bool use_gpu; };        // 消费端声明
    if (ctx_params.use_gpu) { ... }                      // 消费端分支

判定三步与 Python 版一致：

  1. **字段声明**：`struct/class` 体内形如 `bool name = false;`（或枚举成员）。
  2. **消费者**：该名字出现在某个 `if (...)` / 三目 / 布尔表达式里。
  3. **生产者**：该名字出现在某条赋值的**右值**里，或者是某次调用的实参。

"消费者 > 0 且生产者 == 0" ⇒ 该字段在本快照里没有任何地方被写入，分支恒走
默认值那一侧。这与 Python 版的 silently_inert 同义。

**只做正则级结构分析**：不建 AST、不解析 `#ifdef`、不做符号解析。它能做的是
把"看起来没人赋值"的字段一次列全，判断仍要靠人逐条绑行号（同 Python 版）。

用法：
    python3 scan_cpp_flags.py <repo_root> [--json [path]] [--min-members N]
    python3 scan_cpp_flags.py --selftest
"""

import json
import pathlib
import re
import sys

CXX_EXT = {'.h', '.hpp', '.hh', '.cpp', '.cc', '.cxx', '.c', '.cu', '.cuh', '.m', '.mm'}

# struct/class 起止：C++ 结构体体内没有";}"嵌套歧义的大多是声明体，
# 所以用"花括号配平"从 `struct X {` 一直走到配平处即可。
STRUCT_OPEN = re.compile(r'\b(?:struct|class)\s+([A-Za-z_]\w*)\s*(?:final\s*)?\{')
# 成员声明：bool name = true/false;（允许注释、C 风格 bool、std::atomic<bool>）
FIELD_DECL = re.compile(
    r'^\s*(?:const\s+)?(?:static\s+)?(?:std::atomic<\s*)?(?:bool|int|uint32_t|uint8_t)'
    r'(?:\s*>\s*)?\s+([A-Za-z_]\w*)\s*(?:=\s*(?:true|false|-?\d+[uUlL]*))?\s*;',
    re.M)
# 枚举成员需要单独处理：enum xxx_t { A = 0, B = 1 };
ENUM_OPEN = re.compile(r'\benum\s+(?:class\s+)?([A-Za-z_]\w*)\s*\{')

C_KEYWORDS = {
    'if', 'for', 'while', 'switch', 'return', 'sizeof', 'catch', 'else',
    'true', 'false', 'nullptr', 'NULL', 'and', 'or', 'not',
}


def strip_comments_and_strings(text):
    """把注释与字符串字面量替换成空白（保留换行，行号不变）。

    这一步很要紧：`// whether to disable xxx` 这类注释里的名字被当成"出现"
    是 Python 版第一批的经典假阳性（第七类易错点）。
    """
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        two = text[i:i + 2]
        if two == '//':
            j = text.find('\n', i)
            j = n if j < 0 else j
            out.append(' ' * (j - i))
            i = j
        elif two == '/*':
            j = text.find('*/', i + 2)
            j = n if j < 0 else j + 2
            out.append(re.sub(r'[^\n]', ' ', text[i:j]))
            i = j
        elif c == '"' or c == "'":
            q = c
            j = i + 1
            while j < n:
                if text[j] == '\\':
                    j += 2
                    continue
                if text[j] == q:
                    j += 1
                    break
                j += 1
            out.append(re.sub(r'[^\n]', ' ', text[i:j]))
            i = j
        else:
            out.append(c)
            i += 1
    return ''.join(out)


def iter_cxx(root):
    for p in root.rglob('*'):
        if not p.is_file() or p.suffix not in CXX_EXT:
            continue
        parts = set(p.parts)
        if '.git' in parts or 'node_modules' in parts:
            continue
        yield p


def read_clean(p):
    try:
        raw = p.read_text(encoding='utf-8', errors='ignore')
    except OSError:
        return None
    return strip_comments_and_strings(raw)


def struct_spans(text):
    """返回 [(name, start, end)]，按花括号配平。"""
    spans = []
    for m in STRUCT_OPEN.finditer(text):
        depth = 0
        i = m.end() - 1
        while i < len(text):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    break
            i += 1
        spans.append((m.group(1), m.end(), i))
    return spans


def enum_spans(text):
    spans = []
    for m in ENUM_OPEN.finditer(text):
        j = text.find('}', m.end())
        if j < 0:
            continue
        spans.append((m.group(1), m.end(), j))
    return spans


def collect_fields(files):
    """第 1 步：字段声明。返回 {name: {'struct': [...], 'file': ...}}。"""
    fields = {}
    for path, text in files.items():
        for sname, s, e in struct_spans(text):
            body = text[s:e]
            for m in FIELD_DECL.finditer(body):
                name = m.group(1)
                if name in C_KEYWORDS:
                    continue
                default = None
                line = body[m.start():m.end()]
                if '= true' in line:
                    default = 'true'
                elif '= false' in line:
                    default = 'false'
                fields.setdefault(name, []).append(
                    {'struct': sname, 'file': path, 'default': default})
        for sname, s, e in enum_spans(text):
            body = text[s:e]
            for item in body.split(','):
                m = re.match(r'\s*([A-Za-z_]\w*)\s*=', item)
                if m:
                    fields.setdefault('enum:' + m.group(1), []).append(
                        {'struct': sname, 'file': path, 'default': None})
    return fields


INFIX_OPS = set('+-*/%<>!&|^')
ASSIGN_BEFORE = re.compile(r'(?<![=!<>+\-*/%&|^])=(?!=)')
# `!flag` / `flag != x` / `flag &&` are uses of the value, not assignments of it.
NEGATED = re.compile(r'^\s*!\s*')


def is_assignment_line(line):
    """这一行里有没有一条 `=`（非 `==`/`!=`/`<=`/`>=`/`+=`…），返回其位置。"""
    m = ASSIGN_BEFORE.search(line)
    if not m:
        return None
    return m.start()


def collect_uses(files, name, raw_lines=None):
    """第 2、3 步：消费者（条件里出现）与生产者（被赋值 / 被传成实参）。

    一个名字出现在赋值**左边**或右值里都算生产者：`mparams.use_gpu = x` 把值
    写入字段，`f(p.use_gpu)` 把值交给下游。只有"既没被写、也没被传"才是真的
    没人生产。这三类分开记，逐条取证时按行号复核。
    """
    word = re.compile(r'(?<!\w)' + re.escape(name) + r'(?!\w)')
    member = re.compile(r'(?:\.|->)' + re.escape(name) + r'\b')
    consumers, writers, forwarders = [], [], []
    for path in sorted(files):
        text = files[path]
        for lineno, line in enumerate(text.splitlines(), 1):
            # 聚合初始化那一行剥掉注释后只剩 `value,`，标识符不在行里，
            # 所以要先用原始行判一次再决定是否跳过。
            raw_line = raw_lines.get((path, lineno)) if raw_lines else None
            if raw_line is not None and re.search(
                    r'/\*\s*' + re.escape(name) + r'\s*\*/', raw_line):
                writers.append({'file': path, 'line': lineno, 'how': 'aggregate-init'})
                continue
            if name not in line or not (word.search(line) or member.search(line)):
                continue
            # 声明行本身不算（struct 体内的 `bool x = false;`，也包含单行 struct）
            stripped = line.strip()
            decl = re.compile(r'(?:\bstruct\s+\w+\s*\{[^{}]*?)?(?:const\s+)?(?:static\s+)?'
                              r'(?:bool|int|uint32_t|uint8_t)\s+' + re.escape(name) + r'\b\s*(?:=|;)')
            if decl.search(stripped):
                continue
            # 有赋值时先看赋值：单行 `if (v == "x") { p.flag = true; }` 既是条件行
            # 又真的写了字段，先判消费者会把它整个记成"只读"，字段于是被误判成无人写。
            # 只有"没有指向本字段的赋值"的行才可能只是读。
            eq = is_assignment_line(line)
            writes_here = False
            if eq is not None:
                left = line[:eq]
                writes_here = bool(member.search(left) or
                                   re.search(r'(?<!\w)' + re.escape(name) + r'\s*$', left))
            if not writes_here and (re.search(r'\b(?:if|while|switch)\s*\(', line) or '?' in line
                                    or re.search(r'(&&|\|\|)', line)):
                consumers.append({'file': path, 'line': lineno})
                continue
            if eq is not None:
                # `b.use_x = !p.use_x` 是写入；`x = !p.use_x` 是把值读出来交出去。
                left = line[:eq]
                if member.search(left) or re.search(r'(?<!\w)' + re.escape(name) + r'\s*$', left):
                    writers.append({'file': path, 'line': lineno, 'how': 'assigned'})
                    continue
                if word.search(line, eq + 1) or member.search(line, eq + 1):
                    forwarders.append({'file': path, 'line': lineno, 'how': 'read-into-assign'})
                    continue
            # 传成实参：`f(..., params.use_x)` / `f(..., use_x)`
            if re.search(r'[(,]\s*[!\(]*(?:[A-Za-z_]\w*\s*(?:\.|->)\s*)?' + re.escape(name)
                         + r'\s*[,)]', line):
                forwarders.append({'file': path, 'line': lineno, 'how': 'arg'})
                continue
    return consumers, writers, forwarders


def scan(root, min_members=1):
    files, raw_lines = {}, {}
    for p in iter_cxx(root):
        raw = p.read_text(encoding='utf-8', errors='ignore')
        t = read_clean(p)
        if t is not None:
            rel = p.relative_to(root).as_posix()
            files[rel] = t
            for i, line in enumerate(raw.splitlines(), 1):
                if '/*' in line:
                    raw_lines[(rel, i)] = line
    fields = collect_fields(files)
    report = []
    for name, decls in sorted(fields.items()):
        if name.startswith('enum:'):
            continue
        consumers, writers, forwarders = collect_uses(files, name, raw_lines)
        if len(consumers) < min_members:
            continue
        report.append({
            'field': name,
            'decls': decls,
            'consumers': consumers[:20],
            'n_consumers': len(consumers),
            'writers': writers[:20],
            'n_writers': len(writers),
            'forwarders': forwarders[:20],
            'n_forwarders': len(forwarders),
            'unwritten': (len(writers) + len(forwarders)) == 0,
        })
    return {'root': str(root), 'files': len(files), 'fields': report}


def selftest():
    """四条夹具，每条对应一个真实踩过的坑。返回 (ok, [结果说明])。"""
    import tempfile
    cases = []

    def run(src, field, want):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / 'a.h'
            p.write_text(src, encoding='utf-8')
            rep = scan(pathlib.Path(d))
            got = {f['field']: f['unwritten'] for f in rep['fields']}
            cases.append((field, got.get(field), want))

    # 1. 阴：有分支、没赋值 ⇒ unwritten
    run("struct P { bool use_x = false; };\nvoid f(const P& p){ if (p.use_x) {} }\n",
        'use_x', True)
    # 2. 阳：`b.use_x = !p.use_x` 写入了字段 ⇒ 不算 unwritten
    run("struct P { bool use_x = false; };\nvoid g(P& b, const P& p){ b.use_x = !p.use_x; }\n"
        "void f(const P& p){ if (p.use_x) {} }\n", 'use_x', False)
    # 3. 注释里的名字不算出现（第七类易错点）
    run("struct P { bool use_x = false; };\n// whether to disable use_x\n"
        "void f(const P& p){ if (p.use_x) {} }\n", 'use_x', True)
    # 4. 传成实参也算生产（`mtmd_init(m, p.use_x)`）
    run("struct P { bool use_x = false; };\nvoid h(const P& p){ mtmd_init(1, p.use_x); }\n"
        "void f(const P& p){ if (p.use_x) {} }\n", 'use_x', False)
    # 6. 单行 `if (v == "x") { p.use_x = true; }` 真的写了字段 ⇒ 不是 unwritten
    #    （先判消费者会把它整个记成只读，这是本批踩到的误判）
    run('struct P { bool use_x = false; };\nvoid g(P & p, const char * v) {\n'
        '  if (v) { p.use_x = true; }\n}\nvoid f(const P& p){ if (p.use_x) {} }\n', 'use_x', False)
    # 7. 聚合初始化 `/* name */ value,` 是写入（注释被剥掉后标识符不出现）
    run('struct P { bool use_x = false; };\nvoid h(){ P p; p = {\n'
        '    /* use_x */ true,\n  }; }\nvoid f(const P& p){ if (p.use_x) {} }\n', 'use_x', False)
    # 5. 只在条件里用 `!p.use_x` ⇒ 仍是 unwritten（取反不是生产）
    run("struct P { bool use_x = false; };\nvoid f(const P& p){ if (!p.use_x) {} }\n",
        'use_x', True)

    ok = True
    notes = []
    for field, got, want in cases:
        good = got == want
        ok = ok and good
        notes.append('%s %s unwritten=%s want=%s' % ('PASS' if good else 'FAIL', field, got, want))
    return ok, notes


def main():
    argv = sys.argv[1:]
    if '--selftest' in argv:
        ok, notes = selftest()
        for n in notes:
            print(n)
        return 0 if ok else 1
    json_at = argv.index('--json') if '--json' in argv else None
    named = None
    if json_at is not None and json_at + 1 < len(argv) and not argv[json_at + 1].startswith('--'):
        named = argv[json_at + 1]
    skip = {json_at + 1} if named else set()
    min_members = 1
    if '--min-members' in argv:
        k = argv.index('--min-members')
        min_members = int(argv[k + 1])
        skip.add(k + 1)
    args = [a for i, a in enumerate(argv)
            if not a.startswith('--') and i not in skip]
    if len(args) != 1:
        print(__doc__)
        return 2
    root = pathlib.Path(args[0])
    if not root.exists():
        print('找不到目录:', root)
        return 2
    rep = scan(root, min_members=min_members)
    rep['min_members'] = min_members
    text = json.dumps(rep, indent=1, ensure_ascii=False, sort_keys=True)
    if json_at is not None:
        out = pathlib.Path(named) if named else (
            pathlib.Path(__file__).parent / ('scan-cpp-%s.json' % (root.name or 'repo')))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + '\n', encoding='utf-8')
    unwritten = [f for f in rep['fields'] if f['unwritten']]
    print('files=%d fields_with_consumers=%d unwritten=%d'
          % (rep['files'], len(rep['fields']), len(unwritten)))
    for f in unwritten:
        print('  %-40s cons=%2d decls=%s' % (
            f['field'], f['n_consumers'],
            ','.join(sorted({d['struct'] for d in f['decls']}))))
    return 0


if __name__ == '__main__':
    sys.exit(main())
