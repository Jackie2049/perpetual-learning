#!/usr/bin/env python3
"""把 precision_debugger 的每个 stage 映射到"真正能触发它"的装饰器。

问题背景：verl 的 msprobe 集成有个白名单 `_STAGE_TO_ROLE`，文档和 5 份配置注释都
把 6 个 stage 当作可选值列给用户；但整条采集链只有一条入口——`@DistProfiler.annotate(role=...)`
——所以"表里有名字"不等于"有人能触发"。本脚本扫源码，把两者对上。

用法：
    python3 scan_stage_producers.py --root /tmp/verl-main-7454326
    python3 scan_stage_producers.py --root <tree> --json
    python3 scan_stage_producers.py --root <tree> --preset main      # 判定：exit 0/1
    python3 scan_stage_producers.py --selftest                        # 合成树自检

判定口径（三条，缺一不可）：
  1. stage 的触发者只可能是 `X.annotate(..., role="<stage>")` 字面量；
  2. `PrecisionDebuggerProfiler.start()` 的第一个位置参数叫 `stage`，且不读 kwargs["role"]
     —— 所以训练侧 `start_profile(role="train", ...)` 不会变成一次 stage 采集；
  3. `msprobe` 只在 profiler 包里被 import，`PrecisionDebugger` 实例只在 `start()` 里创建
     —— 没有绕开包装器的旁路。

退出码：0 = 与预设一致；1 = 不一致；2 = 机械无效（缺文件/解析失败）。
纯静态、无第三方依赖、Python 3.9+。
"""

import argparse
import ast
import json
import os
import re
import sys

MAP_FILE = os.path.join("verl", "utils", "profiler", "precision_debugger_profile.py")
PROFILE_FILE = os.path.join("verl", "utils", "profiler", "profile.py")
DOC_FILE = os.path.join(
    "docs", "ascend_tutorial", "en", "dev_guide", "precision_analysis", "precision_debugger.md"
)
SKIP_STAGES = {"rollout_generate"}


class MechanicalError(Exception):
    pass


def read_source(root, rel):
    path = os.path.join(root, rel)
    if not os.path.isfile(path):
        raise MechanicalError(f"missing file: {rel}")
    with open(path, encoding="utf-8") as fh:
        return path, fh.read()


def parse_stage_map(root):
    """返回 ({stage: role}, 行号)。键和值都必须是字符串字面量。"""
    path, src = read_source(root, MAP_FILE)
    tree = ast.parse(src, filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "_STAGE_TO_ROLE" for t in node.targets):
                if not isinstance(node.value, ast.Dict):
                    raise MechanicalError("_STAGE_TO_ROLE is not a dict literal")
                mapping = {}
                for k, v in zip(node.value.keys, node.value.values):
                    if not (isinstance(k, ast.Constant) and isinstance(v, ast.Constant)):
                        raise MechanicalError("_STAGE_TO_ROLE contains a non-literal entry")
                    mapping[k.value] = v.value
                return mapping, node.lineno
    raise MechanicalError("_STAGE_TO_ROLE not found")


def iter_py_files(root):
    base = os.path.join(root, "verl")
    if not os.path.isdir(base):
        raise MechanicalError("missing directory: verl/")
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames):
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


def _literal_kwarg(call, name):
    for kw in call.keywords:
        if kw.arg == name and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value
    return None


def collect_producers(root):
    """收集所有 `DistProfiler.annotate(role="...")` 装饰器，带类名/函数名。

    只认 `DistProfiler` 这个包装器：`SkipManager.annotate(role="rollout")` 之类是另一套
    机制，白名单也管不到它们——把它们算进来是第一版的假阳性来源。
    语法版本高于本机解释器的文件走逐行正则兜底，并在结果里标 `via=regex`。
    """
    producers = []
    other_annotate = []
    start_profile_calls = []
    unparsed = []
    for path in iter_py_files(root):
        rel = os.path.relpath(path, root)
        src = None
        try:
            with open(path, encoding="utf-8") as fh:
                src = fh.read()
            tree = ast.parse(src, filename=path)
        except (SyntaxError, UnicodeDecodeError):
            unparsed.append(rel)
            if src:
                producers.extend(_regex_producers(rel, src))
            continue

        class Walker(ast.NodeVisitor):
            def __init__(self):
                self.stack = []

            def visit_ClassDef(self, node):
                self.stack.append(node.name)
                self.generic_visit(node)
                self.stack.pop()

            def visit_FunctionDef(self, node):
                self._check(node)
                self.stack.append(node.name)
                self.generic_visit(node)
                self.stack.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def _check(self, node):
                for dec in node.decorator_list:
                    if not isinstance(dec, ast.Call):
                        continue
                    func = dec.func
                    if not isinstance(func, ast.Attribute) or func.attr != "annotate":
                        continue
                    owner = func.value
                    owner_name = owner.id if isinstance(owner, ast.Name) else ast.dump(owner)
                    role = _literal_kwarg(dec, "role")
                    entry = {
                        "file": rel,
                        "line": dec.lineno,
                        "role": role,
                        "owner": "DistProfiler" if owner_name == "DistProfiler" else owner_name,
                        "owner_class": self.stack[-1] if self.stack else None,
                        "function": node.name,
                        "qualname": ".".join(self.stack + [node.name]),
                        "via": "ast",
                    }
                    if entry["owner"] == "DistProfiler":
                        producers.append(entry)
                    else:
                        other_annotate.append(entry)

            def visit_Call(self, node):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "start_profile":
                    start_profile_calls.append({
                        "file": rel,
                        "line": node.lineno,
                        "kwargs": sorted(kw.arg for kw in node.keywords if kw.arg),
                        "positional": len(node.args),
                    })
                self.generic_visit(node)

        Walker().visit(tree)
    return producers, other_annotate, start_profile_calls, unparsed


_ROLE_RE = re.compile(r"@\s*DistProfiler\.annotate\([^)]*?role\s*=\s*[\"']([^\"']+)[\"']")
_DEF_AFTER_RE = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)")


def _regex_producers(rel, src):
    """AST 不可用时的兜底：逐行找 @DistProfiler.annotate(...role="x")，函数名取下一个 def。"""
    out = []
    if not src:
        return out
    lines = src.splitlines()
    for i, line in enumerate(lines):
        m = _ROLE_RE.search(line)
        if not m:
            continue
        func = None
        for follow in lines[i + 1 : i + 4]:
            dm = _DEF_AFTER_RE.match(follow)
            if dm:
                func = dm.group(1)
                break
        out.append({
            "file": rel,
            "line": i + 1,
            "role": m.group(1),
            "owner": "DistProfiler",
            "owner_class": None,
            "function": func,
            "qualname": func or "<unknown>",
            "via": "regex",
        })
    return out


def check_start_reads_role(root):
    """确认 PrecisionDebuggerProfiler.start() 不把 kwargs['role'] 当成 stage。"""
    path, src = read_source(root, MAP_FILE)
    tree = ast.parse(src, filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "PrecisionDebuggerProfiler":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "start":
                    args = [a.arg for a in item.args.args]
                    src_seg = ast.get_source_segment(src, item) or ""
                    reads_role = ('kwargs.get("role")' in src_seg) or ("kwargs.get('role')" in src_seg)
                    # 去掉 self/cls，取真正承载 stage 的第一个形参
                    params = args[1:] if args and args[0] in ("self", "cls") else args
                    return (params[0] if params else None), reads_role, item.lineno
    raise MechanicalError("PrecisionDebuggerProfiler.start() not found")


def check_msprobe_imports(root):
    """msprobe 只在 profiler 包内被 import / 探测。"""
    hits = []
    for path in iter_py_files(root):
        rel = os.path.relpath(path, root)
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                s = line.strip()
                if s.startswith("#"):
                    continue
                if ("from msprobe" in s) or ("import msprobe" in s) or ('"msprobe"' in s):
                    hits.append({"file": rel, "line": i, "text": s[:120]})
    return hits


def documented_stages(root):
    """从 Ascend 文档的 'Supported stages' 段抽 bullet 列表；缺文档时返回 (None, None)。"""
    try:
        path, src = read_source(root, DOC_FILE)
    except MechanicalError:
        return None, None
    lines = src.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("## supported stages"):
            start = i
            break
    if start is None:
        return None, None
    found = []
    for line in lines[start + 1 :]:
        s = line.strip()
        if s.startswith("## "):
            break
        if s.startswith("*") and "`" in s:
            name = s.split("`")[1]
            found.append(name)
    return found, path


def build_report(root):
    stage_map, map_line = parse_stage_map(root)
    producers, other_annotate, start_profile_calls, unparsed = collect_producers(root)
    first_arg, reads_role, start_line = check_start_reads_role(root)

    by_role = {}
    for p in producers:
        by_role.setdefault(p["role"], []).append(p)

    per_stage = {}
    for stage, role in stage_map.items():
        per_stage[stage] = {
            "role": role,
            "producers": by_role.get(stage, []),
            "skipped_by_design": stage in SKIP_STAGES,
        }

    unmapped = sorted({p["role"] for p in producers if p["role"] not in stage_map and p["role"] is not None})
    no_role = [p for p in producers if p["role"] is None]

    docs, doc_path = documented_stages(root)
    return {
        "root": os.path.abspath(root),
        "stage_map_line": map_line,
        "stage_map": stage_map,
        "per_stage": per_stage,
        "unmapped_roles": unmapped,
        "annotate_without_literal_role": no_role,
        "start_profile_calls_with_stage_kwarg": [
            c for c in start_profile_calls if "stage" in c["kwargs"]
        ],
        "start_profile_role_kwarg_calls": [c for c in start_profile_calls if "role" in c["kwargs"]],
        "start_first_positional": first_arg,
        "start_reads_role": reads_role,
        "start_line": start_line,
        "msprobe_import_sites": check_msprobe_imports(root),
        "documented_stages": docs,
        "documented_stages_file": os.path.relpath(doc_path, root) if doc_path else None,
        "unparsed_files": unparsed,
        "producers_via_regex": [f"{p['file']}:{p['line']}" for p in producers if p["via"] == "regex"],
        "other_profiler_wrappers": [
            {k: v for k, v in p.items() if k in ("file", "line", "role", "owner", "qualname")}
            for p in other_annotate
        ],
    }


EXPECTATIONS = {
    # 2026-11-04，origin/main 74543265：三个 stage 无触发者
    "main": {
        "dead": {"compute_values", "critic_update", "compute_rm_score"},
        "unmapped": {"forward_backward", "train_batch"},
        "docs_equal_map": True,
        "start_first_positional": "stage",
        "start_reads_role": False,
    },
    # 2026-04-20 重构父提交 6eeb5711：critic 两个 stage 当时还活着，RM 从来没有
    "pre6067": {
        "dead": {"compute_rm_score"},
        "unmapped": {
            "env_finish_rollout",
            "env_init",
            "env_init_simulator",
            "env_interact_step",
            "env_reset_envs_to_state_ids",
            "rollout_generate",
            "train_batch",
        },
        "docs_equal_map": None,
        "start_first_positional": "stage",
        "start_reads_role": False,
    },
}


def render(report):
    out = []
    out.append(f"root: {report['root']}")
    out.append(f"_STAGE_TO_ROLE: {report['stage_map_line']} 行，{len(report['stage_map'])} 个 stage")
    out.append("")
    header = f"{'stage':<26}{'role':<14}{'producers'}"
    out.append(header)
    out.append("-" * len(header))
    for stage, info in report["per_stage"].items():
        prods = info["producers"]
        if not prods:
            cell = "-- 无 --"
        else:
            cell = "; ".join(f"{p['file']}:{p['line']} ({p['qualname']})" for p in prods)
        out.append(f"{stage:<26}{info['role']:<14}{cell}")
    out.append("")
    out.append(f"白名单之外还挂了 role 的装饰器: {report['unmapped_roles'] or '无'}")
    if report["annotate_without_literal_role"]:
        for p in report["annotate_without_literal_role"]:
            out.append(f"  ! 非字面量 role: {p['file']}:{p['line']} ({p['qualname']})")
    out.append(
        f"start_profile(stage=...) 调用: {len(report['start_profile_calls_with_stage_kwarg'])} 处"
    )
    out.append(
        f"start_profile(role=...) 调用: {len(report['start_profile_role_kwarg_calls'])} 处"
        f"（role 只是标签，不进入 stage）"
    )
    out.append(
        f"PrecisionDebuggerProfiler.start 首参: {report['start_first_positional']}，"
        f"读 kwargs['role']: {report['start_reads_role']}（{report['start_line']} 行）"
    )
    out.append(f"msprobe 引用点: {len(report['msprobe_import_sites'])} 处")
    for h in report["msprobe_import_sites"]:
        out.append(f"  {h['file']}:{h['line']}  {h['text']}")
    if report["documented_stages"] is not None:
        out.append(f"文档列出的 stage（{report['documented_stages_file']}）: {report['documented_stages']}")
    else:
        out.append("文档 'Supported stages' 段: 未找到")
    out.append(
        f"其它包装器的 annotate（SkipManager 等，不参与本白名单）: "
        f"{sorted({p['owner'] for p in report['other_profiler_wrappers']}) or '无'}"
        f" / roles={sorted({p['role'] for p in report['other_profiler_wrappers']}) or '无'}"
    )
    if report["unparsed_files"]:
        out.append(
            f"AST 解析失败（本机解释器语法版本低于源码）: {len(report['unparsed_files'])} 个文件，"
            f"已用逐行正则兜底"
        )
        for rel in report["unparsed_files"]:
            out.append("  " + rel)
    if report["producers_via_regex"]:
        out.append(f"其中由正则兜底检出的装饰器: {report['producers_via_regex']}")
    return "\n".join(out)


def verify(report, preset):
    exp = EXPECTATIONS[preset]
    problems = []

    dead = {s for s, info in report["per_stage"].items() if not info["producers"]}
    if dead != exp["dead"]:
        problems.append(f"无触发者的 stage: 实测 {sorted(dead)} vs 预期 {sorted(exp['dead'])}")

    unmapped = set(report["unmapped_roles"])
    if unmapped != exp["unmapped"]:
        problems.append(f"白名单外的 role: 实测 {sorted(unmapped)} vs 预期 {sorted(exp['unmapped'])}")

    if report["start_first_positional"] != exp["start_first_positional"]:
        problems.append(
            f"start() 首参: 实测 {report['start_first_positional']!r} vs 预期 {exp['start_first_positional']!r}"
        )
    if report["start_reads_role"] != exp["start_reads_role"]:
        problems.append(f"start() 读 kwargs['role']: 实测 {report['start_reads_role']} vs 预期 {exp['start_reads_role']}")

    overlap = {p["role"] for p in report["other_profiler_wrappers"]} & set(report["stage_map"])
    if overlap:
        problems.append(
            f"其它包装器也在用白名单里的 stage 名 {sorted(overlap)}"
            "——那可能是第二条触发路径，结论要改"
        )

    if report["start_profile_calls_with_stage_kwarg"]:
        problems.append(
            f"发现 start_profile(stage=...) 调用 {len(report['start_profile_calls_with_stage_kwarg'])} 处"
            "——那是一条真实触发路径，结论要改"
        )

    if exp["docs_equal_map"]:
        docs = report["documented_stages"]
        if docs is None:
            problems.append("文档里找不到 'Supported stages' 段")
        elif set(docs) != set(report["stage_map"]):
            problems.append(f"文档 stage 集合: 实测 {sorted(set(docs))} vs 表 {sorted(report['stage_map'])}")

    # msprobe 只应出现在 profiler 包与本脚本认的三个文件里
    allowed_suffixes = (
        os.path.join("verl", "utils", "profiler", "precision_debugger_profile.py"),
        os.path.join("verl", "utils", "import_utils.py"),
    )
    for hit in report["msprobe_import_sites"]:
        if not hit["file"].endswith(allowed_suffixes):
            problems.append(f"msprobe 出现未预期位置: {hit['file']}:{hit['line']} {hit['text']}")

    return problems


def selftest():
    import tempfile

    tmp = tempfile.mkdtemp(prefix="stage-producers-selftest-")
    prof = os.path.join(tmp, "verl", "utils", "profiler")
    os.makedirs(prof)
    work = os.path.join(tmp, "verl", "workers")
    os.makedirs(work)
    for rel in ("verl/__init__.py", "verl/utils/__init__.py", "verl/utils/profiler/__init__.py", "verl/workers/__init__.py"):
        open(os.path.join(tmp, rel), "w").close()

    def write_map(stages):
        body = "\n".join(f'    "{s}": "{r}",' for s, r in stages.items())
        with open(os.path.join(prof, "precision_debugger_profile.py"), "w") as fh:
            fh.write(
                "_STAGE_TO_ROLE = {\n" + body + "\n}\n"
                "class PrecisionDebuggerProfiler:\n"
                "    def start(self, stage=None, global_step=None, model=None, **kwargs):\n"
                "        return False\n"
            )

    def write_worker(roles):
        with open(os.path.join(work, "worker.py"), "w") as fh:
            fh.write("class W:\n")
            for i, role in enumerate(roles):
                fh.write(f'    @DistProfiler.annotate(role="{role}")\n    def m{i}(self):\n        pass\n')

    checks = []
    write_map({"alpha": "actor", "beta": "critic"})
    write_worker(["alpha"])
    r = build_report(tmp)
    dead = {s for s, i in r["per_stage"].items() if not i["producers"]}
    checks.append(("只接一个 stage 时，另一个报无触发者", dead == {"beta"}))

    write_worker(["alpha", "beta"])
    r2 = build_report(tmp)
    dead2 = {s for s, i in r2["per_stage"].items() if not i["producers"]}
    checks.append(("补上装饰器后无触发者集合变空（方向会翻）", dead2 == set()))

    write_worker(["alpha", "beta", "gamma"])
    r3 = build_report(tmp)
    checks.append(("白名单外的 role 被抓出", set(r3["unmapped_roles"]) == {"gamma"}))

    # 关掉"role 被当成 stage"的可能：start 改读 kwargs['role'] 时必须被看见
    with open(os.path.join(prof, "precision_debugger_profile.py"), "a") as fh:
        fh.write('STUB = lambda: kwargs.get("role")\n')
    with open(os.path.join(prof, "precision_debugger_profile.py")) as fh:
        src = fh.read()
    src = src.replace(
        "    def start(self, stage=None, global_step=None, model=None, **kwargs):\n        return False\n",
        "    def start(self, stage=None, global_step=None, model=None, **kwargs):\n"
        '        stage = stage or kwargs.get("role")\n        return False\n',
    )
    with open(os.path.join(prof, "precision_debugger_profile.py"), "w") as fh:
        fh.write(src)
    r4 = build_report(tmp)
    checks.append(("start() 偷读 kwargs['role'] 时被识别", r4["start_reads_role"] is True))

    ok = True
    for name, passed in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="verl 源码树（含 verl/ 与 docs/）")
    ap.add_argument("--json", action="store_true", help="输出 JSON 而非表格")
    ap.add_argument("--preset", choices=sorted(EXPECTATIONS), help="按预设判定，返回 0/1")
    ap.add_argument("--selftest", action="store_true", help="用合成树自检")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    if not args.root:
        ap.error("需要 --root（或用 --selftest）")

    try:
        report = build_report(args.root)
    except MechanicalError as exc:
        print(f"[MECHANICAL] {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render(report))

    if args.preset:
        problems = verify(report, args.preset)
        print()
        if problems:
            print(f"判定（{args.preset}）: {len(problems)} 处不一致")
            for p in problems:
                print(f"  - {p}")
            return 1
        print(f"判定（{args.preset}）: 全部一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
