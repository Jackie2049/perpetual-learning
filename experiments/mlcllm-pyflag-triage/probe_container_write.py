#!/usr/bin/env python3
"""MLC-LLM 编译期 `enable_disaggregation`：谁写进去、谁读出来。

两层读数，都在 CPU 上：

  D1  38 个模型都从 `PagedKVCache.create_generic` 建 KV cache，其中**只有 3 个**
      （llama／llama4／nemotron）把 `enable_disaggregation=` 传下去；其余 35 个
      靠形参默认 `False`。
  D2  `dispatch_kv_cache_creation.py` 里的 `kwargs.pop("enable_disaggregation",
      False)` 与紧随其后的 `kwargs["enable_disaggregation"] = ...`：`kwargs` 是
      `create_tir_paged_kv_cache(self, bb, kwargs)` 的**具名形参**（不是
      `**kwargs`），所以这两行只是把同一个字典读出来又写回去，不构成"生产者给
      下游"——它只是让后续 `TIRPagedKVCache(..., **kwargs)` 收全字段。

结论：这个开关是**编译期按模型**接的，不是运行时开关；模型的 `disaggregation`
默认 `False`，未传的 35 个走静默默认。⇒ 不是缺陷，属设计面（要写证据）。
"""
import ast
import json
import pathlib
import sys
import textwrap

TARGET = "enable_disaggregation"


def models_passing(root):
    out = {"pass": [], "omit": []}
    for p in sorted((root / "python/mlc_llm/model").rglob("*.py")):
        text = p.read_text(encoding="utf-8")
        if "create_generic(" not in text:
            continue
        rel = p.relative_to(root).as_posix()
        if TARGET + "=" in text:
            out["pass"].append(rel)
        else:
            out["omit"].append(rel)
    return out


def dispatch_body(root):
    path = root / "python/mlc_llm/compiler_pass/dispatch_kv_cache_creation.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    # 找到 `kwargs.pop("enable_disaggregation", False)` 所在函数，看 kwargs 指什么
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        src = ast.get_source_segment(text, fn) or ""
        if "kwargs.pop(\"%s\"" % TARGET in src or "kwargs.pop('%s'" % TARGET in src:
            lines = text.splitlines()
            start, end = fn.lineno, fn.end_lineno
            return {
                "file": path.relative_to(root).as_posix(),
                "func": fn.name,
                "line": fn.lineno,
                "params": [a.arg for a in fn.args.args + fn.args.kwonlyargs],
                # `kwargs` 在签名里是**具名形参**（不是 **kwargs），所以
                # `kwargs.pop(...)` / `kwargs[...] = ...` 都只改这个字典本身
                "kwargs_is_named_param": "kwargs" in
                [a.arg for a in fn.args.args + fn.args.kwonlyargs],
                "excerpt": textwrap.dedent("\n".join(lines[start - 1:end]))[:1200],
            }
    return {}


def selftest(root):
    base = run_all(root)
    bad = 0
    cases = [
        ("3 个模型显式传", len(base["d1"]["pass"]) == 3, len(base["d1"]["pass"]), 3),
        ("35 个模型省略", len(base["d1"]["omit"]) == 35, len(base["d1"]["omit"]), 35),
        ("省略的是绝大多数", len(base["d1"]["omit"]) > len(base["d1"]["pass"]),
         True, True),
        ("dispatch 里 kwargs 是具名形参",
         base["d2"].get("kwargs_is_named_param") is True,
         base["d2"].get("kwargs_is_named_param"), True),
        ("dispatch 函数是 create_tir_...",
         "create_tir" in base["d2"].get("func", ""), base["d2"].get("func"), "create_tir_*"),
    ]
    for name, ok, got, want in cases:
        print("%-28s %s  got=%r want=%r" % (name, "PASS" if ok else "FAIL", got, want))
        bad += 0 if ok else 1
    print("\nselftest: %d/%d" % (len(cases) - bad, len(cases)))
    return 0 if bad == 0 else 1


def run_all(root):
    d2 = dispatch_body(root)
    return {
        "root": str(root),
        "d1": models_passing(root),
        "d2": {k: v for k, v in d2.items() if k != "excerpt"},
        "d2_excerpt": d2.get("excerpt", ""),
        "verdict": "COMPILE_TIME_PER_MODEL",
    }


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 2
    root = pathlib.Path(argv[0])
    if argv[0] == "--selftest":
        return selftest(pathlib.Path(argv[1]))
    report = run_all(root)
    text = json.dumps(report, indent=1, ensure_ascii=False)
    if "--json" in argv:
        out = pathlib.Path(argv[argv.index("--json") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
