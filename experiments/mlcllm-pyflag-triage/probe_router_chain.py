#!/usr/bin/env python3
"""MLC-LLM `mlc_llm router --enable-prefix-cache` 到底有没有接到 `Router` 上。

三处镜头，每处都执行**真实源码**（不是 grep 出来的结论）：

  C1  设备端：真实 `Router.__init__`（含它内部的 `start_server` 闭包）在
      `enable_prefix_cache` 两档下，给每个 `PopenServer` 造的
      `EngineConfig.prefix_cache_mode` 是 `radix` 还是 `disable`。
  C2  客户端链：真实 `interface.router.serve` 收到 CLI 解出的值之后，交给
      `Router(...)` 的关键字实参里**有没有** `enable_prefix_cache`。
  C3  真 argparse：`cli.router.main` 的真实参数表能否把 `--enable-prefix-cache`
      解析成 `True`。

只取需要的函数源码（`ast` 定位 `lineno`/`end_lineno` 后原样 `exec`），
第三方依赖用最小替身：`tvm_ffi`、`fastapi`、`uvicorn`、`EngineConfig`、
`PopenServer`、`Tokenizer`。替身列表会写进报告，便于核对覆盖面。

退出码：接线是通的（`C2` 两档如实到达）→ 0；上游把它拆掉 → 2（探针自己报警）。

**本探针第一版给过反向读数**，记在这里：反事实注入用
`src.replace("enable_prefix_cache=enable_prefix_cache,", "")`，那条字符串同时命中
`interface.router.serve` 的**转发行**和它自己的**签名行**，注入后 `Router(...)`
照样收到关键字 ⇒ 探针报 `CHAIN_BROKEN`。直到 `--selftest` 的红灯才暴露
"这是把读的路由看成了死活"。现在注入只删调用点（`drop_router_kwarg`，按 AST 定位）。

    python3 probe_router_chain.py <mlc-llm 根> [--json <out>]
    python3 probe_router_chain.py --selftest <mlc-llm 根>
"""
import argparse
import ast
import json
import pathlib
import sys
import textwrap
import types
from typing import Any, List, Literal, Optional

STUBS = ["tvm_ffi.get_global_func", "fastapi.APIRouter", "uvicorn", "EngineConfig",
         "PopenServer", "Tokenizer", "threading.Thread (真实标准库)"]


# --------------------------------------------------------------------- 源码抽取
def function_source(path, name, cls=None):
    """原样取出 `def name` 的源码（保留缩进）与起始行号。"""
    text = pathlib.Path(path).read_text(encoding="utf-8")
    tree = ast.parse(text)
    body = tree.body
    if cls is not None:
        for node in body:
            if isinstance(node, ast.ClassDef) and node.name == cls:
                body = node.body
                break
        else:
            raise LookupError("class %s not found in %s" % (cls, path))
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            seg = "".join(text.splitlines(True)[node.lineno - 1: node.end_lineno])
            return textwrap.dedent(seg), node.lineno
    raise LookupError("function %s not found in %s" % (name, path))


def _strip_imports(src):
    """替身已经放进命名空间，去掉 import 行（含括号续行）。

    注意：**不能**把调用点一起删掉。第一版用 `replace("enable_prefix_cache=
    enable_prefix_cache,", "")` 做反事实注入，那条字符串同时命中
    `interface.router.serve` 里**转发**那一行和它自己的**签名**那一行，结果
    注入后 `Router(...)` 仍收到了该关键字 ⇒ 探针报 `WIRED`（读数是假阴性）。
    正确做法是只删调用点：先按 AST 找出 `Router(...)` 这次调用，再把它的
    关键字实参行删掉。
    """
    out, depth = [], 0
    for line in textwrap.dedent(src).splitlines():
        stripped = line.strip()
        if depth == 0 and (stripped.startswith("import ") or stripped.startswith("from ")):
            depth += stripped.count("(") - stripped.count(")")
            continue
        out.append(line)
    return "\n".join(out)


def drop_router_kwarg(src, kwarg):
    """把 `Router(...)` 这次调用里的 `kwarg=...` 实参删掉（只删调用点，不动签名）。

    用 AST 定位调用，再按行删；找不到就抛错——注入必须真的改到东西。
    """
    tree = ast.parse(src)
    target = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fname = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        # 真源码里经由 `router_type`（默认 Router）间接构造，所以同时认
        # 直接 `Router(...)` 与 `router_type(...)`
        if fname in ("Router", "router_type") and any(k.arg == kwarg for k in node.keywords):
            target = node
            break
    if target is None:
        raise LookupError("no Router(...) call passing %s" % kwarg)
    kw = next(k for k in target.keywords if k.arg == kwarg)
    lines = src.splitlines(True)
    # 单行实参：整行删；多行实参：从 kw.value 起始行删到末行
    start = kw.value.lineno
    end = getattr(kw.value, "end_lineno", start)
    del lines[start - 1:end]
    return "".join(lines)


# ------------------------------------------------------------------------- 替身
class _EngineConfig:
    """真实 `EngineConfig` 是 dataclass（字段很多）；这里只关心 prefix_cache_mode。"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.prefix_cache_mode = kwargs.get("prefix_cache_mode", "radix")


def _stub_tvm_ffi():
    mod = types.ModuleType("tvm_ffi")
    mod.get_global_func = lambda name: (lambda: [0])
    return mod


def _stub_uvicorn():
    mod = types.ModuleType("uvicorn")
    mod.run = lambda *a, **k: None   # 关键：不让 app.run 真的起 HTTP 服务
    return mod


def _stub_fastapi():
    mod = types.ModuleType("fastapi")

    class _App:
        def post(self, *a, **k):
            return lambda fn: fn

        def add_middleware(self, *a, **k):
            return None

        def include_router(self, *a, **k):
            return None

        def exception_handler(self, *a, **k):
            return lambda fn: fn

    mod.APIRouter = lambda *a, **k: _App()
    mod.Request = object
    mod.FastAPI = lambda *a, **k: _App()
    return mod


# ------------------------------------------------------------------------- 镜头
def _lens_router_from_source(src, label="<router>"):
    """执行 `Router.__init__`，回收每个 `PopenServer` 收到的 `EngineConfig`。"""
    captured = []

    class PopenServer:
        def __init__(self, **kwargs):
            captured.append(kwargs)
            self.started = None

        def start(self, extra_env=None):
            self.started = extra_env

    class Router:
        pass

    ns = {
        "__name__": "probe_lens_router",
        "EngineConfig": _EngineConfig,
        "PopenServer": PopenServer,
        "Tokenizer": lambda *a, **k: None,
        "tvm_ffi": _stub_tvm_ffi(),
        "threading": __import__("threading"),
        "json": json,
        "math": __import__("math"),
        "openai_api_protocol": types.ModuleType("mlc_llm.protocol.openai_api_protocol"),
        "Optional": Optional, "List": List, "Literal": Literal, "Any": Any, "Tuple": tuple,
    }
    exec(compile(_strip_imports(src), label, "exec"), ns)  # noqa: S102
    init = ns["__init__"]
    readings = {}
    for flag in (True, False):
        captured.clear()
        inst = Router()
        init(inst, model="dummy", model_lib=None, hosts=["127.0.0.1", "127.0.0.1"],
             ports=[9124, 9125], num_gpus=[1, 1], enable_prefix_cache=flag,
             router_mode="round-robin", pd_balance_factor=0.0)
        readings[str(flag)] = {
            "n_popen_server": len(captured),
            "prefix_cache_mode": [c["engine_config"].prefix_cache_mode for c in captured],
            "device": [c.get("device") for c in captured],
        }
    return readings


def _lens_interface_from_source(src, label="<interface>"):
    """执行 `interface.router.serve`，回收它调用 `Router(...)` 的关键字实参。"""
    captured = []

    class Router:
        def __init__(inner_self, **kwargs):
            captured.append(kwargs)

    err = types.ModuleType("mlc_llm.protocol.error_protocol")
    err.create_error_response = lambda *a, **k: None
    err.BadRequestError = type("BadRequestError", (Exception,), {})
    err.bad_request_error_handler = lambda *a, **k: None
    engine_utils = types.ModuleType("mlc_llm.serve.engine_utils")
    engine_utils.random_uuid = lambda: "uuid"

    ns = {
        "__name__": "probe_lens_interface",
        "fastapi": _stub_fastapi(),
        "uvicorn": _stub_uvicorn(),
        "CORSMiddleware": object,
        "error_protocol": err,
        "CompletionLogProbs": object,
        "CompletionRequest": object,
        "engine_base": types.ModuleType("mlc_llm.serve.engine_base"),
        "engine_utils": engine_utils,
        "Router": Router,
        "Type": type,
        "Optional": Optional, "List": List, "Literal": Literal, "Any": Any,
    }
    exec(compile(_strip_imports(src), label, "exec"), ns)  # noqa: S102
    serve = ns["serve"]
    readings = {}
    for cli_value in (True, False):
        captured.clear()
        serve(model="dummy", model_lib=None, router_host="127.0.0.1", router_port=9123,
              endpoint_hosts=["127.0.0.1"], endpoint_ports=[9124], endpoint_num_gpus=[1],
              enable_prefix_cache=cli_value, router_mode="round-robin",
              pd_balance_factor=0.0)
        kwargs = dict(captured[0]) if captured else {}
        readings[str(cli_value)] = {
            "n_router_calls": len(captured),
            "router_kwargs_enable_prefix_cache": kwargs.get("enable_prefix_cache", "<absent>"),
        }
    return readings


def _lens_cli_from_source(src, label="<cli>"):
    """执行 `cli.router.main`，看真参数表把该开关解析成什么。"""
    captured = {}

    class _Help(dict):
        def __missing__(self, k):
            return ""

    ns = {
        "__name__": "probe_lens_cli",
        "ArgumentParser": argparse.ArgumentParser,
        "HELP": _Help(),
        "serve": lambda **kwargs: captured.update(kwargs),
    }
    exec(compile(_strip_imports(src), label, "exec"), ns)  # noqa: S102
    main = ns["main"]
    readings = {}
    for argv in (["m", "--enable-prefix-cache"], ["m"]):
        captured.clear()
        main(argv)
        readings[" ".join(argv[1:]) or "<no flag>"] = captured.get(
            "enable_prefix_cache", "<absent>")
    return readings


def _tree(root):
    """快照目录：优先用仓库里存的逐字节副本，没有才用完整克隆。

    `raw/snapshot/` 里放三份**原样副本**（`router/router.py`、
    `interface/router.py`、`cli/router.py`），让 `--selftest` 在只有本仓库的
    机器上也能跑——和 llama.cpp 那批 `raw/snapshot/common/` 同一做法。
    """
    snap = pathlib.Path(__file__).resolve().parent / "raw" / "snapshot"
    if (snap / "router/router.py").exists():
        return snap
    return root / "python/mlc_llm"


def lens_router(root):
    path = _tree(root) / "router/router.py"
    src, lineno = function_source(path, "__init__", "Router")
    return {"source": "%s/router/router.py:%d" % (path.parent.parent.name, lineno),
            "readings": _lens_router_from_source(src, str(path))}


def lens_interface(root):
    path = _tree(root) / "interface/router.py"
    src, lineno = function_source(path, "serve")
    return {"source": "%s/interface/router.py:%d" % (path.parent.parent.name, lineno),
            "readings": _lens_interface_from_source(src, str(path))}


def lens_cli(root):
    path = _tree(root) / "cli/router.py"
    src, lineno = function_source(path, "main")
    return {"source": "%s/cli/router.py:%d" % (path.parent.parent.name, lineno),
            "readings": _lens_cli_from_source(src, str(path))}


def run_all(root):
    report = {"root": str(root), "stubs": STUBS,
              "c1": lens_router(root), "c2": lens_interface(root), "c3": lens_cli(root)}
    c2_true = report["c2"]["readings"]["True"]["router_kwargs_enable_prefix_cache"]
    # CLI → Router 这一跳是通的：两档都要如实到达（False 也必须到，否则
    # "显式关掉"也做不到）。上游哪天拆掉转发，这里会变 CHAIN_BROKEN。
    c2_false = report["c2"]["readings"]["False"]["router_kwargs_enable_prefix_cache"]
    report["verdict"] = "WIRED" if (c2_true is True and c2_false is False) else "CHAIN_BROKEN"
    return report


# ----------------------------------------------------------------------- selftest
def selftest(root):
    """反事实注入：把接线补回去 / 把条件写反，读数必须跟着变。"""
    path_i = _tree(root) / "interface/router.py"
    src_i, _ = function_source(path_i, "serve")
    body_i = _strip_imports(src_i)
    cut = drop_router_kwarg(body_i, "enable_prefix_cache")
    if cut == body_i:
        raise SystemExit("selftest injection did not change the source")
    cut_readings = _lens_interface_from_source(cut, "<injected-interface>")

    path_r = _tree(root) / "router/router.py"
    src_r, _ = function_source(path_r, "__init__", "Router")
    body_r = _strip_imports(src_r)
    anchor = '"radix" if enable_prefix_cache else "disable"'
    if anchor in body_r:
        inverted_src = body_r.replace(anchor, '"disable" if enable_prefix_cache else "radix"')
        note = "Router 里的三目条件改成反向"
    else:
        raise SystemExit("selftest anchor missing in Router.__init__")
    inverted = _lens_router_from_source(inverted_src, "<injected-router>")

    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        base = run_all(root)
    cases = [
        ("C1 flag=True -> radix",
         base["c1"]["readings"]["True"]["prefix_cache_mode"] == ["radix", "radix"],
         base["c1"]["readings"]["True"]["prefix_cache_mode"], ["radix", "radix"]),
        ("C1 flag=False -> disable",
         base["c1"]["readings"]["False"]["prefix_cache_mode"] == ["disable", "disable"],
         base["c1"]["readings"]["False"]["prefix_cache_mode"], ["disable", "disable"]),
        ("C2 CLI True 到达 Router",
         base["c2"]["readings"]["True"]["router_kwargs_enable_prefix_cache"] is True,
         base["c2"]["readings"]["True"]["router_kwargs_enable_prefix_cache"], True),
        ("C2 CLI False 到达 Router",
         base["c2"]["readings"]["False"]["router_kwargs_enable_prefix_cache"] is False,
         base["c2"]["readings"]["False"]["router_kwargs_enable_prefix_cache"], False),
        ("C3 argparse 能解析为 True",
         base["c3"]["readings"]["--enable-prefix-cache"] is True,
         base["c3"]["readings"]["--enable-prefix-cache"], True),
        ("注入1 删转发 -> C2 读成 <absent>(True)",
         cut_readings["True"]["router_kwargs_enable_prefix_cache"] == "<absent>",
         cut_readings["True"]["router_kwargs_enable_prefix_cache"], "<absent>"),
        ("注入1 删转发 -> C2 读成 <absent>(False)",
         cut_readings["False"]["router_kwargs_enable_prefix_cache"] == "<absent>",
         cut_readings["False"]["router_kwargs_enable_prefix_cache"], "<absent>"),
        ("注入2 条件写反 -> C1 互换",
         inverted["True"]["prefix_cache_mode"] == ["disable", "disable"]
         and inverted["False"]["prefix_cache_mode"] == ["radix", "radix"],
         {"True": inverted["True"]["prefix_cache_mode"],
          "False": inverted["False"]["prefix_cache_mode"]},
         {"True": ["disable", "disable"], "False": ["radix", "radix"]}),
    ]
    bad = 0
    print("selftest 注入方式:", note)
    for name, ok, got, want in cases:
        print("%-30s %s  got=%r want=%r" % (name, "PASS" if ok else "FAIL", got, want))
        bad += 0 if ok else 1
    print("\nselftest: %d/%d" % (len(cases) - bad, len(cases)))
    return 0 if bad == 0 else 1


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 2
    if argv[0] == "--selftest":
        return selftest(pathlib.Path(argv[1]))
    root = pathlib.Path(argv[0])
    if not ((_tree(root) / "router/router.py").exists()
            or (root / "python/mlc_llm/router/router.py").exists()):
        print("不像 mlc-llm 根目录:", root)
        return 2
    report = run_all(root)
    text = json.dumps(report, indent=1, ensure_ascii=False)
    if "--json" in argv:
        out = pathlib.Path(argv[argv.index("--json") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    # 结论（接线是通的）成立时 exit 0；上游拆掉后 exit 2，让探针自己报警。
    return 0 if report["verdict"] == "WIRED" else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
