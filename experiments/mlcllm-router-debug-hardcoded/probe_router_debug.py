#!/usr/bin/env python3
"""MLC-LLM `mlc_llm router` 起的子服务器，debug 端点是不是永远开着。

三条读法（每条都执行**真实源码**，不是 grep 出来的结论）：

  C1  路由器 → 子服务器：真实 `Router.__init__`（含它内部的 `start_server` 闭包）
      在 `enable_prefix_cache` 两档下，给每个 `PopenServer` 造的关键字实参里
      `enable_debug` 是什么值。
  C2  子服务器 → 子进程命令行：真实 `PopenServer.start`（只取到 `subprocess.Popen`
      之前）拼出来的 argv 里有没有 `--enable-debug`，两档（True / False）各一遍。
  C3  子进程 → 真决策点：真实 `interface.serve.serve` 里 `enable_debug` 的消费段
      （`server_context.enable_debug = ...` / `if enable_debug:` 挂 debug 路由）在
      两档下是否真的分叉。

C1 的对照口径：同一个闭包里还有 `prefix_cache_mode`，它在两档下**应该**不同——
如果它也读成同一个值，说明探针根本没跑起来（**读数是死的**），而不是"开关恒真"。

退出码：0 = 结论成立（router 恒传 enable_debug=True、CLI 无开关、下游真分叉）；
        2 = 上游已改（探针自己报警，结论失效）。

    python3 probe_router_debug.py <mlc-llm 根> [--json <out>]
    python3 probe_router_debug.py --selftest <mlc-llm 根>
"""
import argparse
import ast
import json
import pathlib
import sys
import textwrap

SNAPSHOT = "44fd06d04f22c87820e205b0f2b2e5e6a1f6ca85"

# 真实源码里被替身顶掉的外部依赖（清单进读数，便于核对覆盖面）
STUBS = ["tvm_ffi.get_global_func", "fastapi(FastAPI/APIRouter)", "uvicorn.run",
         "EngineConfig", "PopenServer(仅 C2 用真源码)", "Tokenizer",
         "aiohttp", "mlc_llm.protocol.openai_api_protocol",
         "mlc_llm.serve.entrypoints.*", "mlc_llm.serve.server.ServerContext"]


# --------------------------------------------------------------- 源码抽取
def node_source(text, lineno, end_lineno):
    """按行号原样切出源码（保留原缩进），再 dedent 掉公共缩进。"""
    seg = "".join(text.splitlines(True)[lineno - 1:end_lineno])
    return textwrap.dedent(seg)


def find_def(tree, name, cls=None):
    body = tree.body
    if cls is not None:
        for node in body:
            if isinstance(node, ast.ClassDef) and node.name == cls:
                body = node.body
                break
        else:
            raise LookupError("class %s not found" % cls)
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise LookupError("def %s not found" % name)


def _strip_imports(src):
    """去掉顶层 import 行（含括号续行）。名字已被替身放进命名空间。"""
    out, depth = [], 0
    for line in textwrap.dedent(src).splitlines():
        s = line.strip()
        if depth == 0 and (s.startswith("import ") or s.startswith("from ")):
            depth += s.count("(") - s.count(")")
            continue
        out.append(line)
    return "\n".join(out)


_DROP_ANNOTATIONS = False


class _StripAnn(ast.NodeTransformer):
    """删掉形参注解。源码里写的是 `model: str`，而替身没在命名空间里放 `str`
    以外的类型名（`Optional[List[str]]` 之类会在 exec 期求值）——这里只关心
    实参怎么走，注解对结论没有信息，删掉。返回注解会写回平铺后的源码，
    位置信息随之整体平移，故本函数的 `lineno` 不再用于对外报行号。"""

    def visit_arg(self, node):
        node.annotation = None
        return node


def body_source(text, node):
    """函数**体**（不含签名）的源码文本。签名里的类型注解会打断 exec，
    而这里只需要函数体，所以按 body 第一行到 end_lineno 切。"""
    lines = text.splitlines(True)
    first = node.body[0].lineno
    return textwrap.dedent("".join(lines[first - 1:node.end_lineno]))


# --------------------------------------------------------------- C1
def probe_c1(root: pathlib.Path) -> dict:
    """真实 Router.__init__（含 start_server 闭包）→ 每个 PopenServer 收到的关键字实参。"""
    path = root / "python/mlc_llm/router/router.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    node = find_def(tree, "__init__", cls="Router")
    body_src = _strip_imports(body_source(text, node))

    captured = []

    class _FakePopenServer:
        def __init__(self, **kwargs):
            captured.append(kwargs)

        def start(self, extra_env=None):
            pass

    class _EngineConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _Thread:
        def __init__(self, target=None, args=()):
            self._t, self._a = target, args

        def start(self):
            self._t(*self._a)

        def join(self):
            pass

    ns = {
        "PopenServer": _FakePopenServer,
        "EngineConfig": _EngineConfig,
        "threading": type("threading", (), {"Thread": _Thread}),
        "tvm_ffi": type("tvm_ffi", (), {
            "get_global_func": staticmethod(lambda _n: (lambda: [[0] * 4]))}),
        "json": json,
        "List": list, "Optional": object, "Literal": object,
        # `__init__` 末尾的 `self.tokenizer = Tokenizer(model)` 与本结论无关，
        # 但少了它会 NameError —— 替身进去，覆盖范围记进 STUBS
        "Tokenizer": type("Tokenizer", (), {"__init__": lambda self, *a, **k: None}),
    }
    # Router.__init__ 里除了建服务还要存一堆属性：给它一个最小 self
    class _Self:
        pass

    src = "def _init(self, model, model_lib, hosts, ports, num_gpus,\n" \
          "          enable_prefix_cache, router_mode, pd_balance_factor):\n" \
        + textwrap.indent(body_src, "    ")
    exec(compile(src, "<router-init>", "exec"), ns)

    out = {}
    for flag in (True, False):
        captured.clear()
        ns["_init"](_Self(), "m", None, ["127.0.0.1"], [8080], [1],
                    flag, "disagg", 0.0)
        out[str(flag)] = [
            {
                "enable_debug": c.get("enable_debug", "<absent>"),
                "prefix_cache_mode": getattr(c.get("engine_config"), "prefix_cache_mode",
                                             "<absent>"),
                "mode": c.get("mode", "<absent>"),
            }
            for c in captured
        ]
    return out


# --------------------------------------------------------------- C2
def probe_c2(root: pathlib.Path) -> dict:
    """真实 PopenServer.start 拼出的 argv（截到 subprocess.Popen 之前）。

    自错记录（本批第一处）：第一版拿 `node_source()`（**连着 `def start(...)` 签名**）
    再在它前面拼一个 `def _start(...)`，结果 `def start` 整个嵌套成了内层函数，
    外层 `_start` 的函数体是空的 ⇒ 调 `_start` 什么都不做、`seen` 恒空。
    现在只取**函数体**（`body_source`）。
    """
    path = root / "python/mlc_llm/serve/server/popen_server.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    node = find_def(tree, "start", cls="PopenServer")
    body = _strip_imports(body_source(text, node))
    # 整段函数体都跑：`self._proc = subprocess.Popen(cmd, ...)` 是**唯一**拼 argv 的
    # 地方，截掉它就什么都抓不到（本批第三处自错，见下）。
    # `Popen` 换成只记 argv 的桩、`wait_until_ready` 之类后续调用一并桩掉，
    # 因此不会有真进程。截断前先确认"整段里 Popen 恰好被调用一次"。
    if body.count("subprocess.Popen(") != 1:
        raise RuntimeError("Popen 调用点不是 1 处 —— 函数体形状变了，探针要重写")

    out = {}
    for flag in (True, False):
        seen = {}

        class _Cfg:
            """形状照 `EngineConfig`：被读到的字段全给具体值（不是 None），
            这样 `--overrides` 段也会拼出来，argv 与真跑一致。"""

            def __init__(self):
                self.additional_models = []
                self.speculative_mode = "disable"
                self.tensor_parallel_shards = 1
                self.pipeline_parallel_stages = 1
                self.context_window_size = 0
                self.sliding_window_size = 0
                self.attention_sink_size = 0
                self.prefill_chunk_size = 0
                self.max_history_size = 0
                self.max_num_sequence = 0
                self.max_total_sequence_length = 0
                self.max_single_sequence_length = 0
                self.prefill_mode = "hybrid"
                self.prefix_cache_mode = "radix"
                self.prefix_cache_max_num_recycling_seqs = 0
                self.spec_draft_length = None
                self.gpu_memory_utilization = 0.8

        class _Self:
            model = "m"
            model_lib = None
            device = "cuda:0"
            mode = "server"
            host = "127.0.0.1"
            port = 8080
            enable_tracing = False
            engine_config = _Cfg()

            def __init__(self):
                self.enable_debug = flag

        class _Popen:
            """替掉 `subprocess.Popen`：只抓 argv，不起进程。

            自错记录（本批第二、三处）：
            ② 上一版写成 `class _Subprocess: Popen = _Popen`——类体赋值在**类作用域**
               求值，而 `_Popen` 是外层函数的局部名，类作用域看不到；
            ③ 更根本的一处：为了"别真起进程"把函数体截到
               `self._proc = subprocess.Popen(` 之前——可 argv **只在那里**拼，
               截掉后 argv 恒为空，读数成了"两支都没有 --enable-debug"，
               看上去还像结论成立。现在整段跑、只桩 `Popen` 与 `requests`。
            """

            def __init__(self, cmd, **kwargs):
                seen["cmd"] = list(cmd)

            def poll(self):
                # None = 子进程还活着。返回 0 会被 `is not None` 判成"进程已退出"，
                # 当场抛 "The server fails to launch"（本批第四处自错）。
                return None

        ns = {
            "sys": sys,
            "os": type("os", (), {"environ": {}}),
            "subprocess": type("_Subprocess", (), {"Popen": _Popen}),
            "Path": pathlib.Path,
            # `str(Path(__file__).resolve().parents[4])` 只是当 cwd 用，与 argv 无关
            "__file__": str(path),
            # 就绪轮询要发真 HTTP 请求，桩掉；它不影响 argv
            "requests": type("requests", (), {
                "get": staticmethod(lambda *_a, **_k: type("R", (), {
                    "status_code": 200})())}),
            "time": type("time", (), {"sleep": staticmethod(lambda _s: None)}),
        }
        exec(compile("def _start(self, extra_env=None):\n"
                     + textwrap.indent(textwrap.dedent(body), "    ")
                     + "\n", "<popen-start>", "exec"), ns)
        # 源码里 self.engine_config 在类上，_Self 每个实例共用同一个 _Cfg 也无所谓
        ns["_start"](_Self(), None)
        cmd = seen.get("cmd", [])
        if not cmd:
            # 抓不到 argv 就不许静默给出"两支都没有"这种读数
            raise RuntimeError("C2 没抓到 argv —— 探针没跑起来，读数无效")
        out[str(flag)] = {
            "has_enable_debug": "--enable-debug" in cmd,
            "n_flags": cmd.count("--enable-debug"),
            "argv": cmd,
        }
    return out


# --------------------------------------------------------------- C3
def probe_c3(root: pathlib.Path) -> dict:
    """真实 interface.serve 的 enable_debug 消费段：挂没挂 debug 路由。"""
    path = root / "python/mlc_llm/interface/serve.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    node = find_def(tree, "serve")
    body = _strip_imports(node_source(text, node.lineno, node.end_lineno))
    lines = body.splitlines()
    start = next(i for i, ln in enumerate(lines)
                 if "server_context.enable_debug = enable_debug" in ln)
    seg = textwrap.dedent("\n".join(lines[start:]))
    # 取到 `app.exception_handler` 之前：后面的 FastAPI 装异常处理器与 `uvicorn.run`
    # 与"挂没挂 debug 路由"无关，截掉可让替身只看必要形状
    seg = seg[:seg.index("app.exception_handler")]

    out = {}
    for flag in (True, False):
        class _App:
            def __init__(self):
                self.included = []

            def include_router(self, r):
                self.included.append(getattr(r, "name", str(r)))

        class _Ctx:
            enable_debug = None

        ns = {
            "app": _App(),
            "server_context": _Ctx(),
            "debug_entrypoints": type("M", (), {"app": type("A", (), {"name": "debug"})}),
            "logger": type("L", (), {"info": staticmethod(lambda *_a, **_k: None)}),
            "uvicorn": type("U", (), {"run": staticmethod(lambda *_a, **_k: None)}),
        }
        exec(compile("def _seg(enable_debug):\n"
                     + textwrap.indent(seg, "    "), "<serve-seg>", "exec"), ns)
        ns["_seg"](flag)
        out[str(flag)] = {
            "context_enable_debug": ns["server_context"].enable_debug,
            "included": list(ns["app"].included),
        }
    return out


def probe_cli(root: pathlib.Path) -> dict:
    """用户有没有办法关掉：**CLI 上的开关 + 它真能走到 `PopenServer`**。

    只查"CLI 有没有一个 debug 开关"是不够的——加了开关却不接进 `Router(...)`，
    和没有开关在行为上一样。所以这一项取两段之**与**：
    ① `cli/router.py` 的 argparse 里有 debug 相关开关；
    ② `interface/router.py::serve` 把这个值原样递给 `Router(...)`（按 AST 找
       关键字实参，名字里含 debug 且值是一个形参），且 `Router.__init__` 收这个形参。
    """
    path = root / "python/mlc_llm/cli/router.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    opts = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument":
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    opts.append(a.value)
    router_debug = [o for o in opts if "debug" in o]

    # ② interface.serve → Router(...) 有没有把**某个入参**转发成 Router 的 debug 键。
    #    只看 `enable_debug=enable_prefix_cache` 这种"键名含 debug"不够——名字叫什么
    #    都行，关键是 Router 那边这一位能不能被外面改。本批自检第 4b 条抓到：
    #    注入写了 `enable_debug=enable_prefix_cache`，Router 的签名里也有这个形参，
    #    但 Router 内部 `PopenServer(enable_debug=True)` 照样钉死 ⇒ 转发形同虚设。
    #    所以判据取 "Router 那一位不是字面量" ∧ "interface 确实传了这个键"。
    router_src = (root / "python/mlc_llm/router/router.py").read_text(encoding="utf-8")
    router_tree = ast.parse(router_src)
    rinit = find_def(router_tree, "__init__", cls="Router")
    router_params = {a.arg for a in rinit.args.args + rinit.args.kwonlyargs}
    pinned_literal = False
    for node in ast.walk(rinit):
        if isinstance(node, ast.Call):
            for k in node.keywords:
                if k.arg == "enable_debug" and isinstance(k.value, ast.Constant):
                    pinned_literal = True

    iface = root / "python/mlc_llm/interface/router.py"
    iface_tree = ast.parse(iface.read_text(encoding="utf-8"))
    forwarded = []
    for node in ast.walk(iface_tree):
        if isinstance(node, ast.Call):
            fname = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if fname != "Router":
                continue
            for k in node.keywords:
                if k.arg in router_params and k.arg.startswith("enable_") \
                        and isinstance(k.value, ast.Name):
                    forwarded.append({"call": fname, "kwarg": k.arg, "expr": k.value.id})

    serve_cli = root / "python/mlc_llm/cli/serve.py"
    st = ast.parse(serve_cli.read_text(encoding="utf-8"))
    serve_opts = []
    for node in ast.walk(st):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument":
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    serve_opts.append(a.value)
    return {"router_cli_debug_flags": router_debug,
            "router_cli_flags": opts,
            "serve_cli_debug_flags": [o for o in serve_opts if "debug" in o],
            "router_serve_forwards_debug": forwarded,
            "router_pins_literal_in_body": pinned_literal}


# --------------------------------------------------------------- 判定
def verdict(c1, c2, c3, cli) -> dict:
    router_always_on = (
        all(r["enable_debug"] is True for r in c1["True"])
        and all(r["enable_debug"] is True for r in c1["False"])
        and len(c1["True"]) == len(c1["False"]) > 0
    )
    # 对照：同一个闭包里的 prefix_cache_mode 必须随档变化，否则读数本身就是死的
    control_live = (
        [r["prefix_cache_mode"] for r in c1["True"]]
        != [r["prefix_cache_mode"] for r in c1["False"]]
    )
    # 子进程真的拿到 flag（True 档）。False 档**不该**拿到——这是同一判据的反面，
    # 合在一条里是因为"两支都一样"既可能是恒真也可能是探针没跑。
    flag_reaches_subprocess = (
        c2["True"]["has_enable_debug"] is True
        and c2["False"]["has_enable_debug"] is False
    )
    # 顺带量出来的一条：start() 在 True 档追加了**两次** --enable-debug。
    # 与"router 钉死"是两个独立缺陷，分开报，避免修掉一个另一个被带着一起说。
    duplicate_append = (
        c2["True"]["n_flags"] == 2 and c2["False"]["n_flags"] == 0
    )
    downstream_forks = (
        c3["True"]["included"] != c3["False"]["included"]
        and c3["True"]["context_enable_debug"] is True
        and c3["False"]["context_enable_debug"] is False
    )
    # 「用户能不能关掉」= CLI 有开关 **且** 那个值真的被转发进 Router(...)。
    # 只看前半截会把"加了个没人用的参数"当成"已经能关"（本批自检第 4 条抓到的）。
    # 再核一遍这一位在 `Router.__init__` 里是不是字面量：注入只改进 `interface` 侧
    # 的转发时，`router_pins_literal_in_body` 仍为 True ⇒ 转发不产生任何行为差异。
    cli_has_off = (bool(cli["router_cli_debug_flags"])
                   and bool(cli["router_serve_forwards_debug"])
                   and not cli["router_pins_literal_in_body"])
    return {
        "router_pins_enable_debug_true": router_always_on,
        "control_prefix_cache_mode_live": control_live,
        "subprocess_gets_flag": flag_reaches_subprocess,
        "duplicate_enable_debug_append": duplicate_append,
        "downstream_really_forks": downstream_forks,
        "router_cli_can_turn_it_off": cli_has_off,
        "finding_holds": bool(router_always_on and control_live
                              and flag_reaches_subprocess and downstream_forks
                              and not cli_has_off),
    }


def run(root: pathlib.Path) -> dict:
    c1 = probe_c1(root)
    c2 = probe_c2(root)
    c3 = probe_c3(root)
    cli = probe_cli(root)
    return {"snapshot": SNAPSHOT, "stubs": STUBS, "C1": c1, "C2": {
        k: {"has_enable_debug": v["has_enable_debug"], "n_flags": v["n_flags"]}
        for k, v in c2.items()}, "C3": c3, "cli": cli, "verdict": verdict(c1, c2, c3, cli)}


# --------------------------------------------------------------- 自检
def selftest(root: pathlib.Path) -> int:
    """每种破坏让**指定的那一项**判据变红。

    判据：`want in red`（目标项确实红了）**且** `finding_holds` 变 False。
    **不**要求"只有目标项红"——多条判据共用同一份源码事实时，一处改动连带
    变红是正常的；但若目标项**不在**红名单里，说明这条判据根本没在看那个东西
    （TM-015／TM-017 都踩过），一律算失败。
    """
    import shutil
    import tempfile

    base = run(root)
    ok = True

    def run_mutated(mutate, name, note=""):
        nonlocal ok
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="mlc-mut-"))
        try:
            shutil.copytree(root / "python", tmp / "python")
            mutate(tmp)
            got = run(tmp)["verdict"]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        red = [k for k, v in got.items() if v is False and k != "finding_holds"]
        good = (name in red) and (got["finding_holds"] is False)
        ok &= good
        print("%-40s %-44s %s  red=%s" % (name, note, "PASS" if good else "FAIL", red))

    # 注入 1：router 不再钉 True，改成透传形参 —— 用户就能关掉了
    def mut1(tmp):
        p = tmp / "python/mlc_llm/router/router.py"
        s = p.read_text(encoding="utf-8")
        s = s.replace("                enable_debug=True,",
                      "                enable_debug=enable_prefix_cache,")
        p.write_text(s, encoding="utf-8")

    run_mutated(mut1, "router_pins_enable_debug_true",
                "router 不再钉 True（改成跟着 prefix_cache 走）")

    # 注入 2：把 prefix_cache_mode 也钉成常量 —— 对照项应红（读数变哑）
    def mut2(tmp):
        p = tmp / "python/mlc_llm/router/router.py"
        s = p.read_text(encoding="utf-8")
        s = s.replace('prefix_cache_mode="radix" if enable_prefix_cache else "disable"',
                      'prefix_cache_mode="radix"')
        p.write_text(s, encoding="utf-8")

    run_mutated(mut2, "control_prefix_cache_mode_live", "对照组也钉成常量")

    # 注入 3：下游 if 拆掉（无论传什么都挂 debug 路由）——下游不再分叉
    def mut3(tmp):
        p = tmp / "python/mlc_llm/interface/serve.py"
        s = p.read_text(encoding="utf-8")
        s = s.replace("        if enable_debug:\n", "        if True:\n")
        p.write_text(s, encoding="utf-8")

    run_mutated(mut3, "downstream_really_forks", "下游 if enable_debug 改成 if True")

    # 注入 4：**只**给 router CLI 加一个 --no-enable-debug，不动 Router 那一段。
    # 期望：`router_cli_can_turn_it_off` 仍然 False（因为值没被转发）、
    # `finding_holds` 仍然 True。这一条验的是"CLI 开关单独不算数"这条隐含判据，
    # 不是验"目标项变红"——所以单独判，并顺手确认 `probe_cli` 真的看见了那个开关
    # （否则探针没读到 = 这条注入空转）。
    def mut4(tmp):
        p = tmp / "python/mlc_llm/cli/router.py"
        s = p.read_text(encoding="utf-8")
        anchor = '    parser.add_argument(\n        "--enable-prefix-cache",'
        assert s.count(anchor) == 1, "锚点不唯一，注入会打偏"
        s = s.replace(anchor,
                      '    parser.add_argument("--no-enable-debug", action="store_true")\n'
                      + anchor, 1)
        p.write_text(s, encoding="utf-8")

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="mlc-mut-"))
    try:
        shutil.copytree(root / "python", tmp / "python")
        mut4(tmp)
        rep4 = run(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    v4 = rep4["verdict"]
    saw_flag = bool(rep4["cli"]["router_cli_debug_flags"])
    still_pinned = (v4["router_cli_can_turn_it_off"] is False
                    and v4["finding_holds"] is True and saw_flag
                    and rep4["cli"]["router_serve_forwards_debug"] == [])
    ok &= still_pinned
    print("%-40s %-44s %s  saw_flag=%s can_off=%s holds=%s" % (
        "router_cli_can_turn_it_off", "只加 CLI 开关、不转发",
        "PASS" if still_pinned else "FAIL", saw_flag,
        v4["router_cli_can_turn_it_off"], v4["finding_holds"]))

    # 注入 4b：把开关真的接通（去掉 router 的钉死 + 加 CLI 开关 + 转进 Router(...)）
    # —— 这时才算"用户能关掉"，`router_cli_can_turn_it_off` 应转为 True，
    # 于是 `finding_holds` 变 False 且红名单里**有**它。
    def mut4b(tmp):
        p = tmp / "python/mlc_llm/router/router.py"
        s = p.read_text(encoding="utf-8")
        anchor = "                enable_debug=True,"
        assert s.count(anchor) == 1, "锚点不唯一"
        s = s.replace(anchor, "                enable_debug=False,", 1)
        p.write_text(s, encoding="utf-8")
        q = tmp / "python/mlc_llm/cli/router.py"
        t = q.read_text(encoding="utf-8")
        a2 = '    parser.add_argument(\n        "--enable-prefix-cache",'
        t = t.replace(a2,
                      '    parser.add_argument("--no-enable-debug", action="store_true")\n'
                      + a2, 1)
        q.write_text(t, encoding="utf-8")
        r = tmp / "python/mlc_llm/interface/router.py"
        u = r.read_text(encoding="utf-8")
        u = u.replace("        enable_prefix_cache=enable_prefix_cache,",
                      "        enable_prefix_cache=enable_prefix_cache,\n"
                      "        enable_debug=enable_prefix_cache,", 1)
        r.write_text(u, encoding="utf-8")

    run_mutated(mut4b, "router_cli_can_turn_it_off",
                "去掉钉死 + CLI 开关 + 转发进 Router")

    # 注入 5：删掉后半段那次重复追加 —— 只剩一次，重复项应红而主结论仍成立
    def mut5(tmp):
        p = tmp / "python/mlc_llm/serve/server/popen_server.py"
        s = p.read_text(encoding="utf-8")
        tail = '        if self.enable_tracing:\n            cmd += ["--enable-tracing"]\n' \
               '        if self.enable_debug:\n            cmd += ["--enable-debug"]\n'
        s = s.replace(tail, '        if self.enable_tracing:\n            cmd += ["--enable-tracing"]\n')
        p.write_text(s, encoding="utf-8")

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="mlc-mut-"))
    try:
        shutil.copytree(root / "python", tmp / "python")
        mut5(tmp)
        got = run(tmp)["verdict"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    only_dup_red = (got["duplicate_enable_debug_append"] is False
                    and got["finding_holds"] is True)
    ok &= only_dup_red
    print("%-40s %-44s %s  dup=%s holds=%s" % (
        "duplicate_enable_debug_append", "删掉后半段那次重复追加",
        "PASS" if only_dup_red else "FAIL",
        got["duplicate_enable_debug_append"], got["finding_holds"]))

    print("\n基线 verdict: %s" % json.dumps(base["verdict"], ensure_ascii=False))
    n = 5 if ok else 0
    print("selftest: %d/5" % n)
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", help="mlc-llm 源码根")
    ap.add_argument("--json")
    ap.add_argument("--selftest", action="store_true")
    ns = ap.parse_args(argv)
    if not ns.root:
        print(__doc__)
        return 2
    root = pathlib.Path(ns.root)
    if ns.selftest:
        return selftest(root)
    rep = run(root)
    if ns.json:
        pathlib.Path(ns.json).write_text(
            json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep["verdict"], indent=2, ensure_ascii=False))
    return 0 if rep["verdict"]["finding_holds"] else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
