#!/usr/bin/env python3
"""MLC-LLM 扩池第六批：把每条候选钉到具体文件行，并做三个反事实注入。

不信任扫描器的结论——它本轮就错了两处（见 readme）。每条判据都指明"哪一行、
什么条件"，`--selftest` 时在**临时副本**上注入，要求目标判据变红。

    python3 verify_findings.py <mlc-llm 根>
    python3 verify_findings.py --selftest <mlc-llm 根>

退出码：0 = 全部判据成立；1 = 有判据不成立（源码变了）；2 = 快照不是预期版本。
"""
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

SNAPSHOT = "44fd06d04f22c87820e205b0f2b2e5e6a1f6ca85"
checks, failed = [], []
VERBOSE = True


def check(name, cond, detail=""):
    checks.append((name, bool(cond), detail))
    if not cond:
        failed.append(name)
    if VERBOSE:
        print("%-46s %s%s" % (name, "PASS" if cond else "FAIL",
                              ("  — " + detail) if detail else ""))


def read(root, rel):
    return (root / rel).read_text(encoding="utf-8")


def lineno_of(text, pattern):
    for i, line in enumerate(text.splitlines(), 1):
        if re.search(pattern, line):
            return i
    return None


def code_of(root, rel):
    """去掉 docstring 后的代码，用来判"这行真的存在于函数体里"。

    第一版 F1c 用 `iface.split(chr(34)*3)[-1]` 取"docstring 之后的尾巴"：
    `interface/router.py` 有两段 docstring，最后一段在函数定义之后，
    于是把真在第 38 行的转发行判成缺失（假红）。改成按 AST 剔除全部 docstring。
    """
    import ast
    text = read(root, rel)
    tree = ast.parse(text)
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) \
                    and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                spans.append((body[0].lineno, body[0].end_lineno))
    lines = text.splitlines()
    keep = [ln for i, ln in enumerate(lines, 1) if not any(a <= i <= b for a, b in spans)]
    return "\n".join(keep)


# ------------------------------------------------------------------ 判据
def run_checks(root, expect_snapshot=True):
    del checks[:]
    del failed[:]

    # S：快照必须干净（沿用 TM-017 的 S0 教训：判据绿不代表读的是原文件）
    if expect_snapshot:
        try:
            head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=True).stdout.strip()
            dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                                   capture_output=True, text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            head, dirty = "", ""
        check("S0 snapshot is %s" % SNAPSHOT[:8], head == SNAPSHOT, head)
        check("S0b tree is clean", dirty == "", dirty[:80] or "(clean)")

    cli_router = read(root, "python/mlc_llm/cli/router.py")
    cli_serve = read(root, "python/mlc_llm/cli/serve.py")
    iface = read(root, "python/mlc_llm/interface/router.py")
    router = read(root, "python/mlc_llm/router/router.py")
    engine = read(root, "python/mlc_llm/serve/engine.py")
    sync = read(root, "python/mlc_llm/serve/sync_engine.py")
    ebase = read(root, "python/mlc_llm/serve/engine_base.py")
    kv = read(root, "python/mlc_llm/nn/kv_cache.py")
    dispatch = read(root, "python/mlc_llm/compiler_pass/dispatch_kv_cache_creation.py")
    gsm = read(root, "python/mlc_llm/bench/evaluation/gsm8k.py")
    chat = read(root, "python/mlc_llm/testing/debug_chat.py")
    base_h = read(root, "cpp/serve/engine_actions/batch_prefill_base.h")
    base_cc = read(root, "cpp/serve/engine_actions/batch_prefill_base.cc")

    # F1 router 前缀缓存：CLI → interface → Router → PopenServer 四跳全在
    check("F1a CLI 声明 --enable-prefix-cache",
          lineno_of(cli_router, r'"--enable-prefix-cache"') is not None)
    check("F1b CLI 把它交给 serve()",
          "enable_prefix_cache=parsed.enable_prefix_cache" in cli_router)
    check("F1c interface.serve 把它递给 Router()",
          "enable_prefix_cache=enable_prefix_cache" in code_of(
              root, "python/mlc_llm/interface/router.py"))
    check("F1d Router 用它决定 prefix_cache_mode",
          'prefix_cache_mode="radix" if enable_prefix_cache else "disable"' in router)
    check("F1e 文档也在教这个参数", "enable_prefix_cache=" in
          read(root, "docs/microserving/tutorial.rst"))

    # F2 enable_tracing：CLI → interface → MLCEngine/AsyncMLCEngine → super → EngineState
    check("F2a CLI 声明 --enable-tracing", '"--enable-tracing"' in cli_serve)
    check("F2b interface.serve 接收并下传",
          "enable_tracing=enable_tracing" in code_of(root, "python/mlc_llm/interface/serve.py"))
    check("F2c MLCEngine 转给 super()",
          engine.count("enable_tracing=enable_tracing") >= 2)
    check("F2d EngineState 真消费", "if enable_tracing:" in ebase and
          "EventTraceRecorder()" in ebase)
    check("F2e sync_engine 直接决定 trace_recorder",
          "EventTraceRecorder() if enable_tracing else None" in sync)

    # F3 enable_disaggregation：3 个模型显式传，35 个省略；dispatch 那两行只改本字典
    models = sorted((root / "python/mlc_llm/model").rglob("*.py"))
    passing = [p for p in models if p.read_text(encoding="utf-8").count("create_generic(")
               and "enable_disaggregation=" in p.read_text(encoding="utf-8")]
    omitting = [p for p in models if p.read_text(encoding="utf-8").count("create_generic(")
                and "enable_disaggregation=" not in p.read_text(encoding="utf-8")]
    check("F3a 恰好 3 个模型显式传", len(passing) == 3, str(len(passing)))
    check("F3b 其余 35 个省略", len(omitting) == 35, str(len(omitting)))
    check("F3c C++ 侧真的按它分支", "int(enable_disaggregation)" in kv)
    check("F3d dispatch 的 kwargs 是具名形参",
          "def create_tir_paged_kv_cache(\n        self,\n        bb: relax.BlockBuilder,\n"
          "        kwargs: Dict[str, Any]," in dispatch.replace("\r", ""))

    # F4 两个测试辅助开关都有 CLI 生产者（扫描器原先报 0）
    check("F4a gsm8k --disable_cot 有生产者", "use_cot=not args.disable_cot" in gsm)
    check("F4b debug_chat --disable-instrument 有生产者",
          "disable_instrument=parsed.disable_instrument" in chat)

    # F5 C++ is_decode：结构体字段 + hybrid 预填时置 true + 三处消费者
    check("F5a 字段声明", "bool is_decode = false;" in base_h)
    check("F5b hybrid 模式下写 true", "/*is_decode=*/true" in base_cc)
    check("F5c 三个消费者", sum(
        (root / p).read_text(encoding="utf-8").count(".is_decode)") for p in (
            "cpp/serve/engine_actions/new_request_prefill.cc",
            "cpp/serve/engine_actions/disagg_prepare_recv.cc",
            "cpp/serve/engine_actions/disagg_remote_send.cc")) == 3)
    return failed


# ------------------------------------------------------------------ 注入
def inject_and_expect(root, name, mutate, expect_fail):
    """在临时副本上改一处，跑判据，要求 expect_fail 这条判据变红。"""
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="mlc-verify-"))
    dst = tmp / "tree"
    shutil.copytree(root, dst, ignore=shutil.ignore_patterns(".git", "3rdparty"), symlinks=True)
    # 注入版不做快照校验（副本没有 .git），也不打印逐条结果
    mutate(dst)
    global VERBOSE
    VERBOSE = False
    try:
        bad = run_checks(dst, expect_snapshot=False)
    finally:
        VERBOSE = True
    caught = expect_fail in bad
    print("   注入 → %s %s（注入后失败项：%s）" % (
        "捕获到" if caught else "未捕获", expect_fail, bad))
    shutil.rmtree(tmp, ignore_errors=True)
    return caught


def selftest(root):
    print("== selftest：4 条注入，每条必须让指定判据变红 ==\n")

    def drop_iface_forward(d):
        p = d / "python/mlc_llm/interface/router.py"
        p.write_text(p.read_text(encoding="utf-8").replace(
            "        enable_prefix_cache=enable_prefix_cache,\n", ""), encoding="utf-8")

    def pin_engine_state(d):
        p = d / "python/mlc_llm/serve/engine_base.py"
        p.write_text(p.read_text(encoding="utf-8").replace("if enable_tracing:",
                                                           "if True:"), encoding="utf-8")

    def flip_router(d):
        p = d / "python/mlc_llm/router/router.py"
        p.write_text(p.read_text(encoding="utf-8").replace(
            'prefix_cache_mode="radix" if enable_prefix_cache else "disable"',
            'prefix_cache_mode="disable" if enable_prefix_cache else "radix"'), encoding="utf-8")

    def drop_hybrid_decode(d):
        p = d / "cpp/serve/engine_actions/batch_prefill_base.cc"
        p.write_text(p.read_text(encoding="utf-8").replace("/*is_decode=*/true",
                                                           "/*is_decode=*/false"), encoding="utf-8")

    cases = [
        ("F1c interface.serve 把它递给 Router()", drop_iface_forward),
        ("F2d EngineState 真消费", pin_engine_state),
        ("F1d Router 用它决定 prefix_cache_mode", flip_router),
        ("F5b hybrid 模式下写 true", drop_hybrid_decode),
    ]
    ok = 0
    for target, mutate in cases:
        print("-- 注入 %s" % target)
        ok += 1 if inject_and_expect(root, target, mutate, target) else 0
        print()
    print("selftest: %d/%d" % (ok, len(cases)))
    return 0 if ok == len(cases) else 1


def main(argv):
    global VERBOSE
    if not argv:
        print(__doc__)
        return 2
    if "--verbose" in argv:
        VERBOSE = True
    if argv[0] == "--selftest":
        return selftest(pathlib.Path(argv[1]))
    root = pathlib.Path(argv[0])
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError:
        head = ""
    if head and head != SNAPSHOT:
        print("快照不符：期望 %s，实际 %s" % (SNAPSHOT, head))
        return 2
    bad = run_checks(root)
    print("%d checks, %d failed" % (len(checks), len(bad)))
    if "--json" in argv:
        out = pathlib.Path(argv[argv.index("--json") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"snapshot": SNAPSHOT,
             "checks": [{"name": n, "ok": o, "detail": d} for n, o, d in checks],
             "failed": bad}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
