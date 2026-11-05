#!/usr/bin/env python3
"""重复的 `--enable-debug` 到底有没有后果。

`PopenServer.start()` 在 `enable_debug=True` 时把 `--enable-debug` 追加了**两次**
（`popen_server.py:71` 与 `:115`）。这里不动源码、只拿快照里那份**真实的**
`support/argparse.py`（`ArgumentParser` 的子类，只覆盖了 `error`）去解析三种 argv，
看结局：

  ① 只出现一次  → 正常
  ② 出现两次    → 正常（`store_true` 累加幂等）
  ③ 出现两次 + 一个位置参数模型名 → 会不会被当成位置参数吃掉

结论预期的形状是"重复本身无害，只是多了一个 token"，所以它**不是**一条独立缺陷，
只作为事实记录进读数——避免把"两处追加"写成"会导致问题"。

    python3 probe_duplicate_flag.py <含 python/mlc_llm/support/argparse.py 的根>
    python3 probe_duplicate_flag.py --selftest <root>
"""
import argparse as stdlib_argparse
import importlib.util
import pathlib
import sys


def load_real_parser(root: pathlib.Path):
    """加载快照里**真实的** mlc_llm.support.argparse。"""
    p = root / "python/mlc_llm/support/argparse.py"
    spec = importlib.util.spec_from_file_location("_mlc_support_argparse", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ArgumentParser


def build(ArgumentParser, model_required: bool):
    """照 `cli/serve.py` 的形状造一个最小参数表（只留与本结论相关的部分）。"""
    p = ArgumentParser("probe")
    if model_required:
        p.add_argument("model", type=str)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--enable-debug", action="store_true")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8082)
    return p


CASES = {
    "once": ["m", "--device", "cuda:0", "--enable-debug", "--host", "127.0.0.1"],
    "twice": ["m", "--device", "cuda:0", "--enable-debug", "--enable-debug",
              "--host", "127.0.0.1"],
    "zero": ["m", "--device", "cuda:0", "--host", "127.0.0.1"],
}


def run(root: pathlib.Path) -> dict:
    ArgumentParser = load_real_parser(root)
    out = {}
    for name, argv in CASES.items():
        try:
            ns = build(ArgumentParser, True).parse_args(argv)
            out[name] = {"ok": True, "enable_debug": ns.enable_debug,
                         "model": getattr(ns, "model", None)}
        except SystemExit:
            out[name] = {"ok": False}
    return out


def selftest(root: pathlib.Path) -> int:
    r = run(root)
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print("%-52s %s%s" % (name, "PASS" if cond else "FAIL",
                              ("  — " + detail) if detail else ""))

    chk("once 解析成功且 enable_debug=True", r["once"]["ok"] and r["once"]["enable_debug"] is True)
    chk("twice 解析成功且 enable_debug=True", r["twice"]["ok"] and r["twice"]["enable_debug"] is True)
    chk("zero 解析成功且 enable_debug=False", r["zero"]["ok"] and r["zero"]["enable_debug"] is False)
    chk("twice 与 once 的解析结果相同（重复幂等）",
        r["twice"] == r["once"], "%s vs %s" % (r["twice"], r["once"]))
    chk("两次出现的开关没有被当成位置参数吃掉（model 仍是 m）",
        r["twice"]["model"] == "m")

    print("\nselftest: %d/5" % (5 if ok else 0))
    return 0 if ok else 1


def main(argv):
    ap = stdlib_argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json")
    ns = ap.parse_args(argv)
    root = pathlib.Path(ns.root)
    if ns.selftest:
        return selftest(root)
    rep = run(root)
    if ns.json:
        import json
        pathlib.Path(ns.json).write_text(
            json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    import json
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
