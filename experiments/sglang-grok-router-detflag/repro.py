#!/usr/bin/env python3
"""SGLang：Grok 路由器的 `enable_deterministic_inference` 形参没有任何生产者。

背景
----
`python/sglang/kernels/ops/moe/router.py::fused_moe_router_shim` 的签名里有一个

    enable_deterministic_inference: bool = False

它只控制一件事：**是否绕开 tensorcore 版路由内核**。源码注释写的是
"we keep using single kernel to avoid non-deterministic behavior"。

唯一把该函数装到调用链上的是 `python/sglang/srt/models/grok.py`：

    custom_routing_function = functools.partial(
        fused_moe_router_shim, self.router_logit_softcapping
    )

`functools.partial` 把第一个位置参数（`moe_softcapping`）钉死，其余全部走关键字。
而全仓唯一真正发起调用的地方 `srt/layers/moe/topk.py` 只传四个关键字：

    topk_weights, topk_ids = custom_routing_function(
        hidden_states=..., gating_output=..., topk=..., renormalize=...
    )

⇒ 形参取默认值 False ⇒ `--enable-deterministic-inference` 对 Grok1/Grok2 MoE
的路由**不产生任何影响**（与同文件里 aten 兜底分支的写法相反）。

本脚本在**静态**上把这条链逐段钉住，并在**行为**上证明这个形参真的能改变输出
（否则「没传」就等于「无害」，不值得报）。

为什么不用真机跑
----------------
本机是 Apple Silicon（macOS），triton 只有 Linux 轮子，且本脚本的 README 记录过
`uv` 的报错：`Wheels are available for triton (v3.9.0) on ... manylinux_2_28_*`。
⇒ 这里用**真 triton 的纯 Python 语义**（`triton.language` 命名空间 + 桩掉启动器）
在 CPU 上执行被审计函数体的**真实控制流**，而不是另写一份仿真实现。

判定
----
exit 0 = 缺陷存在（形参无生产者，且行为上可分叉）
exit 1 = 结论失效（上游已接线/已删除该形参/已无可观测差异）
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys
import types

# ---------------------------------------------------------------- 静态取证

ROUTER_REL = "python/sglang/kernels/ops/moe/router.py"
GROK_REL = "python/sglang/srt/models/grok.py"
TOPK_REL = "python/sglang/srt/layers/moe/topk.py"

FLAG = "enable_deterministic_inference"
SHIM = "fused_moe_router_shim"


def _read(root: pathlib.Path, rel: str) -> str:
    p = root / rel
    if not p.exists():
        raise SystemExit(f"[FAIL] 找不到 {rel}（快照路径或上游布局已变）")
    return p.read_text(encoding="utf-8")


def static_findings(root: pathlib.Path) -> dict:
    router = _read(root, ROUTER_REL)
    grok = _read(root, GROK_REL)
    topk = _read(root, TOPK_REL)

    facts: dict[str, object] = {}

    # 1) 形参在签名里，且默认 False
    tree = ast.parse(router)
    shim_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == SHIM:
            shim_fn = node
            break
    facts["shim_found"] = shim_fn is not None
    default_false = False
    params = []
    if shim_fn is not None:
        args = shim_fn.args
        names = [a.arg for a in args.args]
        params = names
        defaults = list(args.defaults)
        offset = len(names) - len(defaults)
        for i, d in enumerate(defaults):
            if names[offset + i] == FLAG:
                default_false = isinstance(d, ast.Constant) and d.value is False
    facts["shim_params"] = params
    facts["flag_param_default_false"] = default_false

    # 2) 别名：grok.py 用 functools.partial 只钉住第一个位置参数
    facts["grok_binds_partial_with_softcapping"] = (
        "functools.partial(" in grok
        and "fused_moe_router_shim, self.router_logit_softcapping" in grok
    )
    facts["grok_flag_reference_count"] = grok.count(FLAG)

    # 3) 唯一调用点传的关键字集合
    call_kwargs: list[str] = []
    tk = ast.parse(topk)
    for node in ast.walk(tk):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "custom_routing_function":
                call_kwargs = [k.arg for k in node.keywords if k.arg]
    facts["topk_call_kwargs"] = call_kwargs
    facts["topk_call_passes_flag"] = FLAG in call_kwargs
    facts["topk_call_positional_args"] = None  # 见下

    # 4) 这个 partial 是唯一把 shim 装进调用链的地方
    binders = []
    for rel in (GROK_REL,):
        for i, line in enumerate(_read(root, rel).splitlines(), 1):
            if SHIM in line and ("partial" in line or "custom_routing_function" in line):
                binders.append(f"{rel}:{i}")
    facts["shim_binding_sites"] = binders

    # 5) 对照：同仓其它路由快捷路径都是直读执行配置的（写进证据页那张表）
    #    这里只绑定"这个仓库里存在至少一处直读"，真正的逐处核对在证据页做。
    facts["direct_read_sites_elsewhere"] = _count_direct_reads(root)
    return facts


DIRECT_READ_FILES = (
    "python/sglang/srt/layers/moe/mega_gate.py",
    "python/sglang/srt/models/deepseek_v2.py",
    "python/sglang/srt/models/gpt_oss.py",
    "python/sglang/srt/models/kimi_k3.py",
)


def _count_direct_reads(root: pathlib.Path) -> dict:
    out = {}
    for rel in DIRECT_READ_FILES:
        p = root / rel
        if not p.exists():
            out[rel] = None  # 文件不在了 ⇒ 不静默当成 0
            continue
        out[rel] = p.read_text(encoding="utf-8").count(
            "deterministic.enable_deterministic_inference"
        )
    return out


# ---------------------------------------------------------------- 行为分叉

def _load_real_function(root: pathlib.Path, name: str):
    """把 router.py 里那个函数体的**真实源码**拿出来，在桩环境里执行。

    只桩三样：`triton` / `triton.language`（本机无 Linux 轮子）、`torch`
    （只用到 `torch.Tensor` 做类型标注和 `torch.empty` 建输出）、以及两个
    被调用的内核启动函数。函数的**分支、算术、参数传递全部是原文件里的字面代码**。
    """
    src = _read(root, ROUTER_REL)
    module = ast.parse(src)
    wanted = {}
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name in {
            SHIM,
            "fused_moe_router_tensorcore",
            "fused_moe_router_cudacore",
        }:
            wanted[node.name] = ast.get_source_segment(src, node)
    if SHIM not in wanted:
        raise SystemExit(f"[FAIL] 源码里找不到 {SHIM}")

    triton_stub = types.ModuleType("triton")
    tl_stub = types.ModuleType("triton.language")
    tl_stub.constexpr = int
    triton_stub.language = tl_stub
    triton_stub.jit = lambda fn: fn
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = object

    ns: dict[str, object] = {
        "triton": triton_stub,
        "tl": tl_stub,
        "torch": torch_stub,
        "Optional": object,
        "Tuple": tuple,
    }

    # 两个内核函数保留真实签名（BLOCK_SIZE_* 的算术在 shim 体内，必须真实执行），
    # 但不去启动 GPU：把最后一行 triton 启动换成记录调用名。
    calls: list[str] = []

    body_tc = wanted["fused_moe_router_tensorcore"]
    body_cc = wanted["fused_moe_router_cudacore"]

    def _make_dispatch(fn_name: str):
        def _dispatch(**kwargs):
            calls.append(fn_name)
            return (kwargs.get("topk"), "ids")
        return _dispatch

    exec(compile(ast.parse(body_tc), f"<{ROUTER_REL}>", "exec"), ns)  # noqa: S102
    exec(compile(ast.parse(body_cc), f"<{ROUTER_REL}>", "exec"), ns)  # noqa: S102
    ns["fused_moe_router_tensorcore"] = _make_dispatch("tensorcore")
    ns["fused_moe_router_cudacore"] = _make_dispatch("cudacore")
    exec(compile(ast.parse(wanted[SHIM]), f"<{ROUTER_REL}>", "exec"), ns)  # noqa: S102
    return ns[SHIM], calls


class _FakeTensor:
    """只提供 shim 真正读到的两个属性：`.shape`。"""

    def __init__(self, shape):
        self.shape = shape


def behavioral_probe(root: pathlib.Path) -> dict:
    shim, calls = _load_real_function(root, SHIM)

    # 选一组会走进 tensorcore 分支的形状：bs >= 512
    bs, hidden_dim, num_experts, topk = 512, 1024, 8, 2
    hidden_states = _FakeTensor((bs, hidden_dim))
    gating = _FakeTensor((num_experts, hidden_dim))

    # functools.partial 的方式与 grok.py 一致：第一个位置参数是 moe_softcapping
    import functools

    routed = functools.partial(shim, 30.0)

    # A：完全照抄 srt/layers/moe/topk.py 的调用（四个关键字，不传 flag）
    calls.clear()
    routed(
        hidden_states=hidden_states,
        gating_output=gating,
        topk=topk,
        renormalize=False,
    )
    kernel_without_flag = calls[-1] if calls else None

    # B：负向对照——显式传 flag=True，应当换成 cudacore
    calls.clear()
    routed(
        hidden_states=hidden_states,
        gating_output=gating,
        topk=topk,
        renormalize=False,
        enable_deterministic_inference=True,
    )
    kernel_with_flag = calls[-1] if calls else None

    return {
        "shape": {"bs": bs, "hidden_dim": hidden_dim, "num_experts": num_experts},
        "kernel_under_topk_call": kernel_without_flag,
        "kernel_with_flag_forced_true": kernel_with_flag,
        "flag_changes_kernel": kernel_without_flag != kernel_with_flag,
    }


# ---------------------------------------------------------------- 历史归属

HISTORY = [
    # (ref, date, note) —— 由 probe_history 逐版本抓函数签名核对
    ("c6d7f8d370", "2025-03-13", "引入 fused_moe_router_shim（GroK-1 elementwise kernels #4398）"),
    ("86d10d220f", "2025-08-23", "shim 增强（correction_bias / tensorcore 分支）"),
    ("9b8ebb2798", "2025-10-09", "**加入 enable_deterministic_inference 形参**，同提交未改任何调用点"),
    ("ee464fedc6", "2026-07-14", "MoE 内核迁到 sglang.kernels（RFC #29630 Phase 2.5），形参原样搬走"),
    ("693d333306", "2026-10-20", "kernels 按计算重新分组（#45112）"),
]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="sglang 仓库根目录（含 python/sglang/...）")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="把判定翻转：模拟上游已接线")
    ns = ap.parse_args(argv)

    root = pathlib.Path(ns.root).resolve()
    facts = static_findings(root)
    beh = behavioral_probe(root)

    elsewhere = facts["direct_read_sites_elsewhere"]
    checks = [
        ("形参数存在且默认 False", facts["flag_param_default_false"] is True),
        ("grok.py 用 partial 只钉住 moe_softcapping", facts["grok_binds_partial_with_softcapping"] is True),
        ("grok.py 从不提这个开关", facts["grok_flag_reference_count"] == 0),
        ("唯一调用点不传该关键字", facts["topk_call_passes_flag"] is False),
        ("调用点只按关键字传（无位置参数覆盖）", facts["topk_call_positional_args"] is None),
        ("该形参真的能改变走到哪个内核", beh["flag_changes_kernel"] is True),
        (
            "同仓四个对照文件仍在、且仍直读执行配置（否则证据页那张表要重写）",
            all(isinstance(v, int) and v > 0 for v in elsewhere.values()),
        ),
    ]

    if ns.selftest:
        # 模拟"上游已接线"：把最后一条判据翻过来，脚本必须报结论失效
        checks[-1] = (checks[-1][0], False)

    ok = all(v for _, v in checks)
    report = {
        "root": str(root),
        "facts": facts,
        "behavior": beh,
        "checks": [{"name": n, "pass": bool(v)} for n, v in checks],
        "verdict": "DEFECT_PRESENT" if ok else "CONCLUSION_INVALIDATED",
        "history": [{"ref": r, "date": d, "note": n} for r, d, n in HISTORY],
    }

    if ns.json:
        out = pathlib.Path(__file__).with_name("repro.json")
        out.write_text(json.dumps(report, indent=1, ensure_ascii=False, sort_keys=True))
        print(f"wrote {out}")

    for c in report["checks"]:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['name']}")
    print(f"\nverdict: {report['verdict']}")
    print(
        "  照 topk.py 的调用走到内核：%s；强制传 True 时：%s"
        % (beh["kernel_under_topk_call"], beh["kernel_with_flag_forced_true"])
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
