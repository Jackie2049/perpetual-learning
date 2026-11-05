#!/usr/bin/env python3
"""SkyRL HEAD（f854d6d9）pyflag 七条候选的逐条判据。

用法：
    python3 probe.py --snapshot raw/snapshot [--repo /path/to/SkyRL checkout]
    python3 probe.py --snapshot raw/snapshot --selftest

退出码：0 = 全部判据成立；1 = 有判据不成立（上游已变，结论需复核）；
        2 = 快照缺失／自检失败。

设计约束（都是被前几批打过的）：
  * 每条判据都**先指向具体行号**再断言，不靠整文件子串；
  * 「无生产者」的判据不能把声明行（`name=False,` 在 def 参数表里）算成生产者；
  * `--selftest` 的每条注入都必须让**指定的那条**判据变红，只红别处算失败。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile

CHECKS = []  # (id, group, description, fn)  fn(snapshot_root) -> (ok, detail)


def check(cid, group, desc):
    def deco(fn):
        CHECKS.append((cid, group, desc, fn))
        return fn
    return deco


def read(root, rel):
    p = os.path.join(root, rel)
    with open(p, encoding="utf-8") as f:
        return f.read()


def lines(root, rel):
    return read(root, rel).split("\n")


MW = "skyrl/backends/skyrl_train/workers/model_wrapper.py"
FSDP = "skyrl/backends/skyrl_train/workers/fsdp/fsdp_worker.py"
CFG = "skyrl/train/config/config.py"
MONKEY = "skyrl/backends/skyrl_train/distributed/ulysses/monkey_patch.py"
WORKER = "skyrl/backends/skyrl_train/workers/worker.py"
REG = "skyrl-gym/skyrl_gym/envs/registration.py"
PYFUNC = "skyrl-agent/skyrl_agent/tasks/verifiers/py_functional.py"
Q4 = "skyrl/backends/skyrl_train/patches/megatron/qwen4_exp/modules.py"
Q4SPEC = "skyrl/backends/skyrl_train/patches/megatron/qwen4_exp/layer_specs.py"
SETUP = "skyrl/backends/skyrl_train/inference_servers/setup.py"
MAINBASE = "skyrl/train/entrypoints/main_base.py"


def _find(ls, pat, start=1):
    rx = re.compile(pat)
    return [i for i, l in enumerate(ls, 1) if i >= start and rx.search(l)]


# ---------------------------------------------------------------- use_liger_kernel
@check("L1", "use_liger_kernel", "形参 `use_liger_kernel=False` 在 HFModelWrapper.__init__ 形参表里")
def l1(root):
    ls = lines(root, MW)
    ds = [i for i in _find(ls, r"^\s*def __init__\(") ]
    ps = _find(ls, r"^\s*use_liger_kernel=False,\s*$")
    ok = bool(ds) and bool(ps) and ds[0] < ps[0] and ps[0] - ds[0] < 60
    return ok, f"def __init__ @{ds[:1]}, 形参 @{ps[:1]}"


@check("L2", "use_liger_kernel", "分支 `if use_liger_kernel:` 存在，且 import 真实存在")
def l2(root):
    ls = lines(root, MW)
    br = _find(ls, r"^\s*if use_liger_kernel:\s*$")
    imp = _find(ls, r"from liger_kernel\.transformers import AutoLigerKernelForCausalLM")
    ok = bool(br) and bool(imp) and br[0] < imp[0] < br[0] + 5
    return ok, f"分支 @{br[:1]}, import @{imp[:1]}"


@check("L3", "use_liger_kernel", "fsdp_worker 两处 HFModelWrapper(...) 均不传该关键字")
def l3(root):
    ls = lines(root, FSDP)
    calls = _find(ls, r"HFModelWrapper\(")
    if len(calls) < 2:
        return False, f"只找到 {len(calls)} 处 HFModelWrapper(（应为 2）"
    # 取每处调用到闭括号
    bad = []
    for c in calls:
        depth = 0
        seg = []
        for l in ls[c - 1: c + 60]:
            seg.append(l)
            depth += l.count("(") - l.count(")")
            if depth <= 0:
                break
        if "use_liger_kernel" in "\n".join(seg):
            bad.append(c)
    return not bad, f"调用点 @{calls}, 传了该键的 @{bad}"


@check("L4", "use_liger_kernel", "配置 dataclass 里没有 use_liger_kernel 字段")
def l4(root):
    src = read(root, CFG)
    hits = [i for i, l in enumerate(src.split("\n"), 1) if "use_liger_kernel" in l]
    return not hits, f"config.py 命中行 {hits}"


@check("L5", "use_liger_kernel", "声明文件之外的非测试 .py 里没有任何人传这个关键字")
def l5(root):
    hits = []
    me = os.path.dirname(os.path.abspath(__file__))
    # 注入用的临时树拷在本目录下，扫描时往上找到含 skyrl/ 的那一层，别把探针自己算进去
    start = os.path.abspath(root)
    if os.path.commonpath([me, start]) == me:
        cur = start
        while not os.path.isdir(os.path.join(cur, "skyrl")) and os.path.dirname(cur) != me:
            cur = os.path.dirname(cur)
        root = cur if os.path.isdir(os.path.join(cur, "skyrl")) else start
    os_walk_root = root
    for dirpath, _, files in os.walk(os_walk_root):
        for f in files:
            if not f.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, f), root)
            if "test" in rel.lower() or "benchmark" in rel.lower():
                continue
            if rel == "skyrl/backends/skyrl_train/workers/model_wrapper.py":
                continue  # 声明文件本身由 L1/L2 管，这里管「别处有没有人传」
            for i, l in enumerate(open(os.path.join(dirpath, f), encoding="utf-8", errors="replace"), 1):
                if "use_liger_kernel" in l:
                    hits.append((rel, i, l.strip()))
    producers = [h for h in hits if re.search(r"[\w\)]use_liger_kernel\s*=|use_liger_kernel\s*=\s*(True|False)\s*[,\)]", h[2]) and not re.match(r"^\s*use_liger_kernel=False,\s*$", h[2])]
    return not producers, f"全仓非模型包装文件里命中 {len(hits)} 处 {[h[:2] for h in hits]}, 生产者 {producers}"


# --------------------------------------------------------------- use_remove_padding
@check("R1", "use_remove_padding", "apply_monkey_patch 的分支是 `use_remove_padding or ulysses_sp_size > 1`")
def r1(root):
    ls = lines(root, MONKEY)
    hit = _find(ls, r"if use_remove_padding or ulysses_sp_size > 1:")
    dep = _find(ls, r"def apply_monkey_patch\(")
    ok = bool(hit) and bool(dep) and dep[0] < hit[0] < dep[0] + 60
    return ok, f"def @{dep[:1]}, 分支 @{hit[:1]}"


@check("R2", "use_remove_padding", "唯一调用点在 sequence_parallel_size > 1 守卫内且不传该形参")
def r2(root):
    ls = lines(root, WORKER)
    guard = _find(ls, r"if self\.sequence_parallel_size > 1:")
    call = _find(ls, r"apply_monkey_patch\(")
    if not guard or not call:
        return False, f"守卫 @{guard}, 调用 @{call}"
    body = "\n".join(ls[call[0] - 1: call[0] + 3])
    ok = guard[0] < call[0] < guard[0] + 8 and "use_remove_padding" not in body
    return ok, f"守卫 @{guard}, 调用 @{call}, 调用段含该名={('use_remove_padding' in body)}"


# ------------------------------------------------- enable_sample_support_replay
@check("S1", "enable_sample_support_replay", "配置里有字段，worker 取局部变量后按关键字传给 forward")
def s1(root):
    cfg = read(root, CFG)
    field = [i for i, l in enumerate(cfg.split("\n"), 1) if re.match(r"^\s*enable_sample_support_replay: bool = ", l)]
    ls = lines(root, WORKER)
    assign = _find(ls, r"sample_support_replay\s*=")
    kw = _find(ls, r"enable_sample_support_replay=sample_support_replay,")
    n = sum(l.count("enable_sample_support_replay=sample_support_replay,") for l in ls)
    ok = bool(field) and bool(assign) and n == 4
    return ok, f"config 字段 @{field}, worker 赋值 @{assign[:1]}, 关键字实参 {n} 处（期望 4）"


# ------------------------------------------------------------------ enable_pd
@check("P1", "enable_pd", "配置里有 enable_pd 字段，setup/main_base 读 `ie_cfg.enable_pd` 后显式传 True")
def p1(root):
    cfg = read(root, CFG)
    field = [i for i, l in enumerate(cfg.split("\n"), 1) if re.match(r"^\s*enable_pd: bool = ", l)]
    st = lines(root, SETUP)
    reads = _find(st, r"ie_cfg\.enable_pd")
    true_n = sum(l.count("enable_pd=True") for l in st)
    mb = lines(root, MAINBASE)
    mb_read = _find(mb, r"self\.cfg\.generator\.inference_engine\.enable_pd")
    ok = bool(field) and bool(reads) and true_n == 3 and bool(mb_read)
    return ok, f"config @{field}, setup 读 {reads}, 显式 True {true_n} 处（期望 3）, main_base 读 @{mb_read}"


# ------------------------------------------------------------------ use_combine
@check("C1", "use_combine", "基类形参默认 True，子类用位置参数传 False")
def c1(root):
    ls = lines(root, Q4)
    base = _find(ls, r"def __init__\(self, config: TransformerConfig, use_combine: bool = True\)")
    sub = _find(ls, r"super\(\)\.__init__\(config, use_combine=False\)")
    cls_sub = _find(ls, r"^class Qwen4ExpFinalMixer\(")
    ok = bool(base) and bool(sub) and bool(cls_sub) and cls_sub[0] < sub[0] < cls_sub[0] + 15
    return ok, f"基类 @{base}, 子类 @{cls_sub}, super @{sub}"


@check("C2", "use_combine", "生产构造点 Qwen4ExpGatedResidual(config) 走默认 True（只有 2 处）")
def c2(root):
    ls = lines(root, Q4SPEC)
    prod = _find(ls, r"= Qwen4ExpGatedResidual\(config\)\s*$")
    return len(prod) == 2, f"生产构造点 {prod}"


# ------------------------------------------------------------------ use_signals
@check("G1", "use_signals", "形参 docstring 标 (Deprecated)，默认 False，所有调用点都用默认")
def g1(root):
    ls = lines(root, PYFUNC)
    sig = _find(ls, r"^def timeout_limit\(seconds: float, use_signals: bool = False\):")
    dep = _find(ls, r"use_signals: \(Deprecated\)")
    ok = bool(sig) and bool(dep) and sig[0] < dep[0] < sig[0] + 12
    return ok, f"签名 @{sig}, (Deprecated) @{dep}"


# ---------------------------------------------------------------- disable_print
@check("D1", "disable_print", "pprint 是公开 helper：docstring 明写返回字符串而不是打印")
def d1(root):
    ls = lines(root, REG)
    fn = _find(ls, r"^\s*def pprint\(\s*$")
    doc = _find(ls, r"If ``disable_print is True`` a string otherwise", start=(fn[0] + 1 if fn else 1))
    ok = bool(fn) and bool(doc) and fn[0] < doc[0] < fn[0] + 20
    return ok, f"def pprint @{fn}, docstring @{doc}"


# liger_producer_added 同时命中 L3（调用点）与 L5（全仓无生产者）；两者都该红，写在注入自己身上
EXPECTED_RED = {"liger_producer_added": ["L3", "L5"]}

GROUPS = ["use_liger_kernel", "use_remove_padding", "enable_sample_support_replay",
          "enable_pd", "use_combine", "use_signals", "disable_print"]


def run(root):
    results = []
    for cid, group, desc, fn in CHECKS:
        try:
            ok, detail = fn(root)
        except Exception as e:  # 快照缺文件等
            ok, detail = False, f"exception: {type(e).__name__}: {e}"
        results.append({"id": cid, "group": group, "desc": desc, "ok": bool(ok), "detail": detail})
    return results


# ------------------------------------------------------------------- selftest
def _mutate(tmp, rel, fn):
    p = os.path.join(tmp, rel)
    src = open(p, encoding="utf-8").read()
    new = fn(src)
    assert new != src, f"注入没改到 {rel}"
    open(p, "w", encoding="utf-8").write(new)


def _copy_snapshot(src, dst):
    for dirpath, _, files in os.walk(src):
        for f in files:
            s = os.path.join(dirpath, f)
            d = os.path.join(dst, os.path.relpath(s, src))
            os.makedirs(os.path.dirname(d), exist_ok=True)
            open(d, "w", encoding="utf-8").write(open(s, encoding="utf-8").read())


INJECTIONS = [
    # (名字, 改哪个文件, 改法, 必须变红的 check id, 其余 check 必须不变)
    ("liger_producer_added", FSDP,
     lambda s: s.replace("            logprobs_chunk_size=self.cfg.logprobs_chunk_size,\n        )",
                         "            logprobs_chunk_size=self.cfg.logprobs_chunk_size,\n            use_liger_kernel=True,\n        )"),
     "L3"),
    ("liger_config_field", CFG,
     lambda s: s.replace("    enable_sample_support_replay: bool = False",
                         "    use_liger_kernel: bool = False\n    enable_sample_support_replay: bool = False", 1),
     "L4"),
    ("liger_branch_removed", MW,
     lambda s: s.replace("            if use_liger_kernel:", "            if False:", 1),
     "L2"),
    ("remove_padding_guard_dropped", MONKEY,
     lambda s: s.replace("if use_remove_padding or ulysses_sp_size > 1:", "if ulysses_sp_size > 1:", 1),
     "R1"),
    ("remove_padding_passed_at_callsite", WORKER,
     lambda s: s.replace("use_parent_class=use_parent_class\n            )",
                         "use_parent_class=use_parent_class, use_remove_padding=False\n            )", 1),
     "R2"),
    ("ssr_kwarg_removed", WORKER,
     lambda s: s.replace("enable_sample_support_replay=sample_support_replay,", "", 1),
     "S1"),
    ("pd_true_arg_removed", SETUP,
     lambda s: s.replace("                enable_pd=True,", "                enable_pd=False,", 1),
     "P1"),
    ("combine_subclass_now_true", Q4,
     lambda s: s.replace("super().__init__(config, use_combine=False)", "super().__init__(config, use_combine=True)", 1),
     "C1"),
    ("gated_residual_gets_explicit_arg", Q4SPEC,
     lambda s: s.replace("= Qwen4ExpGatedResidual(config)", "= Qwen4ExpGatedResidual(config, use_combine=True)"),
     "C2"),
    ("signals_docstring_unmarked", PYFUNC,
     lambda s: s.replace("use_signals: (Deprecated)", "use_signals: Legacy option"),
     "G1"),
    ("pprint_docstring_changed", REG,
     lambda s: s.replace("If ``disable_print is True`` a string otherwise", "Always prints"),
     "D1"),
]


def selftest(root):
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".selftest-tmp"), exist_ok=True)
    base = run(root)
    base_ids_failing = [r["id"] for r in base if not r["ok"]]
    if base_ids_failing:
        print(f"[selftest] 基线就有判据不成立：{base_ids_failing}", file=sys.stderr)
        return False
    allok = True
    for name, rel, fn, target in INJECTIONS:
        with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(__file__)) + "/.selftest-tmp") as tmp:
            _copy_snapshot(root, tmp)
            try:
                _mutate(tmp, rel, fn)
            except AssertionError as e:
                print(f"[selftest] {name}: 注入失败 {e}", file=sys.stderr)
                allok = False
                continue
            res = run(tmp)
            red = [r["id"] for r in res if not r["ok"]]
            expected = EXPECTED_RED.get(name, [target])
            good = red == expected
            print(f"[selftest] {name}: 变红 {red} 期望 {expected} -> {'PASS' if good else 'FAIL'}")
            allok &= good
    print(f"[selftest] {'ALL PASS' if allok else 'FAILURES'} ({len(INJECTIONS)} 条注入)")
    return allok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if not os.path.isdir(a.snapshot):
        print(f"快照目录不存在：{a.snapshot}", file=sys.stderr)
        return 2
    if a.selftest:
        return 0 if selftest(a.snapshot) else 2
    res = run(a.snapshot)
    for g in GROUPS:
        for r in res:
            if r["group"] == g:
                print(f"{'PASS' if r['ok'] else 'FAIL'} {r['id']:<3} {r['desc']}  |  {r['detail']}")
    bad = [r["id"] for r in res if not r["ok"]]
    print(f"\n{len(res) - len(bad)}/{len(res)} 通过" + (f"；不成立：{bad}" if bad else ""))
    if a.json:
        json.dump({"snapshot": os.path.abspath(a.snapshot), "checks": res,
                   "passed": len(res) - len(bad), "total": len(res)}, open(a.json, "w"), indent=1, ensure_ascii=False)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
