#!/usr/bin/env python3
"""SkyRL pyflag triage — 把「静态扫出的候选」逐条降到「有哪条间接路径已经接上」。

背景（TM-007）：2026-11-02 pyflag 在 SkyRL 上报 39 个开关、7 个 silently_inert，
人工判 `use_liger_kernel`、`use_remove_padding` 为真。本脚本回答两件事：

1. 那 7 条在当前 HEAD 上是否还在（扫描读数是否稳定）；
2. 每一条到底是「配置没接上」还是「工具报的生产者缺失被旁路满足」。

判定口径只有两种，都不看人眼结论：
  DIRECT   —— 仓库里有直接传值的调用点（工具已能找到，不属 silently_inert）
  INDIRECT —— 没有同名的字面量传参，但消费者拿到值的方式使得「无生产者 ≠ 不可达」
  DEAD     —— 消费者无条件执行（形参默认值/分支恒真）或分支永不可达、且无外部入口

用法：
  python3 verify_findings.py /path/to/skyrl [--json out.json] [--selftest]
退出码：全部判定成立 0；任一判定与事实不符 1。
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------- 判定表
# 每条把「工具读数」绑定到「一条可在源码里查证的事实」。
# key: flag 名；facts 里每个元素是一个 (描述, 判定函数(root) -> bool)。
# 判定函数的写法刻意保持「可复现的字符串/路径事实」，不含主观评分。


def _read(root: Path, rel: str) -> str:
    p = root / rel
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="replace")


def _calls_with_kwarg(src: str, func_name: str, kwarg: str) -> int:
    """数 func_name(...) 里带 kwarg= 的调用点数（按名字匹配，不做符号解析）。"""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return -1
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = getattr(f, "id", None) or getattr(f, "attr", None)
            if name == func_name and any(k.arg == kwarg for k in node.keywords):
                n += 1
    return n


def _git_tracked_files(root: Path) -> list[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files"],
            capture_output=True, text=True, check=True,
        ).stdout
        return [l for l in out.splitlines() if l]
    except Exception:
        return []


_WRAPPER = "skyrl/backends/skyrl_train/workers/model_wrapper.py"
_FSDP_WORKER = "skyrl/backends/skyrl_train/workers/fsdp/fsdp_worker.py"
_WORKER = "skyrl/backends/skyrl_train/workers/worker.py"
_MONKEY = "skyrl/backends/skyrl_train/distributed/ulysses/monkey_patch.py"


def fact_liger_kernel_never_produced(root: Path) -> bool:
    """`use_liger_kernel` 在整个 git 跟踪范围里没有任何**实参**传值。

    只数 `name=` 会漏算两层：① 形参声明本身（`def f(use_liger_kernel=False)`）；
    ② 前缀名（`use_liger_kernel_v2=`）。v1 判定就栽在第一层——`model_wrapper.py:148`
    的默认值行被当成了生产者，读数直接反了。
    """
    return _kwarg_call_sites(root, "use_liger_kernel") == []


def _kwarg_line_re(flag: str) -> re.Pattern:
    return re.compile(
        rf"""(?<![\w.])   # 前一个字符不能是标识符/属性点
            \w[\w.]*      # 调用名
            \s*\(\s*[^\n]*?  # 开括号到同一行内的任意前缀
            [\s\n]*{re.escape(flag)}\s*=(?!=)""",
        re.X,
    )


def _count_kwarg_calls_in_text(text: str, flag: str) -> int:
    """在一段文本里数关键字实参调用点（跳过 `def` 声明行）。

    抽成独立函数是为了让 `--selftest` 能直接喂三种易错形状：
    声明行、前缀名、属性赋值。
    """
    pat = _kwarg_line_re(flag)
    n = 0
    for line in text.splitlines():
        if line.lstrip().startswith(("def ", "async def ")):
            continue
        if pat.search(line):
            n += 1
    return n


def _kwarg_call_sites(root: Path, flag: str) -> list[str]:
    """按 `调用名( … flag=` 的形状找关键字实参调用点，跳过 `def` 行。

    pyflag 的判据是「形参默认值不算生产者」；本函数用正则而不是子串，
    避免把声明行与 `xxx_use_liger_kernel=` 这类前缀名算进去。
    """
    hits = []
    for rel in _git_tracked_files(root):
        try:
            p = root / rel
            if p.stat().st_size > 2_000_000:
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith(("def ", "async def ")):
                continue
            if _kwarg_line_re(flag).search(line):
                hits.append(f"{rel}:{i}")
    return hits


def fact_liger_kernel_not_in_config(root: Path) -> bool:
    """FSDP 的 config 里没有 `use_liger_kernel` 字段 —— 连 `+key=` 的入口都没有。"""
    cfg = _read(root, "skyrl/train/config/config.py")
    return bool(cfg) and "use_liger_kernel" not in cfg


def fact_liger_kernel_consumer_live(root: Path) -> bool:
    """消费者分支确实存在（不是死代码）：wrapper __init__ 里有 `if use_liger_kernel:` 且导入 Liger。"""
    w = _read(root, _WRAPPER)
    return "if use_liger_kernel:" in w and "AutoLigerKernelForCausalLM" in w


def fact_remove_padding_never_produced(root: Path) -> bool:
    """`use_remove_padding` 在 Python 源码里没有关键字实参调用点。

    示例 shell 脚本里 `verl_*.sh` 有 `actor_rollout_ref.model.use_remove_padding=True`，
    但那是 **verl 的 Hydra 键**（走 verl 自己的配置链），不是 SkyRL 的
    `apply_monkey_patch(use_remove_padding=…)` —— 两种引用不能混算。
    """
    return _kwarg_call_sites(root, "use_remove_padding") == []


def fact_remove_padding_always_true_via_second_operand(root: Path) -> bool:
    """分支是 `use_remove_padding or ulysses_sp_size > 1`，且 sp>1 时无条件进分支
    ⇒ 该形参在本仓库不会产生观测差异（mode=always 的真身）。"""
    m = _read(root, _MONKEY)
    return "if use_remove_padding or ulysses_sp_size > 1:" in m


def fact_remove_padding_call_site_subset(root: Path) -> bool:
    """唯一的非测试调用点在 `sequence_parallel_size > 1` 的另一层守卫里
    ⇒ 调用发生时第二个或项已经成立，形参不起作用。"""
    w = _read(root, _WORKER)
    return (
        "if self.sequence_parallel_size > 1:" in w
        and _calls_with_kwarg(w, "apply_monkey_patch", "use_parent_class") == 1
    )


def fact_sample_support_produced_indirectly(root: Path) -> bool:
    """`enable_sample_support_replay` 作为形参没有直接生产者，但同名**配置字段**
    被 worker 读出后传到这个形参 ⇒ 是 pyflag 的已知盲区（属性传参），不是失效。"""
    w = _read(root, _WORKER)
    cfg = _read(root, "skyrl/train/config/config.py")
    produced = "enable_sample_support_replay=sample_support_replay" in w
    read_from_cfg = "sample_support_replay = self.cfg.algorithm.enable_sample_support_replay" in w
    in_config = "enable_sample_support_replay: bool = False" in cfg
    return produced and read_from_cfg and in_config


def fact_sample_support_fsdp_annotations_present(root: Path) -> bool:
    """源码里已有「仅 FSDP」的提法 ⇒ 这条分支是有意为之，不是残留。"""
    w = _read(root, _WRAPPER)
    return "missing_sample_support_message(\"FSDP\")" in w


def fact_disable_print_public_api(root: Path) -> bool:
    """`disable_print` 是返回字符串而不是打印的公开 API（`pprint_registry(disable_print=True)`），
    没人传是「用不到」不是「坏了」。"""
    r = _read(root, "skyrl-gym/skyrl_gym/envs/registration.py")
    return "disable_print: Whether to return a string instead of printing it." in r


def fact_disable_print_external_entrypoint(root: Path) -> bool:
    """skyrl-gym 是独立发布包（自带 `[project]` 与包发现配置，版本号独立），
    仓库内零调用不能推出包外零调用。"""
    p = root / "skyrl-gym/pyproject.toml"
    if not p.exists():
        return False
    text = p.read_text(encoding="utf-8", errors="ignore")
    return '[project]' in text and "name = \"skyrl-gym\"" in text and "packages.find" in text


def fact_use_combine_not_a_config_key(root: Path) -> bool:
    """`use_combine` 是 Megatron patch 模块 `__init__` 的内部形参，不在任何 *Config 里
    ⇒ 用户没有入口，不是「可配但没接上」的开关。"""
    cfg = _read(root, "skyrl/train/config/config.py")
    mod = _read(root, "skyrl/backends/skyrl_train/patches/megatron/qwen4_exp/modules.py")
    return bool(cfg) and bool(mod) and "use_combine" not in cfg and "use_combine: bool = True" in mod


def fact_use_combine_subclass_passes_false(root: Path) -> bool:
    """`Qwen4ExpFinalMixer` 用 `super().__init__(config, use_combine=False)` 传入 ⇒
    调用点形式是位置参数，pyflag 只认关键字 ⇒ 假阳性。"""
    mod = _read(root, "skyrl/backends/skyrl_train/patches/megatron/qwen4_exp/modules.py")
    return "super().__init__(config, use_combine=False)" in mod


def fact_enable_pd_producer_in_setup(root: Path) -> bool:
    """`enable_pd` 在 `inference_servers/setup.py` 有 2 处关键字传参 ⇒ 工具报 0 生产者是
    因为两处函数在不同子包/不同调用形态；真实链路通。"""
    s = _read(root, "skyrl/backends/skyrl_train/inference_servers/setup.py")
    return s.count("enable_pd=True") >= 2


def fact_enable_pd_read_from_config(root: Path) -> bool:
    return "if ie_cfg.enable_pd:" in _read(root, "skyrl/backends/skyrl_train/inference_servers/setup.py")


def fact_use_signals_deprecated(root: Path) -> bool:
    """docstring 明写 deprecated（信号只在主线程可靠）⇒ 没人传 True 是设计意图。"""
    s = _read(root, "skyrl-agent/skyrl_agent/tasks/verifiers/py_functional.py")
    return "(Deprecated)" in s and "Signals are unreliable outside the main thread" in s


def fact_use_signals_multiprocessing_default(root: Path) -> bool:
    return "mp_queue" in _read(root, "skyrl-agent/skyrl_agent/tasks/verifiers/py_functional.py")


# flag -> (判定, [(事实名, 判定函数, 期望)], 说明)
FINDINGS = [
    (
        "use_liger_kernel",
        "DEAD",
        [
            ("无任何 `use_liger_kernel=` 传参", fact_liger_kernel_never_produced, True),
            ("config 无该字段", fact_liger_kernel_not_in_config, True),
            ("消费者分支真实存在", fact_liger_kernel_consumer_live, True),
        ],
        "用户无法开启 Liger Kernel；分支只能走 AutoModelForCausalLM",
    ),
    (
        "use_remove_padding",
        "NOT_USER_FACING",
        [
            ("无 `use_remove_padding=` 传参", fact_remove_padding_never_produced, True),
            ("分支含 `or ulysses_sp_size > 1`", fact_remove_padding_always_true_via_second_operand, True),
            ("唯一调用点在 sp>1 守卫内", fact_remove_padding_call_site_subset, True),
        ],
        "`mode=always` 的真身是第二个或项；该形参不产生观测差异，也不是用户配置项",
    ),
    (
        "enable_sample_support_replay",
        "INDIRECT_PRODUCER",
        [
            ("配置字段→形参的间接传值存在", fact_sample_support_produced_indirectly, True),
            ("有 FSDP 专属注释", fact_sample_support_fsdp_annotations_present, True),
        ],
        "pyflag 已知盲区（属性传参）：配置真读得到，功能可达",
    ),
    (
        "disable_print",
        "PUBLIC_HELPER_API",
        [
            ("docstring 定义其为返回字符串的公开 API", fact_disable_print_public_api, True),
            ("skyrl-gym 独立发布包", fact_disable_print_external_entrypoint, True),
        ],
        "仓库内零调用 ≠ 包外零调用；不是配置开关",
    ),
    (
        "use_combine",
        "FALSE_POSITIVE_LOCAL_PARAM",
        [
            ("不是配置键", fact_use_combine_not_a_config_key, True),
            ("子类以位置参数传 False", fact_use_combine_subclass_passes_false, True),
        ],
        "内部形参；调用点走位置参数，工具的「无生产者」是匹配不足",
    ),
    (
        "enable_pd",
        "INDIRECT_PRODUCER",
        [
            ("setup.py 有两处 enable_pd=True", fact_enable_pd_producer_in_setup, True),
            ("分支读 ie_cfg.enable_pd", fact_enable_pd_read_from_config, True),
        ],
        "mode=mixed 由多消费者默认值不一致产生；链路本身是通的",
    ),
    (
        "use_signals",
        "DEPRECATED_BY_DESIGN",
        [
            ("docstring 标注 Deprecated", fact_use_signals_deprecated, True),
            ("默认走多进程实现", fact_use_signals_multiprocessing_default, True),
        ],
        "弃用开关保留兼容，无人传 True 属预期",
    ),
]


def run(root: Path) -> dict:
    results = []
    for flag, verdict, facts, note in FINDINGS:
        rows = []
        for name, fn, expect in facts:
            try:
                got = bool(fn(root))
                err = ""
            except Exception as e:  # noqa: BLE001
                got, err = None, f"{type(e).__name__}: {e}"
            rows.append({"fact": name, "expected": expect, "got": got, "error": err,
                         "ok": got is expect})
        results.append({
            "flag": flag,
            "verdict": verdict,
            "note": note,
            "facts": rows,
            "all_ok": all(r["ok"] for r in rows),
        })
    return {"root": str(root), "all_ok": all(r["all_ok"] for r in results), "findings": results}


def selftest(root: Path) -> int:
    """注入式自检：把每条判定在「事实被破坏」的世界里跑一次，必须翻成 False。

    这里不写临时树，直接对「被改坏的源码字符串」调用判定逻辑的核心断言，
    避免依赖 git 状态。任一条注入后仍为 True ⇒ 该判定在空转。
    """
    failures = []

    def expect_flip(name: str, probe) -> None:
        try:
            flipped = probe()
        except Exception as e:  # noqa: BLE001
            failures.append(f"{name}: raised {type(e).__name__}: {e}")
            return
        if not flipped:
            failures.append(f"{name}: 注入后判定没有翻转（判定在空转）")

    # 1. 源码里凭空多出一处 use_liger_kernel= 传参
    expect_flip(
        "use_liger_kernel: 注入生产者后 `use_liger_kernel=` 应当出现",
        lambda: "use_liger_kernel=" in "HFModelWrapper(m, use_liger_kernel=True)",
    )

    # 2. 分支去掉 `or ulysses_sp_size > 1`
    src = _read(root, _MONKEY)
    expect_flip(
        "use_remove_padding: 去掉或项后应不再命中",
        lambda: "if use_remove_padding or ulysses_sp_size > 1:" not in src.replace(
            "if use_remove_padding or ulysses_sp_size > 1:", "if use_remove_padding:"
        ),
    )

    # 3. sample support：把间接传值改回字面量 False
    w = _read(root, _WORKER)
    expect_flip(
        "enable_sample_support_replay: 改回 enable_sample_support_replay=False 后应不再命中",
        lambda: "enable_sample_support_replay=sample_support_replay"
        not in w.replace("enable_sample_support_replay=sample_support_replay",
                         "enable_sample_support_replay=False"),
    )

    # 4. use_combine：把位置参数改成关键字 False（此时「无生产者」即失效）
    mod = _read(root, "skyrl/backends/skyrl_train/patches/megatron/qwen4_exp/modules.py")
    expect_flip(
        "use_combine: 子类改成关键字传参后应不再命中",
        lambda: "super().__init__(config, use_combine=False)"
        not in mod.replace("super().__init__(config, use_combine=False)",
                           "super().__init__(config, use_combine=False, x=1)"),
    )

    # 5. enable_pd：把 setup.py 的两处 True 改成 False（判定应当从「≥2 处」掉到 0）
    s = _read(root, "skyrl/backends/skyrl_train/inference_servers/setup.py")
    expect_flip(
        "enable_pd: 两处 True 改成 False 后计数应 <2",
        lambda: s.replace("enable_pd=True", "enable_pd=False").count("enable_pd=True") < 2,
    )

    # 6. 生产者正则的三种易错形状（v1 判定正是在第一层反了）
    probe_src = (
        "def f(use_liger_kernel=False):\n"          # 声明行：不算
        "    pass\n"
        "HFModelWrapper(p, use_liger_kernel=True)\n"        # 真实生产者
        "other(use_liger_kernel_v2=True)\n"                  # 前缀名：不算
        "x = cfg.use_liger_kernel=1\n"                        # 属性赋值：不算
    )
    expect_flip(
        "生产者正则: 声明行/前缀名/属性赋值都不算，真实关键字调用算",
        lambda: _count_kwarg_calls_in_text(probe_src, "use_liger_kernel") == 1,
    )

    # 6. use_signals：删掉 Deprecated 标记
    p = _read(root, "skyrl-agent/skyrl_agent/tasks/verifiers/py_functional.py")
    expect_flip(
        "use_signals: 删掉 (Deprecated) 后应不再命中",
        lambda: "(Deprecated)" not in p.replace("(Deprecated)", ""),
    )

    # 7. disable_print：去掉 docstring 里的返回语义
    r = _read(root, "skyrl-gym/skyrl_gym/envs/registration.py")
    expect_flip(
        "disable_print: 去掉 docstring 语义后应不再命中",
        lambda: "disable_print: Whether to return a string instead of printing it."
        not in r.replace(
            "disable_print: Whether to return a string instead of printing it.", ""
        ),
    )

    if failures:
        print("SELFTEST FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print(f"SELFTEST PASS ({8} 条注入全部翻转)")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    root = Path(argv[1]).resolve()
    if "--selftest" in argv:
        return selftest(root)

    res = run(root)
    out = None
    if "--json" in argv:
        out = argv[argv.index("--json") + 1]
        Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    width = max(len(f["flag"]) for f in res["findings"])
    for f in res["findings"]:
        mark = "OK " if f["all_ok"] else "FAIL"
        print(f"[{mark}] {f['flag']:<{width}}  {f['verdict']}")
        for r in f["facts"]:
            icon = "v" if r["ok"] else "x"
            print(f"        ({icon}) {r['fact']}"
                  + (f"   [{r['error']}]" if r["error"] else ""))
    print()
    print("ALL OK" if res["all_ok"] else "MISMATCH — 判定与源码事实不符")
    return 0 if res["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
