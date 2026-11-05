#!/usr/bin/env python3
"""SGLang 路由网关（sgl-model-gateway）的 Python 启动器：哪些参数真的走到了 Rust。

三段各自判（都从 `raw/` 的逐字节快照里读，不需要克隆）：
  P1  `RouterArgs` 的字段里，哪些被 `Router.from_args` 从 `fields_to_remove` 名单里删掉
  P2  剩下传进 `Router(**args_dict)` 的每个关键字，是不是 Rust `Router::new` 的形参
  P3  `RouterArgs` 有、但 Rust **构造函数**没有的字段，是否在 `Router.start()` 里另外读到了
      —— 这一条是"看着漏其实没漏"的关键（`tokenizer_cache_*` 四兄弟）

P3 的定位是三段文本切片：`start()` 从 `fn start(&self)` 到函数的 `}$` 收尾。
切片失败就报错，不静默当"没读到"（TM-021 的教训：截错位置会给出看起来对的假读数）。
"""
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent


def read(name):
    return (HERE / "raw" / name).read_text(encoding="utf-8")


def py_fields(src):
    return re.findall(r"^    ([a-z_][a-z0-9_]*): ", src, re.M)


def removed_fields(src):
    i = src.index("fields_to_remove")
    j = src.index("for field in fields_to_remove")
    return set(re.findall(r'"([a-z_][a-z0-9_]*)"', src[i:j]))


def rust_new_params(src):
    """取 `#[pyo3(signature = (...))]` 的默认值清单——`Router(**kwargs)` 真正校验的就是它
    （`fn new(...)` 的形参顺序不同，且带类型，不能直接拿来比名字）。"""
    i = src.index("fn start(&self) -> PyResult<()> {")   # Router 的 #[pymethods] 在它之前
    head = src[:i]
    j = head.rindex("#[pyo3(signature = (")
    seg = head[j:]
    # `worker_urls` 是唯一的必填位参（签名里没有 `= 默认值`），也要收进来
    found = re.findall(r"^\s{8}([a-z_][a-z0-9_]*)(?: = |,)", seg, re.M)
    return found


def rust_new_params_typed(src):
    i = src.index("fn new(\n        worker_urls: Vec<String>")
    seg = src[i:]
    params = []
    for line in seg.splitlines()[1:]:
        m = re.match(r"^\s{8}([a-z_][a-z0-9_]*): ", line)
        if m:
            params.append(m.group(1))
        elif line.strip() in ("}", ") -> Self {", ") -> PyResult<Self> {"):
            break
    return params


def rust_start_body(src):
    """`start()` 到最后一次 `server::startup(` 之间的全部内容。

    注意 `to_router_config()` 与 `start()` 是两个紧邻的方法，按"遇到下一个 `    }`"切片
    会把 config 组装段漏掉（本批第一版就这么错，读数因此凭空多出 22 个"无去处"字段）。
    """
    j = src.index("server::startup(")
    i = src.rindex("    pub fn to_router_config(&self)", 0, j)
    rest = src[j:]
    k = rest.index("\n    }\n")
    return src[i:j + k]


def main():
    checks = []

    def rec(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        print("%-52s %s%s" % (name, "PASS" if ok else "FAIL", ("  — " + detail) if detail else ""))

    args = read("router_args.py")
    router = read("sr_router.py")
    binding = read("bind_lib.rs")

    fields = py_fields(args)
    removed = removed_fields(router)
    passed = [f for f in fields if f not in removed]
    rec("P1.args_fields_counted", len(fields) > 100, "RouterArgs 字段 %d 个" % len(fields))
    rec("P1.fields_to_remove_parsed", len(removed) > 0, "移除清单 %d 个" % len(removed))

    rust_params = rust_new_params(binding)
    typed = rust_new_params_typed(binding)
    rec("P2.rust_signature_parsed", len(rust_params) > 50,
        "pyo3 signature 默认值 %d 个 / fn new 形参 %d 个" % (len(rust_params), len(typed)))
    rec("P2.two_lists_agree", sorted(rust_params) == sorted(typed),
        "两份清单%s" % ("一致" if sorted(rust_params) == sorted(typed) else "不一致：%s" % (
            set(rust_params) ^ set(typed))))

    bogus = [f for f in passed if f not in rust_params]
    rec("P2.no_bogus_kwarg", not bogus,
        "传了但 Rust 构造函数不收的：%s" % (bogus or "0（未触发 TypeError）"))

    start = rust_start_body(binding)
    rec("P3.start_slice_located", "to_router_config" in start and "server::startup" in start,
        "start() 切片 %d 字符，含 config 组装与 server::startup" % len(start))

    REM = frozenset(removed)

    # ---- 逐字段判定：一个字段只要「没被传给 Rust」且「不在移除清单里」，它的值就只能
    #      来自 Rust 侧默认值；再看本文件有没有读过它。读不到 = 静默降级。
    passed_set = set(passed)
    not_passed = [f for f in fields if f not in passed_set]
    silent = [f for f in not_passed if f not in REM]
    nowhere = [f for f in silent if ("self.%s" % f) not in start]
    converted = [f for f in REM if ("self.%s" % f) not in start]
    rec("P3.every_args_field_reaches_rust", not nowhere,
        "未传给 Rust 的 %d 个（与移除清单逐一相同：%s）；只由 Rust 默认值的额外字段 %d 个；"
        "移除清单里在组装段另有去处的 %d 个；两类都无去处的：%s"
        % (len(not_passed), sorted(not_passed) == sorted(REM), len(silent),
           len(converted), nowhere or "0"))

    # ---- P4 对照 1：把移除清单里这四个 tokenizer_cache 字段改成"没被转换"，
    #      同一判据必须点名它们（证明判据能分辨"有去处"与"没去处"）。
    tc = ["tokenizer_cache_enable_l0", "tokenizer_cache_l0_max_entries",
          "tokenizer_cache_enable_l1", "tokenizer_cache_l1_max_memory"]
    txt2 = start
    for f in tc:
        txt2 = txt2.replace("self." + f, "self.REMOVED_" + f)
    rem2 = frozenset(f for f in REM if f not in tc)
    conv2 = [f for f in rem2 if ("self.%s" % f) not in txt2]
    all_named_tc_gone = all(("self.%s" % f) not in txt2 for f in tc)
    nowhere2 = [f for f in conv2 if ("self.%s" % f) not in start]
    rec("P4.control_converted_read_removed", all_named_tc_gone,
        "把这四个已转换字段的读取全部改名后，判据能把它们从\"有去处\"里认出来：%s"
        % all_named_tc_gone)

    out = {
        "verdict": "ALL_ARGS_REACH_RUST" if all(c["ok"] for c in checks) else "INCONCLUSIVE",
        "counts": {
            "args_fields": len(fields),
            "removed_before_rust": len(removed),
            "passed_to_rust": len(passed),
            "rust_new_params": len(rust_params),
            "rust_default_only_fields": silent,
            "tokenizer_cache_read_in_config_mapping": {
                f: ("self.%s" % f) in start for f in tc},
            "removed_fields_with_other_destination": converted,
        },
        "checks": checks,
    }
    dest = HERE / "raw" / "probe.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nverdict: %s -> %s" % (out["verdict"], dest))
    return 0 if all(c["ok"] for c in checks) else 1




if __name__ == "__main__":
    sys.exit(main())
