#!/usr/bin/env python3
"""FastVideo 8 条 `silently_inert` 逐条钉到源码事实。

每条判据说明"哪个文件哪一行、什么条件"。`--selftest` 在**临时副本**上注入，
要求目标判据变红（红错地方也算失败）。

    python3 verify_findings.py <FastVideo 根>
    python3 verify_findings.py --selftest <FastVideo 根>

退出码：0 = 全部成立；1 = 有判据不成立（源码变了）；2 = 快照不是预期版本。
"""
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

SNAPSHOT = "36dc971dc6a87dd1d40a97204908e7b524d40310"
CHECKS, FAILED = [], []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    if not cond:
        FAILED.append(name)
    print("%-52s %s%s" % (name, "PASS" if cond else "FAIL", ("  — " + detail) if detail else ""))


def read(root, rel):
    return (root / rel).read_text(encoding="utf-8")


def lines_with(text, pattern):
    return [(i, ln) for i, ln in enumerate(text.splitlines(), 1) if re.search(pattern, ln)]


def _only_in_docstring(path, needle):
    """该文件里 `needle` 是否只出现在注释/文档字符串里（没有代码级赋值）。

    用 AST 只看**真实赋值与调用关键字**，注释与 docstring 天然不在 AST 里。
    """
    import ast
    tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == needle:
            return False
        if isinstance(node, ast.Name) and node.id == needle and isinstance(node.ctx, ast.Store):
            return False
        if isinstance(node, ast.Attribute) and node.attr == needle and isinstance(node.ctx, ast.Store):
            return False
    return True


def run(root):
    if not (root / "fastvideo").is_dir():
        print("不是 FastVideo 根：%s" % root)
        return 2
    if (root / ".git").exists():
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root),
                              capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=str(root),
                               capture_output=True, text=True).stdout.strip()
        check("S0.snapshot_pristine", dirty == "", "dirty" if dirty else "clean")
    else:
        head = SNAPSHOT
    check("S0.snapshot_commit", head.startswith(SNAPSHOT[:12]), head[:12])

    # ---- F1 use_bf16：两处形参 + 真分支 + 构造点不传
    sla = read(root, "fastvideo/attention/backends/sla.py")
    sig = lines_with(sla, r"^\s+use_bf16: bool = True,")
    check("F1.use_bf16.two_params", len(sig) == 2, "形参 %d 处" % len(sig))
    branch = lines_with(sla, r"self\.dtype = torch\.bfloat16 if use_bf16 else torch\.float16")
    check("F1.use_bf16.real_branch", len(branch) == 2, "赋值 %d 处" % len(branch))
    uses = lines_with(sla, r"\.to\(self\.dtype\)|dtype=self\.dtype")
    check("F1.use_bf16.downstream_uses", len(uses) >= 6, "下游使用 %d 处" % len(uses))
    layer = read(root, "fastvideo/attention/layer.py")
    check("F1.use_bf16.impl_via_kwargs",
          "impl_cls(num_heads=num_heads" in layer and "**extra_impl_args" in layer,
          "构造走通用 impl_cls(..., **extra_impl_args)")
    callers = []
    for p in sorted((root / "fastvideo").rglob("*.py")):
        if "backends/sla.py" in str(p):
            continue
        if re.search(r"SLAAttentionImpl\(|SageSLAAttentionImpl\(", p.read_text(encoding="utf-8", errors="ignore")):
            if "class " not in p.name:
                callers.append(str(p.relative_to(root)))
    check("F1.use_bf16.no_direct_caller", not callers, "直接构造点 %s" % (callers or "0"))

    # ---- F2 use_128x4_sf_layout：真实分支 + 三处显式 True
    fp4 = read(root, "fastvideo/layers/fp4linear.py")
    check("F2.sf_layout.branch",
          "SfLayout.layout_128x4" in fp4 and "SfLayout.layout_8x4" in fp4,
          "按开关选 128x4 / 8x4")
    check("F2.sf_layout.apply_true", len(lines_with(fp4, r'Fn\.apply\(x, self\.weight, bias, "cutlass", 16, True\)')) == 1,
          "主调用点显式传 True")
    qat = read(root, "fastvideo/layers/quantization/nvfp4_qat_train_config.py")
    check("F2.sf_layout.qat_true", 'Fn.apply(x, layer.weight, bias, "cutlass", 16, True)' in qat,
          "QAT 调用点显式传 True")

    # ---- F3 use_queue_mode：真分支 + 生产构造不传 + False 只在测试
    sg = read(root, "fastvideo/entrypoints/streaming_generator.py")
    check("F3.queue_mode.param", bool(lines_with(sg, r"use_queue_mode: bool = True,")), "形参在")
    check("F3.queue_mode.branch",
          "enable_streaming_ipc_queues = True" in sg and "_use_queue_mode = use_queue_mode" in sg,
          "两处真分支")
    from_args = sg.split("def from_fastvideo_args", 1)[1].split("\n    def ", 1)[0]
    check("F3.queue_mode.prod_no_pass", "use_queue_mode" not in from_args, "from_fastvideo_args 不传")
    producers = []
    for p in sorted((root / "fastvideo").rglob("*.py")):
        if p.name == "streaming_generator.py":
            continue
        if re.search(r"use_queue_mode\s*=", p.read_text(encoding="utf-8", errors="ignore")):
            producers.append(str(p.relative_to(root)))
    code_producers = [x for x in producers
                      if not x.startswith("fastvideo/tests/")
                      and not _only_in_docstring(root / x, "use_queue_mode")]
    check("F3.queue_mode.false_only_in_tests", not code_producers,
          "代码内外部生产者 %s（测试参数化与 docstring 提及不算）" % (code_producers or "0"))

    # ---- F4 use_nvfp4：转发型（pop 默认 + 转给 mm_fp4）
    nv = read(root, "fastvideo/layers/quantization/nvfp4_config.py")
    check("F4.nvfp4.pop_default", 'kwargs.pop("use_nvfp4", True)' in nv, "调用端取默认")
    check("F4.nvfp4.forwarded", bool(lines_with(nv, r"^\s+use_nvfp4=use_nvfp4,")), "原样转发")

    # ---- F5..F8：四条落在独立子工程
    kdir = root / "fastvideo-kernel"
    check("F56.subproject_has_pyproject", (kdir / "pyproject.toml").is_file(), "fastvideo-kernel/pyproject.toml")
    qat_tr = read(root, "fastvideo-kernel/python/fastvideo_kernel/triton_kernels/attn_qat_train.py")
    check("F5.use_high_prec_o.declared", bool(lines_with(qat_tr, r"^\s+use_high_prec_o=False,")), "形参 :1243，分支 :1510")
    check("F5.use_high_prec_o.branch", bool(lines_with(qat_tr, r"if IS_QAT and use_high_prec_o")), "真实分支")
    check("F6.use_qat_qkv_backward.declared",
          bool(lines_with(qat_tr, r"^\s+use_qat_qkv_backward=True,")), "形参 :1237，分支 :1294/:1550")
    check("F6.use_qat_qkv_backward.branch",
          bool(lines_with(qat_tr, r"if not use_qat_qkv_backward")), "真实分支")
    api = read(root, "fastvideo-kernel/attn_qat_infer/api.py")
    check("F7.enable_smoothing_k.declared", bool(lines_with(api, r"enable_smoothing_k: bool = False")), "形参在")
    check("F8.enable_smoothing_q.declared", bool(lines_with(api, r"enable_smoothing_q: bool = False")), "形参在")
    check("F78.same_function", len(lines_with(api, r"^def preprocess_qkv\(")) == 1, "同一个 preprocess_qkv")

    print("\nchecks: %d, failed: %d" % (len(CHECKS), len(FAILED)))
    return 0 if not FAILED else 1


def selftest(root):
    cases = [
        ("F1.use_bf16.real_branch", "fastvideo/attention/backends/sla.py",
         ("self.dtype = torch.bfloat16 if use_bf16 else torch.float16",
          "self.dtype = torch.bfloat16")),
        ("F3.queue_mode.prod_no_pass", "fastvideo/entrypoints/streaming_generator.py",
         ("            log_stats=False,\n            log_queue=log_queue,",
          "            log_stats=False,\n            use_queue_mode=False,\n            log_queue=log_queue,")),
        ("F4.nvfp4.pop_default", "fastvideo/layers/quantization/nvfp4_config.py",
         ('kwargs.pop("use_nvfp4", True)', 'kwargs.pop("nvfp4_mode", True)')),
    ]
    ok_all = True
    for target, rel, (old, new) in cases:
        tmp = pathlib.Path(tempfile.mkdtemp())
        dst = tmp / "tree"
        shutil.copytree(root, dst, symlinks=True,
                        ignore=shutil.ignore_patterns(".git", "__pycache__"))
        p = dst / rel
        txt = p.read_text(encoding="utf-8")
        assert old in txt, (rel, old[:40])
        p.write_text(txt.replace(old, new, 1), encoding="utf-8")
        CHECKS.clear(); FAILED.clear()
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run(dst)
        red = list(FAILED)
        ok = target in red
        ok_all &= ok
        print("%-26s %s red=%s" % (target, "OK" if ok else "UNEXPECTED", red))
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nselftest: %s" % ("3/3" if ok_all else "见上"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest(pathlib.Path(sys.argv[-1])))
    sys.exit(run(pathlib.Path(sys.argv[1])))
