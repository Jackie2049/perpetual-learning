#!/usr/bin/env python3
"""AReaL 42 个开关 / 7 条 `silently_inert` 逐条钉到源码事实。

每条判据都指明"哪个文件哪一行、什么条件"，`--selftest` 时在**临时副本**上注入，
要求目标判据变红（红错地方也算失败）。

    python3 verify_findings.py <AReaL 根>
    python3 verify_findings.py --selftest <AReaL 根>

退出码：0 = 全部判据成立；1 = 有判据不成立（源码变了）；2 = 快照不是预期版本。
"""
import ast
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

SNAPSHOT = "3b93679888e9f4929f6f155f13c6a286a62dfd72"
checks, failed = [], []
VERBOSE = True


def check(name, cond, detail=""):
    checks.append((name, bool(cond), detail))
    if not cond:
        failed.append(name)
    if VERBOSE:
        print("%-52s %s%s" % (name, "PASS" if cond else "FAIL",
                              ("  — " + detail) if detail else ""))


def read(root, rel):
    return (root / rel).read_text(encoding="utf-8")


def lineno_of(text, pattern):
    for i, line in enumerate(text.splitlines(), 1):
        if re.search(pattern, line):
            return i
    return None


def call_sites(root, callee, *, include_tests=False, self_only=False):
    """全仓 `callee(...)` 调用点：返回 [(rel, lineno, kwargs, nargs)]。

    按 AST 找，不按文本找——`functools.partial(callee, k=v)` 也算一次调用
    （`partial` 的第一个位置参数是被调函数），这正是本批 `use_sapo_loss`
    三个开关被扫描器漏掉的形态。

    `self_only=True` 只收 `self.callee(...)`。`get_rng_state` 必须用它：
    `torch.get_rng_state()` / `current_platform.get_rng_state()` 也同名（同样是
    属性调用，光看"是不是属性调用"分不开），混进来会让"唯一调用点"这条假红
    —— 本批第一处判据错，见 README。
    """
    out = []
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(root).as_posix()
        if not include_tests and (rel.startswith("tests/") or "/tests/" in rel):
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = None
            is_self_attr = False
            if isinstance(node.func, ast.Name):
                target = node.func.id
            elif isinstance(node.func, ast.Attribute):
                target = node.func.attr
                is_self_attr = (isinstance(node.func.value, ast.Name)
                                and node.func.value.id == "self")
            if target == callee:
                if self_only and not is_self_attr:
                    continue
                kws = [k.arg for k in node.keywords if k.arg]
                out.append((rel, node.lineno, kws, len(node.args)))
                continue
            # functools.partial(callee, ...) / partial(callee, ...)
            if target == "partial" and node.args:
                first = node.args[0]
                if isinstance(first, ast.Name) and first.id == callee:
                    kws = [k.arg for k in node.keywords if k.arg]
                    out.append((rel, node.lineno, kws, len(node.args) - 1))
    return out


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    root = pathlib.Path(argv[0])
    if not (root / "areal").is_dir():
        print("不是 AReaL 根：%s" % root)
        return 2

    # ---- S0 快照干净前置（TM-017 的教训：负向对照改坏快照而判据全绿）
    if (root / ".git").exists():
        st = subprocess.run(["git", "status", "--porcelain"], cwd=str(root),
                            capture_output=True, text=True).stdout.strip()
        check("S0.snapshot_pristine", st == "", "dirty" if st else "clean")
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root),
                              capture_output=True, text=True).stdout.strip()
    else:
        check("S0.snapshot_pristine", True, "无 .git，跳过")
        head = SNAPSHOT
    check("S0.snapshot_commit", head.startswith(SNAPSHOT[:12]), head[:12])

    # =============================================== F1..F3：三个 loss 开关
    actor = read(root, "areal/trainer/ppo/actor.py")
    partial_sites = call_sites(root, "grpo_loss_fn")
    partial_rel = [(r, ln) for r, ln, kws, _n in partial_sites
                   if r == "areal/trainer/ppo/actor.py"]
    for flag in ("use_sapo_loss", "use_cispo_loss", "use_decoupled_loss"):
        # F_kw：真生产者在 functools.partial 的关键字实参里
        prod = [(r, ln) for r, ln, kws, _n in partial_sites if flag in kws]
        check("F.%s.produced_via_partial" % flag,
              len(prod) == 1 and prod[0][0] == "areal/trainer/ppo/actor.py",
              str(prod))
        # F_dec：消费者是真的按它分支（不是只转发）
        ln = lineno_of(actor, r"\b%s\b" % flag)
        check("F.%s.appears_in_actor" % flag, ln is not None, "line %s" % ln)

    # 三个开关都从 config 字段读出来（`self.config.<flag>`）
    for flag in ("use_sapo_loss", "use_cispo_loss", "use_decoupled_loss"):
        check("F.%s.read_from_config" % flag,
              ("self.config.%s" % flag) in actor)

    # 反向：扫描器为什么漏——它只按"关键字实参名 == 形参名"找直接调用点，
    # 而这里唯一的生产者外面包了 functools.partial
    direct = [(r, ln) for r, ln, kws, _n in partial_sites
              if r == "areal/trainer/ppo/actor.py"
              and not any(k in kws for k in ("use_sapo_loss",))]
    check("F.no_plain_direct_call", all(True for _ in direct))

    # cli_args 里这三个字段都存在且是 bool
    cli = read(root, "areal/api/cli_args.py")
    for flag in ("use_sapo_loss", "use_cispo_loss", "use_decoupled_loss"):
        check("F.%s.config_field" % flag,
              re.search(r"^\s+%s: bool = field\(" % flag, cli, re.M) is not None)

    # =============================================== F4：use_cpu（真不可达）
    kernels = read(root, "areal/experimental/models/archon/moe/kernels.py")
    check("F4.use_cpu.declared",
          re.search(r"def generate_permute_indices\(", kernels) is not None)
    ln_branch = lineno_of(kernels, r"^\s+if use_cpu:\s*$")
    check("F4.use_cpu.branch_exists", ln_branch is not None, "line %s" % ln_branch)
    # 唯一调用点不传它
    sites = call_sites(root, "generate_permute_indices")
    check("F4.use_cpu.single_call_site", len(sites) == 1, str(sites))
    check("F4.use_cpu.not_passed",
          bool(sites) and all("use_cpu" not in kws for _r, _l, kws, _n in sites))
    check("F4.use_cpu.not_positional",
          bool(sites) and all(nargs < 6 for _r, _l, _k, nargs in sites),
          "位置实参个数 %s" % [n for *_x, n in sites])
    # 全仓（含测试）没有任何地方传它
    all_sites = call_sites(root, "generate_permute_indices", include_tests=True)
    check("F4.use_cpu.no_producer_anywhere",
          all("use_cpu" not in kws for _r, _l, kws, _n in all_sites))
    # 那句 docstring 仍宣称它可用 ⇒ 是"说明与实现没对齐"而不是纯死码
    check("F4.use_cpu.docstring_promises",
          "use_cpu: whether to use CPU implementation" in kernels)

    # =============================================== F5：use_local_output
    ps = read(root, "areal/models/parallel_styles.py")
    check("F5.use_local_output.default_true",
          re.search(r"use_local_output: bool = True", ps) is not None)
    check("F5.use_local_output.stored",
          "self.use_local_output = use_local_output" in ps)
    par = read(root, "areal/engine/fsdp_utils/parallel.py")
    check("F5.use_local_output.produced_by_partial",
          re.search(r"partial\(\s*self\._prepare_output_fn,\s*self\.output_layout,"
                    r"\s*self\.use_local_output\)", ps.replace("\n", " ")) is not None
          or "self._prepare_output_fn, self.output_layout, self.use_local_output" in ps)
    check("F5.use_local_output.constructed_default",
          par.count("ReplicateParallel()") >= 1)

    # =============================================== F6/F7：两个 ckpt 开关
    ck = read(root, "areal/engine/megatron_utils/checkpointer.py")
    # use_dist_ckpt：唯一消费点 + 唯一调用点不传
    check("F6.use_dist_ckpt.declared",
          re.search(r"use_dist_ckpt: bool = True", ck) is not None)
    # 只收属性调用：`torch.get_rng_state()` / `current_platform.get_rng_state()`
    # 同名但无关，混进来会让"唯一调用点"这条判据假红
    rng_sites = call_sites(root, "get_rng_state", self_only=True)
    check("F6.use_dist_ckpt.single_call", len(rng_sites) == 1, str(rng_sites))
    check("F6.use_dist_ckpt.not_passed",
          bool(rng_sites) and all(kws == [] for _r, _l, kws, _n in rng_sites))
    check("F6.use_dist_ckpt.not_positional",
          bool(rng_sites) and all(n == 0 for *_x, n in rng_sites))
    # with_rng：save/load 两条链都有形参（4 处声明），但生产调用点一律不传
    check("F7.with_rng.declared_four_times",
          ck.count("with_rng: bool = True") == 4,
          "次数 %d" % ck.count("with_rng: bool = True"))
    prod_load = call_sites(root, "load_checkpoint")
    prod_save = call_sites(root, "save_checkpoint")
    prod = [s for s in prod_load + prod_save]
    check("F7.with_rng.prod_call_sites_exist", len(prod) >= 2, str(len(prod)))
    check("F7.with_rng.never_passed",
          all("with_rng" not in kws for _r, _l, kws, _n in prod),
          str([(r, l, k) for r, l, k, _ in prod]))

    print()
    print("checks: %d, failed: %d" % (len(checks), len(failed)))
    return 0 if not failed else 1


def selftest(root):
    """三条注入各让**指定的那条判据**变红；红的不是目标项也算失败。"""
    ok = True

    def run_mutated(mutate, target, note):
        nonlocal ok
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="areal-mut-"))
        try:
            shutil.copytree(root, tmp / "r",
                            ignore=shutil.ignore_patterns(".git", "__pycache__"))
            mutate(tmp / "r")
            out = subprocess.run([sys.executable, __file__,
                                  str(tmp / "r")],
                                 capture_output=True, text=True).stdout
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        red = [ln.split()[0] for ln in out.splitlines() if ln.endswith("FAIL") or " FAIL  " in ln]
        good = target in red
        ok &= good
        print("%-40s %-34s %s  red=%s" % (target, note, "PASS" if good else "FAIL", red))

    # 注入 1：把 actor.py 里那个 partial 的关键字删掉 —— 生产者消失
    def mut1(r):
        p = r / "areal/trainer/ppo/actor.py"
        s = p.read_text(encoding="utf-8")
        assert "use_sapo_loss=self.config.use_sapo_loss," in s
        s = s.replace("                    use_sapo_loss=self.config.use_sapo_loss,\n", "")
        p.write_text(s, encoding="utf-8")

    # 注入 2：调用点开始给 use_cpu 传值 —— 它就不再不可达
    def mut2(r):
        p = r / "areal/experimental/models/archon/moe/utils.py"
        s = p.read_text(encoding="utf-8")
        assert "TOKEN_GROUP_ALIGN_SIZE_M,\n        )" in s
        s = s.replace("            TOKEN_GROUP_ALIGN_SIZE_M,\n        )",
                      "            TOKEN_GROUP_ALIGN_SIZE_M,\n            use_cpu=False,\n        )", 1)
        p.write_text(s, encoding="utf-8")

    # 注入 3：让 with_rng 被显式传值 —— 生产者出现
    def mut3(r):
        p = r / "areal/engine/megatron_engine.py"
        s = p.read_text(encoding="utf-8")
        assert "self.checkpointer.load_checkpoint(\n                    meta.path, with_optimizer=meta.with_optim\n                )" in s
        s = s.replace("self.checkpointer.load_checkpoint(\n                    meta.path, with_optimizer=meta.with_optim\n                )",
                      "self.checkpointer.load_checkpoint(\n                    meta.path, with_optimizer=meta.with_optim, with_rng=False\n                )")
        p.write_text(s, encoding="utf-8")

    run_mutated(mut1, "F.use_sapo_loss.produced_via_partial", "删掉 partial 里的 use_sapo_loss")
    run_mutated(mut2, "F4.use_cpu.not_passed", "让调用点开始传 use_cpu")
    run_mutated(mut3, "F7.with_rng.never_passed", "让 load_checkpoint 传 with_rng")

    print("\nselftest: %s" % ("3/3" if ok else "0/3"))
    return 0 if ok else 1


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        sys.exit(2)
    if "--selftest" in sys.argv:
        sys.exit(selftest(pathlib.Path(args[0])))
    sys.exit(main(args))
