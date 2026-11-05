#!/usr/bin/env python3
"""抄本 ↔ 上游快照一致性：抄本的每一行都必须能在上游函数体里找到。

--selftest 在**临时副本**上注入，要求目标判据变红（红错地方也算失败）：

  I1  抄本里加一个上游没有的语句 → C2b 变红
  I2  上游快照出现 use_cpu 生产者 → C3 变红（"结论失效"能被发现）
  I3  抄本删掉累加那一行        → C2a/C2b 都变红
"""
import hashlib
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
SNAP = "3b93679888e9f4929f6f155f13c6a286a62dfd72"
KERNELS_SHA16 = "d8f583c6e5aa1d0d"
CHECKS, FAILED = [], []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    if not cond:
        FAILED.append(name)
    print("%-46s %s%s" % (name, "PASS" if cond else "FAIL", ("  — " + detail) if detail else ""), flush=True)


def func_body(text, name):
    """取 `def name(...)` 的**函数体**（跳过签名与 docstring）到下一个顶格 def 之前。"""
    i = text.index("def %s(" % name)
    rest = text[i + 1:]
    end = len(rest)
    for m in range(len(rest)):
        if rest.startswith("\ndef ", m) or rest.startswith("\n@", m):
            end = m
            break
    body = text[i:i + 1 + end]
    # 跳过签名（到第一个 "):" 行）与紧随其后的 docstring
    lines = body.splitlines()
    k = next(n for n, ln in enumerate(lines) if ln.rstrip().endswith(":"))
    k += 1
    if k < len(lines) and lines[k].strip().startswith('"""'):
        while not lines[k].strip().endswith('"""') or len(lines[k].strip()) < 3:
            k += 1
        k += 1
    return "\n".join(lines[k:])


def meaningful_lines(text):
    """**逻辑行**：剥 docstring/注释后按括号深度把续行接回去，避免"多一个 device= 行"
    这种纯格式差异被当成内容差异。"""
    import re as _re
    phys, in_doc = [], False
    for ln in text.splitlines():
        s = ln.strip()
        if in_doc:
            if s.endswith('"""') or s.count('"""') % 2 == 1:
                in_doc = False
            continue
        if s.startswith('"""'):
            if s.count('"""') % 2 == 1:
                in_doc = True
            continue
        if not s or s.startswith("#"):
            continue
        phys.append(_re.sub(r"\s+", " ", s))
    out, buf, depth = [], "", 0
    for s in phys:
        buf = (buf + " " + s).strip() if buf else s
        depth += s.count("(") + s.count("[") + s.count("{") - s.count(")") - s.count("]") - s.count("}")
        if depth <= 0:
            out.append(_re.sub(r"\s+", " ", buf))
            buf, depth = "", 0
    if buf:
        out.append(buf)
    return out


def run(base):
    kernels = (base / "raw" / "upstream" / "kernels.py").read_text(encoding="utf-8")
    utils = (base / "raw" / "upstream" / "utils.py").read_text(encoding="utf-8")
    cpu = (base / "_cpu_impl.py").read_text(encoding="utf-8")
    cpu = cpu[cpu.index("def fill("):]          # 只比函数本体，不比模块级 import
    split = (base / "_naive_split.py").read_text(encoding="utf-8")

    check("C1.snapshot_fingerprint",
          hashlib.sha256(kernels.encode()).hexdigest()[:16] == KERNELS_SHA16,
          "kernels.py sha256[:16]=%s" % hashlib.sha256(kernels.encode()).hexdigest()[:16])

    # C2a：fill_indices_cpu 的每一行代码都在抄本里（允许探针加的三处改名/参数）
    up = meaningful_lines(func_body(kernels, "fill_indices_cpu"))
    mine = meaningful_lines(cpu)
    mine_txt = "\n".join(mine)
    # 抄本的签名行与上游不同（改名 + 探针关键字），逐行比对时跳过签名
    up = [x for x in up if not x.startswith("def ")]
    # 签名可能跨行：从 "def fill(" 起一直跳过到以 "):" 结尾的那一行
    def canon(line):
        t = (line.replace("naive_write_offsets", "write_offsets")
                 .replace("device=tokens_per_expert_group.device,", "")
                 .replace("write_start += length if stride is None else int(stride[e].item())",
                          "write_start += length"))
        t = re.sub(r"\s+", " ", t).strip()
        return re.sub(r"\s*,?\s*\)", ")", t)

    def strip_sig(lines):
        out, skip = [], False
        for x in lines:
            if x.startswith("def fill("):
                skip = not x.rstrip().endswith("):")
                continue
            if skip:
                skip = not x.rstrip().endswith("):")
                continue
            out.append(x)
        return out

    up = [canon(x) for x in strip_sig(up)]
    mine = [canon(x) for x in strip_sig(mine)]
    mine_txt = "\n".join(mine)
    missing = [x for x in up if x not in mine_txt]
    check("C2a.cpu_impl_lines_present", not missing,
          "上游 %d 行，缺 %d 行%s" % (len(up), len(missing),
                                  ("：" + " | ".join(missing[:3])) if missing else ""))

    # C2b：抄本不得多出上游没有的语句（只允许本文件头记载的三处改名/参数）
    # 反方向：先把抄本里三处记载过的改名还原成上游写法，再要求"无多余行"
    allowed_extra = ()
    up_txt = "\n".join(up)
    restored = (mine_txt.replace("naive_write_offsets", "write_offsets")

                        .replace("write_start += length if stride is None else int(stride[e].item())",
                                 "write_start += length")
                        .replace("device=tokens_per_expert_group.device,\n", ""))
    extra = [x for x in restored.splitlines() if x not in up_txt]
    check("C2b.no_unexpected_lines_in_transcript", not extra,
          "多出 %d 行%s" % (len(extra), ("：" + extra[0]) if extra else ""))

    # C3：结论前提——`generate_permute_indices(...)` 的调用点没有传 use_cpu
    call = utils.split("= generate_permute_indices(", 1)[1].split(")", 1)[0]
    n = len([ln for ln in call.splitlines() if "use_cpu" in ln])
    check("C3.no_producer_at_call_site", n == 0, "调用点里 use_cpu 出现 %d 次" % n)

    # C4：naive 偏移确实来自上游那两行
    ok = "m_offsets = torch.cumsum(m_sizes, 0)" in kernels and \
         "m_offsets = torch.cumsum(m_sizes, 0)" in split
    check("C4.naive_offsets_traced", ok, "上游与抄本都按 m_offsets - m_sizes 算起点")
    return 0 if not FAILED else 1


def copy_bundle(dst):
    for p in ("_cpu_impl.py", "_naive_split.py", "_oracle.py"):
        shutil.copy(HERE / p, dst / p)
    (dst / "raw" / "upstream").mkdir(parents=True)
    for p in ("kernels.py", "utils.py"):
        shutil.copy(HERE / "raw" / "upstream" / p, dst / "raw" / "upstream" / p)


def selftest():
    import contextlib
    import io
    cases = [
        ("I1.transcript_extra_statement",
         lambda b: (b / "_cpu_impl.py").write_text(
             (b / "_cpu_impl.py").read_text().replace(
                 "permuted_indices = torch.full(", "permuted_indices = torch.empty(")),
         ["C2a.cpu_impl_lines_present", "C2b.no_unexpected_lines_in_transcript"]),
        ("I2.producer_appears",
         lambda b: (b / "raw" / "upstream" / "utils.py").write_text(
             (b / "raw" / "upstream" / "utils.py").read_text().replace(
                 "TOKEN_GROUP_ALIGN_SIZE_M,\n        )", "TOKEN_GROUP_ALIGN_SIZE_M,\n  use_cpu=False,\n        )", 1)),
         ["C3.no_producer_at_call_site"]),
        ("I3.line_deleted",
         lambda b: (b / "_cpu_impl.py").write_text(
             (b / "_cpu_impl.py").read_text().replace("write_start += length if stride", "pass  # if stride")),
         ["C2a.cpu_impl_lines_present", "C2b.no_unexpected_lines_in_transcript"]),
    ]
    res = []
    for name, mutate, expect in cases:
        tmp = pathlib.Path(tempfile.mkdtemp())
        copy_bundle(tmp)
        mutate(tmp)
        buf = io.StringIO()
        CHECKS.clear(); FAILED.clear()
        with contextlib.redirect_stdout(buf):
            run(tmp)
        red = list(FAILED)
        ok = red == expect
        res.append(ok)
        print("%-28s %s red=%s expect=%s" % (name, "OK" if ok else "UNEXPECTED", red, expect))
        shutil.rmtree(tmp)
    print("\nselftest: %d/%d" % (sum(res), len(res)))
    return 0 if all(res) else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    sys.exit(run(HERE))
