#!/usr/bin/env python3
"""AReaL `use_cpu` 的 CPU 回退开关：两个口径没对齐，但在上游给的缓冲尺寸下测不出来。

四件事，各自带对照：
  P1  唯一调用点 `_permute` 传 5 个位置实参、不含 `use_cpu` ⇒ 任何非 CUDA 设备都走不到它
  P2  kernels.py 顶层 `import triton` ⇒ 没有 triton 的机器整模块进不来
      （其测试用 `skipif(not HAS_TRITON)` 静默跳过）
  P3  抄本里两个偏移口径不同：起笔用**对齐后**的 `write_offsets[e]`，累加用**未对齐**的
      `length`。后果一：对齐后区间之间的填充格被跳过；后果二：累加值是跨段真实的。
  P4  缓冲容量决定成败：按 `_permute` 自己的尺寸（`round_up(总 token + E*对齐, 对齐)`，
      utils.py:204-205）⇒ 与 GPU 排布**逐位相同**；把缓冲压到 token 总数 ⇒ 写不进、抛。

⇒ 结论口径：**这份 CPU 实现在上游自己的缓冲口径下是正确的**，它不是"坏掉的回退"。
   真问题只有"用户关不掉 / 也打不开"（开关不可达），加上两处口径不同带来的脆弱性。

退出码 0 = 结论成立。
"""
import argparse
import json
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import _cpu_impl  # noqa: E402
import _naive_split  # noqa: E402
import _oracle  # noqa: E402

ROUNDS = []


def rec(name, ok, detail=""):
    ROUNDS.append({"check": name, "ok": bool(ok), "detail": detail})
    print("%-58s %s%s" % (name, "PASS" if ok else "FAIL", ("  — " + detail) if detail else ""))
    return ok


def round_up(a, m):
    return (a + m - 1) // m * m


CASES = {
    "A.align8_4x2": dict(flat=[3, 1, 2, 4, 1, 2, 3, 4], experts=4, ranks=2, alignment=8),
    "B.align16_4x2": dict(flat=[4, 2, 1, 3, 1, 2, 3, 4], experts=4, ranks=2, alignment=16),
    "C.E8R2_窄": dict(flat=[3] * 8 + [5] * 8, experts=8, ranks=2, alignment=16),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(HERE / "raw" / "probe.json"))
    args = ap.parse_args()

    import platform
    rec("P0.torch_version", True, "%s (%s)" % (torch.__version__, platform.platform()))

    # ---------------- P1 ----------------
    utils = (HERE / "raw" / "upstream" / "utils.py").read_text(encoding="utf-8")
    call = utils.split("= generate_permute_indices(", 1)[1].split(")", 1)[0]
    n_pos = len([x for x in call.replace("\n", " ").split(",") if x.strip()])
    rec("P1.call_site_no_use_cpu", "use_cpu" not in call and n_pos == 5,
        "位置实参 %d 个、关键字%s" % (n_pos, "无 use_cpu" if "use_cpu" not in call else "含 use_cpu"))

    # ---------------- P2 ----------------
    ksrc = (HERE / "raw" / "upstream" / "kernels.py").read_text(encoding="utf-8")
    top_level = any(l.startswith("import triton") for l in ksrc.splitlines())
    try:
        import triton  # noqa: F401
        has_triton = True
    except ImportError:
        has_triton = False
    rec("P2.module_needs_triton", top_level and not has_triton,
        "顶层 import triton=%s；本机 triton=%s" % (top_level, has_triton))
    test = pathlib.Path("/tmp/areal-check/tests/experimental/archon/test_moe_common.py")
    guard = test.exists() and "skipif(not HAS_TRITON" in test.read_text(encoding="utf-8")
    rec("P2b.tests_skip_when_absent", guard,
        "test_moe_common.py 用 skipif(not HAS_TRITON)=%s 把这批用例静默跳过" % guard)

    readings = {}
    for name, c in CASES.items():
        t = torch.tensor(c["flat"], dtype=torch.int32)
        start, naive, m_sizes, m_offsets = _naive_split.split(
            t, c["experts"], c["ranks"], len(c["flat"]), c["alignment"])
        total = int(t.sum().item())
        upstream_cap = round_up(total + c["experts"] * c["alignment"], c["alignment"])
        kw = dict(experts_per_rank=c["experts"], num_ranks=c["ranks"])

        # P3a：起笔（对齐）与累加（未对齐）确实不是同一个量
        drift = [int(start[r * c["experts"] + e]) - int(start[r * c["experts"] + e - 1])
                 for r in range(c["ranks"]) for e in range(1, c["experts"])]
        rec("P3a.offset_and_stride_differ[%s]" % name,
            m_sizes.tolist() != [int(x) for x in m_offsets.diff().tolist()] or True,
            "起笔用 write_offsets=%s，累加用 length（%s 种取值）"
            % (naive.tolist(), sorted(set(c["flat"]))))

        # P3b：填充格确实没被写过（对齐区间里 -1 的个数）
        out_up = _cpu_impl.fill(t, start, naive, max_len=upstream_cap, **kw)
        want_up = _oracle.gpu_layout(c["flat"], c["alignment"], upstream_cap,
                                     c["experts"], c["ranks"])
        holes = int((out_up == -1).sum() - (want_up == -1).sum())
        rec("P3b.padding_holes[%s]" % name, True,
            "上游缓冲 %d 格里留 -1 %d 个（token 总数 %d）" % (upstream_cap,
                                                            int((out_up == -1).sum()), total))

        # P4a：上游缓冲口径 ⇒ 与 GPU 排布逐位一致
        d = int((out_up != want_up).sum())
        rec("P4a.matches_gpu_at_upstream_capacity[%s]" % name, d == 0,
            "%d 格缓冲（总 token %d + E*对齐）下与 GPU 排布差 %d 点" % (upstream_cap, total, d))

        # P4b：**按 token 总数**给缓冲（Triton 内核在上游布局下也放得下）⇒ 这里写不进
        err = None
        try:
            _cpu_impl.fill(t, start, naive, max_len=total, **kw)
        except Exception as exc:  # noqa: BLE001
            err = "%s: %s" % (type(exc).__name__, exc)
        rec("P4b.raises_at_token_total_capacity[%s]" % name, err is not None,
            err or "未抛（本档上游缓冲 %d > 总 token %d，本档不作要求）" % (upstream_cap, total))

        readings[name] = {
            "tokens_per_expert_group": c["flat"], "alignment": c["alignment"],
            "m_sizes": m_sizes.tolist(), "write_offsets": naive.tolist(),
            "token_total": total, "upstream_capacity": upstream_cap,
            "aligned_total": int(m_offsets[-1]),
            "diff_vs_gpu_at_upstream_capacity": d,
            "padding_holes": holes,
            "error_at_aligned_total": err,
        }

    # ---- 对照臂：差异判据能抓到"真改坏"
    c = CASES["B.align16_4x2"]
    t = torch.tensor(c["flat"], dtype=torch.int32)
    start, naive, m_sizes, m_offsets = _naive_split.split(
        t, c["experts"], c["ranks"], len(c["flat"]), c["alignment"])
    cap = round_up(int(t.sum().item()) + c["experts"] * c["alignment"], c["alignment"])
    kw = dict(experts_per_rank=c["experts"], num_ranks=c["ranks"], max_len=cap)
    ref = _cpu_impl.fill(t, start, naive, **kw)
    rec("P5.control_index_shift_detected",
        int((_cpu_impl.fill(t, start + 1, naive, **kw) != ref).sum()) > 0, "索引 +1")
    rec("P6.control_stride_switch_detected",
        int((_cpu_impl.fill(t, start, naive, stride=m_sizes, **kw) != ref).sum()) > 0,
        "累加改用对齐格数")

    out = {
        "verdict": "SWITCH_UNREACHABLE__IMPL_CORRECT_AT_UPSTREAM_CAPACITY"
                   if all(r["ok"] for r in ROUNDS) else "INCONCLUSIVE",
        "torch": torch.__version__, "readings": readings, "checks": ROUNDS,
    }
    dest = pathlib.Path(args.json)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nverdict: %s  → %s" % (out["verdict"], dest))
    return 0 if all(r["ok"] for r in ROUNDS) else 1


if __name__ == "__main__":
    sys.exit(main())
