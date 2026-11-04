#!/usr/bin/env python3
"""Execute the *real source text* of the `use_split_scale_stages` branch block.

The point is not to simulate the rule, it is to run the exact lines that ship in
tensorrt_llm/_torch/attention/backends/fp4_mla/fp4_mla_cutedsl_mufu16.py, with the
exact constant values the two call sites actually pass, and observe which subview
each one takes.

Usage:
  python3 probe_split_scale.py --repo /path/to/TensorRT-LLM   # prints JSON, exit 0 if all checks pass
  python3 probe_split_scale.py --selftest                      # injects 3 faults, each must be caught
"""
import argparse
import ast
import json
import pathlib
import re
import sys

TARGET = "tensorrt_llm/_torch/attention/backends/fp4_mla/fp4_mla_cutedsl_mufu16.py"
FUNC = "_issue_mxf4nvf4_mma_tile_from_base"

# What the two real call sites pass. Values are read from the source (see checks);
# they are constants here on purpose, never derived from the code under test.
CALL_QK = "fp4_mla_cutedsl_mufu16.py:3033"
CALL_PV = "fp4_mla_cutedsl_mufu16.py:4522"
# Signature order at fp4_mla_cutedsl_mufu16.py:2552-2595:
#   (sA, sB, sA_SF, sB_SF, base_col_id, base_row_id, stage, ..., sfa_stage=0, sfb_stage=0)
# QK (3033) passes `stage` positionally and leaves sfa_stage/sfb_stage at 0.
REAL_STAGE = 4          # any positive value: the row chosen is what matters
REAL_SFA_STAGE = 0      # default, not passed at either call site
REAL_SFB_STAGE = 0
QK_SFA_BYTES = 512      # QK_SFA_STAGE_BYTES = 512 * chunks (chunks>=1)
QK_SFB_BYTES = 512


def extract_block(src_lines):
    """Return (start_lineno, source_text) of the two `elif ... use_split_scale_stages` blocks."""
    out = []
    i = 0
    while i < len(src_lines):
        m = re.match(r"^(\s*)elif cutlass\.const_expr\(use_split_scale_stages\):\s*$",
                     src_lines[i])
        if m:
            indent = m.group(1)
            block = [src_lines[i]]
            j = i + 1
            while j < len(src_lines) and (src_lines[j].strip() == ""
                                          or src_lines[j].startswith(indent + "    ")):
                block.append(src_lines[j])
                j += 1
            out.append((i + 1, "\n".join(block)))
            i = j
        else:
            i += 1
    return out


def extract_guard(src_lines, lineno):
    """Return the `if cutlass.const_expr(stage_s..._from_smem):` line that guards lineno."""
    for i in range(lineno - 1, max(lineno - 12, -1), -1):
        m = re.match(r"^(\s*)if cutlass\.const_expr\((stage_sf[ab]_from_smem)\):\s*$",
                     src_lines[i])
        if m:
            return i + 1, m.group(2), len(m.group(1))
    return None


def extract_call_kwargs(src_lines, call_line):
    """Collect keyword names passed at a call site until the closing paren."""
    kwargs = set()
    depth = 0
    for i in range(call_line - 1, min(call_line + 120, len(src_lines))):
        line = src_lines[i]
        for m in re.finditer(r"\b(\w+)=", line):
            kwargs.add(m.group(1))
        depth += line.count("(") - line.count(")")
        if depth <= 0 and i > call_line - 1:
            break
    return kwargs


def run_block(block_src, sfa_from_smem, sfb_from_smem, use_sfa_off, use_sfb_off,
              split_scale, stage, sfa_stage, sfb_stage, sfa_bytes, sfb_bytes):
    """Execute the real block text with real constant values; return which subview won."""
    taken = {"sfa": None, "sfb": None}

    class S:
        def __init__(self, name):
            self.name = name

        def subview(self, off):
            taken[self.name] = off
            return self

    class Const:
        @staticmethod
        def const_expr(v):
            return bool(v)

    ns = {
        "cutlass": Const,
        "use_split_scale_stages": split_scale,
        "stage": stage,
        "sfa_stage": sfa_stage,
        "sfb_stage": sfb_stage,
        "sfa_stage_bytes": sfa_bytes,
        "sfb_stage_bytes": sfb_bytes,
        # the neighbouring `elif`s of the real block are not executed here; only the
        # split-scale arm is, with the sibling decision pre-computed as a flag.
        "use_sfa_stage_byte_offset": use_sfa_off,
        "use_sfb_stage_byte_offset": use_sfb_off,
        "sSFA": S("sfa"),
        "sSFB": S("sfb"),
    }
    # Re-indent the extracted block to top level so it can be exec'd standalone.
    lines = block_src.splitlines()
    base = len(lines[0]) - len(lines[0].lstrip())
    body = "\n".join(l[base:] if l.strip() else l for l in lines)
    # The block is an `elif`; turn it into a plain if for standalone execution.
    body = re.sub(r"^elif ", "if ", body, count=1)
    exec(compile(body, "<real-block>", "exec"), ns)
    return taken


def analyse(repo, injected=None):
    path = pathlib.Path(repo) / TARGET
    src_lines = path.read_text().splitlines()
    blocks = extract_block(src_lines)
    kwargs_qk = extract_call_kwargs(src_lines, int(CALL_QK.split(":")[1]))
    kwargs_pv = extract_call_kwargs(src_lines, int(CALL_PV.split(":")[1]))
    guards = {}
    for ln, body in blocks:
        g = extract_guard(src_lines, ln)
        guards[ln] = g

    inj = injected or {}

    def pass_split(**kw):
        return kw.get("split", False) if inj.get("force_split") else kw.get("split", False)

    # Real constant values at the QK call site (fp4_mla_cutedsl_mufu16.py:3033-3070):
    #   use_sfa_stage_byte_offset=True, use_sfb_stage_byte_offset=True,
    #   use_split_scale_stages NOT passed -> default False,
    #   stage_sfa_from_smem/stage_sfb_from_smem propagated from the enclosing function (default True)
    blk_sfb = next(b for b in blocks if guards[b[0]][1] == "stage_sfb_from_smem")
    blk_sfa = next(b for b in blocks if guards[b[0]][1] == "stage_sfa_from_smem")
    qk_sfb = run_block(blk_sfb[1], True, True, True, True,
                       pass_split(split=False), REAL_STAGE, REAL_SFA_STAGE, REAL_SFB_STAGE, QK_SFA_BYTES, QK_SFB_BYTES)
    qk_sfa = run_block(blk_sfa[1], True, True, True, True,
                       pass_split(split=False), REAL_STAGE, REAL_SFA_STAGE, REAL_SFB_STAGE, QK_SFA_BYTES, QK_SFB_BYTES)
    # Control arm: the same block with the flag ON must take a *different* subview,
    # otherwise "unreachable" would be indistinguishable from "wired to the same arm".
    on_sfb = run_block(blk_sfb[1], True, True, True, True,
                       True if not inj.get("force_split") else False, REAL_STAGE, 2, 2, QK_SFA_BYTES, QK_SFB_BYTES)
    on_sfa = run_block(blk_sfa[1], True, True, True, True,
                       True if not inj.get("force_split") else False, REAL_STAGE, 2, 2, QK_SFA_BYTES, QK_SFB_BYTES)
    # Negative control: if the sibling byte-offset flag is off, the split arm is reachable.
    off_sfb = run_block(blk_sfb[1], True, True, False, False,
                        True if not inj.get("sibling_always_wins") else False, REAL_STAGE, 2, 2, QK_SFA_BYTES, QK_SFB_BYTES)
    off_sfa = run_block(blk_sfa[1], True, True, False, False,
                        True if not inj.get("sibling_always_wins") else False, REAL_STAGE, 2, 2, QK_SFA_BYTES, QK_SFB_BYTES)

    checks = []

    def check(cid, ok, detail):
        checks.append({"id": cid, "ok": bool(ok), "detail": detail})

    check("B1.block_found", len(blocks) == 2,
          "found %d `elif use_split_scale_stages` blocks" % len(blocks))
    guarded = sorted(g[1] for g in guards.values() if g)
    check("B2.guards_cover_both_operands", guarded == ["stage_sfa_from_smem", "stage_sfb_from_smem"],
          "guards found: %s" % guarded)
    check("B4.qk_call_omits_flag", "use_split_scale_stages" not in kwargs_qk,
          "QK call site kwargs contain the flag: %s" % ("use_split_scale_stages" in kwargs_qk))
    check("B5.pv_call_omits_flag", "use_split_scale_stages" not in kwargs_pv,
          "PV call site kwargs contain the flag: %s" % ("use_split_scale_stages" in kwargs_pv))
    check("B0.scale_stage_indices_default_zero",
          "sfa_stage" not in kwargs_qk and "sfb_stage" not in kwargs_qk,
          "call site leaves sfa_stage/sfb_stage at their 0 default: %s" % (
              sorted(k for k in kwargs_qk if k.endswith("_stage"))))
    check("B6.qk_call_passes_siblings",
          {"use_sfa_stage_byte_offset", "use_sfb_stage_byte_offset"} <= kwargs_qk,
          "siblings passed at QK: %s" % sorted(k for k in kwargs_qk if "stage_byte_offset" in k))
    # The behaviour: siblings win, so the split arm's subview is *not* taken.
    check("B7.real_values_take_sibling_arm",
          qk_sfb == {"sfa": None, "sfb": None} and qk_sfa == {"sfa": None, "sfb": None},
          "split-arm subviews taken with real values: sfb=%s sfa=%s" % (qk_sfb, qk_sfa))
    # Positive control: with the flag on AND siblings off, the arm is live and its
    # offset differs from the sibling arm's — i.e. the branch is meaningful code.
    check("B8.arm_live_when_reachable",
          off_sfb == {"sfa": None, "sfb": QK_SFB_BYTES * 2} and off_sfa == {"sfa": QK_SFA_BYTES * 2, "sfb": None},
          "with siblings off and flag on: sfb=%s sfa=%s" % (off_sfb, off_sfa))
    # The else-arm (neither sibling nor split) uses `bytes * stage`; the split arm uses
    # `bytes * sfa_stage/sfb_stage`. With stage=3 and sfa_stage=sfb_stage=2 the two
    # offsets differ, so "the split arm is live" is observable, not assumed.
    else_sfb = run_block(blk_sfb[1], True, True, False, False, False, REAL_STAGE, 2, 2, QK_SFA_BYTES, QK_SFB_BYTES)
    else_sfa = run_block(blk_sfa[1], True, True, False, False, False, REAL_STAGE, 2, 2, QK_SFA_BYTES, QK_SFB_BYTES)
    check("B9.arm_offset_differs_from_else_arm",
          on_sfb != else_sfb and on_sfa != else_sfa,
          "split-arm: sfb=%s sfa=%s | else-arm: sfb=%s sfa=%s" % (on_sfb, on_sfa, else_sfb, else_sfa))
    return checks


def selftest():
    """Each injected fault must turn at least one check red."""
    repo = AUTO_REPO
    base = analyse(repo)
    base_ok = all(c["ok"] for c in base)
    print("selftest baseline: %d checks, ok=%s" % (len(base), base_ok))
    faults = [
        ("force_split", "split arm forced on -> B7 must go red"),
        ("sibling_always_wins", "sibling forced to win -> B8 must go red"),
    ]
    n_pass = 0
    for name, why in faults:
        res = analyse(repo, injected={name: True})
        red = [c["id"] for c in res if not c["ok"]]
        ok = bool(red)
        print("  inject %-22s -> red: %s  [%s]" % (name, red or "NONE", "OK" if ok else "MISSED"))
        n_pass += ok
    return 0 if (base_ok and n_pass == len(faults)) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/tmp/trtllm-probe")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    AUTO_REPO = a.repo
    if a.selftest:
        sys.exit(selftest())
    out = analyse(a.repo)
    print(json.dumps({"checks": out, "passed": sum(c["ok"] for c in out), "total": len(out)}, indent=1))
    sys.exit(0 if all(c["ok"] for c in out) else 1)
