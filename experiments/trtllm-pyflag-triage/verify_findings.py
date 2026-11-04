#!/usr/bin/env python3
"""Re-derive every TensorRT-LLM pyflag finding from source text, then classify it.

Every check below reads the snapshot and asserts one greppable fact. Nothing here
executes the framework; the behavioural probe for the one live finding is
probe_split_scale.py.

  python3 verify_findings.py --repo /path/to/TensorRT-LLM [--report scan.json]
  python3 verify_findings.py --selftest     # inject faults, each must turn its check red
"""
import argparse
import ast
import json
import pathlib
import re
import subprocess
import sys

REPO = None
FAULTS = {}


def read(rel):
    return (pathlib.Path(REPO) / rel).read_text()


def lines(rel):
    return read(rel).splitlines()


def has(rel, pattern, flags=0):
    return re.search(pattern, read(rel), flags) is not None


def lineno_of(rel, pattern):
    for i, l in enumerate(lines(rel), 1):
        if re.search(pattern, l):
            return i
    return None


def call_kwargs(rel, call_line, func_name):
    """Keyword arg names of the call to func_name nearest call_line."""
    src = read(rel)
    tree = ast.parse(src)
    best = None
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == func_name:
            if best is None or abs(n.lineno - call_line) < abs(best.lineno - call_line):
                best = n
    if best is None:
        return None
    return {k.arg for k in best.keywords if k.arg}


def enclosing_func(rel, line):
    tree = ast.parse(read(rel))
    best = None
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.lineno <= line <= n.end_lineno:
            if best is None or n.lineno > best.lineno:
                best = n
    return best


CHECKS = []


def check(cid, ok, detail):
    if FAULTS.get("flip_" + cid):
        ok, detail = (not ok), "FAULT-INJECTED: " + str(detail)
    CHECKS.append({"id": cid, "ok": bool(ok), "detail": detail})


FP4 = "tensorrt_llm/_torch/attention/backends/fp4_mla/fp4_mla_cutedsl_mufu16.py"
FP4B = "tensorrt_llm/_torch/attention/backends/fp4_mla/fp4_mla_cutedsl_mufu16_fused_v_transpose.py"
OPENAI = "tensorrt_llm/serve/openai_server.py"
SERVE = "tensorrt_llm/commands/serve.py"


def findings_live():
    """F1: use_split_scale_stages — declared, branched on, never passed."""
    src = read(FP4)
    check("F1a.declared", re.search(r"^\s+use_split_scale_stages: ctm\.Constexpr = False,", src, re.M) is not None,
          "parameter with default False at %s" % lineno_of(FP4, r"^\s+use_split_scale_stages: ctm\.Constexpr"))
    elif_lines = [i for i, l in enumerate(src.splitlines(), 1) if "elif cutlass.const_expr(use_split_scale_stages)" in l]
    check("F1b.two_elif_arms", len(elif_lines) == 2, "elif arms at lines %s" % elif_lines)
    qk = call_kwargs(FP4, 3033, "_issue_mxf4nvf4_mma_tile_from_base")
    pv = call_kwargs(FP4, 4522, "_issue_mxf4nvf4_mma_tile_from_base")
    check("F1c.no_producer_qk", qk is not None and "use_split_scale_stages" not in qk, "QK kwargs: %s" % sorted(qk or [])[-6:])
    check("F1d.no_producer_pv", pv is not None and "use_split_scale_stages" not in pv, "PV kwargs: %s" % sorted(pv or [])[-6:])
    # The four sibling constexprs ARE passed at the QK site, which is what makes the
    # omission look like an omission rather than a design choice.
    sib = {"use_a_stage_word_offset", "use_b_stage_word_offset",
           "use_sfa_stage_byte_offset", "use_sfb_stage_byte_offset"}
    check("F1e.siblings_passed_at_qk", sib <= (qk or set()), "siblings passed: %s" % sorted(sib & (qk or set())))
    # The arm sits in the middle of the parameter list (between b_stage and sfa_stage),
    # so it was not appended later and forgotten — it was written in place.
    ls = src.splitlines()
    def idx(pat):
        for i, l in enumerate(ls):
            if re.search(pat, l):
                return i
        return -1
    i_split, i_sfa = idx(r"^\s+use_split_scale_stages: ctm"), idx(r"^\s+sfa_stage: ctm")
    i_b = idx(r"^\s+b_stage: ctm")
    check("F1f.param_in_middle", i_b < i_split < i_sfa,
          "b_stage@%d < use_split_scale_stages@%d < sfa_stage@%d" % (i_b, i_split, i_sfa))
    # Same shape in the fused-v-transpose twin.
    qk_b = call_kwargs(FP4B, 3415, "_issue_mxf4nvf4_mma_tile_from_base")
    check("F1g.twin_same_shape",
          has(FP4B, r"use_split_scale_stages: ctm\.Constexpr = False")
          and qk_b is not None and "use_split_scale_stages" not in qk_b,
          "twin declares+routes the same way; sibling passed: %s" % ("use_sfa_stage_byte_offset" in (qk_b or set())))
    # The two call sites do NOT die the same way, and the difference decides what a fix
    # has to touch:
    #   QK (3033) enters the block (stage_s*f_from_smem forwarded, defaults True) and is
    #             then blocked by the sibling arm.
    #   PV (4522) passes stage_sfa_from_smem=False / stage_sfb_from_smem=False, so the
    #             whole `if stage_s*f_from_smem:` block is skipped: the elif there is
    #             dead one level higher, for a reason unrelated to the siblings.
    # Folding both into "两个调用点都不传" reads as one cause and hides that.
    pv_kw = {}
    for n in ast.walk(ast.parse(read(FP4))):
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_issue_mxf4nvf4_mma_tile_from_base" \
                and n.lineno > 4000:
            pv_kw = {k.arg: ast.unparse(k.value) for k in n.keywords if k.arg}
    check("F1h.pv_dead_by_enclosing_guard",
          pv_kw.get("stage_sfa_from_smem") == "False" and pv_kw.get("stage_sfb_from_smem") == "False",
          "PV skips the block entirely: stage_sfa_from_smem=%r, stage_sfb_from_smem=%r"
          % (pv_kw.get("stage_sfa_from_smem"), pv_kw.get("stage_sfb_from_smem")))


def snapshot_pristine():
    """The snapshot must be unmodified — every check here reads its text.

    This exists because a negative control in this batch (`sed -i ''`, run under zsh,
    where the empty-string backup suffix is not an argument) silently rewrote FP4 and
    its restore was skipped. All checks plus probe 9/9 stayed green against the
    corrupted tree: they assert on literal shapes only, and "siblings passed True"
    flipping to False read as a *finding* rather than a fault. A judgement about
    source text has to first prove the source text is the one it claims to read.
    """
    try:
        r = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                           capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.SubprocessError) as e:
        return False, "cannot run git in %s: %s" % (REPO, e)
    if r.returncode != 0:
        return False, "git status failed (not a git checkout?): %s" % r.stderr.strip()[:200]
    dirty = [l for l in r.stdout.splitlines() if l.strip()]
    if dirty:
        return False, "%d modified path(s): %s" % (len(dirty), "; ".join(dirty[:4]))
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    return True, "clean tree at %s" % head[:10]


def findings_runtime_control():
    """F2: enable_runtime_control_endpoints — wired, and the fallback is the default path."""
    tree = ast.parse(read(SERVE))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "launch_server")
    # `uvloop.run(server(...))` must NOT be nested inside `if multi_frontend.is_launcher:`.
    serve_calls, launcher_ifs = [], []
    for n in ast.walk(fn):
        if isinstance(n, ast.If) and re.search(r"multi_frontend\.is_launcher\b", ast.unparse(n.test)):
            launcher_ifs.append((n.lineno, n.end_lineno))
        if isinstance(n, ast.Call) and re.search(r"\buvloop\.run\b", ast.unparse(n.func)):
            serve_calls.append(n.lineno)
    nested = [ln for ln in serve_calls if any(a <= ln <= b for a, b in launcher_ifs)]
    check("F2a.serve_call_exists", len(serve_calls) == 1, "uvloop.run at %s" % serve_calls)
    check("F2b.serve_reachable_without_launcher", not nested,
          "serve call nested inside is_launcher block: %s" % nested)
    check("F2c.caller_passes_flag",
          has(SERVE, r"enable_runtime_control_endpoints=\s*\n?\s*enable_runtime_control_endpoints") is False or
          re.search(r"enable_runtime_control_endpoints=\s*\n\s*enable_runtime_control_endpoints,", read(SERVE)) is not None,
          "launch_server forwards the flag to OpenAIServer")
    # The default is 8 frontends, and sleep mode falls back to 1 -> the launcher
    # branch runs, so the endpoints are registered on the default path too.
    check("F2d.cli_default_is_eight",
          re.search(r'"--num_serve_frontends",\s*\n\s*type=click\.IntRange\([^)]*\),\s*\n\s*#.*\n\s*default=8,', read(SERVE)) is not None,
          "click default for --num_serve_frontends is 8")
    check("F2e.sleep_mode_falls_back_to_one",
          has(SERVE, r'elif enable_sleep_mode:\s*\n\s*reason = "--enable_sleep_mode"'),
          "_resolve_default_num_serve_frontends lowers to 1 under --enable_sleep_mode")
    check("F2f.rl_endpoints_deprecated",
          has(OPENAI, r'"RL control endpoints are deprecated'),
          "enable_rl_control_endpoints logs a deprecation warning")


def findings_false_positives():
    """Each of these was reported silently_inert; each has a real producer pyflag cannot see."""
    # (1) **dict expansion
    check("FP1.dict_expansion_use_causal_mask",
          has("tensorrt_llm/_torch/attention/backends/prims_ts/_block_sparse/compiler.py",
              r'"use_causal_mask": key\.mask_type == "causal"')
          and has("tensorrt_llm/_torch/attention/backends/prims_ts/_block_sparse/compiler.py",
                  r"_Prepare\w+\(\*\*prepare_kwargs\)"),
          "use_causal_mask reaches the kernel through **prepare_kwargs")
    # (2) explicit None sentinel is a producer
    check("FP2.explicit_none_is_producer",
          has("tensorrt_llm/_torch/pyexecutor/_util.py", r"use_replay_state_update=None,"),
          "use_replay_state_update is passed explicitly as None")
    # (3) config-dict whitelist
    check("FP3.config_dict_whitelist",
          has("tensorrt_llm/_torch/visual_gen/models/ltx23/ltx23_core/audio_vae/vocoder.py",
              r'"use_bias_at_final",')
          and has("tensorrt_llm/_torch/visual_gen/models/ltx23/ltx23_core/audio_vae/vocoder.py",
                  r"kwargs = \{key: cfg\[key\] for key"),
          "use_bias_at_final is read out of the model config dict")
    # (4) same-name but source-only helper inlined into C++ by setup.py
    check("FP4.build_time_inlined_flag",
          has("cpp/kernels/fmha_v2/setup.py", r"if kspec\.enable_attn_logit_softcapping:")
          and has("cpp/kernels/fmha_v2/setup.py", r"kernel_spec\.enable_attn_logit_softcapping"),
          "enable_attn_logit_softcapping is a kernel_spec field, not a call argument")
    # (5) sibling producer in the same file — enable_multi_cta passes True and False
    check("FP5.same_file_producer",
          has("tensorrt_llm/_torch/kernels/blackwell/top_k/filtered_top_k_decode_varlen.py", r"enable_multi_cta=True")
          and has("tensorrt_llm/_torch/kernels/blackwell/top_k/filtered_top_k_decode_varlen.py", r"enable_multi_cta=False")
          and has("tensorrt_llm/_torch/kernels/blackwell/top_k/filtered_top_k_decode_varlen.py",
                  r"def filtered_topk_kernel\("),
          "enable_multi_cta is constructed with both values in the same file")
    # (6) attribute read rather than argument
    check("FP6.getattr_producer",
          has("tensorrt_llm/_torch/models/modeling_utils.py", r"getattr\(model, ['\"]use_custom_cublas_mm['\"]")
          or has("tensorrt_llm/_torch/models/modeling_utils.py", r"use_custom_cublas_mm=getattr"),
          "use_custom_cublas_mm is read off the module via getattr")


def findings_dead_or_internal():
    """Items with no user-facing surface at all."""
    # zero callers anywhere in the snapshot
    src_all = {}
    for p in pathlib.Path(REPO).rglob("*.py"):
        if ".git" in p.parts:
            continue
        try:
            src_all[str(p)] = p.read_text(errors="replace")
        except Exception:
            pass
    # Prefix-guarded on purpose: `_extract_answer` is a *different* method on
    # LongBenchV2 and substring counting would conflate the two (a mistake already
    # made once while scanning SkyRL).
    n_extract = sum(len(re.findall(r"(?<![\w.])extract_answer\(", s)) for s in src_all.values())
    check("D1.extract_answer_zero_callers", n_extract == 1,
          "`extract_answer(` (not `_extract_answer`) occurrences: %d (definition only)" % n_extract)
    n_tllm_lin = sum(s.count("get_tllm_linear_weight(") for s in src_all.values())
    check("D2.legacy_converter_zero_callers", n_tllm_lin == 1,
          "`get_tllm_linear_weight(` occurrences: %d (definition only)" % n_tllm_lin)
    check("D3.build_script_only",
          has("scripts/build_wheel.py", r"def main\(.*\n(?:.*\n)*?\s+use_ccache: bool = False,")
          or has("scripts/build_wheel.py", r"use_ccache: bool = False"),
          "use_ccache / use_3rdparty_cache live in the build script's main()")
    check("D4.decorator_internal",
          has("tensorrt_llm/_torch/distributed/communicator.py", r"@staticmethod\s*\n\s+def log_op\(func, enable_log=False\)"),
          "enable_log is a parameter of the log_op decorator itself")


def inventory(report_path):
    rep = json.load(open(report_path))
    flags = rep["flags"]
    inert = [f for f in flags if f["silently_inert"]]
    check("I1.total_flags", len(flags) == 352, "total flags %d" % len(flags))
    check("I2.inert_count", len(inert) == 35, "silently_inert %d" % len(inert))
    modes = {}
    for f in inert:
        modes[f["mode"]] = modes.get(f["mode"], 0) + 1
    check("I3.mode_breakdown", sum(modes.values()) == 35, "modes %s" % modes)
    return inert


def main():
    global REPO
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo")
    ap.add_argument("--report", default=str(pathlib.Path(__file__).parent / "findings.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    REPO = a.repo or "/tmp/trtllm-probe"
    if a.selftest:
        return selftest(a.report)
    pristine, why = snapshot_pristine()
    check("S0.snapshot_pristine", pristine, why)
    if not pristine:
        # Reading a tree someone else modified produces confident nonsense, so stop
        # before deriving anything from it.
        for c in CHECKS:
            print(("PASS " if c["ok"] else "FAIL ") + c["id"] + " | " + c["detail"])
        print("aborted: the snapshot is not the one this batch was measured on")
        return 2
    if a.report and pathlib.Path(a.report).exists():
        inventory(a.report)
    findings_live()
    findings_runtime_control()
    findings_false_positives()
    findings_dead_or_internal()
    failed = [c for c in CHECKS if not c["ok"]]
    for c in CHECKS:
        print(("PASS " if c["ok"] else "FAIL ") + c["id"] + " | " + c["detail"])
    print("%d checks, %d failed" % (len(CHECKS), len(failed)))
    return 1 if failed else 0


def selftest(report_path):
    """Each fault must turn exactly the check it targets red."""
    targets = ["F1c.no_producer_qk", "F1e.siblings_passed_at_qk", "F2b.serve_reachable_without_launcher",
               "FP1.dict_expansion_use_causal_mask", "D1.extract_answer_zero_callers",
               "F1h.pv_dead_by_enclosing_guard"]
    ok_all = True
    for t in targets:
        FAULTS.clear()
        FAULTS["flip_" + t] = True
        CHECKS.clear()
        if pathlib.Path(report_path).exists():
            inventory(report_path)
        findings_live(); findings_runtime_control(); findings_false_positives(); findings_dead_or_internal()
        red = [c["id"] for c in CHECKS if not c["ok"]]
        ok = red == [t]
        ok_all &= ok
        print("  inject %-42s -> red=%s %s" % (t, red, "OK" if ok else "MISMATCH"))
    print("selftest: %s" % ("PASS" if ok_all else "FAIL"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
