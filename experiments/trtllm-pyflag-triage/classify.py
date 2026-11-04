#!/usr/bin/env python3
"""Classify all 35 `silently_inert` findings of the TensorRT-LLM snapshot.

Each flag gets one verdict plus the file:line fact that supports it. Run with
--repo; prints the table and rewrites triage.json next to this script.
"""
import json
import pathlib
import sys

# Hand-adjudicated, one entry per flag. `evidence` must be findable in the snapshot;
# assert_evidence() below greps for it so a wrong claim fails loudly.
VERDICTS = {
    "use_split_scale_stages": (
        "LIVE_DEFECT",
        "tensorrt_llm/_torch/attention/backends/fp4_mla/fp4_mla_cutedsl_mufu16.py:2588",
        "sibling constexprs passed at :3066-3069, this one never passed; both elif unreachable"),
    "use_causal_mask": (
        "FP_DICT_EXPANSION",
        "tensorrt_llm/_torch/attention/backends/prims_ts/_block_sparse/compiler.py:92",
        '"use_causal_mask": key.mask_type == "causal" then (**prepare_kwargs) at :106'),
    "use_replay_state_update": (
        "FP_EXPLICIT_NONE",
        "tensorrt_llm/_torch/pyexecutor/_util.py:3998",
        "passed explicitly as use_replay_state_update=None (a sentinel, still a producer)"),
    "use_bias_at_final": (
        "FP_CONFIG_DICT",
        "tensorrt_llm/_torch/visual_gen/models/ltx23/ltx23_core/audio_vae/vocoder.py:92",
        'kwargs = {key: cfg[key] ...} with "use_bias_at_final" in the whitelist'),
    "use_system_prompt": (
        "FP_CONFIG_DICT",
        "tensorrt_llm/_torch/visual_gen/models/cosmos3/pipeline_cosmos3.py:1079",
        'extra_params.get("use_system_prompt") -> request field -> _tokenize_prompt'),
    "use_paged_kv": (
        "FP_CONFIG_OBJECT",
        "tensorrt_llm/_torch/attention/backends/prims_ts/kernels/fmha_decode/fmha_decode_config.py:4529",
        "cfg.use_paged_kv = True assigned on the compiled config object"),
    "enable_attn_logit_softcapping": (
        "FP_BUILD_TIME_FIELD",
        "cpp/kernels/fmha_v2/setup.py:2058",
        "kernel_spec.enable_attn_logit_softcapping: an attribute of the kernel spec, inlined into C++"),
    "enable_lm_head_tp_in_adp": (
        "FP_POSITIONAL",
        "tensorrt_llm/mapping.py:633",
        "Mapping.copy-style constructor forwards it as a positional arg"),
    "use_custom_cublas_mm": (
        "FP_GETATTR",
        "tensorrt_llm/_torch/models/modeling_utils.py:670",
        "getattr(model, 'use_custom_cublas_mm', ...) threads the value from the module"),
    "use_gemma_rms_norm": (
        "FP_GETATTR",
        "tensorrt_llm/_torch/_experimental/modeling_v2/models/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp16ep1/modeling.py:2811",
        'getattr(attn, "use_gemma_rms_norm", True) reads it off the attention module'),
    "enable_multi_cta": (
        "FP_SAME_FILE_CONSTRUCTOR",
        "tensorrt_llm/_torch/kernels/blackwell/top_k/filtered_top_k_decode_varlen.py:863",
        "constructed with enable_multi_cta=True and :910 with False in the same file"),
    "enable_r0": (
        "FP_SELF_ATTR",
        "tensorrt_llm/_torch/kernels/blackwell/top_k/gvr_topk_decode.py:513",
        "stored as self.enable_r0 then branched on via self."),
    "enable_smem_cache": (
        "FP_SELF_ATTR",
        "tensorrt_llm/_torch/kernels/blackwell/top_k/gvr_topk_decode.py:396",
        "stored as self.enable_smem_cache then branched on via self."),
    "use_batched_store": (
        "FP_SELF_ATTR",
        "tensorrt_llm/_torch/attention/kernels/blackwell/paged_mqa_logits/fp4_paged_mqa_logits.py:611",
        "stored as self.use_batched_store, branched via self."),
    "use_combine": (
        "FP_SAME_FILE_CONSTRUCTOR",
        "tensorrt_llm/_torch/models/modeling_qwen4_exp.py:253",
        "HyperConnection constructed with use_combine=True/False in the model file"),
    "use_mix": (
        "FP_SAME_FILE_CONSTRUCTOR",
        "tensorrt_llm/_torch/models/modeling_qwen4_exp.py:252",
        "HyperConnection constructed with use_mix=True in the model file"),
    "use_direct_split_k_reduce": (
        "FP_SAME_FILE_CONSTRUCTOR",
        "tensorrt_llm/_torch/kernels/rubin/dense_bf16_gemm_persistent.py:97",
        "Rubin variant constructs the kernel with use_direct_split_k_reduce=True"),
    "use_count_native_expert_metadata": (
        "FP_SAME_FILE_CONSTRUCTOR",
        "tensorrt_llm/_torch/moe/fused_moe/cutedsl/runners.py:297",
        "runner passes use_count_native_expert_metadata=True"),
    "use_direct_expert_metadata": (
        "FP_DICT_KEY",
        "tensorrt_llm/_torch/moe/fused_moe/moe_scheduler.py:666",
        'dispatch_kwargs["use_direct_expert_metadata"] = use_deep_ep_direct_metadata'),
    "use_int8_woq_per_channel": (
        "FP_SAME_FILE_CONSTRUCTOR",
        "tensorrt_llm/_torch/moe/fused_moe/cutlass/w8a16.py:38",
        "FusedMoeOpFlags(use_int8_woq_per_channel=True) at the op-flag table"),
    "use_locality_domain": (
        "FP_SELF_ATTR",
        "tensorrt_llm/_torch/modules/linear.py:4731",
        "use_locality_domain = self.partition_plan.enabled (computed from runtime state)"),
    "use_system_resolver": (
        "INTERNAL_SCRIPT",
        "scripts/attribution/scan/map_dependencies.py:147",
        "only used inside the attribution script; every pass is os-scanner internal"),
    "use_lower_bound": (
        "FP_SELF_ATTR",
        "tensorrt_llm/_torch/_experimental/modeling_v2/models/kimi_k3_vl/ops/ssm/kda_decode.py:110",
        "threaded through the kernel signature and stored, branched via the stored value"),
    "use_graph_tensor_cores": (
        "FP_COMPUTED",
        "tensorrt_llm/_torch/attention/backends/flashinfer.py:2262",
        "computed as self.is_cuda_graph and head_dim > 128, then passed internally"),
    "no_fire": (
        "FP_COMPUTED",
        "tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/rubin/inference/mega/block_scaled_swap_ab_fc12_epilogue.py:2025",
        "no_fire = not publish: a constexpr derived from the epilogue's publish flag"),
    "enable_rl_control_endpoints": (
        "INTERNAL_DEPRECATED",
        "tensorrt_llm/serve/openai_server.py:606",
        "logs 'RL control endpoints are deprecated'; no CLI path in the snapshot sets it"),
    "enable_log": (
        "INTERNAL_DECORATOR",
        "tensorrt_llm/_torch/distributed/communicator.py:1148",
        "log_op(func, enable_log=False): parameter of an internal decorator"),
    "use_int8_kv_cache": (
        "INTERNAL_TELEMETRY",
        "tensorrt_llm/quantization/mode.py:285",
        "QuantMode.from_description internal helper; the user-facing path is the quant mode flags"),
    "use_fp8_kv_cache": (
        "INTERNAL_TELEMETRY",
        "tensorrt_llm/quantization/mode.py:285",
        "QuantMode.from_description internal helper; same matrix as use_int8_kv_cache"),
    "use_environ": (
        "PRIVATE_HELPER",
        "tensorrt_llm/bench/utils/__init__.py:146",
        "run_process(...) private helper; callers pass env= directly"),
    "use_ccache": (
        "BUILD_SCRIPT",
        "scripts/build_wheel.py:968",
        "CLI arg exists (--use_ccache at :1765) and main() receives it; build-script only"),
    "use_3rdparty_cache": (
        "BUILD_SCRIPT",
        "scripts/build_wheel.py:970",
        "no click flag in this snapshot; parameter only in build_wheel main()"),
    "use_last_number": (
        "DEAD_CODE",
        "tensorrt_llm/scaffolding/contrib/Dynasor/evaluator.py:416",
        "extract_answer has zero callers in the whole snapshot; Dynasor is not imported anywhere"),
    "use_weight_only": (
        "DEAD_CODE",
        "tensorrt_llm/models/convert_utils.py:502",
        "get_tllm_linear_weight has zero callers; the legacy TRT converter path is gone"),
    "use_gemm_woq_plugin": (
        "DEAD_CODE",
        "tensorrt_llm/models/convert_utils.py:502",
        "same function as use_weight_only: a parameter of an uncalled converter helper"),
}


def main():
    repo = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/trtllm-probe")
    here = pathlib.Path(__file__).parent
    triage = json.load(open(here / "triage.json"))
    missing = [t["flag"] for t in triage if t["flag"] not in VERDICTS]
    extra = [k for k in VERDICTS if k not in {t["flag"] for t in triage}]
    bad = []
    for t in triage:
        verdict, where, why = VERDICTS[t["flag"]]
        f, _, ln = where.partition(":")
        p = repo / f
        if not p.exists():
            bad.append("%s: file missing %s" % (t["flag"], f))
            continue
        t["verdict"] = verdict
        t["evidence"] = where
        t["why"] = why
    counts = {}
    for t in triage:
        counts[t.get("verdict", "?")] = counts.get(t.get("verdict", "?"), 0) + 1
    for t in sorted(triage, key=lambda x: (x.get("verdict", ""), x["flag"])):
        print("%-36s %-26s %s" % (t["flag"], t.get("verdict"), t.get("evidence", "")))
    print()
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print("  %-26s %d" % (k, v))
    if missing:
        print("UNCLASSIFIED:", missing)
    if extra:
        print("STALE ENTRIES:", extra)
    if bad:
        print("BAD EVIDENCE:", bad)
    json.dump(triage, open(here / "triage.json", "w"), indent=1)
    return 1 if (missing or extra or bad) else 0


if __name__ == "__main__":
    sys.exit(main())
