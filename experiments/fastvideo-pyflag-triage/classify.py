#!/usr/bin/env python3
"""FastVideo 的 8 条 `silently_inert` 逐条给档位与理由；漏一条 exit 1。

    python3 classify.py raw/scan-fastvideo.json
"""
import json
import sys

TIERS = {
    "use_bf16": ("R3", "attention/backends/sla.py:202/:404 形参 → :217/:423 `self.dtype = bf16 if use_bf16 else fp16`，"
                       ":303-319/:506 真用它 cast 与 autocast；构造走 attention/layer.py:83-90 的 **extra_impl_args，"
                       "仓库内调用方零生产者 ⇒ 恒 True（设计意图）"),
    "use_128x4_sf_layout": ("R3", "layers/fp4linear.py:36 形参 → :58/:68 选 SfLayout.layout_128x4 / layout_8x4；"
                                 "三处调用点（fp4linear.py:113、nvfp4_qat_train_config.py:50、测试 :89/:94）"
                                 "全部显式传 True ⇒ 值恒定、不是没人传"),
    "use_queue_mode": ("R3", "entrypoints/streaming_generator.py:81 形参 → :82-86 真分支（开 IPC 队列/拷贝 args）；"
                             "生产构造 :104-110 的 from_fastvideo_args 不传 ⇒ 恒 True；唯一 False 在测试参数化里"),
    "use_nvfp4": ("R3", "layers/quantization/nvfp4_config.py:306 的 _mm_fp4_op 形参 → :325 原样转给 mm_fp4；"
                        "调用端 :416 kwargs.pop('use_nvfp4', True) → :424 传下 ⇒ 有生产者但值恒为默认"),
    "use_high_prec_o": ("R5", "fastvideo-kernel/.../triton_kernels/attn_qat_train.py:1230 的 forward；"
                              "fastvideo-kernel 是独立子工程（自带 pyproject），形参只由同文件内部包装传"),
    "use_qat_qkv_backward": ("R5", "同一个 forward（attn_qat_train.py:1230），默认 True，子工程内部调用方不传"),
    "enable_smoothing_k": ("R5", "fastvideo-kernel/attn_qat_infer/api.py:88 的 preprocess_qkv"),
    "enable_smoothing_q": ("R5", "同上（同一函数的第二个参数）"),
}


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    data = json.load(open(argv[1], encoding="utf-8"))
    inert = [f for f in data["flags"] if f.get("silently_inert")]
    missing = []
    tiers = {}
    for f in sorted(inert, key=lambda x: x["flag"]):
        name = f["flag"]
        tier, why = TIERS.get(name, ("?", "未归因"))
        if tier == "?":
            missing.append(name)
        tiers[tier] = tiers.get(tier, 0) + 1
        print("%-22s %-7s %-3s %s" % (name, f["mode"], tier, why))
    print("\n%d 条候选：%s" % (len(inert), tiers))
    if missing:
        print("未归因：%s" % missing)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
