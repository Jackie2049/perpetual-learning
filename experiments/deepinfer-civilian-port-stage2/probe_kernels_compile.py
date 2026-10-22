"""探针：多个真实 DeepInfer tilelang kernel 文本级去 PDL 后能否在 SM89 (4090) 编译。
只验证编译（PDL 是唯一堵点的普适性），数值对照按 kernel 语义另做。

方法：服务器 cuda 包目录已有 *_pdlfree.py（由原文件 sed 生成，仅替换 pdl_sync->sync_grid、
pdl_trigger->pass，未改动原文件）。直接导入各文件的工厂函数并调用编译。
"""
import importlib
import sys
import time

sys.path.insert(0, "/home/zxw/deepinfer-civilian")
PKG = "deepinfer.model_executor.layers.tilelang_ops.cuda"

# (模块名, 工厂函数名, 最小构造参数) —— 参数取源码默认/最小合法值
CASES = [
    ("kv_index_cat", "kv_index_cat_tl", dict(dtype="float16", window_size=64, topk_tokens=16, out_stride=64)),
    ("dflash", "dflash_input_store_tl", dict(n_embed=1024, hc_mult=4)),
    ("get_kv_index", "load_kv_index_tl", dict(index_length=32, block_size=0, has_block_tables=False)),
    ("fused_sample", "fused_sample_tl", dict(vocab_size=128, do_sample=True, do_reject_sampling=False)),
]

for mod_name, fn_name, kwargs in CASES:
    mod = importlib.import_module(f"{PKG}.{mod_name}_pdlfree")
    fn = getattr(mod, fn_name)
    t0 = time.time()
    try:
        fn(**kwargs)
        print(f"[OK]   {fn_name} compiled in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"[FAIL] {fn_name}: {type(e).__name__}: {str(e)[:200]} (after {time.time()-t0:.1f}s)")
