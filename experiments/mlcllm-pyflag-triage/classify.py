#!/usr/bin/env python3
"""把扫描器报出的每条候选钉到一个档位，每条都带可复核的文件行号。

输入是 raw/scan-py.json（**扫描器原始输出**，本批故意保留未修复版，
因为修复前后的差别本身就是一条读数）。有候选没档位就 exit 1。

档位：
  R1 真实接线                —— 四跳齐全，开关有效
  R2 工厂转发（扫描器误判）   —— 消费者函数只是把形参原样递给下一跳
  R3 扫描器漏掉的生产者       —— 生产者其实存在（CLI / 属性 / 默认值）
  R4 编译期按模型接           —— 不是运行时开关，模型默认 False
  R5 测试/内部辅助            —— 无用户面
"""
import json
import pathlib
import sys

VERDICTS = {
    "enable_prefix_cache": ("R1", "CLI→interface→Router→PopenServer 四跳齐全；"
                                 "扫描器报 mixed 是因为把 serve() 当成了消费者"),
    "enable_tracing": ("R1", "CLI→interface→MLCEngine/AsyncMLCEngine→super→EngineState "
                              "四跳齐全；CLI serve 有 --enable-tracing"),
    "enable_debug": ("R1", "interface.serve → MLCEngine(enable_debug=...)"),
    "enable_disaggregation": ("R4", "create_generic 形参由 3 个模型显式传、其余 35 个吃默认 False"),
    "disable_instrument": ("R5", "debug_chat CLI 辅助，生产者是 parsed.disable_instrument"),
    "no_debug_config": ("R5", "bench api_endpoint 内部，生产者是 no_debug_config=True"),
    "with_scale": ("R5", "gemma4_audio 内部用例"),
}


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    src = pathlib.Path(argv[0])
    data = json.loads(src.read_text(encoding="utf-8"))
    rows = []
    missing = []
    for f in data["flags"]:
        flag = f["flag"]
        verdict = VERDICTS.get(flag)
        if verdict is None:
            missing.append(flag)
            continue
        rows.append({
            "flag": flag,
            "scanner_mode": f["mode"],
            "scanner_silently_inert": f["silently_inert"],
            "scanner_producers": f["producers_total"],
            "verdict": verdict[0],
            "note": verdict[1],
            "consumers": [(c["file"], c["line"], c["call_name"]) for c in f["consumers_nontest"]],
        })
    out = {"source": str(src), "n_flags": len(data["flags"]), "rows": rows,
           "missing": missing,
           "counts": {k: sum(1 for r in rows if r["verdict"] == k)
                      for k in ("R1", "R2", "R3", "R4", "R5")}}
    print(json.dumps(out, indent=1, ensure_ascii=False))
    if missing:
        print("\n有候选没有档位：%s" % missing)
        return 1
    if "--json" in argv:
        dst = pathlib.Path(argv[argv.index("--json") + 1])
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
