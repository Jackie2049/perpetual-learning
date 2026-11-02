#!/usr/bin/env python3
"""把 DAPO A/B 的 gen 差拆成「固定等待」与「多生成的组」两笔。

Usage: python3 decompose-gen.py summary-rerun.json

模型：gen_i = a_arm + s · workload_i（各臂独立截距、共同斜率）。截距差即轮询量子
贡献的固定等待差，斜率乘以工作量差即多生成带来的真实计算差。
workload 给两种口径：生成组数（summary 的 filter.generated_values）与
生成 token 数（组数 × response_length/mean）。
"""
import json
import random
import statistics
import sys


def solve(a, b):
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for i in range(len(m)):
        piv = max(range(i, len(m)), key=lambda r: abs(m[r][i]))
        m[i], m[piv] = m[piv], m[i]
        for r in range(len(m)):
            if r != i:
                f = m[r][i] / m[i][i]
                for c in range(i, len(m[0])):
                    m[r][c] -= f * m[i][c]
    return [m[i][-1] / m[i][i] for i in range(len(m))]


def fit(rows, control):
    """rows: (mode, workload, gen)。返回 {mode: (intercept, slope, cost)}。"""
    modes = sorted({r[0] for r in rows})
    sx = [[0.0] * (len(modes) + 1) for _ in range(len(modes) + 1)]
    sy = [0.0] * (len(modes) + 1)
    for mode, x, y in rows:
        v = [1.0 if mode == m else 0.0 for m in modes] + [x]
        for i in range(len(v)):
            sy[i] += v[i] * y
            for j in range(len(v)):
                sx[i][j] += v[i] * v[j]
    beta = solve(sx, sy)
    slope = beta[-1]
    resid = [y - (sum(beta[i] for i, m in enumerate(modes) if m == mode) + slope * x) for mode, x, y in rows]
    return modes, beta[:-1], slope, statistics.stdev(resid), resid


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "summary-rerun.json"
    data = json.loads(open(path).read())
    modes = [m for m in data if "timing" in data[m] and m != "arms"]
    control = next(m for m in modes if data[m].get("resolved_poll_interval") in (None, 2.0))
    print(f"control = {control};  runs = {modes}\n")

    for label, use_tokens in (("每步生成组数", False), ("每步生成 token 数", True)):
        rows = []
        for m in modes:
            v = data[m]
            gen = v["gen_values"]
            f = v["filter"]["generated_values"]
            n = v["numeric"]["response_length/mean"]
            w = [f[i] * n[i] for i in range(len(gen))] if use_tokens else f
            rows += [(m, w[i], gen[i]) for i in range(len(gen))]
        modes_f, intercepts, slope, sd_res, _ = fit(rows, control)
        base_a = intercepts[modes_f.index(control)]
        base_w = statistics.fmean(r[1] for r in rows if r[0] == control)
        base_g = statistics.fmean(data[control]["gen_values"])
        print(f"== 模型：{label}（共同斜率 {slope * (1e6 if use_tokens else 1e3):.1f} "
              f"{'μs/token' if use_tokens else 'ms/组'}，残差 sd {sd_res:.3f}s）")
        for m in modes_f:
            a = intercepts[modes_f.index(m)]
            w = statistics.fmean(r[1] for r in rows if r[0] == m)
            g = statistics.fmean(data[m]["gen_values"])
            step = data[m]["timing"]["step"]["mean"]
            print(f"  {m:9s} 截距 {a:6.3f}s  工作量 {w:8.0f}  gen {g:6.3f}s  整步 {step:6.3f}s")
            if m != control:
                extra = slope * (w - base_w)
                print(f"            vs {control}: Δgen {g - base_g:+.3f}s = 固定等待 {a - base_a:+.3f}s"
                      f" + 工作量 {extra:+.3f}s（残差 {g - base_g - (a - base_a) - extra:+.3f}s）")
        # 按臂内 step 重采样做 bootstrap，给"固定等待差"一个区间
        by_mode = {m: [r for r in rows if r[0] == m] for m in modes_f}
        boots = []
        rng = random.Random(0)
        for _ in range(2000):
            sample = []
            for m, rs in by_mode.items():
                sample += [rs[rng.randrange(len(rs))] for _ in rs]
            midx = [x[0] for x in sample]
            mi = sorted(set(midx))
            if len(mi) < 2:
                continue
            ms, ic, _sl, _sd, _r = fit(sample, control)
            if control not in ms:
                continue
            boots.append((ic[ms.index(control)], ms, ic))
        if boots:
            print("  bootstrap（n=2000，臂内重采样）固定等待差的 2.5%/97.5% 分位：")
            for m in modes_f:
                if m == control:
                    continue
                vals = sorted(ic[ic_m.index(m)] - ic[ic_m.index(control)] for _b, ic_m, ic in boots if m in ic_m)
                if vals:
                    print(f"    {m:9s} {statistics.fmean(vals):+.3f}s  [{vals[int(0.025 * len(vals))]:+.3f}, "
                          f"{vals[int(0.975 * len(vals)) - 1]:+.3f}]")
        print()


if __name__ == "__main__":
    main()
