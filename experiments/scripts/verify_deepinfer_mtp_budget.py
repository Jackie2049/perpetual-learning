#!/usr/bin/env python3
"""Compile the real MTPVarlen header with only Sequence/TORCH_CHECK stubs.

Usage: python3 verify_deepinfer_mtp_budget.py /path/to/DeepInfer [--cxx clang++]
No DeepInfer dependencies, device code, model or performance measurement.
The single sequence.hpp include is replaced; scheduler/estimator bodies are intact.
"""

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile


PRELUDE = r"""
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error(#condition); } while (0)
struct MtpFixture {
    std::vector<int> draft_tokens;
    std::vector<float> accept_rates, prev_accept_rates;
    int process_cnt = 2, schedule_cnt = 2;
    bool has_draft_info = true;
};
struct Sequence { MtpFixture mtp; };
"""

TESTS = r"""
int checks = 0;
void check(bool ok, const char *label) {
    ++checks;
    if (!ok) throw std::runtime_error(label);
}
bool near(double a, double b) { return std::abs(a - b) < 1e-5; }
using Batch = std::vector<std::shared_ptr<Sequence>>;
Batch batch(int n) {
    Batch b;
    for (int i = 0; i < n; ++i) {
        b.push_back(std::make_shared<Sequence>());
        b.back()->mtp.draft_tokens.resize(2);
    }
    return b;
}
std::vector<int> run(const MTPVarlen &m, const Batch &b) {
    std::vector<int> q{999}; // also verifies output reset
    m.schedule(b, q);
    return q;
}
int total(const std::vector<int> &q) { return std::accumulate(q.begin(), q.end(), 0); }
double expected(const Batch &b, const std::vector<int> &q, bool old) {
    double value = b.size();
    for (size_t i = 0; i < b.size(); ++i) {
        const auto &r = old ? b[i]->mtp.prev_accept_rates : b[i]->mtp.accept_rates;
        for (int j = 0; j < q[i] - 1; ++j) value += r[j];
    }
    return value;
}
int main() {
    SamplerRuntimeEstimator sampler;
    for (int i = 1; i <= 4; ++i) sampler.update(i);
    check(!sampler.get(), "sampler warmup");
    sampler.update(5);
    check(near(*sampler.get(), 3), "sampler median");
    for (int i = 6; i <= 10; ++i) sampler.update(i);
    check(near(*sampler.get(), 6), "sampler ring eviction");

    ForwardRuntimeEstimator forward;
    for (int i = 0; i < 9; ++i) forward.update(4, 100);
    check(near(forward.get(), 0), "forward per-graph warmup");
    forward.update(4, 100);
    for (int i = 0; i < 100; ++i) forward.update(8, 2);
    check(near(forward.get(), 51), "graph medians equally weighted, not sample weighted");
    for (int i = 0; i < 5000; ++i) forward.update(8, 2);
    check(near(forward.get(), 2), "forward window eviction");

    MTPVarlen m;
    auto b = batch(2);
    check(run(m, b) == std::vector<int>({3, 3}), "no curve means all drafts");
    m.set_decode_perf_curve({{2, 1}, {3, 1.1f}, {4, 1.2f}, {6, 3}});
    b[0]->mtp.prev_accept_rates = {.9f, .8f};
    b[1]->mtp.prev_accept_rates = {.2f, .1f};
    b[0]->mtp.accept_rates = {.2f, .1f};
    b[1]->mtp.accept_rates = {.9f, .8f};
    auto q = run(m, b);
    check(q == std::vector<int>({1, 3}), "old rates budget=4, latest rates allocation");
    MTPVarlen feedback;
    feedback.set_decode_perf_curve(m.get_decode_perf_curve());
    for (int i = 0; i < 4; ++i) feedback.update_runtime(99, 4, 1.2f, 100);
    check(total(run(feedback, b)) == 4, "sampler feedback not used before five samples");
    feedback.update_runtime(99, 4, 1.2f, 100);
    check(total(run(feedback, b)) == 6, "global sampler fallback amortizes cost over larger budget");
    for (int i = 0; i < 5; ++i) feedback.update_runtime(2, 4, 1.2f, 0);
    check(total(run(feedback, b)) == 4, "same request-count sampler overrides fallback");
    for (int i = 0; i < 10; ++i) feedback.update_runtime(2, 4, 41.2f, 0);
    check(total(run(feedback, b)) == 2, "positive forward residual reduces speculation");
    MTPVarlen negative;
    negative.set_decode_perf_curve(m.get_decode_perf_curve());
    for (int i = 0; i < 10; ++i) negative.update_runtime(2, 4, .2f, 0);
    check(total(run(negative, b)) == 4, "negative forward slope is clamped to baseline");
    std::swap(b[0]->mtp.accept_rates, b[1]->mtp.accept_rates);
    check(run(m, b) == std::vector<int>({3, 1}), "latest rates change allocation, not total");
    for (auto &s : b) s->mtp.prev_accept_rates = {0, 0};
    check(run(m, b) == std::vector<int>({1, 1}), "zero older rates disable extra verification");
    for (auto &s : b) {
        s->mtp.accept_rates.clear();
        s->mtp.prev_accept_rates.clear();
    }
    check(total(run(m, b)) == 4, "cold with draft uses ones");
    for (auto &s : b) s->mtp.has_draft_info = false;
    check(total(run(m, b)) == 2, "cold without draft uses zeros");

    // An outstanding ZOS step changes which CPU buffer represents N-2.
    b[0]->mtp.accept_rates = {.9f, .8f};
    b[1]->mtp.accept_rates = {.2f, .1f};
    for (auto &s : b) {
        s->mtp.prev_accept_rates = {0, 0};
        ++s->mtp.schedule_cnt;
    }
    check(run(m, b) == std::vector<int>({3, 1}), "inflight uses accept_rates as N-2");
    b[0]->mtp.schedule_cnt += 1;
    bool rejected = false;
    try { run(m, b); } catch (const std::runtime_error &) { rejected = true; }
    check(rejected, "more than one outstanding step rejected");

    // Exact ties favor lower sequence index; equal throughput keeps smaller budget.
    b = batch(2);
    for (auto &s : b) s->mtp.accept_rates = s->mtp.prev_accept_rates = {.5f, .5f};
    m.set_decode_perf_curve({{2, 1}, {3, 1}, {4, 10}, {6, 20}});
    check(run(m, b) == std::vector<int>({2, 1}), "equal confidence sequence-index tie");
    m.set_decode_perf_curve({{2, 1}, {3, 1.25f}, {4, 1.5f}, {6, 2}});
    check(total(run(m, b)) == 2, "equal throughput keeps smaller total_q");

    // Exhaust all 3^3 valid per-request prefix lengths, independently of the
    // header's flattened sort, for seeded synthetic cumulative probabilities.
    std::mt19937 rng(20261020);
    std::uniform_real_distribution<float> uniform(.01f, .99f);
    const std::vector<std::pair<int, float>> curve{{3, 1}, {4, 1.1f}, {6, 1.4f}, {9, 2.2f}};
    m.set_decode_perf_curve(curve);
    for (int trial = 0; trial < 200; ++trial) {
        auto rows = batch(3);
        for (auto &s : rows) {
            for (auto *r : {&s->mtp.accept_rates, &s->mtp.prev_accept_rates}) {
                float first = uniform(rng);
                *r = {first, first * uniform(rng)};
            }
        }
        auto got = run(m, rows);
        double best_tp = -1, best_latest = -1;
        int best_total = 0;
        for (int a = 1; a <= 3; ++a) for (int c = 1; c <= 3; ++c) for (int d = 1; d <= 3; ++d) {
            std::vector<int> candidate{a, c, d};
            auto point = std::find_if(curve.begin(), curve.end(), [&](auto p) { return p.first >= total(candidate); });
            double tp = expected(rows, candidate, true) / point->second;
            if (tp > best_tp + 1e-7 || (near(tp, best_tp) && total(candidate) < best_total)) {
                best_tp = tp;
                best_total = total(candidate);
            }
            if (total(candidate) == total(got)) best_latest = std::max(best_latest, expected(rows, candidate, false));
        }
        check(total(got) == best_total, "budget matches exhaustive prefix optimum");
        check(near(expected(rows, got, false), best_latest), "allocation maximizes latest expected tokens at fixed budget");
    }
    std::cout << "PASS " << checks << " checks (real header bodies; fixture Sequence; CPU only)\n";
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--cxx", default="clang++")
    parser.add_argument("--cxxflag", action="append", default=[], help="extra compiler flag; use --cxxflag=-flag")
    args = parser.parse_args()
    compiler = shutil.which(args.cxx)
    if not compiler:
        parser.error(f"compiler not found: {args.cxx}")
    path = args.source / "csrc/engine/mtp_varlen.hpp"
    raw = path.read_bytes()
    source = raw.decode()
    include = '#include "sequence.hpp"'
    if source.count(include) != 1:
        parser.error("expected exactly one sequence.hpp include; inspect source drift")
    print(f"mtp_varlen.hpp SHA256 {hashlib.sha256(raw).hexdigest()}", flush=True)
    print(subprocess.check_output([compiler, "--version"], text=True).splitlines()[0], flush=True)
    # The stubs use STL containers, so insert them at the existing include after
    # the header's standard includes, not before those includes.
    with tempfile.TemporaryDirectory(prefix="deepinfer-mtp-cpu-") as scratch:
        cpp = Path(scratch) / "contract.cpp"
        binary = Path(scratch) / "contract"
        cpp.write_text(source.replace(include, PRELUDE) + TESTS)
        subprocess.run([compiler, "-std=c++17", "-O2", *args.cxxflag, str(cpp), "-o", str(binary)], check=True, timeout=60)
        subprocess.run([str(binary)], check=True, timeout=30)


if __name__ == "__main__":
    main()
