#!/usr/bin/env python3
"""Execute immutable original DeepInfer output-commit loop in a CPU harness.

Real KvCacheState/SequenceScheduleState and original per-request loop/_check_stop.
Sequence storage, tensor access, SO status, queues and slot release are doubles.
No full Scheduler, sampler/kernel, KV storage, model, Gloo or device validation.
"""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

REVISION = '774ce244b87cc0eb8d4b94ab9f5de2ab22049f01'
PRELUDE = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <deque>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/types.h>
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error(#c); } while (0)
'''
DOUBLES = r'''
struct Data {
    std::vector<int> token_ids;
    int prompt_len=10;
    int get_len() const { return token_ids.size(); }
};
struct Sequence {
    struct { Data tokens, compressed_tokens; } input;
    KvCacheState cache;
    SequenceScheduleState schedule;
    struct { int last_output_token=-1; std::vector<int> accepted_tokens, draft_tokens; } mtp;
    struct { int status=0; } structured_output;
    struct { struct { bool ignore_eos=false; int max_tokens=100; } stopping; } generation;
    struct { double arrival_time=0; } request;
    bool is_finished() const { return ::is_finished(schedule.status); }
    void append_token_id(int t) { input.tokens.token_ids.push_back(t); }
    void append_compressed_token_id(int t) { input.compressed_tokens.token_ids.push_back(t); }
    int get_last_token_id() const { return input.tokens.token_ids.back(); }
    int get_output_len() const { return input.tokens.get_len()-input.tokens.prompt_len; }
    int get_nextn_len() const { return mtp.draft_tokens.size(); }
};
using SequencePtr=std::shared_ptr<Sequence>;
enum class EngineType { NORMAL };
struct SchedulerOutputs { bool prompt_run=false; std::vector<SequencePtr> scheduled_seqs; std::vector<int> seqlens_q; };
struct Status { int value=0; template<class T> T cast() const { return value; } };
struct Probe {
    EngineType engine_type=EngineType::NORMAL;
    int eos_token_id=11, max_model_len=1000, released=0;
    std::vector<SequencePtr> running;
    std::deque<SequencePtr> waiting;
    std::vector<int> pushed_tokens_before_finish;
    void release_mtp_info_slot(Sequence&) { ++released; }
    int map_engram(int t) { return t; } // identity adapter, not tokenizer execution
_CHECK_STOP_
    void process(SchedulerOutputs& scheduler_outputs, const std::vector<int>& accepted, const std::vector<int>& main,
                 bool so_status_bool=false, std::vector<Status> so_status={}) {
        auto output_token_ids_data=main;
        auto accepted_token_ids_ptr=accepted.data();
        bool has_accepted_tokens=!accepted.empty();
        int flat_offset=0;
        ssize_t num_output_tokens=main.size();
        double ts=0;
        std::vector<double> first_token_latencies;
        std::vector<int> accepted_token_ids_vec, token_ids;
        pushed_tokens_before_finish.clear();
_LOOP_
        assert(num_output_tokens == static_cast<ssize_t>(scheduler_outputs.scheduled_seqs.size()));
    }
};
SequencePtr make_seq(int cached=9, SequenceStatus status=SequenceStatus::RUNNING) {
    auto s=std::make_shared<Sequence>();
    s->input.tokens.token_ids.assign(10,7);
    s->input.compressed_tokens.token_ids.assign(10,7);
    s->cache.set_num_tokens(cached);
    s->schedule.status=status;
    s->mtp.draft_tokens={21,22,23};
    return s;
}
void check(const SequencePtr& s, int cached, int length, int previous) {
    assert(s->cache.num_tokens==cached && s->input.tokens.get_len()==length);
    assert(s->cache.previous_num_tokens==previous);
    assert(s->input.compressed_tokens.get_len()==length);
}
'''
CONTROLS = r'''
int main() {
    int count=0;
    // rejection, partial acceptance, all accepted: final sampled token is not an accepted draft.
    for (int a=0; a<=3; ++a) {
        Probe p; auto s=make_seq(); s->schedule.start_inflight_step(4);
        SchedulerOutputs o{false,{s},{4}};
        std::vector<int> accepted(4,-1);
        for (int j=0;j<a;++j) accepted[j]=21+j;
        p.process(o,accepted,{99});
        check(s,10+a,11+a,9);
        assert(s->mtp.accepted_tokens.size()==static_cast<size_t>(a));
        assert(p.pushed_tokens_before_finish==std::vector<int>{a+1});
        assert(s->schedule.total_inflight_tokens==0);
        ++count;
    }
    // Stop among accepted drafts: ledger retains verification prefix, append count is shorter.
    {
        Probe p; auto s=make_seq(); SchedulerOutputs o{false,{s},{4}};
        p.process(o,{11,22,23,-1},{99});
        check(s,10,11,9);
        assert(s->mtp.accepted_tokens.size()==3 && s->mtp.last_output_token==99);
        assert(s->get_last_token_id()==11 && s->schedule.status==SequenceStatus::FINISHED_STOPPED);
        assert(p.pushed_tokens_before_finish==std::vector<int>{1} && p.released==1); ++count;
    }
    {
        Probe p; auto s=make_seq(); s->generation.stopping.max_tokens=2;
        SchedulerOutputs o{false,{s},{4}}; p.process(o,{21,22,23,-1},{99});
        check(s,11,12,9);
        assert(s->mtp.accepted_tokens.size()==3 && p.pushed_tokens_before_finish==std::vector<int>{2});
        assert(s->schedule.status==SequenceStatus::FINISHED_LENGTH_CAPPED); ++count;
    }
    // Persisted SO failure is sufficient, even without a new status report.
    {
        Probe p; auto s=make_seq(); s->structured_output.status=-1;
        SchedulerOutputs o{false,{s},{4}}; p.process(o,{21,22,-1,-1},{99});
        check(s,10,11,9);
        assert(s->get_last_token_id()==0 && s->schedule.status==SequenceStatus::FINISHED_SO_FAILED);
        assert(p.pushed_tokens_before_finish==std::vector<int>{1}); ++count;
    }
    // Partial prefill advances by q without appending; final prefill must not commit by output count.
    for (bool final: {false,true}) {
        Probe p; auto s=make_seq(final?6:0, final?SequenceStatus::RUNNING:SequenceStatus::WAITING);
        s->mtp.draft_tokens.clear(); SchedulerOutputs o{true,{s},{4}}; p.process(o,{}, {99});
        check(s,final?10:4,final?11:10,final?6:0);
        assert(p.pushed_tokens_before_finish==std::vector<int>{final?1:0});
        assert(final?p.running.size()==1:p.waiting.size()==1); ++count;
    }
    // Skipped rows still advance flat offset; they must not advance cache or append tokens.
    for (auto status: {SequenceStatus::FINISHED_ABORTED,SequenceStatus::WAITING}) {
        Probe p; auto stale=make_seq(9,status), live=make_seq();
        if (status==SequenceStatus::WAITING) {
            // Execute the original primitive reset methods, not full _preempt/free_seq.
            stale->schedule.start_inflight_step(4);
            stale->cache.reset_for_recompute(); stale->schedule.reset_for_recompute();
        }
        SchedulerOutputs o{false,{stale,live},{2,4}};
        p.process(o,{31,-1,21,22,-1,-1},{98,99});
        check(stale,status==SequenceStatus::WAITING?0:9,10,status==SequenceStatus::WAITING?0:9);
        check(live,12,13,9);
        assert(live->mtp.accepted_tokens==std::vector<int>({21,22}));
        assert(p.pushed_tokens_before_finish==std::vector<int>({0,3})); ++count;
    }
    // Two reserved steps: retiring the older one retains only the latest reservation.
    {
        Probe p; auto s=make_seq(); s->schedule.start_inflight_step(4); s->schedule.start_inflight_step(4);
        SchedulerOutputs o{false,{s},{4}}; p.process(o,{21,-1,-1,-1},{99});
        check(s,11,12,9); assert(s->schedule.total_inflight_tokens==4);
        p.process(o,{-1,-1,-1,-1},{98});
        check(s,12,13,11); assert(s->schedule.total_inflight_tokens==0); ++count;
    }
    assert(count==12);
    std::cout << "PASS " << count << " original C++ loop/state controls; Sequence/tensor/resource doubles.\n";
}
'''


def read(root, path):
    data = subprocess.check_output(['git', 'show', f'{REVISION}:{path}'], cwd=root)
    print(path, 'sha256=' + hashlib.sha256(data).hexdigest(), flush=True)
    return data.decode()


def between(source, start, end):
    return source[source.index(start):source.index(end, source.index(start))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_root', type=Path)
    parser.add_argument('--cxx', default='clang++')
    args = parser.parse_args()
    seq = read(args.source_root, 'csrc/engine/sequence.hpp')
    scheduler = read(args.source_root, 'csrc/engine/scheduler.hpp')
    types = between(seq, 'enum class SequenceStatus', 'inline std::string get_finished_reason')
    types += between(seq, 'struct KvCacheState', 'struct StructuredOutputState')
    types += between(seq, 'struct SequenceScheduleState', '/// C++ 内部的命名构造参数')
    stop = between(scheduler, '    void _check_stop(', '    void set_schedulable(')
    loop = between(scheduler, '        for (ssize_t i = 0; i < num_output_tokens;', '        // Virtual sequences')
    program = PRELUDE + types + DOUBLES.replace('_CHECK_STOP_', stop).replace('_LOOP_', loop) + CONTROLS
    flags = ['-std=c++20', '-O0']
    if sys.platform == 'darwin':
        sdk = subprocess.check_output(['xcrun', '--show-sdk-path'], text=True).strip()
        flags += ['-isysroot', sdk, '-isystem', str(Path(sdk) / 'usr/include/c++/v1')]
    compiler = shutil.which(args.cxx)
    if not compiler:
        raise RuntimeError('C++ compiler not found')
    with tempfile.TemporaryDirectory(prefix='deepinfer-commit-slice-') as folder:
        cpp, binary = Path(folder)/'probe.cpp', Path(folder)/'probe'
        cpp.write_text(program)  # generated build output, not an upstream edit
        subprocess.run([compiler, *flags, str(cpp), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True)
    print('No full extension, sampler, allocation, output formatter, GPU/collective or physical KV verification.')


if __name__ == '__main__':
    main()
