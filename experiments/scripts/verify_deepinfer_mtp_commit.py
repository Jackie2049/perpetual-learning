#!/usr/bin/env python3
"""Execute immutable original DeepInfer output-commit loop in a CPU harness.

Real KvCacheState/SequenceScheduleState and original per-request loop/_check_stop.
Sequence storage, tensor access, SO status, queues and slot release are doubles.
No full Scheduler, sampler/kernel, KV storage, model, Gloo or device validation.
--check-output-identity also runs original formatter loop/RequestOutput with a
heap-pool adapter, plus original Python SO-mask method with hardware doubles.
"""
import argparse
import ast
from collections import defaultdict
from contextlib import nullcontext
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS

REVISION = '774ce244b87cc0eb8d4b94ab9f5de2ab22049f01'
PRELUDE = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <deque>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/types.h>
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error(#c); } while (0)
namespace c10 { template<class T> using optional=std::optional<T>; constexpr auto nullopt=std::nullopt; }
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
    struct { int last_output_token=-1; std::vector<int> accepted_tokens, draft_tokens; std::vector<float> prev_confidence_raw; } mtp;
    struct { int status=0; } structured_output;
    struct { struct { bool ignore_eos=false; int max_tokens=100; } stopping; struct { int top_logprobs=0; } sampling; } generation;
    struct { double arrival_time=0; std::string request_id="r"; } request;
    bool is_finished() const { return ::is_finished(schedule.status); }
    void append_token_id(int t) { input.tokens.token_ids.push_back(t); }
    void append_compressed_token_id(int t) { input.compressed_tokens.token_ids.push_back(t); }
    int get_last_token_id() const { return input.tokens.token_ids.back(); }
    int get_output_len() const { return input.tokens.get_len()-input.tokens.prompt_len; }
    int get_prompt_len() const { return input.tokens.prompt_len; }
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
_FORMAT_
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
void check_commit() {
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

POOL = r'''
// Not the upstream pool: ordinary heap ownership keeps objects alive for assertions.
struct HeapPool {
    std::vector<std::unique_ptr<RequestOutput>> owned;
    template<class... Args> uint64_t allocate(Args&&... args) {
        owned.push_back(std::make_unique<RequestOutput>(std::forward<Args>(args)...));
        return reinterpret_cast<uint64_t>(owned.back().get());
    }
};
'''
FORMAT = r'''
    std::vector<RequestOutput> format(SchedulerOutputs& scheduler_outputs, const std::vector<int>& accepted,
                                      const std::vector<int>& main, const std::vector<int>& ftp) {
        HeapPool pool; auto request_output_pool=&pool;
        std::vector<uint64_t> request_outputs;
        int num_real_seqs=main.size(), total_q=accepted.size(), top_logprobs_stride=1, max_num_batched_tokens=32;
        bool has_accepted_tokens=!accepted.empty();
        auto ftp_ptr=ftp.data(); auto accepted_token_ids_base_ptr=accepted.data(); auto token_ids_ptr=main.data();
        std::vector<float> accepted_lp(accepted.size(),-0.1f), main_lp(main.size(),-0.2f);
        float *accepted_token_logprobs_base_ptr=accepted_lp.data(), *token_logprobs_ptr=main_lp.data();
        float *accepted_token_last_logprobs_base_ptr=nullptr, *accepted_sum_smaller_probs_base_ptr=nullptr;
        float *accepted_top_logprobs_base_ptr=nullptr, *top_logprobs_base_ptr=nullptr, *sum_smaller_probs_base_ptr=nullptr;
        int *accepted_top_token_ids_base_ptr=nullptr, *top_token_ids_base_ptr=nullptr;
        std::optional<int> accepted_token_last_logprobs_, accepted_sum_smaller_probs_, accepted_top_logprobs_, top_logprobs_;
        std::optional<std::vector<std::optional<PromptLogprobs>>> prompt_logprobs;
_OUTPUT_LOOP_
        std::vector<RequestOutput> result;
        for (auto ptr: request_outputs) result.push_back(*reinterpret_cast<RequestOutput*>(ptr));
        return result;
    }
'''
OUTPUT_CONTROLS = r'''
void check_output() {
    int count=0;
    for (int mode=0; mode<8; ++mode) {
        Probe p; auto s=make_seq(); SchedulerOutputs o{false,{s},{4}};
        std::vector<int> accepted={21,22,-1,-1}, main={99}, ftp={0};
        bool report=false;
        if (mode==1) { accepted={11,22,23,-1}; } // early EOS control
        if (mode==2) { s->generation.stopping.max_tokens=2; accepted={21,22,23,-1}; }
        if (mode==3 || mode==4 || mode==5) s->structured_output.status=-1;
        if (mode==4) accepted={-1,-1,-1,-1};
        if (mode==5) { accepted.clear(); o.seqlens_q={1}; s->mtp.draft_tokens.clear(); }
        if (mode==6) report=true; // new failure report, ledger was healthy
        if (mode==7) s->schedule.status=SequenceStatus::FINISHED_SO_FAILED;
        p.process(o,accepted,main,report,report?std::vector<Status>{{-1}}:std::vector<Status>{});
        auto outputs=p.format(o,accepted,main,ftp);
        if (mode==0) {
            assert(outputs.size()==3 && outputs[0].token_id==21 && outputs[1].token_id==22 && outputs[2].token_id==99);
        } else if (mode==1) {
            assert(outputs.size()==1 && outputs[0].token_id==11 && outputs[0].finished);
            assert(outputs[0].accept_tok==4 && p.pushed_tokens_before_finish[0]==1);
        } else if (mode==2) {
            assert(outputs.size()==2 && outputs[1].token_id==22 && outputs[1].finished);
        } else if (mode==7) {
            assert(outputs.empty() && p.pushed_tokens_before_finish[0]==0);
        } else {
            int expected=(mode==4 || mode==5)?99:21;
            assert(s->get_last_token_id()==0 && outputs.size()==1 && outputs[0].token_id==expected);
            assert(outputs[0].finished && outputs[0].finish_reason=="xgrammar runtime errors or timeout");
            std::cout << "conditional SO identity: ledger=0 output=" << expected << " mode=" << mode << '\n';
        }
        ++count;
    }
    std::cout << "PASS " << count << " original formatter/commit controls; injected sampling payloads, not engine reachability.\n";
}
'''


def check_so_mask(root):
    """Original mask method only; CUDA, matcher and tensor objects are doubles."""
    source = read(root, 'deepinfer/model_executor/layers/sampler.py')
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'Sampler')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_get_so_mask')
    method.decorator_list = []

    class Tensor:
        def __init__(self, data=None, shape=None): self.data, self.shape = data, shape
        def cuda(self, **kwargs): return self
        def record_stream(self, stream): pass

    class Event:
        def record(self, *args): pass

    stream = NS(wait_event=lambda *args: None)
    namespace = {'torch': NS(int32='int32', tensor=lambda data, **kw: Tensor(data=list(data)),
                            cuda=NS(current_stream=lambda: stream, Event=Event, Stream=lambda: stream,
                                    stream=lambda s: nullcontext())),
                 'allocate_vocab_mask': lambda n, v, **kw: Tensor(shape=(n,v)),
                 'logger': NS(warning=lambda *a: None, exception=lambda *a: None)}
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[])), '<original SO mask>', 'exec'), namespace)
    healthy = NS(is_aborted=lambda: False, accept_token=lambda t: True, constrained=lambda: True,
                 fill_vocab_mask=lambda *a: True, rollback=lambda n: None)
    aborted = NS(is_aborted=lambda: True)
    cases = [([-1], [None], None, None), ([2], [None], [-1], [[],[],[]]),
             ([2], [aborted], [-1], [[],[],[]]), ([-1,2], [None,healthy], [-1,2], [[1],[1],[1]])]
    for statuses, matchers, expected, indices in cases:
        metadata = [NS(so_status=s, so_matcher=m, last_output_token=7, last_accepted_token=[],
                       last_next_n_token=[8,9], request_id=str(i)) for i,(s,m) in enumerate(zip(statuses,matchers))]
        result = namespace['_get_so_mask'](NS(bitmasks_cpu=[], vocab_size=32, max_MTP_value=2, so_copy_stream=None), metadata)
        actual_indices = None if result[0] is None else [t.data for t in result[0]]
        assert result[2] == expected and actual_indices == indices, (statuses, result[2], actual_indices)
    print('PASS 4 original SO-mask controls; no actual masks, xgrammar, CUDA or sampling payload generation.')


def check_offline_output(root):
    source = read(root, 'deepinfer/entrypoints/llm.py')
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'LLM')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_run_engine')
    assert not method.decorator_list
    namespace = {'defaultdict': defaultdict, 'RequestOutput': NS(clone=lambda x: NS(**vars(x)))}
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[])), '<original offline collection>', 'exec'), namespace)
    for decode in (False, True):
        state, decoded = NS(done=False), []
        row = NS(request_id='0', token_id=21, finished=True, finish_reason='xgrammar runtime errors or timeout')

        def step():
            state.done = True
            return [row]  # injected formatter-shaped object, not a native pybind pointer

        def tokenizer(ids):
            decoded.append(ids)
            return 'tokenizer double'

        engine = NS(has_unfinished_requests=lambda: not state.done, step=step,
                    flush_inflight_reqs=lambda: [], tokenizer=NS(decode=tokenizer))
        results = namespace['_run_engine'](NS(llm_engine=engine, decode_sequence=decode), False)
        if decode:
            assert decoded == [[21]] and results == ['tokenizer double']
        else:
            assert results[0][0].token_id == 21 and results[0][0].finish_reason == row.finish_reason
    print('PASS 2 original offline collection controls; engine/clone/tokenizer doubles, no actual API service.')


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
    parser.add_argument('--check-output-identity', action='store_true')
    args = parser.parse_args()
    seq = read(args.source_root, 'csrc/engine/sequence.hpp')
    scheduler = read(args.source_root, 'csrc/engine/scheduler.hpp')
    types = between(seq, 'enum class SequenceStatus', 'inline std::string get_finished_reason')
    types += between(seq, 'struct KvCacheState', 'struct StructuredOutputState')
    types += between(seq, 'struct SequenceScheduleState', '/// C++ 内部的命名构造参数')
    stop = between(scheduler, '    void _check_stop(', '    void set_schedulable(')
    loop = between(scheduler, '        for (ssize_t i = 0; i < num_output_tokens;', '        // Virtual sequences')
    output_types = formatter = output_controls = ''
    if args.check_output_identity:
        output = read(args.source_root, 'csrc/engine/outputs.hpp')
        types += between(seq, 'inline std::string get_finished_reason', 'struct SequenceData')
        output_types = between(output, 'typedef std::vector<float> PromptLogprobs;', 'class RequestOutputPool') + POOL
        output_loop = between(output, '    for (int i = 0; i < num_real_seqs;', '    for (auto &seq : scheduler_outputs.ignored_seqs)')
        formatter = FORMAT.replace('_OUTPUT_LOOP_', output_loop)
        output_controls = OUTPUT_CONTROLS
    program = (PRELUDE + types + output_types + DOUBLES.replace('_CHECK_STOP_', stop).replace('_LOOP_', loop)
               .replace('_FORMAT_', formatter) + CONTROLS + output_controls
               + ('int main() { check_commit(); check_output(); }' if args.check_output_identity else 'int main() { check_commit(); }'))
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
    if args.check_output_identity:
        check_so_mask(args.source_root)
        check_offline_output(args.source_root)
    print('No full extension, real sampler, pool/pybind/API/engine, GPU/collective or physical KV verification.')


if __name__ == '__main__':
    main()
