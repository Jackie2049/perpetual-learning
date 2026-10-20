#!/usr/bin/env python3
"""Compile immutable original C++ chunk-sizing slices for DeepInfer PR #108.

No torch, extension, real Sequence/Scheduler, allocation, Gloo, model or device.
Sequence storage/admission context are a small harness; original sizing and image
boundary methods are retained. TORCH_CHECK becomes a local exception (messages
not evaluated). Source slices end BEFORE status checks/KV allocation/execution.
"""
import argparse
import ast
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

BASE = '774ce244b87cc0eb8d4b94ab9f5de2ab22049f01'
HEAD = 'ab47ee4c62ab704431db1b06a66f01c57619faef'
RELATED = ('2a5db94adbc70f62ea205306efa677fe3331c954', '6a934611d538bcce54434c73a886459bca36b021')
SOURCE_PATH = 'csrc/engine/scheduler.hpp'
PRELUDE = r'''
#include <algorithm>
#include <deque>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/types.h>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error(#condition); } while (0)
struct Seq {
    struct { int num_tokens = 0; } cache;
    struct { std::deque<int> chunk_checkpoints; } schedule;
    std::vector<int> token_ids;
    int get_len() const { return token_ids.size(); }
    const std::vector<int>& get_token_ids() const { return token_ids; }
};
'''
CLASS = r'''
struct Probe_LABEL_ {
    int lookahead_tokens, cp_align, max_num_batched_tokens;
    bool chunked_prefill;
    const int MM_TOKEN_IMAGE = 1002, MM_TOKEN_IMAGE_PAD = 1001;
_HELPERS_
    std::pair<int,int> attempt(Seq* seq, int used, bool previous) {
        int num_prompt_tokens = seq->get_len() - seq->cache.num_tokens;
        int num_batched_tokens = used;
        std::vector<int> scheduled(previous ? 1 : 0);
        while (true) {
_SIZING_
            return {1, seqlen_q};
        }
        return {0, 0};
    }
};
'''
MAIN = r'''
template <class Probe> int run() {
    Probe probe;
    Seq seq;
    std::string command;
    while (std::cin >> command) {
        if (command == "new") {
            int length, cached, count, image_begin, image_end;
            std::cin >> length >> cached >> count;
            seq.token_ids.assign(length, 7);
            seq.cache.num_tokens = cached;
            seq.schedule.chunk_checkpoints.clear();
            for (int i=0, cp; i<count; ++i) { std::cin >> cp; seq.schedule.chunk_checkpoints.push_back(cp); }
            std::cin >> image_begin >> image_end;
            if (image_begin >= 0) {
                seq.token_ids.at(image_begin) = 1002;
                for (int i=image_begin+1; i<image_end; ++i) seq.token_ids.at(i)=1001;
            }
        } else if (command == "try") {
            int used; bool previous;
            std::cin >> probe.max_num_batched_tokens >> probe.lookahead_tokens >> probe.cp_align
                     >> probe.chunked_prefill >> used >> previous;
            std::pair<int,int> result;
            try { result=probe.attempt(&seq,used,previous); }
            catch (const std::runtime_error&) { result={2,0}; }
            // Harness-only progress: real cache/output/queue transitions are not run.
            if (result.first==1) seq.cache.num_tokens += result.second;
            std::cout << result.first << ' ' << result.second << ' ' << seq.cache.num_tokens
                      << ' ' << seq.schedule.chunk_checkpoints.size();
            for (int cp: seq.schedule.chunk_checkpoints) std::cout << ' ' << cp;
            std::cout << '\n';
        } else return 3;
    }
    return 0;
}
int main(int argc,char** argv) {
    if (argc!=2) return 2;
    return std::string(argv[1])=="base" ? run<Probe_base>() : run<Probe_head>();
}
'''


def source(root, revision):
    data = subprocess.check_output(['git', 'show', f'{revision}:{SOURCE_PATH}'], cwd=root)
    print(revision, 'sha256=' + hashlib.sha256(data).hexdigest())
    return data.decode()


def slices(text):
    helpers = text[text.index('    bool _is_inside_image_pad_region('):text.index('    /// 构造 SchedulerOutputs')]
    sizing = text[text.index('            int seq_workload ='):text.index('            TORCH_CHECK(seq->schedule.status == SequenceStatus::WAITING')]
    assert '_adjust_chunk_size' in helpers and 'num_batched_tokens +=' in sizing
    return helpers, sizing


def check_caller(root, binary, session):
    """Run the original Python wait helper against precomputed sizing traces.

    Engine.step and Sequence remain fakes, not actual engine execution. The C++
    harness advances cache immediately; real output/cache commits are not tested.
    """
    data = subprocess.check_output(['git', 'show', f'{BASE}:examples/benchmark.py'], cwd=root)
    assert data == subprocess.check_output(['git', 'show', f'{HEAD}:examples/benchmark.py'], cwd=root)
    print('benchmark.py sha256=' + hashlib.sha256(data).hexdigest())
    tree = ast.parse(data.decode())
    helper = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == 'prefill_cached_prefix')
    namespace = {'Sequence': object}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), '<original benchmark helper>', 'exec'), namespace)
    controls = [
        # cached is an absolute coordinate, so the final q is 2, not 10.
        ('absolute_target', 'new 20 4 1 10 -1 -1', ['4 1 1 1 0 0'] * 2, 4, 10, False, 2),
        ('safe_image_boundary', 'new 20 0 1 8 3 8', ['32 1 1 1 0 0'], 0, 8, False, 1),
        ('image_interior_overshoot', 'new 20 0 1 5 3 8', ['32 1 1 1 0 0'], 0, 5, True, 1),
        ('already_at_target', None, [], 5, 5, False, 0),
    ]
    for variant in ('base', 'head'):
        for name, setup, attempts, cached, target, expect_assert, count in controls:
            trace = [] if setup is None else session(binary, variant, setup, attempts)
            assert all(row[0] == 1 for row in trace), (name, trace)
            seq = SimpleNamespace(num_cached_tokens=cached, chunk_checkpoints=[])
            calls = []

            def step():
                row = trace[len(calls)]  # bounded fixture, not an upstream timeout
                calls.append(row)
                seq.num_cached_tokens = row[2]
                seq.chunk_checkpoints = row[4:]

            namespace['llm'] = SimpleNamespace(llm_engine=SimpleNamespace(step=step))
            raised = False
            try:
                namespace['prefill_cached_prefix'](seq, target)
            except AssertionError:
                raised = True
            assert raised == expect_assert and len(calls) == count, (variant, name, raised, calls)
            if name == 'already_at_target':
                assert seq.chunk_checkpoints == [target]  # helper did not call step
            elif not raised:
                assert seq.num_cached_tokens == target and not seq.chunk_checkpoints
            print('caller', variant, name, 'PASS calls=', len(calls), 'assertion=', raised)
    print('PASS 8 original Python caller/controlled C++ trace checks; no actual engine or fork execution.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_root', type=Path)
    parser.add_argument('--cxx', default='clang++')
    parser.add_argument('--check-caller', action='store_true', help='also execute the original Python prefix wait helper on sizing traces')
    args = parser.parse_args()
    original = source(args.source_root, BASE)
    updated = source(args.source_root, HEAD)
    for revision in RELATED:
        assert slices(source(args.source_root, revision)) == slices(original), 'related sizing changed; inspect'
    program = PRELUDE
    for label, text in (('base', original), ('head', updated)):
        helpers, sizing = slices(text)
        program += CLASS.replace('_LABEL_', '_' + label).replace('_HELPERS_', helpers).replace('_SIZING_', sizing)
    program += MAIN
    cxx = shutil.which(args.cxx)
    if cxx is None: raise RuntimeError('C++ compiler not found')
    flags = ['-std=c++20', '-O0']
    if sys.platform == 'darwin':
        sdk = subprocess.check_output(['xcrun', '--show-sdk-path'], text=True).strip()
        flags += ['-isysroot', sdk, '-isystem', str(Path(sdk) / 'usr/include/c++/v1')]

    def session(binary, variant, setup, attempts):
        data = setup + '\n' + '\n'.join('try ' + row for row in attempts) + '\n'
        output = subprocess.check_output([str(binary), variant], input=data, text=True)
        return [list(map(int, line.split())) for line in output.splitlines()]

    # Result rows: admitted/blocked/exception status (1/0/2), q, harness cached, checkpoint count, checkpoints.
    cases = [
        ('checkpoint_flag_off', 'new 20 0 1 5 -1 -1', ['32 0 1 0 0 0'] * 2,
         [[0,0,0,0],[1,20,20,0]], [[1,5,5,0],[1,15,20,0]]),
        ('all_checkpoints', 'new 12 0 2 4 8 -1 -1', ['32 0 1 1 0 0'] * 3,
         [[1,4,4,1,8],[1,4,8,0],[1,4,12,0]], None),
        ('no_budget_preserves_future', 'new 12 0 1 6 -1 -1', ['16 0 1 1 16 1'],
         [[0,0,0,1,6]], None),
        ('lookahead_remainder', 'new 10 0 0 -1 -1', ['8 4 1 1 0 0'] * 2,
         [[1,8,8,0],[2,0,8,0]], [[1,6,6,0],[1,4,10,0]]),
        ('infeasible_tail_checkpoint', 'new 10 0 1 8 -1 -1', ['32 4 1 1 0 0'],
         [[1,8,8,0]], [[1,10,10,0]]),
        ('too_short_checkpoint', 'new 20 0 1 2 -1 -1', ['32 4 1 1 0 0'],
         [[0,0,0,0]], [[1,20,20,0]]),
        ('image_bailout_preserves', 'new 12 0 1 4 0 8', ['6 0 1 1 2 1'],
         [[0,0,0,0]], [[0,0,0,1,4]]),
        ('image_grows_past_checkpoint', 'new 20 0 1 5 3 8', ['32 0 1 1 0 0'],
         [[1,8,8,0]], None),
        ('image_shrinks_before_checkpoint', 'new 20 0 1 5 3 8', ['6 0 1 1 0 0'],
         [[1,3,3,0]], None),
        ('ordinary_chunks', 'new 12 0 0 -1 -1', ['4 0 1 1 0 0'] * 3,
         [[1,4,4,0],[1,4,8,0],[1,4,12,0]], None),
    ]
    with tempfile.TemporaryDirectory(prefix='deepinfer-chunk-slice-') as folder:
        cpp, binary = Path(folder) / 'probe.cpp', Path(folder) / 'probe'
        cpp.write_text(program)  # generated build product, not an upstream source edit
        subprocess.run([cxx, *flags, str(cpp), '-o', str(binary)], check=True)
        for name, setup, attempts, expected_base, expected_head in cases:
            expected_head = expected_base if expected_head is None else expected_head
            old, new = session(binary, 'base', setup, attempts), session(binary, 'head', setup, attempts)
            assert old == expected_base, (name, 'base', old, expected_base)
            assert new == expected_head, (name, 'head', new, expected_head)
            print(name, 'PASS base=', old, 'head=', new)
        if args.check_caller:
            check_caller(args.source_root, binary, session)
    print('PASS 10 paired native C++ sizing-slice controls; #95/#98 slices equal base.')
    print('No full Scheduler/extension/upstream pytest, allocation, process_outputs, DP voting, Gloo or device proof.')


if __name__ == '__main__':
    main()
