#!/usr/bin/env python3
"""负向对照：`check_producers.py` 的写入方正则能不能真的抓到写入方。

「计数为 0」有两种可能：① 确实没有写入方；② 正则写错、什么也匹配不上。
本脚本把四种真实写法喂给同一个正则，必须全部命中；再把只读/无关写法喂进去，必须全部不命中。
任一期望不满足即退出码 1——这条对照是「没有生产者」这个结论的前提。
"""
import os
import re
import sys

# 与 check_producers.py 保持同一份正则；改那边必须同步改这里。
# 三种写入写法都要命中：位置参数、key=/val= 关键字、直接把键名当 kwargs。
PRODUCER_RE = re.compile(
    r'assign_non_tensor(?:_data)?\s*\((?:[^)]*["\']force_group_size["\']|'
    r'[^)]*\bforce_group_size\s*=)'
)

SHOULD_MATCH = [
    ('test_actual_writer', 'tu.assign_non_tensor_data(batch, "force_group_size", force_group_size)'),
    ('positional_int', 'tu.assign_non_tensor(data, "force_group_size", 4)'),
    ('single_quoted', "tu.assign_non_tensor_data(td, 'force_group_size', n)"),
    ('kwargs_key_form', 'tu.assign_non_tensor(data, force_group_size=4)'),
    ('key_val_form', 'tu.assign_non_tensor(data, key="force_group_size", val=4)'),
]

SHOULD_NOT_MATCH = [
    ('reader_default', 'force_group_size = tu.get_non_tensor_data(data=data, key="force_group_size", default=1)'),
    ('signature_default', '    force_group_size=1,'),
    ('config_read', 'if self.engine_config.force_group_size > 1:'),
    ('comment_mentions', '# force_group_size stays at its default here'),
    ('other_key_with_word', 'tu.assign_non_tensor_data(batch, "micro_batch_group_size", 2)'),
    ('in_docstring', '    """Tests for prepare_micro_batches with force_group_size > 1"""'),
]


def main():
    ok = True
    for name, sample in SHOULD_MATCH:
        hit = bool(PRODUCER_RE.search(sample))
        print(f'{"PASS" if hit else "FAIL"}  should-match   {name:20s} -> {hit}')
        ok &= hit
    for name, sample in SHOULD_NOT_MATCH:
        hit = bool(PRODUCER_RE.search(sample))
        print(f'{"PASS" if not hit else "FAIL"}  should-not      {name:20s} -> {hit}')
        ok &= not hit
    print('\n负向对照：', '全部通过' if ok else '有失败项——「没有写入方」的结论不可用')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
