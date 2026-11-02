#!/usr/bin/env python3
"""`check_producers.py` 的单元测试：证据能否被识别，以及结论会不会被旧结果污染。

三条必须守住的线：
  1. 负向对照：写入方的三种写法都命中，只读/无关写法都不命中（见 negative_control.py，
     这里再断言一次，免得改脚本时只改了其中一处）。
  2. 正向可行性：把一棵干净的源码树追加一个写入方，判决必须从「无写入方」翻成「有」。
     缺少这条时，「计数为 0」既可能是真没有、也可能是脚本根本没在扫。
  3. 无缓存：同一进程里对同一个 root 连续两次 build()，第二次数到的必须是磁盘上的当前状态。
     本脚本第一版正是栽在这里——`scan` 挂了 lru_cache，注入写入方后仍读到旧的 0。
"""
import os
import shutil
import tempfile
import unittest

import check_producers as cp
import negative_control as nc

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_VERL = os.path.abspath(os.path.join(HERE, '..', '..', 'verl'))


class TestRegexCoverage(unittest.TestCase):
    """与 negative_control.py 共用同一份正则语义。"""

    def test_writer_forms_all_match(self):
        for name, sample in nc.SHOULD_MATCH:
            with self.subTest(name=name):
                self.assertTrue(cp.producer_re.search(sample), f'漏抓写入写法：{name}')

    def test_reader_forms_none_match(self):
        for name, sample in nc.SHOULD_NOT_MATCH:
            with self.subTest(name=name):
                self.assertFalse(cp.producer_re.search(sample), f'误判为写入方：{name}')


class TestScanOnSyntheticTree(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='fgprobe-')
        os.makedirs(os.path.join(self.root, 'verl', 'workers', 'engine'))
        os.makedirs(os.path.join(self.root, 'tests', 'utils'))
        with open(os.path.join(self.root, 'verl', 'workers', 'engine', 'utils.py'), 'w') as fh:
            fh.write(
                'def prepare_micro_batches(data):\n'
                '    force_group_size = tu.get_non_tensor_data(\n'
                '        data=data, key="force_group_size", default=1)\n'
            )
        with open(os.path.join(self.root, 'tests', 'utils', 'test_x.py'), 'w') as fh:
            fh.write('tu.assign_non_tensor_data(batch, "force_group_size", n)\n')

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_reader_counted_as_reader_not_producer(self):
        producers, readers, test_producers, missing = cp.scan(self.root)
        self.assertFalse(missing)
        self.assertEqual(producers, [])
        self.assertEqual([r['path'] for r in readers], ['verl/workers/engine/utils.py'])
        self.assertEqual([t['path'] for t in test_producers], ['tests/utils/test_x.py'])

    def test_injected_writer_flips_verdict(self):
        before = cp.build(self.root)
        self.assertTrue(before['verdict']['no_producer_in_production_code'])
        with open(os.path.join(self.root, 'verl', 'workers', 'engine', 'utils.py'), 'a') as fh:
            fh.write('\n\ndef g(td):\n    tu.assign_non_tensor(td, force_group_size=4)\n')
        after = cp.build(self.root)
        # 这一条同时守住「无缓存」：若 scan 有缓存，after 会等于 before 而失败。
        self.assertFalse(after['verdict']['no_producer_in_production_code'])
        self.assertEqual(after['producer_count'], 1)
        self.assertEqual(after['producer_files'], ['verl/workers/engine/utils.py'])

    def test_missing_root_is_reported_not_silently_empty(self):
        producers, readers, test_producers, missing = cp.scan(os.path.join(self.root, 'nope'))
        self.assertTrue(missing)
        self.assertEqual(producers, [])
        # missing 为真时判决必须是 False，不能把「没扫到」当成「没有」。
        data = cp.build(os.path.join(self.root, 'nope'))
        self.assertFalse(data['verdict']['no_producer_in_production_code'])
        self.assertFalse(data['verdict']['always_default_1_is_structural'])


@unittest.skipUnless(os.path.isdir(REAL_VERL), 'verl checkout 不在默认位置，跳过真实树断言')
class TestRealTree(unittest.TestCase):
    def test_real_checkout_has_no_producer(self):
        data = cp.build(REAL_VERL)
        self.assertFalse(data['verl_missing'])
        self.assertEqual(data['producer_count'], 0)
        self.assertTrue(data['verdict']['no_producer_in_production_code'])
        self.assertTrue(data['verdict']['anchors_all_hold'])
        self.assertTrue(data['verdict']['default_keys_never_inject_it'])
        self.assertTrue(data['verdict']['tq_writeback_cannot_carry_it'])
        self.assertTrue(data['verdict']['always_default_1_is_structural'])

    def test_engine_config_has_no_such_field(self):
        # 用户无法从 yaml 配到它 —— 这是「无写入方」的第三根支柱，单独钉死。
        path = os.path.join(REAL_VERL, 'verl', 'workers', 'config', 'engine.py')
        with open(path, 'r', encoding='utf-8') as fh:
            self.assertNotIn('force_group_size', fh.read())


if __name__ == '__main__':
    unittest.main(verbosity=2)
