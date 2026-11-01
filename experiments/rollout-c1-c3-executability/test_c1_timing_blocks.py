#!/usr/bin/env python3
"""c1_timing_blocks 的单元测试。

要点不在"脚本能跑"，而在三件事被守住：
  1. 每条判定引用的源码证据必须在固定 commit 上真实命中（不是凭记忆写的行号）；
  2. 等级函数对"缺件"是真敏感的——注入一个缺证据的块，等级必须掉下来；
  3. C1 与 C3 的接缝成立：没有任何计时块能分开给出 t1 与 t2，
     所以 §21.5 的"默认整体报"在语料四个框架里同样适用。

    python3 -m unittest test_c1_timing_blocks -v
"""

import unittest

import c1_timing_blocks as m


class TestSourceEvidence(unittest.TestCase):
    def test_all_anchors_hit(self):
        """所有 anchor 与 proof 都必须在只读 checkout 里命中。"""
        bad = []
        for blk in m.BLOCKS:
            blk = dict(blk)
            blk['framework_key'] = {'verl': 'verl', 'openrlhf': 'openrlhf',
                                    'nemo': 'nemo_rl', 'skyrl': 'skyrl'}[blk['id'].split('.')[0]]
            r = m.check_anchor(blk)
            if not r['ok']:
                bad.append((blk['id'], r['reason']))
        self.assertEqual(bad, [], 'anchors/proofs that no longer hit: %s' % bad)

    def test_checkouts_are_at_recorded_commits(self):
        """固定 commit 是 C1 第 ① 项的前提；checkout 漂了这条结论就不成立。"""
        import os
        import subprocess
        for key, (root, commit) in m.CHECKOUTS.items():
            if not os.path.isdir(os.path.join(root, '.git')):
                self.skipTest('%s: no .git (只读快照缺失，本机不可核)' % key)
            head = subprocess.check_output(
                ['git', '-C', root, 'rev-parse', 'HEAD'], text=True).strip()
            self.assertEqual(head, commit, key)

    def test_missing_proof_drops_grade(self):
        """负向对照：把某块的一条 proof 删掉，等级必须掉出 A。"""
        blk = dict(m.BLOCKS[0])
        blk['framework_key'] = 'verl'
        self.assertNotIn('c1_missing', blk)
        base = m.c1_grade(blk)
        broken = dict(blk)
        broken['proof'] = {'waiting': blk['proof']['waiting']}  # 缺 overlap
        if base == 'A':
            self.assertEqual(m.item_status(broken, 'overlap'), m.PARTIAL)
            self.assertEqual(m.c1_grade(broken), 'B')


class TestBlockIdentity(unittest.TestCase):
    def test_row_only_timer_is_partial(self):
        """只有裸行间计时、没有命名键的块，第 ① 项不能判为 is_decidable。"""
        blk = [b for b in m.BLOCKS if b['id'] == 'openrlhf.timing_generation'][0]
        self.assertIsNone(blk.get('timer_key'))
        self.assertEqual(m.item_block_identity(blk), m.PARTIAL)
        self.assertNotEqual(m.c1_grade(blk), 'A')

    def test_named_timer_is_decidable(self):
        blk = [b for b in m.BLOCKS if b['id'] == 'skyrl.sync_weights'][0]
        self.assertEqual(m.item_block_identity(blk), m.DECIDABLE)


class TestC1C3Bridge(unittest.TestCase):
    def test_no_block_separates_t1_from_t2(self):
        """关键接缝：没有任何计时块能独立给出 t1；能给的都只能到 t1∪t2。"""
        exact = [b['id'] for b in m.BLOCKS
                 if b.get('stamps') == ['t1']]
        self.assertEqual(exact, [])
        pooled = [b['id'] for b in m.BLOCKS if b.get('stamps') == ['t1_or_t2']]
        self.assertGreater(len(pooled), 0, '若真能分开给 t1，§21.5 的结论需要重写')

    def test_grade_does_not_imply_c2_splittability(self):
        """A 级块不等于能拆 B2/B3：两件事的口径不同，不能混用。"""
        a_grades = [b['id'] for b in m.BLOCKS if m.c1_grade(b) == 'A']
        self.assertTrue(a_grades)
        for b in m.BLOCKS:
            if m.c1_grade(b) == 'A':
                self.assertNotEqual(b.get('stamps'), ['t1'])


class TestC3Definitions(unittest.TestCase):
    def test_no_framework_reports_c3_literal_eta(self):
        """C3 的 η 定义（消费版本−生成版本）在四个快照里都只能近似或不可得。"""
        self.assertEqual(m.build_summary()['c3']['literal_count'], 0)

    def test_staleness_proofs_hit(self):
        """C3 判定的源码证据同样必须命中，否则"可近似"这一级也不成立。"""
        proof = m.build_summary()['c3']['staleness_proofs']['skyrl']
        self.assertTrue(proof['all_ok'], proof)

    def test_s_gap_when_trajectories_are_dropped(self):
        """生成侧口径会高估 S：丢弃的组仍被算进"生成量"。"""
        s = m.s_accepted_trajectories({'steps': 10, 'n_gen': 32, 'n_drop': 8})
        self.assertAlmostEqual(s['relative_gap'], 8.0 / 24.0, places=9)
        s0 = m.s_accepted_trajectories({'steps': 10, 'n_gen': 32, 'n_drop': 0})
        self.assertAlmostEqual(s0['relative_gap'], 0.0, places=12)


if __name__ == '__main__':
    unittest.main()
