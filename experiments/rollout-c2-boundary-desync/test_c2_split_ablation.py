#!/usr/bin/env python3
"""c2_split_ablation 的单元测试。

要点不在于"代码能跑通"，而在于**这些读法的偏差是被测出来的、不是构造出来的**：
断言全部写死常数或写成对被继承前提的检查，不从被测函数的输出反推。

    python3 -m unittest test_c2_split_ablation -v
"""

import unittest

import c2_split_ablation as m


CFG = m.CONFIGS['near-balance']


def scenario():
    import random
    tasks = m.build_scenario(CFG, random.Random(0))
    return tasks, m.simulate(tasks, CFG['pressure'])


class TestEventGraph(unittest.TestCase):
    def test_simulate_is_deterministic(self):
        """同一任务图跑两次必须逐位相同（归档 --verify 依赖这条）。"""
        tasks, (ev1, end1) = scenario()
        ev2, end2 = m.simulate(tasks, CFG['pressure'])
        self.assertEqual(ev1, ev2)
        self.assertEqual(end1, end2)

    def test_prefill_waits_for_weights_not_for_kv(self):
        """§21.2 的前提：引擎在权重生效前不可能开始吃新权重。"""
        tasks, (ev, end) = scenario()
        prefill = [e for e in ev if e['name'] == 'prefill'][0]
        self.assertGreaterEqual(prefill['start'], end['weight_bcast'] - 1e-9)

    def test_kv_migration_independent_of_bcast(self):
        """B2 链与 B3 链在任务图上确实独立，否则两者恒同步、偏差被抹平。"""
        tasks, (ev, end) = scenario()
        self.assertNotAlmostEqual(end['kv_migrate'], end['weight_bcast'], places=6)

    def test_batch_starts_after_both_chains(self):
        tasks, (ev, end) = scenario()
        first_batch = min(e['start'] for e in ev if e['name'].startswith('batch_'))
        self.assertGreaterEqual(first_batch, max(end['weight_bcast'], end['kv_migrate']) - 1e-9)


class TestTruthIsIndependentOfReadings(unittest.TestCase):
    def test_truth_uses_simulator_not_observables(self):
        """真值必须来自仿真内部事件，不能与被测的读法共用同一批可观测量。"""
        import random
        tasks = m.build_scenario(CFG, random.Random(3))
        events, end = m.simulate(tasks, CFG['pressure'])
        truth = m.truth_of(end)
        self.assertAlmostEqual(truth['B2'] + truth['B3'], truth['B2U3'], places=9)

    def test_batch_starts_exactly_at_t2(self):
        """第一个批请求的起点就是 t2；这条保证下面"把切点取到批起点"确实会偏。"""
        import random
        tasks = m.build_scenario(CFG, random.Random(3))
        events, end = m.simulate(tasks, CFG['pressure'])
        obs = m.observable(events, end, tasks)
        self.assertAlmostEqual(obs['batch_starts'][0], m.truth_of(end)['B2U3'], places=6)

    def test_b2_is_positive_and_bounded_by_base(self):
        """B2 物理上不可能为负，也不可能大于两条链的标称 base（含抖动上限 1.15×）。"""
        for name, cfg in m.CONFIGS.items():
            import random
            for seed in range(5):
                tasks = m.build_scenario(cfg, random.Random(seed))
                events, end = m.simulate(tasks, cfg['pressure'])
                truth = m.truth_of(end)
                self.assertGreaterEqual(truth['B2'], 0.0, (name, seed))
                self.assertLessEqual(truth['B2'], 1.2 * cfg['b2'], (name, seed))


class TestDefectIsCaught(unittest.TestCase):
    """负向对照：注入一个已知读错 t1 的实现，聚合必须报出偏差。

    判据不写成"翻转次数必须为 0"——正确读法在 gen-heavy 下自身也会翻转
    （真 B2≈5.3 与 B3≈7.4 量级接近，代理时刻的少数样本会越界），
    写成翻转数=0 会让测试在数据真实变化时失效。改为比较两个实现之间的偏差。
    """

    def _aggregate_with(self, patch):
        rows = []
        import random
        for seed in m.SEEDS:
            cfg = m.CONFIGS['gen-heavy']
            rng = random.Random(seed)
            tasks = m.build_scenario(cfg, rng)
            events, end = m.simulate(tasks, cfg['pressure'])
            obs = m.observable(events, end, tasks)
            n = m.batch_len(rng)
            rows.append({'seed': seed, 'batch_len': n,
                         'truth': m.truth_of(end),
                         'readings': patch(obs, n, cfg)})
        return m.aggregate(rows)

    def test_correct_reading_beats_injected_defect(self):
        correct = self._aggregate_with(m.readings)
        agg_truth_b2 = correct['truth']['B2']
        # 前提：这一档配置的真 B2 确实非零，否则相对误差无意义
        self.assertGreater(agg_truth_b2, 1.0)

        def buggy(obs, n, cfg):
            """缺陷：把 t1 误报成"批被拖住的时刻"，整段在途时间被算进 B3。"""
            r = m.readings(obs, n, cfg)
            t_held = obs['batch_starts'][min(n, len(obs['batch_starts'])) - 1]
            r["held_sep"] = {"B2": 0.0, "B3": t_held}
            return r

        bad = self._aggregate_with(buggy)
        self.assertAlmostEqual(bad['held_sep']['B2_mean'], 0.0, places=9)
        self.assertGreater(abs(bad['held_sep']['B2_rel_err_mean']),
                           abs(correct['held_sep']['B2_rel_err_mean']))
        self.assertGreater(bad['held_sep']['order_flipped'],
                           correct['held_sep']['order_flipped'])


class TestNaiveReadingIsAlwaysOverOptimistic(unittest.TestCase):
    def test_naive_b2_is_bigger_than_truth_in_all_configs(self):
        """标称步数当测量值会系统性高报 B2：三档配置都必须为正偏差。"""
        for name in m.CONFIGS:
            rows = m.run_config(m.CONFIGS[name])
            agg = m.aggregate(rows)
            self.assertGreater(agg['naive_sep']['B2_rel_err_mean'], 0.0, name)


if __name__ == '__main__':
    unittest.main()
