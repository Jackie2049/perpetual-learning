#!/usr/bin/env python3
"""Drive the real `ReplayBuffer.sample()` loop on CPU and count its blocking polls.

Why this exists
---------------
TM-008's measurements are reproducible but one number is not explained: the
per-step wait saving is about 0.98 x poll_interval on 2 cards and 0.59 x on 4
cards (DAPO, both at `max_inflight_gen_batches=2`).  The loop can sleep more than
once per step (each `sample()` call loops until enough groups are terminal *and*
nothing is in flight), so "wait per step" = (blocking polls) x (mean residual
phase).  The archives record the total `timing_s/gen` but never the poll count,
so the two factors cannot be separated from them.

`poll-count.patch` adds that missing counter (`training/sampler/blocking_polls`).
This harness proves the counter is right before it is used on a scarce GPU
window: the class under test is the *real* `verl/trainer/ppo/v1/replay_buffer.py`
from a tree, executed unmodified; only the data layer (TransferQueue), the clock
and the post-loop batch materialisation are stubs.  It is not a performance
model -- arrival times are scripted inputs.

Usage
-----
  sim-loop.py --tree /tmp/verl-31cfccb4 --selftest          # unpatched tree
  sim-loop.py --tree /tmp/m5-pollcount-tree --selftest      # patched tree
  sim-loop.py --tree /tmp/m5-pollcount-tree --out raw/sim-polls.json
"""

import argparse
import json
import sys
import ast
from collections import Counter, defaultdict

CLASS_START = "class ReplayBuffer:"
CLASS_ASYNC = "class ReplayBufferAsync(ReplayBuffer):"


# ----------------------------------------------------------------------------
# stubs: virtual clock, fake TransferQueue, trivial KVBatchMeta
# ----------------------------------------------------------------------------
class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = 0
        self.on_advance = None

    def time(self):
        return self.now

    def sleep(self, dt):
        self.sleeps += 1
        self.advance(dt)

    def advance(self, dt):
        self.now += dt
        if self.on_advance is not None:
            self.on_advance(self.now)


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def sum(self):
        return self

    def item(self):
        return self.value


class FakeKVBatchMeta:
    def __init__(self, partition_id, keys, tags):
        self.partition_id = partition_id
        self.keys = keys
        self.tags = tags


class FakeTQ:
    """Minimal TransferQueue: prompt tags plus per-trajectory rm_scores."""

    def __init__(self, clock):
        self.clock = clock
        self.prompts = {}  # uid -> {status, finish_at, global_steps, reward}
        self.traj = {}  # key -> reward value
        self.clears = 0
        self.metric = "reward"  # key the DAPO classifier reads out of reward_extra_info

    def add_prompt(self, uid, finish_at, global_steps, reward, n_traj=4, filtered=None):
        # `filtered` decides DAPO filtering directly: a group is filtered when all
        # trajectories share the same metric, which is what the real classifier reads.
        if filtered is None:
            filtered = True
        self.prompts[uid] = {
            "status": "pending",
            "finish_at": finish_at,
            "global_steps": global_steps,
            "reward": reward,
            "n_traj": n_traj,
            "spread": 0.0 if filtered else 1.0,
        }

    def settle(self, now):
        for uid, p in self.prompts.items():
            if p["status"] == "pending" and p["finish_at"] <= now:
                p["status"] = "finished"
                for session in range(p["n_traj"]):
                    key = f"{uid}_{session}_0"
                    # Trajectories inside one group differ only when not filtered.
                    self.traj[key] = p["reward"] + (0.0 if p["spread"] == 0 else 1e-3 * session)

    def kv_list(self):
        items = {}
        for uid, p in self.prompts.items():
            items[uid] = {"is_prompt": True, "status": p["status"], "global_steps": p["global_steps"]}
        for key, value in self.traj.items():
            items[key] = {"rm": value}
        return {"train": items}

    def kv_batch_get(self, keys, partition_id, select_fields):
        if select_fields == ["extra_fields"]:
            # Trees without the canonical-reward shortcut classify a group by reading every
            # trajectory's `reward_extra_info` (24f25b03). The metric key has to be in there.
            return {
                "extra_fields": [
                    {"reward_extra_info": {self.metric: self.traj[key]}} for key in keys
                ]
            }
        assert select_fields == ["rm_scores"], select_fields
        return {"rm_scores": [FakeTensor(self.traj[key]) for key in keys]}

    def kv_clear(self, partition_id, keys):
        self.clears += 1
        for key in keys:
            if key in self.prompts:
                for traj_key in [k for k in self.traj if k.split("_")[0] == key]:
                    del self.traj[traj_key]
                del self.prompts[key]
            else:
                self.traj.pop(key, None)


class SkipManagerStub:
    @staticmethod
    def annotate_tq(**_kwargs):
        def decorate(fn):
            return fn

        return decorate


def load_replay_buffer(tree):
    """exec the real class source; nothing inside the class is reimplemented.

    Both classes are loaded: the sync one counts polls inside `sample()`, the async one
    inside `wait_for_sampleable()`, and the shipped patch touches both.
    """
    path = f"{tree}/verl/trainer/ppo/v1/replay_buffer.py"
    source = open(path).read()
    start = source.index(CLASS_START)
    source.index(CLASS_ASYNC)  # both classes must be present before the slice is trusted
    module = ast.parse(source)
    helpers = "\n\n".join(
        ast.get_source_segment(source, node)
        for node in module.body
        if isinstance(node, ast.FunctionDef)
    )
    class_source = helpers + "\n\n" + source[start:]

    clock = Clock()
    tq = FakeTQ(clock)
    clock.on_advance = tq.settle
    namespace = {
        "logging": __import__("logging"),
        "os": __import__("os"),
        "time": clock,
        "Counter": Counter,
        "defaultdict": defaultdict,
        "np": __import__("numpy"),
        "DictConfig": dict,
        "tq": tq,
        "KVBatchMeta": FakeKVBatchMeta,
        "SkipManager": SkipManagerStub,
        "logger": __import__("logging").getLogger("sim"),
        "VERL_REPLAY_BUFFER_DEBUG_INTERVAL_SECONDS": 3600,
        "DAPO_FILTERED_REWARD_COUNTS_KEY": "_dapo_filtered_reward_counts",
        "FILTER_GROUPS_REWARD_METRIC": "reward",
    }
    exec(compile(class_source, path, "exec"), namespace)
    return {"sync": namespace["ReplayBuffer"], "async": namespace["ReplayBufferAsync"]}, clock, tq


def load_aggregator(tree):
    """exec the real `MetricsAggregator` class (utils.py), stubbing only torch tensors."""
    path = f"{tree}/verl/trainer/ppo/v1/utils.py"
    source = open(path).read()
    module = ast.parse(source)
    cls = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "MetricsAggregator"
    )
    class_source = ast.get_source_segment(source, cls)

    class StubTensor:
        pass

    class StubTorch:
        Tensor = StubTensor

    namespace = {
        "Counter": Counter,
        "defaultdict": defaultdict,
        "Any": object,
        "np": __import__("numpy"),
        "torch": StubTorch,
        "DAPO_FILTERED_REWARD_COUNTS_KEY": "_dapo_filtered_reward_counts",
    }
    exec(compile(class_source, path, "exec"), namespace)
    return namespace["MetricsAggregator"]


# ----------------------------------------------------------------------------
# scenario driver
# ----------------------------------------------------------------------------
class Scenario:
    def __init__(self, tree, poll_interval, inflight=1, dapo=True, train_batch_size=8, gen_batch_size=1,
                 mode="sync"):
        classes, self.clock, self.tq = load_replay_buffer(tree)
        self.cls = classes[mode]
        self.mode = mode
        self.poll_interval = poll_interval
        self.inflight = inflight
        self.dapo = dapo
        self.train_batch_size = train_batch_size
        self.gen_batch_size = gen_batch_size
        self.next_uid = 0
        # scripted arrival model: each dispatched prompt finishes `duration` later
        self.duration = None
        self.filtered_prob = 0.0
        self.keep_prob_if_not_filtered = 1.0
        self.dispatched = []
        self.buffer = self.cls(
            trainer_mode=self.mode,
            trainer_config={},
            # The trainer's yaml default. A positive int is inert for these scripts (fresh
            # prompts are never stale); it cannot be None on trees before the `int | None`
            # change (24f25b03 asserts), and the arm under test runs with the default anyway.
            max_off_policy_threshold=8,
            max_off_policy_strategy="drop",
            sampler_kwargs={},
            poll_interval=poll_interval,
            refill_fn=self.dispatch,
            filter_groups_metric="reward" if dapo else None,
            train_batch_size=train_batch_size,
            gen_batch_size=gen_batch_size if dapo else train_batch_size,
            max_inflight_gen_batches=inflight,
            sync_refill_failed_groups=False,
        )

    def dispatch(self, n):
        n -= n % self.gen_batch_size
        assert n > 0, n
        for _ in range(n):
            uid = f"p{self.next_uid}"
            # Deterministic pseudo-random filtering (uid * Knuth), stable across processes.
            bucket = (self.next_uid * 2654435761) % 1000 / 1000.0
            filtered = bucket < self.filtered_prob
            self.next_uid += 1
            self.tq.add_prompt(
                uid,
                finish_at=self.clock.now + self.duration,
                global_steps=self.step,
                reward=0.0,
                filtered=filtered,
            )
            self.dispatched.append(uid)
        return n

    def step_once(self, step):
        """One trainer step: dispatch one batch, block in sample(), then train."""
        self.step = step
        self.dispatch(self.train_batch_size)
        start = self.clock.now
        _batch, metrics = self.buffer.sample(step, "train", self.train_batch_size)
        gen = self.clock.now - start
        return gen, metrics


def run_scenario(tree, name, *, poll_interval, inflight, dapo, duration, steps=6,
                 train_batch_size=8, filtered_prob=0.0, train_time=1.0, mode="sync"):
    sc = Scenario(tree, poll_interval, inflight=inflight, dapo=dapo, train_batch_size=train_batch_size,
                  mode=mode)
    sc.duration = duration
    sc.filtered_prob = filtered_prob
    rows = []
    for step in range(1, steps + 1):
        before = sc.clock.sleeps
        gen, metrics = sc.step_once(step)
        rows.append(
            {
                "step": step,
                "gen_s": round(gen, 6),
                "clock_sleeps": sc.clock.sleeps - before,
                "metric_polls": metrics.get("training/sampler/blocking_polls"),
                "evicted": metrics.get("training/filter_groups/evicted_samples", 0),
                "surplus": metrics.get("training/filter_groups/discarded_surplus_samples", 0),
            }
        )
        sc.clock.advance(train_time)
    return {"name": name, "poll_interval": poll_interval, "inflight": inflight, "rows": rows}


# ----------------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------------
def checks(tree, patched):
    """The virtual clock is the oracle: it counts the sleeps the loop really took.

    On a patched tree every count must also equal `training/sampler/blocking_polls`;
    on an unpatched tree that key must be absent (negative control).
    """
    results = []

    def record(name, ok, detail):
        results.append({"check": name, "ok": bool(ok), "detail": detail})

    def polls(run):
        return [r["clock_sleeps"] for r in run["rows"]]

    def metric_matches(run):
        return all(
            (r["metric_polls"] == r["clock_sleeps"]) if patched else (r["metric_polls"] is None)
            for r in run["rows"]
        )

    # 1. Prefilled buffer: everything terminal before sample() -> no sleep at all.
    sc = Scenario(tree, poll_interval=2.0, inflight=1, dapo=False, train_batch_size=8)
    sc.duration = 0.5
    sc.step = 1
    sc.dispatch(8)
    sc.clock.advance(1.0)  # settles all 8
    _b, metrics = sc.buffer.sample(1, "train", 8)
    fast = {"rows": [{"step": 1, "clock_sleeps": sc.clock.sleeps,
                      "metric_polls": metrics.get("training/sampler/blocking_polls")}]}
    record("fast path takes 0 polls", polls(fast) == [0] and metric_matches(fast),
           {"sleeps": polls(fast), "metric": fast["rows"][0]["metric_polls"]})

    # 2. One quantum: arrivals land inside the first sleep.
    one = run_scenario(tree, "one-quantum", poll_interval=2.0, inflight=1, dapo=False,
                       duration=1.0, steps=3)
    record("arrival inside first quantum -> 1 poll per step",
           polls(one) == [1, 1, 1] and metric_matches(one), polls(one))

    # 3. Three quanta: arrivals at +5.0s are only visible at polls 2/4/6.
    three = run_scenario(tree, "three-quanta", poll_interval=2.0, inflight=1, dapo=False,
                         duration=5.0, steps=1)
    record("arrival at 5.0s with P=2.0 -> 3 polls",
           polls(three) == [3] and metric_matches(three), polls(three))

    # 4. DAPO filtering forces refill rounds, so a step must block more than once.
    dapo = run_scenario(tree, "dapo-refill", poll_interval=2.0, inflight=2, dapo=True,
                        duration=0.6, steps=4, train_batch_size=8, filtered_prob=0.75)
    record("DAPO refill: >1 blocking poll per step", all(p > 1 for p in polls(dapo)) and metric_matches(dapo),
           {"polls": polls(dapo), "evicted": [r["evicted"] for r in dapo["rows"]]})

    # 5. Same schedule, finer poll: cannot take fewer polls.
    fine = run_scenario(tree, "dapo-refill-p010", poll_interval=0.1, inflight=2, dapo=True,
                        duration=0.6, steps=4, train_batch_size=8, filtered_prob=0.75)
    record("P=0.1 takes at least as many polls as P=2.0",
           all(f >= c for f, c in zip(polls(fine), polls(dapo))) and metric_matches(fine),
           {"p010": polls(fine), "p200": polls(dapo)})

    # 6. Deeper in-flight window cannot increase the poll count on the same arrivals.
    d1 = run_scenario(tree, "depth-1", poll_interval=2.0, inflight=1, dapo=True,
                      duration=1.0, steps=4, train_batch_size=8, filtered_prob=0.5)
    d2 = run_scenario(tree, "depth-2", poll_interval=2.0, inflight=2, dapo=True,
                      duration=1.0, steps=4, train_batch_size=8, filtered_prob=0.5)
    record("inflight=2 polls <= inflight=1 polls",
           all(b <= a for a, b in zip(polls(d1), polls(d2))) and metric_matches(d2),
           {"inflight1": polls(d1), "inflight2": polls(d2)})

    # 7. The counter is per sample() call, not cumulative: identical steps -> identical counts.
    record("counter resets per step", len(set(polls(dapo))) == 1, polls(dapo))

    # 8. Presence/absence of the metric key is the patch's own signature.
    record("metric key present iff tree is patched",
           all((r["metric_polls"] is not None) for r in dapo["rows"]) == patched,
           {"patched": patched, "metrics": [r["metric_polls"] for r in dapo["rows"]]})

    # 9. The aggregator must *sum* the key across parameter_sync_step iterations, otherwise a
    #    step-level reading is the average of per-call counts. Checked by running the real
    #    `MetricsAggregator` class source, not by looking for the key name: a key registered
    #    under the wrong rule (or not at all) reads differently. Two iterations of 2 and 3
    #    polls sum to 5; the unregistered default is a sample-weighted average of 2.5.
    aggregator = load_aggregator(tree)
    agg = aggregator()
    for per_call in (2, 3):
        agg.add_step_metrics({"training/sampler/blocking_polls": per_call}, sample_count=8)
    reduced = agg.get_aggregated_metrics().get("training/sampler/blocking_polls")
    want = 5.0 if patched else 2.5
    record("aggregator reduces the poll key by sum", reduced == want,
           {"reading": reduced, "expected": want, "patched": patched})

    # 10. A registration is only worth carrying if some path logs it. `_validate()` is the only
    #     user of partition "val" and it discards the buffer's metric dict (`batch, _ = ...`),
    #     so a `validation/...` rule is dead wiring. The invariant asserted is the pair:
    #     registered-for-validation iff the val call site keeps its metrics.
    trainer_source = open(f"{tree}/verl/trainer/ppo/v1/trainer_base.py").read()
    utils_source = open(f"{tree}/verl/trainer/ppo/v1/utils.py").read()
    lines = trainer_source.splitlines()
    val_idx = next(
        (i for i, line in enumerate(lines) if 'partition_id="val"' in line and "batch_size=len(batch)" in line),
        None,
    )
    val_call = lines[val_idx - 1].strip() if val_idx else ""
    val_discards = bool(val_idx) and val_call.startswith("batch, _ =")
    val_registered = "validation/sampler/blocking_polls" in utils_source
    record("validation poll key registered only if val metrics are logged",
           val_registered != val_discards,
           {"val_discards_metrics": val_discards, "validation_key_registered": val_registered,
            "val_call_site": val_call.strip()})

    # 11. Async mode counts polls inside `wait_for_sampleable()`, and `sample()` forwards that
    #     dict: the count must match the clock there too, and reset per call.
    async_run = run_scenario(tree, "async-wait", poll_interval=2.0, inflight=1, dapo=False,
                             duration=5.0, steps=2, train_batch_size=8, mode="async")
    record("async wait_for_sampleable counts its own polls",
           all(p == 3 for p in polls(async_run)) and metric_matches(async_run),
           {"polls": polls(async_run), "metric": [r["metric_polls"] for r in async_run["rows"]]})

    # 12. Same negative control as (8), on the async path.
    record("async metric key present iff tree is patched",
           all(r["metric_polls"] is not None for r in async_run["rows"]) == patched,
           {"patched": patched, "metric": [r["metric_polls"] for r in async_run["rows"]]})
    return results


def matrix(tree):
    """A fixed battery of schedules, used to prove the counter patch is inert."""
    out = []
    for dapo in (False, True):
        for inflight in (1, 2, 3):
            for poll in (2.0, 0.5, 0.1):
                for duration in (0.6, 1.0, 2.5, 5.0):
                    for filtered in (0.0, 0.5, 0.75):
                        if not dapo and (inflight != 1 or filtered != 0.0):
                            continue
                        run = run_scenario(
                            tree,
                            f"dapo={dapo},inflight={inflight},P={poll},d={duration},f={filtered}",
                            poll_interval=poll,
                            inflight=inflight,
                            dapo=dapo,
                            duration=duration,
                            steps=4,
                            train_batch_size=8,
                            filtered_prob=filtered,
                        )
                        out.append(run)
    return out


def compare(tree_patched, tree_plain):
    """The patch must not change any behaviour: same trace with and without it."""
    a = matrix(tree_patched)
    b = matrix(tree_plain)
    assert len(a) == len(b)
    diffs = []
    for ra, rb in zip(a, b):
        for row_a, row_b in zip(ra["rows"], rb["rows"]):
            for key in ("gen_s", "clock_sleeps", "evicted", "surplus"):
                if row_a[key] != row_b[key]:
                    diffs.append({"scenario": ra["name"], "step": row_a["step"], "field": key,
                                  "patched": row_a[key], "plain": row_b[key]})
            if row_b["metric_polls"] is not None:
                diffs.append({"scenario": ra["name"], "step": row_a["step"],
                              "field": "metric_on_plain_tree", "patched": None, "plain": row_b["metric_polls"]})
    return {
        "scenarios": len(a),
        "rows": sum(len(r["rows"]) for r in a),
        "diffs": diffs,
        "inert": not diffs,
        "poll_summary": [
            {"scenario": r["name"], "polls_per_step": [x["clock_sleeps"] for x in r["rows"]]}
            for r in a
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--compare", default=None, help="unpatched tree; asserts identical behaviour")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    patched = "_blocking_polls" in open(f"{args.tree}/verl/trainer/ppo/v1/replay_buffer.py").read()
    results = checks(args.tree, patched)
    failed = [r for r in results if not r["ok"]]
    payload = {"tree": args.tree, "patched": patched, "checks": results, "failed": len(failed)}
    if args.compare:
        payload["compare"] = compare(args.tree, args.compare)
        if not payload["compare"]["inert"]:
            failed.append({"check": "patch is inert", "ok": False,
                           "detail": payload["compare"]["diffs"][:5]})
            payload["failed"] = len(failed)
    print(json.dumps({k: v for k, v in payload.items() if k != "compare"}, ensure_ascii=False, indent=2)[:4000])
    if args.compare:
        c = payload["compare"]
        print(f"compare: {c['scenarios']} scenarios / {c['rows']} rows / inert={c['inert']}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
    if args.selftest:
        passed = sum(1 for r in results if r["ok"])
        print(f"selftest: {passed}/{len(results)} passed{'' if not args.compare else ' (+ patch inertia)'}")
        return 0 if not failed else 1
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
