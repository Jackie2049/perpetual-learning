"""CPU 侧替身：把 verl 的 `ReplayBuffer` 源码原封不动跑起来，不装 ray / torch / numpy。

`replay_buffer.py` 在模块级只依赖三样外物：`transfer_queue`（tq 的数据面）、
`verl.utils.skip.SkipManager`（装饰器）与 `omegaconf.DictConfig`（类型标注）。
本文件只替换前两个，**`ReplayBuffer` 本身一行不改**——所以读到的是上游真实
判定逻辑，不是本实验重写的一份。

替身的语义边界（写清楚，供读数解释时用）：
- `kv_batch_get(select_fields=["extra_fields"])` 返回轨迹写入时给的
  `extra_fields` 原文；`["rm_scores"]` 返回包了 `.sum().item()` 的壳。
- 不模拟 TransferQueue 的并发/租约/时序，只做「写进去、读出来」。
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


class _SumShell:
    """`rm_scores` 的替身：真实代码只对它做 `.sum().item()`。"""

    def __init__(self, value: float):
        self.value = value

    def sum(self):
        return self

    def item(self):
        return self.value


class FakeTransferQueue:
    """极简键值分区：`kv_put` / `kv_list` / `kv_batch_get` / `kv_clear`。"""

    def __init__(self):
        self.partitions: dict[str, dict[str, dict]] = {}

    # --- tq API -----------------------------------------------------------
    def kv_put(self, key, partition_id="default", fields=None, tag=None, **_ignored):
        part = self.partitions.setdefault(partition_id, {})
        entry = part.setdefault(key, {"tag": {}, "fields": {}})
        if tag is not None:
            entry["tag"] = dict(tag)
        if fields:
            entry["fields"].update(fields)

    def kv_list(self):
        if not self.partitions:
            return None
        return {pid: {k: dict(v["tag"]) for k, v in items.items()} for pid, items in self.partitions.items()}

    def kv_batch_get(self, keys, partition_id="default", select_fields=None):
        rows = [self.partitions[partition_id][k]["fields"] for k in keys]
        out: dict[str, list] = {}
        for field in select_fields or []:
            if field == "rm_scores":
                out[field] = [_SumShell(float((row.get("rm_scores") or 0.0))) for row in rows]
            else:
                out[field] = [row.get(field) for row in rows]
        return out

    def kv_clear(self, partition_id="default", keys=None, **_ignored):
        for key in keys or []:
            self.partitions.get(partition_id, {}).pop(key, None)

    # --- 生产端（模拟 rollout 侧落盘） -------------------------------------
    def put_group(self, uid: str, n_sessions: int, extra_info: dict | None, rm_score: float = 1.0):
        """按真实顺序写：先每条轨迹，再把 prompt 标成 finished。"""
        for session in range(n_sessions):
            self.kv_put(
                key=f"{uid}_{session}_0",
                partition_id="train",
                fields={"extra_fields": {"reward_extra_info": dict(extra_info or {})}, "rm_scores": rm_score},
                tag={"is_prompt": False, "seq_len": 3, "global_steps": 0},
            )
        self.kv_put(
            key=uid,
            partition_id="train",
            tag={"is_prompt": True, "status": "finished", "global_steps": 0},
        )


class KVBatchMeta(dict):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.__dict__.update(kw)


def load_replay_buffer_module(tree: Path):
    """import `verl/trainer/ppo/v1/replay_buffer.py`，返回模块对象。"""
    tree = Path(tree)
    src = tree / "verl/trainer/ppo/v1/replay_buffer.py"
    if not src.exists():
        raise FileNotFoundError(src)

    def pkg(name, path):
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]
        sys.modules[name] = mod
        return mod

    fake_tq = FakeTransferQueue()
    tq_mod = types.ModuleType("transfer_queue")
    tq_mod.kv_put = fake_tq.kv_put
    tq_mod.kv_list = fake_tq.kv_list
    tq_mod.kv_batch_get = fake_tq.kv_batch_get
    tq_mod.kv_clear = fake_tq.kv_clear
    tq_mod.KVBatchMeta = KVBatchMeta
    sys.modules["transfer_queue"] = tq_mod

    class SkipManager:
        @classmethod
        def annotate_tq(cls, role, phase):
            def deco(fn):
                return fn

            return deco

    if "verl" not in sys.modules:
        verl = pkg("verl", tree / "verl")
        pkg("verl.utils", tree / "verl/utils")
        pkg("verl.trainer", tree / "verl/trainer")
        pkg("verl.trainer.ppo", tree / "verl/trainer/ppo")
        pkg("verl.trainer.ppo.v1", tree / "verl/trainer/ppo/v1")
        skip = types.ModuleType("verl.utils.skip")
        skip.SkipManager = SkipManager
        sys.modules["verl.utils.skip"] = skip
        sys.modules["verl.utils"].skip = skip
        verl.utils = sys.modules["verl.utils"]
        verl.trainer = sys.modules["verl.trainer"]
        sys.modules["verl.trainer"].ppo = sys.modules["verl.trainer.ppo"]
        sys.modules["verl.trainer.ppo"].v1 = sys.modules["verl.trainer.ppo.v1"]

    name = "verl.trainer.ppo.v1.replay_buffer"
    spec = importlib.util.spec_from_file_location(name, src)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.tq = tq_mod
    return module, fake_tq
