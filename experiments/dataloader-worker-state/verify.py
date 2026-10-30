"""Real CPU spawn-worker state across two epochs, not a throughput benchmark.

Run as an importable file: python -B experiments/dataloader-worker-state/verify.py
Requires the existing PyTorch environment; no data download or device discovery.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, get_worker_info
from torch.utils.data.dataloader import _BaseDataLoaderIter
from torch.utils.data._utils.worker import _worker_loop


class Records(Dataset):
    def __init__(self):
        self.epoch = 0
        self.initialized_epoch = -1
        self.init_calls = 0

    def __len__(self):
        return 8

    def __getitem__(self, index):
        info = get_worker_info()
        assert info is not None
        return {
            "index": index,
            "dataset_epoch": self.epoch,
            "initialized_epoch": self.initialized_epoch,
            "worker_init_calls": self.init_calls,
            "worker": info.id,
            "pid": os.getpid(),
            "seed": torch.initial_seed(),
            "draw": torch.rand(()).item(),
        }


def initialize_worker(_worker_id):
    info = get_worker_info()
    info.dataset.initialized_epoch = info.dataset.epoch
    info.dataset.init_calls += 1


def case(persistent):
    dataset = Records()
    # Explicit single replica: this uses the real sampler without a process group.
    sampler = DistributedSampler(dataset, num_replicas=1, rank=0, seed=7, shuffle=True)
    loader = DataLoader(
        dataset, batch_size=2, sampler=sampler, num_workers=2,
        persistent_workers=persistent, multiprocessing_context="spawn",
        generator=torch.Generator().manual_seed(2026), worker_init_fn=initialize_worker,
        prefetch_factor=2, pin_memory=False, timeout=20,
    )
    epochs = []
    iterator = None
    try:
        for epoch in (0, 1):
            dataset.epoch = epoch
            sampler.set_epoch(epoch)
            expected_order = list(sampler)
            iterator = iter(loader)
            rows = []
            for batch in iterator:
                for offset in range(len(batch["index"])):
                    rows.append({key: value[offset].item() for key, value in batch.items()})
            assert [row["index"] for row in rows] == expected_order
            expected_dataset_epoch = 0 if persistent else epoch
            assert {row["dataset_epoch"] for row in rows} == {expected_dataset_epoch}
            assert {row["initialized_epoch"] for row in rows} == {expected_dataset_epoch}
            assert {row["worker_init_calls"] for row in rows} == {1}
            if not persistent:
                assert all(not worker.is_alive() for worker in iterator._workers)
            epochs.append(rows)
    finally:
        # Private API only to join the workers this bounded experiment owns.
        # Not a recommended application-level DataLoader lifecycle API.
        if iterator is not None:
            iterator._shutdown_workers()
            assert all(not worker.is_alive() for worker in iterator._workers)

    assert [row["index"] for row in epochs[0]] != [row["index"] for row in epochs[1]]
    summaries = []
    for epoch, rows in enumerate(epochs):
        by_worker = {}
        for worker in (0, 1):
            selected = [row for row in rows if row["worker"] == worker]
            assert len(selected) == 4
            seeds = {row["seed"] for row in selected}
            assert len(seeds) == 1
            seed = next(iter(seeds))
            previous = [row for row in epochs[0] if row["worker"] == worker]
            if epoch == 1:
                assert (seed == previous[0]["seed"]) == persistent
            generator = torch.Generator().manual_seed(seed)
            offset = 4 if persistent and epoch == 1 else 0
            expected = [torch.rand((), generator=generator).item() for _ in range(offset + 4)][offset:]
            actual = [row["draw"] for row in selected]
            assert actual == expected
            # JSON consumers may use binary64 numbers; keep every 64-bit seed digit.
            by_worker[str(worker)] = {"seed": str(seed), "draws": actual, "rng_offset": offset}
        summaries.append({
            "indices": [row["index"] for row in rows],
            "dataset_epochs": sorted({row["dataset_epoch"] for row in rows}),
            "worker_init_epochs": sorted({row["initialized_epoch"] for row in rows}),
            "worker_init_call_counts": sorted({row["worker_init_calls"] for row in rows}),
            "workers": by_worker,
        })
    same_pids = all(
        {r["pid"] for r in epochs[0] if r["worker"] == worker}
        == {r["pid"] for r in epochs[1] if r["worker"] == worker}
        for worker in (0, 1)
    )
    if persistent:
        assert same_pids
    return {"same_worker_pids": same_pids, "owned_workers_stopped": True, "epochs": summaries}


def main():
    torch.set_num_threads(1)
    result = {
        "python": platform.python_version(), "torch": torch.__version__,
        "torch_git": torch.version.git_version, "start_method": "spawn",
        "scope": "8 map-style records, 2 CPU workers, batch 2, two fully consumed epochs, one explicit sampler replica",
        "source_sha256": {
            method.__qualname__: hashlib.sha256(inspect.getsource(method).encode()).hexdigest()
            for method in (DataLoader.__iter__, _BaseDataLoaderIter._reset, _worker_loop)
        },
        "nonpersistent": case(False), "persistent": case(True),
    }
    for epoch in (0, 1):
        assert result["nonpersistent"]["epochs"][epoch]["indices"] == result["persistent"]["epochs"][epoch]["indices"]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
