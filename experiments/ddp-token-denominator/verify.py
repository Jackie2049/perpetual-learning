"""Local CPU/Gloo DDP denominator controls; no trainer or device experiment."""

import datetime
import json
import os
import platform
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def local_data(rank, empty):
    if rank == 0:
        x = [] if empty else [2.0]
        y = [] if empty else [0]
    else:
        x, y = [1.0] * 3, [1] * 3
    return torch.tensor(x, dtype=torch.float64).reshape(-1, 1), torch.tensor(y, dtype=torch.long)


def summed_loss(model, x, y):
    score = model(x)
    logits = torch.cat([score, torch.zeros_like(score)], dim=1)
    return torch.nn.functional.cross_entropy(logits, y, reduction="sum")


def new_model():
    model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.zero_()
    return model


def worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=20))
    try:
        identities = [None, None]
        dist.all_gather_object(identities, (rank, os.getpid()))
        assert len({pid for _, pid in identities}) == 2
        results = []
        for empty, mode in [(False, "rank_mean"), (False, "sum_over_global"),
                            (False, "world_sum_over_global"), (True, "world_sum_over_global")]:
            x, y = local_data(rank, empty)
            count = torch.tensor(len(y), dtype=torch.int64)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
            model = DDP(new_model(), device_ids=None)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            numerator = summed_loss(model, x, y)
            assert torch.isfinite(numerator)
            if mode == "rank_mean":
                loss = numerator / len(y)
            else:
                loss = numerator / count.item()
                if mode == "world_sum_over_global":
                    loss = loss * dist.get_world_size()
            loss.backward()
            gradient = model.module.weight.grad.item()
            optimizer.step()

            full_x = torch.cat([local_data(r, empty)[0] for r in range(2)])
            full_y = torch.cat([local_data(r, empty)[1] for r in range(2)])
            reference = new_model()
            (summed_loss(reference, full_x, full_y) / len(full_y)).backward()
            target_gradient = reference.weight.grad.item()
            expected = target_gradient if mode == "world_sum_over_global" else (
                -0.25 if mode == "rank_mean" else target_gradient / 2)
            assert abs(gradient - expected) < 1e-12
            row = {"rank": rank, "local_tokens": len(y), "gradient": gradient,
                   "theta_after_sgd": model.module.weight.item()}
            gathered = [None, None]
            dist.all_gather_object(gathered, row)
            assert all(abs(r["gradient"] - expected) < 1e-12 for r in gathered)
            assert all(abs(r["theta_after_sgd"] + 0.1 * expected) < 1e-12 for r in gathered)
            results.append({"layout": "empty_rank" if empty else "unequal", "mode": mode,
                            "global_tokens": count.item(), "reference_gradient": target_gradient,
                            "ranks": gathered})
        if rank == 0:
            print(json.dumps({"python": platform.python_version(), "torch": torch.__version__,
                              "torch_git": torch.version.git_version, "backend": dist.get_backend(),
                              "distinct_worker_processes": 2, "results": results}, sort_keys=True), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    # Pin Gloo to loopback: this experiment never contacts another machine.
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if platform.system() == "Darwin" else "lo"
    with tempfile.TemporaryDirectory(prefix="pl-ddp-denominator-") as scratch:
        mp.spawn(worker, args=("file://" + os.path.join(scratch, "rendezvous"),), nprocs=2, join=True)
