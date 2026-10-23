"""Minimal deep_ep stub for civilian-port smoke (SM89): keep imports alive, raise on use."""
import torch
import torch.distributed as dist


class _NotImplemented:
    def __init__(self, name):
        self._name = name

    def __call__(self, *a, **k):
        raise NotImplementedError(f"deep_ep.{self._name} not available on civilian port (SM89)")

    def __getattr__(self, k):
        return _NotImplemented(f"{self._name}.{k}")


class EPBuffer:
    pass


class EPHandle:
    pass


class EventHandle:
    pass


class EventOverlap:
    pass


class BucketBuffer:
    def __init__(self, group=None, num_bytes=0, explicitly_destroy=False):
        self.group = group
        self.num_bytes = num_bytes
        self.explicitly_destroy = explicitly_destroy


class BucketSession:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


class EngramBuffer:
    @staticmethod
    def get_storage_size_hint(*a, **k):
        return 0, 0

    @staticmethod
    def get_theoretical_config(*a, **k):
        return {}, {}

    def __init__(self, *a, **k):
        pass


def get_num_rdma_alignment():
    return 1


def get_num_allocation_alignment():
    return 1


def get_physical_domain_size(group):
    # civilian single-GPU: return (rank_in_group, group_size) shaped tuple (lo, hi)
    try:
        size = dist.get_world_size(group)
    except Exception:
        size = 1
    return (size, size)
