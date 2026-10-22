#!/usr/bin/env python3
"""Run #7366's unchanged CPU tests with real torch/ZMQ/shared memory.

Python 3.12+, pytest, torch, pyzmq. Supply a local Git object store and optional
directory containing pyzmq. No dependency installation, fetch or device probing.
Package-root/device imports are CPU facades, not native verl integration. The
original tests additionally replace device synchronization and iterator helpers.
The added observation uses actual in-process ZMQ sockets, not socket mocks.
"""

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REV = "48d91130abff6b6704aadac3f382e0c3d344de93"
SOURCE = "verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py"
TEST = "tests/workers/rollout/test_bucketed_weight_transfer_on_cpu.py"

BOOTSTRAP = '''import sys
import types
for name in ("verl", "verl.utils"):
    package = types.ModuleType(name)
    package.__path__ = []
    sys.modules[name] = package
class CpuDevice:
    def synchronize(self): pass
    def empty_cache(self): pass
def reject_device_id():
    raise AssertionError("Device/IPC path is outside CPU test scope")
device = types.ModuleType("verl.utils.device")
device.get_torch_device = lambda: CpuDevice()
device.get_device_name = lambda: "cpu"
device.get_device_id = reject_device_id
device.is_support_ipc = lambda: False
sys.modules[device.__name__] = device
'''

OBSERVE = '''import asyncio
import importlib.util
import time
import uuid
import zmq
spec = importlib.util.spec_from_file_location("bwt", SOURCE)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
context = zmq.Context()
request, reply = context.socket(zmq.REQ), context.socket(zmq.REP)
for socket in (request, reply): socket.setsockopt(zmq.LINGER, 0)
endpoint = "inproc://cpu-ack-" + uuid.uuid4().hex
request.bind(endpoint)
reply.connect(endpoint)
request.send(b"request")
assert reply.recv() == b"request"
sender = native.BucketedWeightSender("unused", ack_timeout_ms=150)
sender.socket = request
async def observe():
    started = time.monotonic()
    async def tick():
        await asyncio.sleep(.01)
        return time.monotonic() - started
    async def block():
        sender._receive_ack("CPU event-loop observation")
    async def deadline():
        try:
            await asyncio.wait_for(block(), timeout=.02)
        except RuntimeError as error:
            assert "timed out waiting 150ms" in str(error)
            return type(error).__name__
        raise AssertionError("Expected the original ACK error, not a successful receive")
    delay, error = await asyncio.gather(tick(), deadline())
    assert delay >= .12, delay
    print("real ZMQ poll: 10ms tick delayed", round(delay, 4),
          "seconds; 20ms asyncio deadline yielded", error, flush=True)
try:
    asyncio.run(observe())
finally:
    request.close()
    reply.close()
    context.term()
import torch
print("Python", __import__("sys").version.split()[0], "torch", torch.__version__,
      "pyzmq", zmq.__version__, "libzmq", zmq.zmq_version(), flush=True)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    parser.add_argument("--deps", type=Path)
    args = parser.parse_args()
    env = dict(os.environ, GIT_NO_LAZY_FETCH="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               PYTHONDONTWRITEBYTECODE="1")
    archive = subprocess.check_output(
        ["git", "-C", str(args.source_repo.resolve()), "archive", REV, SOURCE, TEST], env=env)
    with tempfile.TemporaryDirectory(prefix="perpetual-bwt-") as directory:
        root = Path(directory)
        subprocess.run(["tar", "-x", "-C", directory], input=archive, check=True)
        for relative in (SOURCE, TEST):
            data = (root / relative).read_bytes()
            compile(data, relative, "exec")
            print(relative, "SHA256", hashlib.sha256(data).hexdigest(), flush=True)
        (root / "sitecustomize.py").write_text(BOOTSTRAP)
        env["PYTHONPATH"] = directory + (os.pathsep + str(args.deps.resolve()) if args.deps else "")
        observation = "SOURCE = " + repr(str(root / SOURCE)) + "\n" + OBSERVE
        subprocess.run([sys.executable, "-B", "-c", observation], env=env, cwd=root, check=True)
        result = subprocess.run(
            [sys.executable, "-B", "-m", "pytest", "--noconftest", "-p", "no:cacheprovider",
             "-v", "--tb=short", TEST], env=env, cwd=root)
        print("NOT RUN: native verl/config imports, CUDA IPC, NPU, vLLM, Ray or training.", flush=True)
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
