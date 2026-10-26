"""Small CPU file semantics control. No timing, RSS, cold-cache or GPU claims."""

import argparse
import hashlib
import json
import mmap
from pathlib import Path
import platform
import struct
import tempfile

import safetensors
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    tensors = {
        "weights": torch.arange(8192, dtype=torch.float32),
        "optimizer.momentum": torch.tensor([0.0, 0.5, 1.0, -1.0]),
        "bool_mask": torch.tensor([True, False, True]),
        "fp8": torch.tensor([0.0, 1.0, -2.0, 0.5]).to(torch.float8_e4m3fn),
    }
    result = {"scope": __doc__, "python": platform.python_version(),
              "torch": torch.__version__, "safetensors": safetensors.__version__,
              "script_sha256": digest(Path(__file__)), "routes": {}}
    with tempfile.TemporaryDirectory(prefix="weight-file-mapping-") as folder:
        root = Path(folder)
        st, pt = root / "tiny.safetensors", root / "tiny.pt"
        save_file(tensors, st)
        torch.save(tensors, pt)
        # Header inspection is a format check, not a lazy-read benchmark.
        raw = st.read_bytes()
        size = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8:8 + size])
        result["header"] = header
        assert header["bool_mask"]["data_offsets"][1] - header["bool_mask"]["data_offsets"][0] == 3
        assert header["bool_mask"]["dtype"] == "BOOL"
        assert header["fp8"]["dtype"] == "F8_E4M3"
        assert header["fp8"]["data_offsets"][1] - header["fp8"]["data_offsets"][0] == 4

        def selected_safetensors():
            with safe_open(st, framework="pt", device="cpu", backend="mmap") as opened:
                selected = opened.get_tensor("weights")
            # The returned tensor remains valid after the context closes.
            return selected

        assert torch.serialization.get_default_mmap_options() == mmap.MAP_PRIVATE
        result["torch_mmap_options"] = "MAP_PRIVATE"
        routes = {
            "safetensors_cpu_mmap": (st, selected_safetensors),
            "torch_cpu_mmap": (pt, lambda: torch.load(pt, map_location="cpu", weights_only=True, mmap=True)["weights"]),
            "torch_cpu_eager": (pt, lambda: torch.load(pt, map_location="cpu", weights_only=True, mmap=False)["weights"]),
        }
        for name, (path, read) in routes.items():
            before = digest(path)
            value = read()
            assert torch.equal(value, tensors["weights"])
            total = value.sum().item()
            value[0] = -7
            reopened = read()
            assert reopened[0].item() == 0.0
            assert digest(path) == before
            result["routes"][name] = {"elements": value.numel(), "sum_before_mutation": total,
                                       "mutated_value": value[0].item(), "reopened_value": reopened[0].item(),
                                       "file_hash_unchanged": True}
        loaded = load_file(st, device="cpu", backend="mmap")
        assert torch.equal(loaded["fp8"].view(torch.uint8), tensors["fp8"].view(torch.uint8))
        assert torch.equal(loaded["bool_mask"], tensors["bool_mask"])
        assert torch.equal(loaded["optimizer.momentum"], tensors["optimizer.momentum"])
        result["fp8_roundtrip_bytes_equal"] = True
        result["optimizer_named_tensor_roundtrip"] = True
    result["scratch_removed"] = not root.exists()
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
