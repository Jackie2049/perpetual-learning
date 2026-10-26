"""CPU eager-backend compile and export controls, not Inductor/CUDA validation."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import platform

import torch
from torch import nn


def select_after_computing(x):
    return torch.where(x > 0, x.sqrt(), x.square())


def structured_branch(x):
    return torch.cond(x > 0, lambda a: a.sqrt(), lambda a: a.square(), (x,))


def select_with_safe_input(x):
    positive = x > 0
    sqrt_input = torch.where(positive, x, torch.ones_like(x))
    return torch.where(positive, sqrt_input.sqrt(), x.square())


class Wrapper(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = {"scope": __doc__, "python": platform.python_version(),
              "torch": torch.__version__, "torch_git": torch.version.git_version,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "torch_cond_source_sha256": hashlib.sha256(inspect.getsource(torch.cond).encode()).hexdigest(),
              "backend": "eager", "fullgraph": True, "dtype": "CPU float64", "cases": {}, "exports": {}}
    functions = {"where": select_after_computing, "cond": structured_branch, "safe_where": select_with_safe_input}
    for name, fn in functions.items():
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        result["cases"][name] = []
        for value in (-4.0, 4.0):
            x = torch.tensor(value, dtype=torch.float64, device="cpu", requires_grad=True)
            y = compiled(x)
            grad = torch.autograd.grad(y, x)[0]
            expected_y, expected_grad = (16.0, -8.0) if value < 0 else (2.0, 0.25)
            assert y.item() == expected_y
            if name == "where" and value < 0:
                assert torch.isnan(grad)
            else:
                assert grad.item() == expected_grad
            result["cases"][name].append({"input": value, "output": y.item(),
                                          "grad_finite": bool(torch.isfinite(grad)),
                                          "grad": grad.item() if torch.isfinite(grad) else None})
        # Public export, not generated machine code or an exported backward graph.
        ep = torch.export.export(Wrapper(fn), (torch.tensor(-4.0, dtype=torch.float64, device="cpu"),), strict=True)
        graphs = {key or "root": [str(n.target) for n in mod.graph.nodes if n.op == "call_function"]
                  for key, mod in ep.graph_module.named_modules() if isinstance(mod, torch.fx.GraphModule)}
        result["exports"][name] = graphs
        if name == "cond":
            assert "cond" in graphs["root"]
            assert len(graphs) == 3
        else:
            assert "aten.where.self" in graphs["root"]
            assert "aten.sqrt.default" in graphs["root"]
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
