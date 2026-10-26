"""Small real CPU MHA/SDPA contract controls; no model, CUDA or performance test."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import platform

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def identity_mha(dropout=0.0):
    layer = nn.MultiheadAttention(1, 1, dropout=dropout, bias=False,
                                 batch_first=True, dtype=torch.float64, device="cpu")
    with torch.no_grad():
        layer.in_proj_weight.fill_(1.0)
        layer.out_proj.weight.fill_(1.0)
    return layer


def describe(value):
    return {"finite": bool(torch.isfinite(value).all()),
            "all_nan": bool(torch.isnan(value).all()),
            "scalar": value.item() if torch.isfinite(value).all() else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    q = torch.zeros(1, 1, 1, dtype=torch.float64, device="cpu")
    k = torch.zeros(1, 3, 1, dtype=torch.float64, device="cpu")
    v = torch.tensor([0., 2., 8.], dtype=torch.float64, device="cpu").view(1, 3, 1)
    allowed = torch.tensor([[True, False, True]], device="cpu")
    additive = torch.zeros(1, 3, dtype=torch.float64, device="cpu").masked_fill(~allowed, -torch.inf)

    def sdpa(mask, p=0.0):
        return F.scaled_dot_product_attention(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1),
                                             attn_mask=mask, dropout_p=p, is_causal=False)

    result = {"scope": __doc__, "python": platform.python_version(), "torch": torch.__version__,
              "torch_git": torch.version.git_version, "dtype": "float64", "backend": "CPU MATH",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "source_hashes": {name: hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
                                for name, fn in (("MHA.forward", nn.MultiheadAttention.forward),
                                                 ("multi_head_attention_forward", F.multi_head_attention_forward),
                                                 ("_canonical_mask", F._canonical_mask))}}

    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        mha = identity_mha().eval()
        polarity = {
            "sdpa_bool_allowed": sdpa(allowed),
            "mha_same_bool": mha(q, k, v, attn_mask=allowed)[0],
            "mha_inverted_bool": mha(q, k, v, attn_mask=~allowed)[0],
            "sdpa_additive": sdpa(additive),
            "mha_additive": mha(q, k, v, attn_mask=additive)[0],
        }
        assert polarity["mha_same_bool"].item() == 2.0
        assert all(value.item() == 4.0 for key, value in polarity.items() if key != "mha_same_bool")
        result["polarity"] = {key: describe(value) for key, value in polarity.items()}

        blocked = torch.ones_like(allowed)
        rows = {
            "sdpa_all_false": sdpa(~blocked),
            "mha_all_true_need_weights_true": mha(q, k, v, attn_mask=blocked, need_weights=True)[0],
            "mha_all_true_need_weights_false": mha(q, k, v, attn_mask=blocked, need_weights=False)[0],
        }
        assert rows["sdpa_all_false"].item() == 0.0
        assert rows["mha_all_true_need_weights_false"].item() == 0.0
        assert torch.isnan(rows["mha_all_true_need_weights_true"]).all()
        result["fully_blocked"] = {key: describe(value) for key, value in rows.items()}

        class FunctionalWrapper(nn.Module):
            def __init__(self, respect_training):
                super().__init__()
                self.respect_training = respect_training

            def forward(self):
                p = 1.0 if not self.respect_training or self.training else 0.0
                return sdpa(allowed, p=p)

        # p=1 is a deterministic endpoint, not a stochastic RNG/seed comparison.
        dropout_layer = identity_mha(dropout=1.0)
        dropout = {
            "functional_wrapper_eval_hardcoded_p1": FunctionalWrapper(False).eval()(),
            "functional_wrapper_eval_gated_p": FunctionalWrapper(True).eval()(),
            "mha_eval_p1": dropout_layer.eval()(q, k, v, attn_mask=~allowed, need_weights=False)[0],
            "mha_train_p1": dropout_layer.train()(q, k, v, attn_mask=~allowed, need_weights=False)[0],
        }
        assert dropout["functional_wrapper_eval_hardcoded_p1"].item() == 0.0
        assert dropout["functional_wrapper_eval_gated_p"].item() == 4.0
        assert dropout["mha_eval_p1"].item() == 4.0
        assert dropout["mha_train_p1"].item() == 0.0
        result["dropout"] = {key: describe(value) for key, value in dropout.items()}

    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
