import torch


def topk(
    input,
    k,
    sorted=True,
    *,
    indices_type=torch.int32,
    end=None,
    output_idx=None,
    output_idx_offset=None,
    idx_oob_fill_value=None,
    output_value=None,
    return_value=True,
    **kw,
):
    """torch fallback for deep_select.topk (SM89 civilian port).

    Selects top-k over the last dim; supports per-row valid `end`, an index
    offset, and OOB fill. Exact numerics do not matter for the mock chain.
    """
    vals = input.to(torch.float32)
    vals = torch.where(torch.isfinite(vals), vals, torch.full_like(vals, float("-inf")))
    if end is not None:
        K = vals.shape[-1]
        arange = torch.arange(K, device=vals.device).unsqueeze(0)
        mask = arange < end.unsqueeze(-1)
        vals = torch.where(mask, vals, torch.full_like(vals, float("-inf")))
    kk = min(int(k), vals.shape[-1])
    topv, topi = torch.topk(vals, kk, dim=-1, largest=True, sorted=bool(sorted))
    if idx_oob_fill_value is not None:
        invalid = topv == float("-inf")
        topi = torch.where(invalid, torch.full_like(topi, idx_oob_fill_value), topi)
    # keep every valid index strictly below the per-row `end` (the consumer maps
    # them via block tables and would read OOB otherwise)
    if end is not None:
        lim = end.reshape(-1, 1).to(topi.dtype)
        over = (topi >= 0) & (topi >= lim)
        topi = torch.where(over, torch.zeros_like(topi), topi)
    if output_idx_offset is not None:
        off = output_idx_offset.reshape(-1)
        if off.numel() == topi.shape[0]:
            off = off.unsqueeze(-1)
        else:
            # Full per-query offset array (caller indexes by query row); the mock
            # profile run reaches the first chunk, so head rows apply.
            off = off[: topi.shape[0]].unsqueeze(-1)
        topi = topi + off
    import os as _os, sys as _sys
    if _os.environ.get("DEEPINFER_MOCK_DEBUG"):
        print(f"[dbg] topk input={tuple(input.shape)} k={k} end={None if end is None else tuple(end.shape)} "
              f"topi={tuple(topi.shape)} off={None if output_idx_offset is None else tuple(output_idx_offset.shape)} "
              f"out_idx={None if output_idx is None else tuple(output_idx.shape)}", file=_sys.stderr, flush=True)
    topi = topi.to(indices_type)
    if output_idx is not None:
        output_idx[:, :kk].copy_(topi)
        if output_idx.shape[1] > kk:
            output_idx[:, kk:].fill_(idx_oob_fill_value if idx_oob_fill_value is not None else 0)
    if output_value is not None:
        output_value.copy_(topv)
    # Callers index the return (e.g. topk(...)[1]); always return (values, indices).
    return topv, topi
