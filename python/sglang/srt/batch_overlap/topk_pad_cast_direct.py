"""Opt-in small HCU TopK pad/cast. No host reads, synchronization or remap."""
import logging
import os
import torch
import triton
import triton.language as tl

_LOGGED = False
_log = logging.getLogger(__name__)
_MAX_ROWS = int(os.getenv('SGLANG_HCU_DIRECT_TOPK_MAX_ROWS', '65536'))


@triton.jit
def _pad_cast(IDS, VALID, OUT, N: tl.constexpr, K: tl.constexpr,
              HAS_VALID: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = i < N
    value = tl.load(IDS + i, active, other=0).to(tl.int64)
    if HAS_VALID:
        valid = tl.load(VALID).to(tl.int64)
        padded = (i // K) >= valid
        value = tl.where(padded, -1, value)
        # Match the existing torch implementation: mask the original IDs
        # in-place before an optional int32 -> int64 allocation.
        tl.store(IDS + i, -1, active & padded)
    tl.store(OUT + i, value, active)


def try_pad_cast(ids, valid, mapping, output_int64):
    global _LOGGED
    if (mapping is not None or not output_int64 or not torch.version.hip
            or not ids.is_cuda or ids.ndim != 2 or not ids.is_contiguous()
            or ids.dtype not in (torch.int32, torch.int64)
            or not 0 < ids.shape[0] <= min(_MAX_ROWS, 65536) or not 0 < ids.shape[1] <= 32):
        return None
    if valid is not None and (
        not isinstance(valid, torch.Tensor) or valid.device != ids.device
        or valid.dtype not in (torch.int32, torch.int64)
        or valid.numel() != 1 or not valid.is_contiguous()
    ):
        return None
    if ids.dtype == torch.int64:
        out = ids
        if valid is None:
            return out
    else:
        out = torch.empty(ids.shape, dtype=torch.int64, device=ids.device)
    with torch.cuda.device(ids.device):
        _pad_cast[(triton.cdiv(ids.numel(), 256),)](
            ids, valid if valid is not None else ids, out,
            ids.numel(), ids.shape[1], valid is not None, 256, num_warps=4)
    if not _LOGGED:
        _log.info("TOPK_DIRECT_PAD_CAST active device=%s shape=%s dtype=%s",
                  ids.device, tuple(ids.shape), ids.dtype)
        _LOGGED = True
    return out
