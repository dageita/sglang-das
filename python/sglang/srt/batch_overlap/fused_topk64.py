"""Experimental HIP softmax TopK -> DeepEP IDs. Explicitly opt-in."""
import logging
import torch
import triton
import triton.language as tl

_logged = False


@triton.jit
def _route(LOGITS, VALID, WEIGHTS, IDS, E: tl.constexpr, K: tl.constexpr,
           RENORM: tl.constexpr, HAS_VALID: tl.constexpr,
           BLOCK: tl.constexpr, KB: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(LOGITS + row * E + cols, cols < E, other=-float('inf')).to(tl.float32)
    p = tl.exp(x - tl.max(x, 0))
    p = p / tl.sum(p, 0)
    positions = tl.arange(0, KB)
    selected = tl.full((KB,), 0, tl.float32)
    selected_ids = tl.full((KB,), -1, tl.int64)
    for j in range(K):
        best = tl.max(p, 0)
        # Deterministic low-index tie break; validated against the deployed AOT.
        index = tl.min(tl.where((p == best) & (cols < E), cols, 2147483647), 0)
        selected = tl.where(positions == j, best, selected)
        selected_ids = tl.where(positions == j, index.to(tl.int64), selected_ids)
        p = tl.where(cols == index, -float('inf'), p)
    if RENORM:
        selected = selected / tl.sum(selected, 0)
    if HAS_VALID:
        valid = tl.load(VALID).to(tl.int64)
        selected_ids = tl.where(row >= valid, -1, selected_ids)
    tl.store(WEIGHTS + row * K + positions, selected, positions < K)
    tl.store(IDS + row * K + positions, selected_ids, positions < K)


def route(logits, valid, k, renormalize):
    global _logged
    if (not torch.version.hip or not logits.is_cuda or logits.ndim != 2
            or not logits.is_contiguous()
            or logits.dtype not in (torch.float32, torch.bfloat16, torch.float16)
            or not 0 < logits.shape[0] <= 65536
            or logits.shape[1] != 256 or k != 8):
        return None
    if valid is not None and (not isinstance(valid, torch.Tensor)
            or valid.device != logits.device or valid.numel() != 1
            or valid.dtype not in (torch.int32, torch.int64) or not valid.is_contiguous()):
        return None
    weights = torch.empty((logits.shape[0],k),device=logits.device,dtype=torch.float32)
    ids = torch.empty((logits.shape[0],k),device=logits.device,dtype=torch.int64)
    with torch.cuda.device(logits.device):
        _route[(logits.shape[0],)](logits, valid if valid is not None else ids,
            weights,ids,256,k,renormalize,valid is not None,256,8,num_warps=4)
    if not _logged:
        logging.getLogger(__name__).info('FUSED_TOPK64 active shape=%s dtype=%s',
                                       tuple(logits.shape), logits.dtype)
        _logged = True
    return weights, ids
