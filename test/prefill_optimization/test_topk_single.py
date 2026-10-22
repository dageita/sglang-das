"""GPU differential test of select_experts, including the real compiled mask.

Only the gate is injected with fixed IDs/weights to isolate postprocessing;
this is not a model-accuracy test. No distributed/model processes are started.
"""
import json
from types import SimpleNamespace
import torch
import sglang.srt.layers.moe.topk as m
# Exercise all shape/dtype combinations without falling back after eight
# compiler variants. Test-only; production compiler settings are unchanged.
torch._dynamo.config.recompile_limit = 64

class Backend:
    def __getattr__(self, key):
        return lambda: key in ('is_deepep', 'is_deep_gemm')

m.get_moe_a2a_backend = Backend
m.get_moe_runner_backend = Backend
m.get_global_expert_distribution_recorder = lambda: SimpleNamespace(
    on_select_experts=lambda **kw: None)
m.get_global_experts_capturer = lambda: None
assert m._is_hcu and m._is_hip and not m._is_cuda
cfg = m.TopKConfig(top_k=8)
calls = []
original = m._topk_ids_postprocess
def counted(*args, **kwargs):
    calls.append(kwargs.get('output_int64', args[3] if len(args) > 3 else False))
    return original(*args, **kwargs)
m._topk_ids_postprocess = counted
total = 0
with torch.inference_mode():
    for n in (1, 7, 32, 128):
        for dtype in (torch.int32, torch.int64):
            ids = torch.arange(n*8, device='cuda').reshape(n,8).remainder(256).to(dtype)
            weights = torch.full((n,8), 0.125, device='cuda')
            m.fused_topk = lambda **kw: (weights.clone(), ids.clone())
            x = torch.zeros((n,3072),device='cuda',dtype=torch.bfloat16)
            logits = torch.zeros((n,256),device='cuda')
            for pad in (None, 0, max(0,n-1), n):
                count = None if pad is None else torch.tensor([pad],device='cuda',dtype=torch.int32)
                got = []
                for enabled in (False, True):
                    m._single_hcu_topk_postprocess = enabled
                    calls.clear()
                    out = m.select_experts(x,logits,cfg,layer_id=0,num_token_non_padded=count)
                    got.append((out.topk_ids.clone(),out.topk_weights.clone(),len(calls)))
                assert torch.equal(got[0][0],got[1][0])
                assert torch.equal(got[0][1],got[1][1])
                assert got[1][0].dtype == torch.int64
                assert got[1][2] == 1
                if pad is not None:
                    assert got[0][2] == 2
                    assert (got[1][0][pad:] == -1).all()
                total += 1
    # A routing observer must retain the old early-padding semantics.
    captures = []
    m.get_global_experts_capturer = lambda: SimpleNamespace(
        capture=lambda **kw: captures.append(kw['topk_indices'].clone()))
    m._single_hcu_topk_postprocess = True
    calls.clear()
    count = torch.tensor([n-1],device='cuda',dtype=torch.int32)
    m.select_experts(x,logits,cfg,layer_id=0,num_token_non_padded=count)
    assert len(calls) == 2 and captures and (captures[0][n-1:] == -1).all()
print(json.dumps(dict(passed=True,cases=total,capture_fallback=True)))
