"""Compare experimental routing against the installed HIP AOT, not torch.topk."""
import json
import torch
from sgl_kernel import topk_softmax
from fused_topk64 import route

torch.manual_seed(20261020)
total = 0
max_error = 0.0
with torch.inference_mode():
    for n in (1, 7, 32, 128, 512, 1024):
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for kind in ('random', 'zeros', 'ties', 'peaked'):
                logits = torch.randn((n,256),device='cuda',dtype=dtype)
                if kind == 'zeros': logits.zero_()
                if kind == 'ties': logits.copy_(torch.arange(256,device='cuda').div(4,rounding_mode='floor').to(dtype))
                if kind == 'peaked': logits.mul_(10)
                for renorm in (False, True):
                    w = torch.empty((n,8),device='cuda',dtype=torch.float32)
                    ids = torch.empty((n,8),device='cuda',dtype=torch.int32)
                    topk_softmax(w,ids,logits,renorm)
                    for count in (None, 0, n//2, n):
                        valid = None if count is None else torch.tensor([count],device='cuda',dtype=torch.int32)
                        expected = ids.to(torch.int64)
                        if count is not None: expected[count:] = -1
                        fw, fi = route(logits,valid,8,renorm)
                        if not torch.equal(fi,expected):
                            raise AssertionError((n,str(dtype),kind,renorm,count,
                                fi[0].tolist(),expected[0].tolist()))
                        assert torch.allclose(fw,w,rtol=2e-5,atol=2e-6), (
                            n,str(dtype),kind,renorm,float((fw-w).abs().max()))
                        max_error = max(max_error,float((fw-w).abs().max()))
                        total += 1
print(json.dumps(dict(passed=True,cases=total,max_weight_abs=max_error)))
