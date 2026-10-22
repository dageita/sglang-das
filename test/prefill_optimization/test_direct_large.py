import json
import torch
from topk_pad_cast_direct import try_pad_cast

count = 0
for n in (129,257,512,1024,2048,8192,65536):
    for dtype in (torch.int32,torch.int64):
        for valid_n in (0,n//2,n):
            ids = torch.arange(n*8,device='cuda',dtype=dtype).reshape(n,8).remainder(256)
            valid = torch.tensor([valid_n],device='cuda',dtype=torch.int32)
            ref = ids.to(torch.int64).clone()
            ref[valid_n:] = -1
            out = try_pad_cast(ids,valid,None,True)
            assert torch.equal(ref,out)
            assert (ids[valid_n:] == -1).all()
            count += 1
print(json.dumps(dict(passed=True,large_shape_cases=count)))
