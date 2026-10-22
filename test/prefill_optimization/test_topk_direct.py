"""Small GPU correctness check, no model or distributed workers."""
import json
import torch
from topk_pad_cast_direct import try_pad_cast


def reference(ids, count):
    if count is not None:
        ids[torch.arange(ids.shape[0], device=ids.device) >= count, :] = -1
    return ids.to(torch.int64)


def main():
    cases = 0
    torch.manual_seed(20261020)
    for n in (1, 7, 32, 128):
        for k in (1, 8, 32):
            for dtype in (torch.int32, torch.int64):
                for count_dtype in (torch.int32, torch.int64):
                    for count in (None, -1, 0, n-1, n, n+1):
                        x = torch.randint(-1, 256, (n,k),device='cuda',dtype=dtype)
                        if dtype == torch.int64:
                            x[0,0] = 2**40 + 123  # Do not narrow int64 IDs.
                        valid = None if count is None else torch.tensor(
                            [count],device='cuda',dtype=count_dtype)
                        a, b = x.clone(), x.clone()
                        expected = reference(a, valid)
                        actual = try_pad_cast(b, valid, None, True)
                        assert actual is not None and actual.dtype == torch.int64
                        assert torch.equal(expected, actual)
                        assert torch.equal(a, b), 'input mutation differs'
                        assert (actual.data_ptr() == b.data_ptr()) == (dtype == torch.int64)
                        cases += 1
    x = torch.ones((32,8),device='cuda',dtype=torch.int32)
    assert try_pad_cast(x,None,object(),True) is None
    assert try_pad_cast(x,None,None,False) is None
    assert try_pad_cast(x.t(),None,None,True) is None
    assert try_pad_cast(x.cpu(),None,None,True) is None
    assert try_pad_cast(x,torch.tensor([2]),None,True) is None
    assert try_pad_cast(x,torch.tensor([2,3],device='cuda'),None,True) is None
    assert try_pad_cast(x.float(),None,None,True) is None
    assert try_pad_cast(torch.ones((65537,8),device='cuda',dtype=x.dtype),None,None,True) is None
    # Capture once, then change both data and GPU-resident valid count.
    valid = torch.tensor([31],device='cuda',dtype=torch.int32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        try_pad_cast(x,valid,None,True)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):
        out = try_pad_cast(x,valid,None,True)
    torch.cuda.current_stream().wait_stream(stream)
    for count in (0, 17, 32):
        x.fill_(7)
        valid.fill_(count)
        graph.replay()
        expected = reference(torch.full_like(x,7),valid)
        assert torch.equal(out, expected)
    print(json.dumps(dict(passed=True,cases=cases,fallback_checks=8,
                          graph_replays=3,input_mutation_and_alias_checked=True)))


if __name__ == '__main__':
    main()
