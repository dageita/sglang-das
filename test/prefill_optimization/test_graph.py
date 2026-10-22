"""Standalone single-GPU replay correctness, no model/server or communication."""
import argparse
import torch
from prefill_router_shared_graph import RouterSharedGraph

p = argparse.ArgumentParser()
p.add_argument('--device', type=int, default=0)
a = p.parse_args()
torch.cuda.set_device(a.device)
assert torch.version.hip, 'This test targets HIP'
torch.manual_seed(42)
gate = torch.nn.Linear(128, 32, bias=False, device='cuda', dtype=torch.bfloat16)
up = torch.nn.Linear(128, 256, bias=False, device='cuda', dtype=torch.bfloat16)
down = torch.nn.Linear(128, 128, bias=False, device='cuda', dtype=torch.bfloat16)
shared_gate = torch.nn.Linear(128, 1, bias=False, device='cuda', dtype=torch.bfloat16)

def fn(x):
    u,v = up(x).chunk(2, dim=-1)
    return gate(x), down(torch.nn.functional.silu(u)*v)*torch.sigmoid(shared_gate(x))

with torch.inference_mode():
    state = RouterSharedGraph(target_tokens=32, layer_id='test')
    assert state.run(torch.randn(1,128,device='cuda',dtype=torch.bfloat16),fn) is None
    assert state.key is None and state.graph is None
    for i in range(5):
        x = torch.randn(32,128,device='cuda',dtype=torch.bfloat16)
        before = x.clone()
        expected = fn(x)
        actual = state.run(x, fn)
        for aa,ee in zip(actual,expected):torch.testing.assert_close(aa,ee,rtol=0,atol=0)
        torch.testing.assert_close(x,before,rtol=0,atol=0)
    assert state.hits == 5
    saved_graph = state.graph
    assert state.run(torch.randn(1,128,device='cuda',dtype=torch.bfloat16),fn) is None
    assert state.graph is saved_graph
    assert state.run(torch.randn(27,128,device='cuda',dtype=torch.bfloat16),fn) is None
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        assert state.run(torch.randn(32,128,device='cuda',dtype=torch.bfloat16),fn) is None
print('PASS: startup1 skipped, target32 changing-input replay, unchanged input, shape/stream fallback. Not full-model validation.')
