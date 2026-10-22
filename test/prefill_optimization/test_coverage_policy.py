"""No GPU allocation: admission, history and capacity bounds."""
from types import SimpleNamespace
import prefill_router_shared_graph as m

m.MAX_TOKENS = 2048
m.MIN_TOKENS = 8
m.MIN_SEEN = 3
m.MAX_SHAPES = 2
m.TOKEN_BUDGET = 160
m.torch.cuda.mem_get_info = lambda device: (8*1024**3,16*1024**3)
cache = m.ExactShapeCache(1)
def x(n):
    return SimpleNamespace(shape=(n,3072), device='cuda:0')
assert cache.select(x(1)) is None and not cache.states
assert cache.select(x(32)) is None
assert cache.select(x(32)) is None
s = cache.select(x(32))
assert s is not None and s.target_tokens == 32
s.graph = object()
assert cache.select(x(32)) is s
for _ in range(3): t = cache.select(x(128))
assert t is not None and set(cache.states) == {32,128}
for _ in range(4): assert cache.select(x(64)) is None
assert len(cache.states) == 2 and sum(cache.states) == 160
assert cache.select(x(4096)) is None
other = m.ExactShapeCache(1)
m.torch.cuda.mem_get_info = lambda device: (0,16*1024**3)
for _ in range(4): assert other.select(x(32)) is None
assert not other.states
for n in range(8,2000): other.select(x(n))
assert len(other.seen) <= 256
print('PASS: exact shape, hot admission, token/slot/history bounds, memory fallback')
