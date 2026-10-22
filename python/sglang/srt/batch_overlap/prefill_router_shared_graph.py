"""Opt-in fixed-shape router/shared-expert graph island, HIP Prefill only.

No collectives, TopK, routed experts, norm or DeepEP calls are captured.
Single-stream inference and immutable model weights are required. Restart after
weight/LoRA changes. Returned tensors are graph-owned and consumed immediately.
"""
import logging
import os
import json
from collections import Counter

import torch

log = logging.getLogger(__name__)
ENABLED = os.getenv('SGLANG_PREFILL_ROUTER_SHARED_GRAPH', '0') == '1'
MAX_TOKENS = int(os.getenv('SGLANG_PREFILL_ROUTER_SHARED_GRAPH_MAX_TOKENS', '128'))
TARGET_TOKENS = int(os.getenv('SGLANG_PREFILL_ROUTER_SHARED_GRAPH_TARGET_TOKENS', '32'))
AUDIT = os.getenv('SGLANG_PREFILL_ISLAND_AUDIT', '0') == '1'
VALIDATE_CALLS = int(os.getenv('SGLANG_PREFILL_ISLAND_VALIDATE_CALLS', '0'))
MULTI_SHAPE = os.getenv('SGLANG_PREFILL_GRAPH_MULTI_SHAPE', '0') == '1'
MAX_SHAPES = int(os.getenv('SGLANG_PREFILL_GRAPH_MAX_SHAPES', '4'))
TOKEN_BUDGET = int(os.getenv('SGLANG_PREFILL_GRAPH_TOKEN_BUDGET', '2048'))
MIN_SEEN = int(os.getenv('SGLANG_PREFILL_GRAPH_MIN_SEEN', '3'))
MIN_TOKENS = int(os.getenv('SGLANG_PREFILL_GRAPH_MIN_TOKENS', '8'))
FREE_MARGIN = int(os.getenv('SGLANG_PREFILL_GRAPH_FREE_MB', '2048')) * 1024**2
if MULTI_SHAPE and min(MAX_SHAPES, TOKEN_BUDGET, MIN_SEEN, MIN_TOKENS, FREE_MARGIN) <= 0:
    raise ValueError('Multi-shape cache bounds must be positive')
if VALIDATE_CALLS < 0:
    raise ValueError('VALIDATE_CALLS must be nonnegative')
if ENABLED and not 0 < TARGET_TOKENS <= MAX_TOKENS:
    raise ValueError('Graph TARGET_TOKENS must be positive and <= MAX_TOKENS')


class RouterSharedGraph:
    def __init__(self, target_tokens=TARGET_TOKENS, layer_id=None):
        if target_tokens <= 0:
            raise ValueError('target_tokens must be positive')
        self.target_tokens = target_tokens
        self.layer_id = layer_id
        self.key = None
        self.graph = None
        self.hits = 0
        self.misses = 0
        self.reasons = {}
        self.by_shape = Counter()
        self.validated = 0

    def audit(self, x, action):
        self.by_shape[(tuple(x.shape), action)] += 1
        # Layer0 witnesses each model pass on each device. This is a separate
        # audit mode, not a zero-overhead performance measurement.
        if AUDIT and self.layer_id == 0:
            log.info('PREFILL_ISLAND_AUDIT %s', json.dumps(dict(
                pid=os.getpid(), layer=self.layer_id, device=str(x.device),
                shape=list(x.shape), action=action, hits=self.hits,
                misses=self.misses, count=self.by_shape[(tuple(x.shape), action)])))

    def fallback(self, reason, x):
        self.misses += 1
        self.audit(x, reason)
        self.reasons[reason] = self.reasons.get(reason, 0) + 1
        if self.reasons[reason] == 1:
            log.info('PREFILL_ROUTER_SHARED_GRAPH fallback layer=%s reason=%s '
                     'target_tokens=%s requested=%s device=%s hits=%s misses=%s',
                     self.layer_id, reason, self.target_tokens, tuple(x.shape),
                     x.device, self.hits, self.misses)
        return None

    @staticmethod
    def signature(x):
        return (tuple(x.shape), tuple(x.stride()), x.dtype, x.device,
                torch.cuda.current_stream(x.device).cuda_stream)

    def run(self, x, fn):
        # Check BEFORE allocating input/graph or binding a stream. Startup/health
        # requests must not occupy the single cache slot.
        if x.shape[0] != self.target_tokens:
            return self.fallback('non_target_tokens', x)
        key = self.signature(x)
        if self.key is not None and key != self.key:
            return self.fallback('signature_mismatch', x)
        if self.graph is None:
            self.static_input = torch.empty_like(x)
            self.static_input.copy_(x)
            current = torch.cuda.current_stream(x.device)
            # Shared per-device capture stream avoids one extra queue per layer.
            stream = capture_stream(x.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(2):
                    fn(self.static_input)
            stream.synchronize()  # Only on first eligible call; never in replay.
            self.graph = torch.cuda.CUDAGraph()
            # Unsupported vendor ops fail explicitly; do not hide capture errors.
            with torch.cuda.graph(self.graph, stream=stream):
                self.outputs = fn(self.static_input)
            current.wait_stream(stream)
            self.key = key
            log.info('PREFILL_ROUTER_SHARED_GRAPH captured shape=%s device=%s layer=%s',
                     tuple(x.shape), x.device, self.layer_id)
        self.static_input.copy_(x)
        self.graph.replay()
        self.hits += 1
        self.audit(x, 'replay')
        # Sparse logs: no per-replay logging overhead in the timed loop.
        if self.hits in (1, 4) or self.hits % 100 == 0:
            log.info('PREFILL_ROUTER_SHARED_GRAPH replay layer=%s shape=%s '
                     'device=%s hits=%s misses=%s', self.layer_id,
                     tuple(x.shape), x.device, self.hits, self.misses)
        return self.outputs


_CAPTURE_STREAMS = {}


class ExactShapeCache:
    """No padding, eviction or shared graph memory pools. Bounded per layer."""
    def __init__(self, layer_id):
        self.layer_id = layer_id
        self.states = {}
        self.seen = Counter()
        self.reasons = Counter()
        self.calls = self.replays = 0

    def select(self, x):
        self.calls += 1
        n = x.shape[0]
        state = self.states.get(n)
        if state is not None:
            self.replays += int(state.graph is not None)
        elif n < MIN_TOKENS or n > MAX_TOKENS:
            self.reasons['outside_range'] += 1
        else:
            # Bound admission history too; arbitrary shapes must not grow it.
            if len(self.seen) < 256 or n in self.seen:
                self.seen[n] += 1
            if self.seen.get(n, 0) < MIN_SEEN:
                self.reasons['cold_shape'] += 1
            elif len(self.states) >= MAX_SHAPES or sum(self.states) + n > TOKEN_BUDGET:
                self.reasons['cache_budget'] += 1
            elif torch.cuda.mem_get_info(x.device)[0] < FREE_MARGIN:
                self.reasons['free_memory_margin'] += 1
            else:
                state = self.states[n] = RouterSharedGraph(n, self.layer_id)
                log.info('PREFILL_MULTI_ADMIT layer=%s rows=%s cached=%s',
                         self.layer_id, n, sorted(self.states))
        if self.layer_id == 0 and (self.calls in (1, 32) or self.calls % 128 == 0):
            log.info('PREFILL_MULTI_COVERAGE %s', json.dumps(dict(
                device=str(x.device), calls=self.calls, replay_candidates=self.replays,
                cached_rows=sorted(self.states), reasons=dict(self.reasons),
                common_shapes=self.seen.most_common(8))))
        return state

def compare_outputs(actual, reference, k):
    """Strict diagnostic comparison; no relaxed tolerance silently hides drift."""
    if len(actual) != 2 or len(reference) != 2:
        raise ValueError('Expected router and shared outputs')
    checks = []
    snapshots = []
    for name, got, expected in zip(('router', 'shared'), actual, reference):
        if got.shape != expected.shape or got.dtype != expected.dtype:
            raise ValueError(f'{name} shape/dtype differs')
        # Reuse immutable CPU snapshots for ALL checks. Running two additional
        # GPU topk kernels tests a different operation, not graph equivalence.
        g, e = got.detach().float().cpu().clone(), expected.detach().float().cpu().clone()
        snapshots.append((g, e))
        finite = bool(torch.isfinite(g).all() and torch.isfinite(e).all())
        equal = bool(torch.equal(g, e)) and finite
        checks.append(dict(name=name, exact=equal, finite=finite,
                           max_abs=float((g-e).abs().max()),
                           mismatches=int((g != e).sum())))
    g, e = snapshots[0]
    if g.ndim != 2 or not 0 < k <= g.shape[-1]:
        raise RuntimeError('Numerical validation requires valid owner.top_k')
    # Canonical rule: descending score, lower expert index first on a tie.
    # This does not change production routing and does not validate the vendor
    # GPU TopK implementation or full SGLang padding/expert-remapping path.
    go = torch.argsort(g, dim=-1, descending=True, stable=True)
    eo = torch.argsort(e, dim=-1, descending=True, stable=True)
    gi = go[..., :k].sort(dim=-1).values
    ei = eo[..., :k].sort(dim=-1).values
    for label, values, order in (('actual', g, go), ('reference', e, eo)):
        scores = values.gather(-1, order)
        ties = int((scores[..., k-1] == scores[..., k]).sum()) if k < g.shape[-1] else 0
        checks[0][f'{label}_topk_boundary_tie_rows'] = ties
    checks[0]['topk_check'] = 'cpu_stable_score_desc_expert_index_asc'
    topk_equal = bool(torch.equal(gi, ei))
    return all(c['exact'] for c in checks) and topk_equal, topk_equal, checks


def capture_stream(device):
    if device not in _CAPTURE_STREAMS:
        _CAPTURE_STREAMS[device] = torch.cuda.Stream(device=device)
    return _CAPTURE_STREAMS[device]


def maybe_run_prefill_router_shared(owner, x, batch):
    if not ENABLED:
        return None
    mode = batch.forward_mode
    if (not torch.version.hip or not x.is_cuda or torch.is_grad_enabled()
            or not mode.is_extend() or mode.is_decode()
            or torch.cuda.is_current_stream_capturing()
            or x.ndim != 2 or x.shape[0] <= 0
            or (not MULTI_SHAPE and x.shape[0] > MAX_TOKENS)
            or x.dtype != torch.bfloat16 or not x.is_contiguous()
            or getattr(owner, 'is_nextn', False)
            or getattr(owner, 'enable_shared_expert_fusion', False)
            or owner.shared_expert is None):
        return None
    # Explicitly require the replicated, non-collective shared MLP path.
    shared = owner.shared_expert
    if (getattr(shared.gate_up_proj, 'tp_size', None) != 1
            or getattr(shared.down_proj, 'tp_size', None) != 1
            or getattr(shared.down_proj, 'reduce_results', True)
            or getattr(shared.down_proj, 'use_decode_attn_tp', False)):
        return None
    if MULTI_SHAPE:
        cache = getattr(owner, '_prefill_exact_shape_cache', None)
        if cache is None:
            cache = owner._prefill_exact_shape_cache = ExactShapeCache(
                getattr(owner, 'layer_id', None))
        state = cache.select(x)
        if state is None:
            return None
    else:
        state = getattr(owner, '_prefill_router_shared_graph', None)
        if state is None:
            state = RouterSharedGraph(layer_id=getattr(owner, 'layer_id', None))
            owner._prefill_router_shared_graph = state

    def compute(hidden):
        logits, _ = owner.gate(hidden)
        return logits, owner._forward_shared_experts(hidden)

    outputs = state.run(x, compute)
    if outputs is not None and state.validated < VALIDATE_CALLS:
        # Diagnostic only: same real weights, hidden states and vendor FP8
        # shared MLP. Clone graph outputs BEFORE eager reference can reuse any
        # vendor workspace. Host copies/synchronization are intentional here.
        actual = tuple(v.clone() for v in outputs)
        reference = tuple(v.clone() for v in compute(x))
        k = int(getattr(owner, 'top_k', 0))
        # Canonical CPU expert-set check only; real production routing unchanged.
        passed, topk_equal, checks = compare_outputs(actual, reference, k)
        log.info('PREFILL_ISLAND_NUMERICS %s', json.dumps(dict(
            pid=os.getpid(), layer=state.layer_id, device=str(x.device),
            shape=list(x.shape), call=state.validated+1,
            passed=passed, topk_set_equal=topk_equal,
            topk_check='cpu_canonical_not_production_routing', checks=checks)))
        if not passed:
            raise RuntimeError('Prefill island numerical mismatch: see PREFILL_ISLAND_NUMERICS')
        state.validated += 1
        # Return saved actual values: the eager diagnostic must not cause a
        # hidden alias of the graph output to change the downstream input.
        return actual
    return outputs
