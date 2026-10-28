"""Opt-in exact/bucketed router/shared-expert graph island, HIP Prefill only.

No collectives, TopK, routed experts, norm or DeepEP calls are captured.
Single-stream inference and immutable model weights are required. Restart after
weight/LoRA changes. Returned tensors are graph-owned and consumed immediately.
"""
import logging
import os
import json
import statistics
import time
import atexit
from collections import Counter

import torch

from sglang.srt.distributed.utils import get_group_rank_size
from sglang.srt.batch_overlap.prefill_graph_policy import BucketPolicy

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
STATS_INTERVAL = int(os.getenv('SGLANG_PREFILL_GRAPH_STATS_INTERVAL', '128'))
ADAPTIVE = os.getenv('SGLANG_PREFILL_GRAPH_ADAPTIVE', '0') == '1'
BUCKETS = tuple(sorted(set(int(n) for n in os.getenv(
    'SGLANG_PREFILL_GRAPH_BUCKETS', '512,1024,1536,2048').split(','))))
POLICY = BucketPolicy(
    BUCKETS,
    int(os.getenv('SGLANG_PREFILL_GRAPH_MAX_PADDING_ROWS', '64')),
    float(os.getenv('SGLANG_PREFILL_GRAPH_MAX_PADDING_RATIO', '0.03')),
    MIN_TOKENS,
)
# A shape-dependent vendor FP8/GEMM path can change when M is padded. Check
# a bounded number of distinct padded sizes during warmup by default.
VERIFY_PADDING = os.getenv('SGLANG_PREFILL_GRAPH_VERIFY_PADDING', '1') == '1'
PADDING_VERIFY_CALLS = int(os.getenv('SGLANG_PREFILL_GRAPH_PADDING_VERIFY_CALLS', '2'))
MEMORY_BUDGET = int(os.getenv('SGLANG_PREFILL_GRAPH_MEMORY_MB', '2048')) * 1024**2
if ADAPTIVE and (not MULTI_SHAPE or BUCKETS[-1] > MAX_TOKENS):
    raise ValueError('Adaptive graph requires multi-shape and buckets <= MAX_TOKENS')
if MEMORY_BUDGET <= 0:
    raise ValueError('Graph device memory budget must be positive')
if PADDING_VERIFY_CALLS <= 0:
    raise ValueError('Padding validation count must be positive')
if MULTI_SHAPE and min(MAX_SHAPES, TOKEN_BUDGET, MIN_SEEN, MIN_TOKENS, FREE_MARGIN) <= 0:
    raise ValueError('Multi-shape cache bounds must be positive')
if VALIDATE_CALLS < 0:
    raise ValueError('VALIDATE_CALLS must be nonnegative')
if STATS_INTERVAL < 0:
    raise ValueError('STATS_INTERVAL must be nonnegative')
if ENABLED and not 0 < TARGET_TOKENS <= MAX_TOKENS:
    raise ValueError('Graph TARGET_TOKENS must be positive and <= MAX_TOKENS')


_STATS_REGISTRY = {}
_CONFIG_LOGGED = False


def _final_stats():
    for device in {device for device, _ in _STATS_REGISTRY}:
        layers = [v for (d, _), v in _STATS_REGISTRY.items() if d == device]
        if layers:
            max(layers, key=lambda v: v.layer_id).maybe_log(force=True)


atexit.register(_final_stats)


def _bounded_count(counter, key, max_keys=512):
    """Keep production diagnostics bounded even for arbitrary batch shapes."""
    if key in counter or len(counter) < max_keys:
        counter[key] += 1
    else:
        counter['overflow'] += 1


class PrefillGraphStats:
    """Low-overhead counters; no device synchronization on the replay path."""

    def __init__(self, layer_id, device):
        self.layer_id = layer_id
        self.device = str(device)
        self.total_calls = self.total_rows = 0
        self.eligible_calls = self.eligible_rows = 0
        self.capture_count = self.capture_then_replay = 0
        self.steady_replays = self.steady_replay_rows = 0
        self.capture_wall_ms = 0.0
        self.graph_memory_delta_bytes = 0
        self.graph_allocated_delta_bytes = 0
        self.graph_reserved_delta_bytes = 0
        self.padding_rows = self.graph_executed_rows = 0
        self.padding_checks = self.padding_failures = 0
        self.disabled_buckets = set()
        self.fallbacks = Counter()
        self.actual_rows = Counter()
        self.replayed_rows = Counter()
        self.cached_shapes = set()
        _STATS_REGISTRY[(self.device, self.layer_id)] = self

    def begin(self, rows):
        self.total_calls += 1
        self.total_rows += rows
        _bounded_count(self.actual_rows, rows)

    def eligible(self, rows):
        self.eligible_calls += 1
        self.eligible_rows += rows

    def fallback(self, reason):
        self.fallbacks[reason] += 1

    def captured(self, rows, wall_ms, memory_delta_bytes, allocated=0, reserved=0):
        self.capture_count += 1
        self.capture_wall_ms += wall_ms
        self.graph_memory_delta_bytes += memory_delta_bytes
        self.graph_allocated_delta_bytes += allocated
        self.graph_reserved_delta_bytes += reserved
        self.cached_shapes.add(rows)

    def captured_replayed(self):
        self.capture_then_replay += 1

    def replayed(self, rows, bucket=None):
        self.steady_replays += 1
        self.steady_replay_rows += rows
        _bounded_count(self.replayed_rows, rows)

    def executed(self, rows, bucket):
        self.padding_rows += bucket - rows
        self.graph_executed_rows += bucket

    @staticmethod
    def _rate(num, den):
        return round(num / den, 6) if den else 0.0

    def maybe_log(self, force=False):
        if not _STATS_REGISTRY or (STATS_INTERVAL == 0 and not force):
            return
        last_layer = max(v.layer_id for (d, _), v in _STATS_REGISTRY.items()
                         if d == self.device)
        if self.layer_id != last_layer:
            return
        # During the very first forward, the registry grows layer by layer.
        if not force and self.total_calls == 1 and self.layer_id != 0:
            return
        if not force and self.total_calls not in (1, 32):
            if self.total_calls % STATS_INTERVAL:
                return

        layers = [
            value for (device, _), value in _STATS_REGISTRY.items()
            if device == self.device
        ]
        eligible_calls = sum(x.eligible_calls for x in layers)
        eligible_rows = sum(x.eligible_rows for x in layers)
        steady_replays = sum(x.steady_replays for x in layers)
        steady_replay_rows = sum(x.steady_replay_rows for x in layers)
        fallbacks = Counter()
        call_rates = []
        token_rates = []
        for item in layers:
            fallbacks.update(item.fallbacks)
            if item.eligible_calls:
                call_rates.append(item.steady_replays / item.eligible_calls)
            if item.eligible_rows:
                token_rates.append(item.steady_replay_rows / item.eligible_rows)

        def distribution(values):
            if not values:
                return dict(min=0.0, median=0.0, max=0.0)
            return dict(min=round(min(values), 6),
                        median=round(statistics.median(values), 6),
                        max=round(max(values), 6))

        cached_shape_sets = Counter(
            tuple(sorted(item.cached_shapes)) for item in layers
        )
        payload = dict(
            schema_version=2,
            pid=os.getpid(),
            device=self.device,
            workload=dict(
                total_calls=self.total_calls,
                total_rows=self.total_rows,
                eligible_calls=self.eligible_calls,
                eligible_rows=self.eligible_rows,
                eligibility_rate=self._rate(self.eligible_calls, self.total_calls),
                actual_rows_top=self.actual_rows.most_common(16),
                actual_rows_histogram={str(k): v for k, v in self.actual_rows.items()},
                witness_layer=self.layer_id,
            ),
            graph_layers=dict(
                count=len(layers),
                eligible_calls=eligible_calls,
                eligible_rows=eligible_rows,
                capture_count=sum(x.capture_count for x in layers),
                capture_then_replay=sum(x.capture_then_replay for x in layers),
                steady_replays=steady_replays,
                steady_replay_rows=steady_replay_rows,
                call_hit_rate=self._rate(steady_replays, eligible_calls),
                token_hit_rate=self._rate(steady_replay_rows, eligible_rows),
                per_layer_call_hit_rate=distribution(call_rates),
                per_layer_token_hit_rate=distribution(token_rates),
                capture_wall_ms=round(sum(x.capture_wall_ms for x in layers), 3),
                graph_memory_delta_bytes=sum(
                    x.graph_memory_delta_bytes for x in layers
                ),
                graph_allocated_delta_bytes=sum(x.graph_allocated_delta_bytes for x in layers),
                graph_reserved_delta_bytes=sum(x.graph_reserved_delta_bytes for x in layers),
                padding_rows=sum(x.padding_rows for x in layers),
                graph_executed_rows=sum(x.graph_executed_rows for x in layers),
                padding_overhead_ratio=self._rate(
                    sum(x.padding_rows for x in layers),
                    sum(x.graph_executed_rows - x.padding_rows for x in layers)),
                padding_checks=sum(x.padding_checks for x in layers),
                padding_failures=sum(x.padding_failures for x in layers),
                disabled_bucket_layers=sum(len(x.disabled_buckets) for x in layers),
                fallbacks=dict(fallbacks),
                cached_shape_sets=[
                    dict(shapes=list(shapes), layers=count)
                    for shapes, count in cached_shape_sets.most_common()
                ],
                replayed_rows_top=Counter({
                    key: sum(x.replayed_rows.get(key, 0) for x in layers)
                    for key in set().union(*(x.replayed_rows for x in layers))
                }).most_common(16),
            ),
        )
        log.info('PREFILL_GRAPH_STATS %s', json.dumps(payload, sort_keys=True))


class RouterSharedGraph:
    """Callable-based segment executor; no model fields are accessed here."""

    def __init__(self, target_tokens=TARGET_TOKENS, layer_id=None, stats=None, padded=False):
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
        self.stats = stats
        self.padded = padded
        self.disabled = False
        self.verified_rows = set()

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
        if self.stats is not None:
            self.stats.fallback(reason)
        self.audit(x, reason)
        self.reasons[reason] = self.reasons.get(reason, 0) + 1
        if self.reasons[reason] == 1:
            log.info('PREFILL_ROUTER_SHARED_GRAPH fallback layer=%s reason=%s '
                     'target_tokens=%s requested=%s device=%s hits=%s misses=%s',
                     self.layer_id, reason, self.target_tokens, tuple(x.shape),
                     x.device, self.hits, self.misses)
        return None

    def signature(self, x):
        return (self.target_tokens, tuple(x.shape[1:]), tuple(x.stride()), x.dtype, x.device,
                torch.cuda.current_stream(x.device).cuda_stream)

    def run(self, x, fn):
        # Check BEFORE allocating input/graph or binding a stream. Startup/health
        # requests must not occupy the single cache slot.
        if self.disabled:
            return self.fallback('disabled_bucket', x)
        if x.shape[0] != self.target_tokens and not (self.padded and x.shape[0] < self.target_tokens):
            return self.fallback('non_target_tokens', x)
        key = self.signature(x)
        if self.key is not None and key != self.key:
            return self.fallback('signature_mismatch', x)
        was_captured = self.graph is not None
        if self.graph is None:
            free_before = torch.cuda.mem_get_info(x.device)[0]
            allocated_before = torch.cuda.memory_allocated(x.device)
            reserved_before = torch.cuda.memory_reserved(x.device)
            capture_started = time.monotonic()
            self.static_input = torch.zeros(
                (self.target_tokens, *x.shape[1:]), dtype=x.dtype, device=x.device)
            self.static_input[:x.shape[0]].copy_(x)
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
            try:
                with torch.cuda.graph(self.graph, stream=stream):
                    self.outputs = fn(self.static_input)
            except Exception:
                if self.stats is not None:
                    self.stats.fallback('capture_failure')
                raise
            current.wait_stream(stream)
            self.key = key
            if self.stats is not None:
                self.stats.captured(
                    self.target_tokens,
                    (time.monotonic() - capture_started) * 1000,
                    max(0, free_before - torch.cuda.mem_get_info(x.device)[0]),
                    max(0, torch.cuda.memory_allocated(x.device) - allocated_before),
                    max(0, torch.cuda.memory_reserved(x.device) - reserved_before),
                )
            log.info('PREFILL_ROUTER_SHARED_GRAPH captured shape=%s device=%s layer=%s',
                     tuple(self.static_input.shape), x.device, self.layer_id)
        self.static_input[:x.shape[0]].copy_(x)
        if x.shape[0] < self.target_tokens:
            self.static_input[x.shape[0]:].zero_()
        self.graph.replay()
        self.hits += 1
        if self.stats is not None:
            self.stats.executed(x.shape[0], self.target_tokens)
            if was_captured:
                self.stats.replayed(x.shape[0])
            else:
                self.stats.captured_replayed()
        self.audit(x, 'replay')
        # Sparse logs: no per-replay logging overhead in the timed loop.
        if self.hits in (1, 4) or self.hits % 100 == 0:
            log.info('PREFILL_ROUTER_SHARED_GRAPH replay layer=%s shape=%s '
                     'device=%s hits=%s misses=%s', self.layer_id,
                     tuple(x.shape), x.device, self.hits, self.misses)
        return tuple(value[:x.shape[0]] for value in self.outputs)


_CAPTURE_STREAMS = {}


class AdaptiveShapeCache:
    """Online demand capture with pooled bucket frequency and bounded padding.

    No model executes on background threads. A bucket is captured once the
    aggregate demand across its allowed real sizes reaches MIN_SEEN. Memory
    accounting is per process/device, across all layers; the byte limit is an
    admission threshold, not a hard cap on a single capture's transient peak.
    """

    def __init__(self, layer_id, stats):
        self.layer_id, self.stats = layer_id, stats
        self.states = {}
        self.seen = Counter()

    def select(self, x):
        bucket, reason = POLICY.select(x.shape[0])
        if reason:
            self.stats.fallback(reason)
            return None
        state = self.states.get(bucket)
        if state is not None:
            if state.disabled:
                self.stats.fallback('disabled_bucket')
                return None
            return state
        self.seen[bucket] += 1
        if self.seen[bucket] < MIN_SEEN:
            reason = 'cold_bucket'
        elif len(self.states) >= MAX_SHAPES or sum(self.states) + bucket > TOKEN_BUDGET:
            reason = 'cache_budget'
        else:
            layers = [v for (device, _), v in _STATS_REGISTRY.items()
                      if device == str(x.device)]
            used = sum(v.graph_allocated_delta_bytes for v in layers)
            if used >= MEMORY_BUDGET:
                reason = 'device_graph_memory_budget'
            elif torch.cuda.mem_get_info(x.device)[0] < FREE_MARGIN:
                reason = 'free_memory_margin'
            else:
                state = self.states[bucket] = RouterSharedGraph(
                    bucket, self.layer_id, self.stats, padded=True)
                log.info('PREFILL_BUCKET_ADMIT layer=%s rows=%s bucket=%s cached=%s',
                         self.layer_id, x.shape[0], bucket, sorted(self.states))
                return state
        self.stats.fallback(reason)
        return None


class ExactShapeCache:
    """No padding, eviction or shared graph memory pools. Bounded per layer."""
    def __init__(self, layer_id, stats):
        self.layer_id = layer_id
        self.states = {}
        self.seen = Counter()
        self.reasons = Counter()
        self.calls = self.replays = 0
        self.stats = stats

    def select(self, x):
        self.calls += 1
        n = x.shape[0]
        state = self.states.get(n)
        if state is not None:
            self.replays += int(state.graph is not None)
        elif n < MIN_TOKENS or n > MAX_TOKENS:
            self.reasons['outside_range'] += 1
            self.stats.fallback('outside_range')
        else:
            # Bound admission history too; arbitrary shapes must not grow it.
            if len(self.seen) < 256 or n in self.seen:
                self.seen[n] += 1
            if self.seen.get(n, 0) < MIN_SEEN:
                self.reasons['cold_shape'] += 1
                self.stats.fallback('cold_shape')
            elif len(self.states) >= MAX_SHAPES or sum(self.states) + n > TOKEN_BUDGET:
                self.reasons['cache_budget'] += 1
                self.stats.fallback('cache_budget')
            elif torch.cuda.mem_get_info(x.device)[0] < FREE_MARGIN:
                self.reasons['free_memory_margin'] += 1
                self.stats.fallback('free_memory_margin')
            else:
                state = self.states[n] = RouterSharedGraph(
                    n, self.layer_id, self.stats)
                log.info('PREFILL_MULTI_ADMIT layer=%s rows=%s cached=%s',
                         self.layer_id, n, sorted(self.states))
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


def _linear_parallel_size(linear):
    """Return the constructed linear partition size, conservatively.

    SGLang linear layers expose their resolved partition through ``tp_group``;
    they do not expose a ``tp_size`` attribute.  A replicated linear has a
    ``None`` group, which ``get_group_rank_size`` correctly maps to size one.
    Treat unknown/legacy linear implementations as unsafe for graph capture.
    """
    if not hasattr(linear, 'tp_group'):
        return None
    return get_group_rank_size(linear.tp_group)[1]


def maybe_run_prefill_router_shared(owner, x, batch):
    """Qwen2-compatible segment adapter; policy/executor are model independent."""
    if not ENABLED:
        return None
    # PCG capture/replay owns its execution region, including Dynamo tracing.
    # Do not mutate the eager island cache inside that region.
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        is_in_tc_piecewise_cuda_graph,
    )
    if is_in_tc_piecewise_cuda_graph():
        return None
    global _CONFIG_LOGGED
    if not _CONFIG_LOGGED:
        log.info('PREFILL_GRAPH_CONFIG %s', json.dumps(dict(
            pid=os.getpid(), adaptive=ADAPTIVE, multi_shape=MULTI_SHAPE,
            buckets=BUCKETS if ADAPTIVE else [], max_shapes=MAX_SHAPES,
            token_budget=TOKEN_BUDGET, min_seen=MIN_SEEN,
            min_rows=MIN_TOKENS, max_rows=MAX_TOKENS,
            max_padding_rows=POLICY.max_padding_rows,
            max_padding_ratio=POLICY.max_padding_ratio,
            memory_admission_bytes=MEMORY_BUDGET, free_margin_bytes=FREE_MARGIN,
            verify_padding=VERIFY_PADDING, padding_verify_calls=PADDING_VERIFY_CALLS,
            validate_calls=VALIDATE_CALLS, stats_interval=STATS_INTERVAL)))
        _CONFIG_LOGGED = True
    layer_id = getattr(owner, 'layer_id', None)
    stats = getattr(owner, '_prefill_graph_stats', None)
    if stats is None:
        stats = owner._prefill_graph_stats = PrefillGraphStats(layer_id, x.device)
    rows = int(x.shape[0]) if x.ndim > 0 else 0
    stats.begin(rows)

    def reject(reason):
        stats.fallback(reason)
        stats.maybe_log()
        return None

    mode = batch.forward_mode
    if not torch.version.hip or not x.is_cuda:
        return reject('unsupported_device')
    if torch.is_grad_enabled():
        return reject('grad_enabled')
    if mode.is_decode():
        return reject('decode')
    if not mode.is_extend():
        return reject('not_extend')
    if torch.cuda.is_current_stream_capturing():
        return reject('already_capturing')
    if x.ndim != 2 or rows <= 0:
        return reject('invalid_shape')
    if not MULTI_SHAPE and rows > MAX_TOKENS:
        return reject('outside_range')
    if x.dtype != torch.bfloat16:
        return reject('dtype_mismatch')
    if not x.is_contiguous():
        return reject('non_contiguous')
    if getattr(owner, 'is_nextn', False):
        return reject('nextn')
    if getattr(owner, 'enable_shared_expert_fusion', False):
        return reject('shared_expert_fusion')
    if owner.shared_expert is None:
        return reject('no_shared_expert')
    # Explicitly require the replicated, non-collective shared MLP path.
    # Linear layers do not have a tp_size attribute; inspect their resolved
    # tp_group instead.  Keep each rejection distinct so coverage reports show
    # exactly which assumption prevents capture.
    shared = owner.shared_expert
    if _linear_parallel_size(shared.gate_up_proj) != 1:
        return reject('gate_up_partitioned')
    if _linear_parallel_size(shared.down_proj) != 1:
        return reject('down_partitioned')
    if getattr(shared.down_proj, 'reduce_results', True):
        return reject('down_reduce_results')
    if getattr(shared.down_proj, 'use_decode_attn_tp', False):
        return reject('down_decode_attn_tp')
    stats.eligible(rows)
    if MULTI_SHAPE:
        attribute = '_prefill_bucket_cache' if ADAPTIVE else '_prefill_exact_shape_cache'
        cache = getattr(owner, attribute, None)
        if cache is None:
            cache = (AdaptiveShapeCache if ADAPTIVE else ExactShapeCache)(layer_id, stats)
            setattr(owner, attribute, cache)
        state = cache.select(x)
        if state is None:
            stats.maybe_log()
            return None
    else:
        state = getattr(owner, '_prefill_router_shared_graph', None)
        if state is None:
            state = RouterSharedGraph(layer_id=layer_id, stats=stats)
            owner._prefill_router_shared_graph = state

    def compute(hidden):
        logits, _ = owner.gate(hidden)
        return logits, owner._forward_shared_experts(hidden)

    outputs = state.run(x, compute)
    check_padding = (
        outputs is not None and ADAPTIVE and VERIFY_PADDING
        and x.shape[0] < state.target_tokens
        and len(state.verified_rows) < PADDING_VERIFY_CALLS
        and x.shape[0] not in state.verified_rows
    )
    if outputs is not None and (state.validated < VALIDATE_CALLS or check_padding):
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
            bucket_rows=state.target_tokens, padding_check=check_padding,
            passed=passed, topk_set_equal=topk_equal,
            topk_check='cpu_canonical_not_production_routing', checks=checks)))
        if check_padding:
            stats.padding_checks += 1
            state.verified_rows.add(x.shape[0])
        if not passed and ADAPTIVE:
            # Preserve the eager snapshots before invalidating the bucket.
            # No distributed operation is captured, so ranks may safely use
            # eager independently without changing communication ordering.
            stats.padding_failures += 1
            stats.disabled_buckets.add(state.target_tokens)
            stats.fallback('numerical_bucket_disabled')
            state.disabled = True
            if state.hits > 1:
                stats.steady_replays -= 1
                stats.steady_replay_rows -= x.shape[0]
                stats.replayed_rows[x.shape[0]] -= 1
            else:
                stats.capture_then_replay -= 1
            stats.padding_rows -= state.target_tokens - x.shape[0]
            stats.graph_executed_rows -= state.target_tokens
            log.error('PREFILL_BUCKET_DISABLED layer=%s bucket=%s rows=%s: '
                      'strict numerical check failed; returning eager result',
                      state.layer_id, state.target_tokens, x.shape[0])
            stats.maybe_log(force=True)
            return reference
        if not passed:
            raise RuntimeError('Prefill island numerical mismatch: see PREFILL_ISLAND_NUMERICS')
        state.validated += 1
        # Return saved actual values: the eager diagnostic must not cause a
        # hidden alias of the graph output to change the downstream input.
        stats.maybe_log()
        return actual
    stats.maybe_log()
    return outputs
