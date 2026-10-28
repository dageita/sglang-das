"""CPU-only control-plane checks; GPU numerical equivalence needs real weights."""
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parent
if (ROOT / 'prefill_graph_policy.py').exists():
    SOURCE = ROOT
else:
    SOURCE = ROOT.parent / 'python/sglang/srt/batch_overlap'


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, str(SOURCE / filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


policy = load('adaptive_policy_test_target', 'prefill_graph_policy.py')


class PolicyTests(unittest.TestCase):
    def test_observed_mixed_shapes_share_one_bucket(self):
        p = policy.BucketPolicy((512, 1024, 1536, 2048))
        for rows in (2009, 2012, 2020, 2033, 2048):
            self.assertEqual(p.select(rows), (2048, None))

    def test_padding_limits_and_range(self):
        p = policy.BucketPolicy((512, 2048))
        self.assertEqual(p.select(1)[1], 'outside_range')
        self.assertEqual(p.select(2049)[1], 'outside_range')
        self.assertEqual(p.select(1984)[1], 'padding_limit')
        self.assertEqual(p.select(1990), (2048, None))
        self.assertEqual(p.select(500), (512, None))

    def test_exact_only_and_bad_config(self):
        p = policy.BucketPolicy((512,), 0, 0)
        self.assertEqual(p.select(511)[1], 'padding_limit')
        self.assertEqual(p.select(512), (512, None))
        for buckets in ((), (0,), (512, 512), (2048, 512)):
            with self.assertRaises(ValueError):
                policy.BucketPolicy(buckets)


class FakeTensor:
    def __init__(self, rows, width=2):
        self.data = rows
        self.width = width
        self.dtype, self.device = 'bf16', 'cuda:0'
        self.is_cuda = True
        self.ndim = 2

    @property
    def shape(self):
        return (len(self.data), self.width)

    def stride(self):
        return (self.width, 1)

    def __getitem__(self, index):
        return FakeTensor(self.data[index], self.width)

    def copy_(self, other):
        for dest, src in zip(self.data, other.data):
            dest[:] = src

    def zero_(self):
        for row in self.data:
            row[:] = [0] * self.width

    def is_contiguous(self):
        return True

    def clone(self):
        return FakeTensor([row[:] for row in self.data], self.width)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.torch = MagicMock()
        self.torch.bfloat16 = 'bf16'
        self.torch.cuda.mem_get_info.return_value = (16 * 1024**3, 32 * 1024**3)
        self.torch.cuda.memory_allocated.return_value = 0
        self.torch.cuda.memory_reserved.return_value = 0
        self.torch.cuda.is_current_stream_capturing.return_value = False
        self.torch.is_grad_enabled.return_value = False
        self.torch.cuda.current_stream.return_value.cuda_stream = 1
        self.torch.zeros.side_effect = lambda shape, **kw: FakeTensor(
            [[0] * shape[1] for _ in range(shape[0])], shape[1])
        modules = {
            'torch': self.torch,
            'sglang.srt.distributed.utils': types.SimpleNamespace(
                get_group_rank_size=lambda group: (0, 1)),
            'sglang.srt.batch_overlap.prefill_graph_policy': policy,
            'sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph':
                types.SimpleNamespace(is_in_tc_piecewise_cuda_graph=lambda: False),
        }
        self.module_patch = patch.dict(sys.modules, modules)
        self.module_patch.start()
        with patch.dict(os.environ, {
            'SGLANG_PREFILL_ROUTER_SHARED_GRAPH': '1',
            'SGLANG_PREFILL_GRAPH_MULTI_SHAPE': '1',
            'SGLANG_PREFILL_GRAPH_ADAPTIVE': '1',
            'SGLANG_PREFILL_ROUTER_SHARED_GRAPH_MAX_TOKENS': '2048',
            'SGLANG_PREFILL_GRAPH_STATS_INTERVAL': '0',
            'SGLANG_PREFILL_GRAPH_TOKEN_BUDGET': '5120',
        }):
            self.m = load('adaptive_runtime_test_target', 'prefill_router_shared_graph.py')
        self.stats = self.m.PrefillGraphStats(0, 'cuda:0')

    def tearDown(self):
        self.m._STATS_REGISTRY.clear()
        self.module_patch.stop()

    def test_different_rows_pool_admission_frequency(self):
        cache = self.m.AdaptiveShapeCache(0, self.stats)
        for rows in (2012, 2020):
            self.assertIsNone(cache.select(FakeTensor([[1, 1]] * rows)))
        state = cache.select(FakeTensor([[1, 1]] * 2033))
        self.assertEqual(state.target_tokens, 2048)
        self.assertIs(cache.select(FakeTensor([[1, 1]] * 2015)), state)
        self.assertEqual(self.stats.fallbacks['cold_bucket'], 2)

    def test_memory_and_disabled_bucket_fallback(self):
        cache = self.m.AdaptiveShapeCache(0, self.stats)
        cache.seen[2048] = 3
        self.stats.graph_allocated_delta_bytes = self.m.MEMORY_BUDGET
        self.assertIsNone(cache.select(FakeTensor([[1, 1]] * 2033)))
        self.assertEqual(self.stats.fallbacks['device_graph_memory_budget'], 1)
        self.stats.graph_allocated_delta_bytes = 0
        state = cache.select(FakeTensor([[1, 1]] * 2033))
        state.disabled = True
        self.assertIsNone(cache.select(FakeTensor([[1, 1]] * 2020)))

    def test_persistent_buffer_tail_zeroing_and_output_slice(self):
        state = self.m.RouterSharedGraph(8, 0, self.stats, padded=True)
        state.graph = MagicMock()
        state.static_input = FakeTensor([[9, 9] for _ in range(8)])
        state.outputs = (FakeTensor([[2, 2] for _ in range(8)]),
                         FakeTensor([[3, 3] for _ in range(8)]))
        x = FakeTensor([[1, 1] for _ in range(6)])
        state.key = state.signature(x)
        outputs = state.run(x, None)
        self.assertEqual(outputs[0].shape, (6, 2))
        self.assertEqual(state.static_input.data, [[1, 1]] * 6 + [[0, 0]] * 2)
        # A shorter input must clear rows that held real tokens previously.
        state.run(FakeTensor([[4, 4] for _ in range(4)]), None)
        self.assertEqual(state.static_input.data, [[4, 4]] * 4 + [[0, 0]] * 4)
        self.assertEqual(self.stats.steady_replay_rows, 10)
        self.assertEqual(self.stats.padding_rows, 6)

    def test_signature_change_never_reuses_graph(self):
        state = self.m.RouterSharedGraph(8, 0, self.stats, padded=True)
        state.graph = MagicMock()
        state.key = state.signature(FakeTensor([[1, 1]] * 6))
        self.assertIsNone(state.run(FakeTensor([[1, 1, 1]] * 6, 3), None))
        state.graph.replay.assert_not_called()

    def test_capture_once_then_replay_with_another_real_size(self):
        state = self.m.RouterSharedGraph(8, 0, self.stats, padded=True)
        compute = MagicMock(side_effect=lambda x: (x.clone(), x.clone()))
        first = state.run(FakeTensor([[1, 1] for _ in range(6)]), compute)
        second = state.run(FakeTensor([[1, 1] for _ in range(5)]), compute)
        self.assertEqual(compute.call_count, 3)  # two warmups and one capture
        self.assertEqual(first[0].shape, (6, 2))
        self.assertEqual(second[0].shape, (5, 2))
        self.assertEqual(self.stats.capture_count, 1)
        self.assertEqual(self.stats.capture_then_replay, 1)
        self.assertEqual(self.stats.steady_replays, 1)
        self.assertEqual(self.stats.cached_shapes, {8})

    def test_pcg_context_bypasses_local_adapter(self):
        pcg = sys.modules['sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph']
        with patch.object(pcg, 'is_in_tc_piecewise_cuda_graph', return_value=True):
            self.assertIsNone(self.m.maybe_run_prefill_router_shared(None, None, None))

    def test_failed_padding_returns_eager_and_disables_bucket(self):
        owner = types.SimpleNamespace(
            layer_id=0, shared_expert=types.SimpleNamespace(
                gate_up_proj=types.SimpleNamespace(tp_group=None),
                down_proj=types.SimpleNamespace(tp_group=None, reduce_results=False)),
            gate=lambda x: (x.clone(), None),
            _forward_shared_experts=lambda x: x.clone(), top_k=1)
        batch = types.SimpleNamespace(forward_mode=types.SimpleNamespace(
            is_decode=lambda: False, is_extend=lambda: True))
        cache = self.m.AdaptiveShapeCache(0, self.stats)
        cache.seen[2048] = 3
        owner._prefill_bucket_cache = cache
        x = FakeTensor([[1, 1] for _ in range(2033)])
        state = cache.select(x)
        def bad_run(x, fn):
            state.hits = 1
            self.stats.capture_then_replay += 1
            self.stats.executed(x.shape[0], 2048)
            return (FakeTensor([[9, 9] for _ in range(2033)]), x.clone())
        owner._prefill_graph_stats = self.stats
        with patch.object(state, 'run', side_effect=bad_run), patch.object(
            self.m, 'compare_outputs', return_value=(False, False, [])):
            result = self.m.maybe_run_prefill_router_shared(owner, x, batch)
        self.assertEqual(result[0].data, x.data)
        self.assertTrue(state.disabled)
        self.assertEqual(self.stats.padding_failures, 1)
        self.assertEqual(self.stats.capture_then_replay, 0)


if __name__ == '__main__':
    unittest.main()
