"""CPU-only cache regression; graph capture/replay is mocked, no GPU touched."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from prefill_router_shared_graph import RouterSharedGraph


class PolicyTest(unittest.TestCase):
    def tensor(self, n):
        return SimpleNamespace(shape=(n, 3072), device='cuda:0')

    def test_startup_does_not_bind(self):
        state = RouterSharedGraph(target_tokens=32)
        for n in (1, 16, 64, 1):
            self.assertIsNone(state.run(self.tensor(n), lambda _: self.fail()))
            self.assertIsNone(state.key)
            self.assertIsNone(state.graph)
        self.assertEqual(state.misses, 4)

    def test_target_hits_and_other_requests_preserve_cache(self):
        state = RouterSharedGraph(target_tokens=32)
        replay = []
        state.key = 'target'
        state.graph = SimpleNamespace(replay=lambda: replay.append(True))
        state.static_input = SimpleNamespace(copy_=lambda _: None)
        state.outputs = ('router', 'shared')
        with patch.object(state, 'signature', return_value='target'):
            for n in (1, 32, 32, 16, 32):
                out = state.run(self.tensor(n), lambda _: self.fail())
                self.assertEqual(out, state.outputs if n == 32 else None)
        with patch.object(state, 'signature', return_value='other-stream'):
            self.assertIsNone(state.run(self.tensor(32), lambda _: self.fail()))
        self.assertEqual(len(replay), 3)
        self.assertEqual(state.hits, 3)
        self.assertEqual(state.misses, 3)
        self.assertEqual(state.key, 'target')

    def test_invalid_target(self):
        with self.assertRaises(ValueError):
            RouterSharedGraph(target_tokens=0)


if __name__ == '__main__':
    unittest.main()
