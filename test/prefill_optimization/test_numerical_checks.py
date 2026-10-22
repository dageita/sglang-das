import unittest
import torch
from prefill_router_shared_graph import compare_outputs


class NumericalChecks(unittest.TestCase):
    def test_equal(self):
        x=(torch.tensor([[1.,2.,3.]]), torch.tensor([[5.,6.]]))
        self.assertTrue(compare_outputs(x, tuple(v.clone() for v in x), 2)[0])

    def test_shared_drift(self):
        x=(torch.tensor([[1.,2.,3.]]), torch.tensor([[5.,6.]]))
        y=(x[0], torch.tensor([[5.,7.]]))
        passed, topk, checks = compare_outputs(x,y,2)
        self.assertFalse(passed)
        self.assertTrue(topk)
        self.assertEqual(checks[1]['max_abs'],1.)

    def test_router_and_nan(self):
        x=(torch.tensor([[1.,2.,3.]]), torch.tensor([[5.,6.]]))
        y=(torch.tensor([[9.,2.,3.]]), x[1])
        self.assertFalse(compare_outputs(x,y,2)[1])
        z=(x[0], torch.tensor([[float('nan'),6.]]))
        self.assertFalse(compare_outputs(z,z,2)[0])

    def test_boundary_ties(self):
        x=(torch.tensor([[3.,2.,2.,1.], [0.,0.,0.,0.]]), torch.ones(2,5))
        passed, topk, checks=compare_outputs(x,tuple(v.clone() for v in x),2)
        self.assertTrue(passed and topk)
        self.assertEqual(checks[0]['actual_topk_boundary_tie_rows'],2)

    def test_does_not_call_gpu_topk(self):
        from unittest.mock import patch
        x=(torch.tensor([[1.,2.,3.]]),torch.ones(1,5))
        with patch.object(torch.Tensor,'topk',side_effect=AssertionError('unexpected topk')):
            self.assertTrue(compare_outputs(x,x,2)[0])

    def test_invalid_shape_dtype_k(self):
        x=(torch.tensor([[1.,2.,3.]]),torch.ones(1,5))
        with self.assertRaises(ValueError):
            compare_outputs(x,(x[0].double(),x[1]),2)
        with self.assertRaises(RuntimeError):
            compare_outputs(x,x,4)


if __name__=='__main__':
    unittest.main()
