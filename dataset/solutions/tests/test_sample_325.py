import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from kymatio.scattering2d.frontend.torch_frontend import ScatteringTorch2D
from sample_325 import compute_scattering


class TestSample325(unittest.TestCase):
    def test_compute_scattering_returns_correct_types(self):
        input_tensor = torch.randn(1, 1, 32, 32)
        scattering_object, scattering_output = compute_scattering(input_tensor)
        self.assertIsInstance(scattering_object, ScatteringTorch2D)
        self.assertIsInstance(scattering_output, torch.Tensor)
    
    def test_compute_scattering_output_shape(self):
        input_tensor = torch.randn(1, 1, 32, 32)
        _, scattering_output = compute_scattering(input_tensor)
        # Kymatio Scattering2D returns a 5D tensor: (batch, channel, y, x, order_plus_one)
        self.assertEqual(len(scattering_output.shape), 5)  # Should be a 5D tensor
        self.assertEqual(scattering_output.shape[0], 1)    # Batch size preserved
        # The number of channels in the output depends on the scattering parameters
        # For J=2, L=8, we expect 1 + 2*8 = 17 channels
        self.assertEqual(scattering_output.shape[1], 1 + 2*8)
    
    def test_compute_scattering_deterministic(self):
        input_tensor = torch.randn(1, 1, 32, 32)
        _, output1 = compute_scattering(input_tensor)
        _, output2 = compute_scattering(input_tensor)
        self.assertTrue(torch.allclose(output1, output2))
    
    def test_compute_scattering_different_inputs(self):
        input1 = torch.randn(1, 1, 32, 32)
        input2 = torch.randn(1, 1, 32, 32)
        self.assertFalse(torch.allclose(input1, input2))
        _, output1 = compute_scattering(input1)
        _, output2 = compute_scattering(input2)
        self.assertFalse(torch.allclose(output1, output2))


if __name__ == '__main__':
    unittest.main()
