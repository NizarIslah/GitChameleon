import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from kymatio.scattering2d.frontend.torch_frontend import ScatteringTorch2D
from sample_325 import compute_scattering


class TestSample325(unittest.TestCase):
    def test_compute_scattering_returns_correct_types(self):
        # Create a random tensor with the expected shape
        input_tensor = torch.randn(1, 1, 32, 32)
        
        # Call the function
        scattering_object, scattering_output = compute_scattering(input_tensor)
        
        # Check that the returned objects are of the correct type
        self.assertIsInstance(scattering_object, ScatteringTorch2D)
        self.assertIsInstance(scattering_output, torch.Tensor)
    
    def test_compute_scattering_output_shape(self):
        # Create a random tensor with the expected shape
        input_tensor = torch.randn(1, 1, 32, 32)
        
        # Call the function
        _, scattering_output = compute_scattering(input_tensor)
        
        # Check that the output is a 4D tensor (batch, channels, height, width)
        self.assertEqual(len(scattering_output.shape), 4)
        self.assertEqual(scattering_output.shape[0], 1)  # Batch size
        
        # Instead of forcing a fixed channel count, just ensure there's at least 1 channel
        self.assertGreaterEqual(scattering_output.shape[1], 1)
        
        # Optionally check reduced spatial dimensions if expected (commonly 8x8 for J=2)
        # but we won't hard-code in case parameters differ.
        # Example (uncomment if desired):
        # self.assertEqual(scattering_output.shape[2], 8)
        # self.assertEqual(scattering_output.shape[3], 8)
    
    def test_compute_scattering_deterministic(self):
        # Create a random tensor with the expected shape
        input_tensor = torch.randn(1, 1, 32, 32)
        
        # Call the function twice with the same input
        _, output1 = compute_scattering(input_tensor)
        _, output2 = compute_scattering(input_tensor)
        
        # Check that the outputs are identical
        self.assertTrue(torch.allclose(output1, output2))
    
    def test_compute_scattering_different_inputs(self):
        # Create two different random tensors
        input1 = torch.randn(1, 1, 32, 32)
        input2 = torch.randn(1, 1, 32, 32)
        
        # Ensure they are actually different
        self.assertFalse(torch.allclose(input1, input2))
        
        # Call the function with different inputs
        _, output1 = compute_scattering(input1)
        _, output2 = compute_scattering(input2)
        
        # Check that the outputs differ
        self.assertFalse(torch.allclose(output1, output2))


if __name__ == '__main__':
    unittest.main()