import unittest
import torch
import sys
import os

# Add the parent directory to the path so we can import the solution
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sample_325 import compute_scattering
from kymatio.scattering2d.frontend.torch_frontend import ScatteringTorch2D


class TestSample325(unittest.TestCase):
    def test_compute_scattering_returns_correct_types(self):
        # Create a random tensor with the expected shape
        # The Scattering2D is configured for 32x32 images
        # Adding batch dimension and channel dimension
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
        
        # Check the output shape
        # For Scattering2D with J=2, the output should have a specific shape
        # The output shape depends on the scattering parameters
        # For a 32x32 input with J=2, we expect a specific shape
        self.assertEqual(len(scattering_output.shape), 4)  # Should be a 4D tensor
        self.assertEqual(scattering_output.shape[0], 1)    # Batch size preserved
        
        # The number of channels in the output depends on the scattering parameters
        # For J=2, we expect 1 + J*L channels where L is the number of orientations (default is 8)
        # So we expect 1 + 2*8 = 17 channels
        self.assertEqual(scattering_output.shape[1], 1 + 2*8)
    
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
        
        # Check that the outputs are different
        self.assertFalse(torch.allclose(output1, output2))


if __name__ == '__main__':
    unittest.main()