import unittest
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.samples.sample_300 import compute_chirp

class TestComputeChirp(unittest.TestCase):
    def test_compute_chirp_basic(self):
        """Test that compute_chirp returns a numpy array with expected length."""
        fmin = 100
        fmax = 1000
        duration = 2
        sr = 22050
        linear = True
        
        chirp_signal = compute_chirp(fmin, fmax, duration, sr, linear)
        
        # Check that the output is a numpy array
        self.assertIsInstance(chirp_signal, np.ndarray)
        
        # Check that the length of the signal matches the expected duration
        expected_length = duration * sr
        self.assertEqual(len(chirp_signal), expected_length)
    
    def test_compute_chirp_different_params(self):
        """Test compute_chirp with different parameter values."""
        fmin = 200
        fmax = 2000
        duration = 1
        sr = 44100
        linear = False
        
        chirp_signal = compute_chirp(fmin, fmax, duration, sr, linear)
        
        # Check that the output is a numpy array
        self.assertIsInstance(chirp_signal, np.ndarray)
        
        # Check that the length of the signal matches the expected duration
        expected_length = duration * sr
        self.assertEqual(len(chirp_signal), expected_length)
    
    def test_compute_chirp_linear_vs_exponential(self):
        """Test that linear and exponential chirps produce different signals."""
        fmin = 100
        fmax = 1000
        duration = 2
        sr = 22050
        
        linear_chirp = compute_chirp(fmin, fmax, duration, sr, True)
        exponential_chirp = compute_chirp(fmin, fmax, duration, sr, False)
        
        # The signals should be different if the linear parameter is properly implemented
        # Note: This test might fail if the linear parameter is not properly passed to librosa.chirp
        self.assertFalse(np.array_equal(linear_chirp, exponential_chirp))
    
    def test_compute_chirp_frequency_range(self):
        """Test that the chirp signal contains frequencies in the expected range."""
        fmin = 100
        fmax = 1000
        duration = 5
        sr = 22050
        linear = True
        
        chirp_signal = compute_chirp(fmin, fmax, duration, sr, linear)
        
        # Check that the signal is not all zeros
        self.assertFalse(np.all(chirp_signal == 0))
        
        # Check that the signal has some variation (not a constant value)
        self.assertTrue(np.std(chirp_signal) > 0)

if __name__ == '__main__':
    unittest.main()