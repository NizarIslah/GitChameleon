import unittest
import lightgbm as lgb
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_86 import get_params

class TestSample86(unittest.TestCase):
    def test_get_params(self):
        # Create a simple dataset
        data = np.random.rand(100, 10)  # 100 samples, 10 features
        label = np.random.randint(0, 2, 100)  # Binary labels
        
        # Create a LightGBM dataset with some parameters
        params = {
            'max_bin': 255,
            'metric': 'binary_logloss',
            'feature_pre_filter': False
        }
        
        # Create the dataset
        lgb_dataset = lgb.Dataset(data, label, params=params)
        
        # Get the parameters using our function
        result_params = get_params(lgb_dataset)
        
        # Verify that the parameters are returned correctly
        self.assertIsInstance(result_params, dict)
        
        # Check that our specified parameters are in the result
        for key, value in params.items():
            self.assertIn(key, result_params)
            self.assertEqual(result_params[key], value)

if __name__ == '__main__':
    unittest.main()