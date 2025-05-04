import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import sample_86
import unittest
import lightgbm as lgb
import numpy as np
from sample_86 import get_params


class TestSample86(unittest.TestCase):
    def test_get_params(self):
        # Create a simple dataset
        data = np.random.rand(100, 10)  # 100 samples, 10 features
        label = np.random.randint(0, 2, 100)  # Binary labels

        # Create a LightGBM dataset with some parameters
        params = {
            "max_bin": 255,
            "metric": "binary_logloss",  # note: not stored in dataset.params
            "feature_pre_filter": False,
        }

        # Create the dataset
        lgb_dataset = lgb.Dataset(data, label, params=params)

        # Get the parameters using our function
        result_params = get_params(lgb_dataset)

        # It should be a dict
        self.assertIsInstance(result_params, dict)

        # 'metric' is not a dataset parameter and should not appear
        self.assertNotIn("metric", result_params)

        # Only compare keys that are in result_params (i.e., dataset params)
        for k, v in result_params.items():
            self.assertEqual(lgb_dataset.params.get(k), v)


if __name__ == "__main__":
    unittest.main()
