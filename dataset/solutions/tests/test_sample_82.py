import sys
import os
import unittest
import numpy as np
import pytest
import lightgbm as lgb

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sample_82


class TestLightGBMCV(unittest.TestCase):
    def test_dataset_creation(self):
        """Test that the Dataset is created correctly with the right dimensions."""
        # Check that X and y have the expected shapes from the constants
        X, y = sample_82.X, sample_82.y

        self.assertEqual(X.shape[0], sample_82.NUM_SAMPLES)
        self.assertEqual(X.shape[1], sample_82.NUM_FEATURES)
        self.assertEqual(len(y), sample_82.NUM_SAMPLES)

        # Check that the Dataset object is created correctly
        train_data = sample_82.train_data
        self.assertIsInstance(train_data, lgb.Dataset)

        # Instead of calling `train_data.get_data()` (which fails if the data was freed),
        # we use `num_data()` & `num_feature()` to check data dimensions.
        self.assertEqual(train_data.num_data(), X.shape[0])
        self.assertEqual(train_data.num_feature(), X.shape[1])

        # Check that the labels match in length
        y_data = train_data.get_label()
        self.assertEqual(len(y_data), len(y))


if __name__ == "__main__":
    unittest.main()
