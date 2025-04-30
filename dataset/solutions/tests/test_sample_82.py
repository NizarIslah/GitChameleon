import sys
import os
import numpy as np
import pytest
import lightgbm as lgb
from unittest.mock import patch, MagicMock

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

        # Retrieve data and labels from the Dataset
        X_data = train_data.get_data()
        y_data = train_data.get_label()

        # Assert that they match the original arrays
        self.assertEqual(X_data.shape, X.shape)
        self.assertEqual(y_data.shape, y.shape)


if __name__ == "__main__":
    unittest.main()






