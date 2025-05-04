import sys
import os
import numpy as np
import pytest
from unittest.mock import Mock, patch
import unittest

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_81 import predict_start
import sample_81


class TestPredictStart(unittest.TestCase):
    def setUp(self):
        self.mock_model = Mock()
        # Default prediction
        self.test_data = np.array([1, 2, 3])

    def test_predict_start_returns_correct_predictions(self):
        """Test that predict_start returns the correct predictions on successive calls."""
        preds1 = np.array([1, 2, 3])
        self.mock_model.predict.side_effect = [preds1]
        r1 = sample_81.predict_start(self.mock_model, self.test_data, 0)
        np.testing.assert_array_equal(r1, preds1)


if __name__ == "__main__":
    unittest.main()
