import sys
import os
import unittest
import numpy as np
from unittest.mock import Mock

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sample_81
from sample_81 import predict_start


class TestPredictStart(unittest.TestCase):
    def setUp(self):
        self.mock_model = Mock()
        # Default prediction
        self.mock_model.predict = Mock(return_value=np.array([10, 20, 30]))
        self.test_data = np.array([1, 2, 3])

    def test_predict_start_calls_model_predict_with_correct_params(self):
        """Test that predict_start calls model.predict with the correct parameters."""
        result = predict_start(self.mock_model, self.test_data)
        self.mock_model.predict.assert_called_once_with(self.test_data)
        np.testing.assert_array_equal(result, np.array([10, 20, 30]))

    def test_predict_start_returns_correct_predictions(self):
        """Test that predict_start returns the correct predictions on successive calls."""
        preds1 = np.array([1, 2, 3])
        preds2 = np.array([4, 5, 6])
        self.mock_model.predict.side_effect = [preds1, preds2]

        r1 = predict_start(self.mock_model, self.test_data)
        r2 = predict_start(self.mock_model, self.test_data)

        np.testing.assert_array_equal(r1, preds1)
        np.testing.assert_array_equal(r2, preds2)

    def test_predict_start_with_empty_data(self):
        """Test predict_start with empty data returns empty array."""
        empty = np.array([])
        self.mock_model.predict.return_value = empty

        result = predict_start(self.mock_model, empty)
        np.testing.assert_array_equal(result, empty)


if __name__ == "__main__":
    unittest.main()