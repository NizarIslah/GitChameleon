import sys
import os
import numpy as np
import pytest
from unittest.mock import Mock

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sample_81

@pytest.fixture
def mock_model():
    model = Mock()
    # Default prediction
    model.predict = Mock(return_value=np.array([10, 20, 30]))
    return model

def test_predict_start_calls_model_predict_with_correct_params(mock_model):
    """Test that predict_start calls model.predict with the correct parameters."""
    test_data = np.array([1, 2, 3])
    result = sample_81.predict_start(mock_model, 5, test_data)
    # Ensure model.predict was called with data and start_iteration=5
    mock_model.predict.assert_called_with(test_data, start_iteration=5)
    # And that the return value is passed back
    np.testing.assert_array_equal(result, np.array([10, 20, 30]))

def test_predict_start_with_different_start_iterations(mock_model):
    """Test predict_start with start_iter = 0."""
    test_data = np.array([1, 2, 3])
    sample_81.predict_start(mock_model, 0, test_data)
    mock_model.predict.assert_called_with(test_data, start_iteration=0)

def test_predict_start_returns_correct_predictions(mock_model):
    """Test that predict_start returns the correct predictions on successive calls."""
    preds1 = np.array([1, 2, 3])
    preds2 = np.array([4, 5, 6])
    mock_model.predict.side_effect = [preds1, preds2]
    test_data = np.array([1, 2, 3])
    r1 = sample_81.predict_start(mock_model, 5, test_data)
    r2 = sample_81.predict_start(mock_model, 6, test_data)
    np.testing.assert_array_equal(r1, preds1)
    np.testing.assert_array_equal(r2, preds2)

def test_predict_start_with_empty_data(mock_model):
    """Test predict_start with empty data returns empty array."""
    empty = np.array([])
    mock_model.predict.return_value = empty
    result = sample_81.predict_start(mock_model, 5, empty)
    np.testing.assert_array_equal(result, empty)

if __name__ == "__main__":
    pytest.main()
