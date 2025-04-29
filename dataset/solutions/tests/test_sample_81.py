import numpy as np
import pytest
from unittest.mock import Mock

from sample_81 import predict_start

@pytest.fixture
def mock_model():
    model = Mock()
    model.predict = Mock(return_value=np.array([10, 20, 30]))
    return model

@pytest.fixture
def test_data():
    return np.array([1, 2, 3])

def test_predict_start_calls_model_predict_with_correct_params(mock_model, test_data):
    result = predict_start(mock_model, 5, test_data)
    mock_model.predict.assert_called_with(test_data, start_iteration=5)
    np.testing.assert_array_equal(result, np.array([10, 20, 30]))

def test_predict_start_with_different_start_iterations(mock_model, test_data):
    predict_start(mock_model, 0, test_data)
    mock_model.predict.assert_called_with(test_data, start_iteration=0)

def test_predict_start_returns_correct_predictions(test_data):
    mock_model = Mock()
    preds1 = np.array([1, 2, 3])
    preds2 = np.array([4, 5, 6])
    mock_model.predict.side_effect = [preds1, preds2]
    r1 = predict_start(mock_model, 5, test_data)
    r2 = predict_start(mock_model, 6, test_data)
    np.testing.assert_array_equal(r1, preds1)
    np.testing.assert_array_equal(r2, preds2)

def test_predict_start_with_empty_data(mock_model):
    empty = np.array([])
    mock_model.predict.return_value = empty
    result = predict_start(mock_model, 5, empty)
    np.testing.assert_array_equal(result, empty)
