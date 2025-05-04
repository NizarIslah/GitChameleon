import json

# Add the parent directory to import sys
import os
import sys

import numpy as np
import pytest
import flask  # <-- Added this import

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_171 import MyCustomJSONHandler, app, data, eval_app
from scipy.stats import hmean


class TestSample171:
    @pytest.fixture
    def setup_app(self):
        # Configure the app for testing
        app.config["TESTING"] = True
        app.json_provider_class = MyCustomJSONHandler
        app.json = app.json_provider_class(app)
        return app

    @pytest.fixture
    def sample_data(self):
        # Create sample data for testing
        return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def test_data_route_returns_json(self, setup_app, sample_data):
        """Test that the data route returns JSON with the provided numbers."""
        with setup_app.test_request_context():
            response = data(sample_data.tolist())
            response_data = json.loads(response.get_data(as_text=True))

            assert "numbers" in response_data
            assert response_data["numbers"] == sample_data.tolist()

    def test_eval_app_function(self, setup_app, sample_data):
        """Test that eval_app correctly processes the request and returns data."""
        result = eval_app(setup_app, data, sample_data.tolist())
        response_data = json.loads(result)

        assert "numbers" in response_data
        assert response_data["numbers"] == sample_data.tolist()

    def test_custom_json_handler_with_ndarray(self, setup_app):
        """Test that the custom JSON handler correctly processes NumPy arrays."""
        test_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected_result = hmean(test_array, axis=1).tolist()

        # Create an instance of the handler
        handler = MyCustomJSONHandler(setup_app)
        result = handler.default(test_array)

        assert result == expected_result

    def test_custom_json_handler_with_non_ndarray(self, setup_app):
        """Test that the custom JSON handler correctly handles non-ndarray objects."""
        # For non-ndarray objects, it should raise TypeError (default behavior)
        handler = MyCustomJSONHandler(setup_app)

        with pytest.raises(TypeError):
            handler.default(object())

    def test_json_serialization_with_ndarray(self, setup_app):
        """Test end-to-end JSON serialization with NumPy arrays."""
        test_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected_result = hmean(test_array, axis=1).tolist()

        with setup_app.test_request_context():
            # Create a response with the array
            response = flask.jsonify(test_array)
            result = json.loads(response.get_data(as_text=True))

            assert result == expected_result

    def test_with_empty_array(self, setup_app):
        """Test handling of empty arrays."""
        empty_array = np.array([])

        with setup_app.test_request_context():
            # This should not raise an error
            response = data(empty_array.tolist())
            response_data = json.loads(response.get_data(as_text=True))

            assert "numbers" in response_data
            assert response_data["numbers"] == []

    def test_with_single_row_array(self, setup_app):
        """Test handling of single row arrays."""
        single_row = np.array([[1, 2, 3]])
        expected_result = hmean(single_row, axis=1).tolist()

        # Test the custom handler directly
        handler = MyCustomJSONHandler(setup_app)
        result = handler.default(single_row)

        assert result == expected_result
