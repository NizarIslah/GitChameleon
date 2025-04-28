import unittest
import json
import numpy as np
from scipy import linalg
import flask
from dataset.solutions.sample_170 import app, data, eval, MyCustomJSONHandler


class TestSample170(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.app.testing = True
        self.client = self.app.test_client()

    def test_data_route(self):
        """Test the /data route with a simple array"""
        # Create a test array
        test_arr = [1, 2, 3, 4, 5]
        
        # Test using the eval function
        result = eval(self.app, data, test_arr)
        result_json = json.loads(result)
        
        self.assertIn('numbers', result_json)
        self.assertEqual(result_json['numbers'], test_arr)

    def test_custom_json_encoder_with_ndarray(self):
        """Test the custom JSON encoder with a 3D ndarray where last two dimensions are equal"""
        # Create a test array with shape (2, 3, 3) where each item is a 3x3 matrix
        matrices = np.zeros((2, 3, 3))
        
        # First 3x3 matrix with determinant 1
        matrices[0] = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        
        # Second 3x3 matrix with determinant -2
        matrices[1] = np.array([
            [1, 2, 3],
            [0, 1, 4],
            [5, 0, 1]
        ])
        
        # Expected determinants
        expected_dets = [1.0, -2.0]
        
        # Test the encoder directly
        encoder = MyCustomJSONHandler()
        result = encoder.default(matrices)
        
        # Check if the result is close to expected determinants
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0], expected_dets[0], places=5)
        self.assertAlmostEqual(result[1], expected_dets[1], places=5)
        
    def test_custom_json_encoder_with_other_objects(self):
        """Test the custom JSON encoder with objects it doesn't handle specially"""
        encoder = MyCustomJSONHandler()
        
        # Test with a regular list
        with self.assertRaises(TypeError):
            encoder.default([1, 2, 3])
        
        # Test with a 2D ndarray
        with self.assertRaises(TypeError):
            encoder.default(np.array([[1, 2], [3, 4]]))
        
        # Test with a 3D ndarray where last two dimensions are not equal
        with self.assertRaises(TypeError):
            encoder.default(np.zeros((2, 3, 4)))

    def test_integration_with_ndarray(self):
        """Test the integration of the route with the custom JSON encoder"""
        # Create a test array with shape (2, 3, 3)
        matrices = np.zeros((2, 3, 3))
        
        # First 3x3 matrix with determinant 1
        matrices[0] = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        
        # Second 3x3 matrix with determinant -2
        matrices[1] = np.array([
            [1, 2, 3],
            [0, 1, 4],
            [5, 0, 1]
        ])
        
        # Expected determinants
        expected_dets = [1.0, -2.0]
        
        # Test using the eval function
        result = eval(self.app, data, matrices)
        result_json = json.loads(result)
        
        self.assertIn('numbers', result_json)
        self.assertEqual(len(result_json['numbers']), 2)
        self.assertAlmostEqual(result_json['numbers'][0], expected_dets[0], places=5)
        self.assertAlmostEqual(result_json['numbers'][1], expected_dets[1], places=5)


if __name__ == '__main__':
    unittest.main()