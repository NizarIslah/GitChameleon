import unittest
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.solutions.sample_270 import custom_make_subplots

import plotly
import plotly.graph_objects as go


class TestSample270(unittest.TestCase):
    
    def test_custom_make_subplots(self):
        """Test that custom_make_subplots returns a Figure with correct rows and cols."""
        # Test with 2 rows and 3 columns
        fig = custom_make_subplots(rows=2, cols=3)
        
        # Check that the return value is a plotly Figure
        self.assertIsInstance(fig, go.Figure)
        
        # Check that the figure has the correct number of rows and columns
        # In plotly 4.0.0, we can check the _grid_ref attribute
        self.assertEqual(fig._grid_ref.shape[0], 2)  # rows
        self.assertEqual(fig._grid_ref.shape[1], 3)  # cols
        
    def test_custom_make_subplots_single(self):
        """Test custom_make_subplots with a single row and column."""
        fig = custom_make_subplots(rows=1, cols=1)
        
        # Check that the return value is a plotly Figure
        self.assertIsInstance(fig, go.Figure)
        
        # Check that the figure has the correct number of rows and columns
        self.assertEqual(fig._grid_ref.shape[0], 1)  # rows
        self.assertEqual(fig._grid_ref.shape[1], 1)  # cols


if __name__ == "__main__":
    unittest.main()