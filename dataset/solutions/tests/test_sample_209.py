# Add the parent directory to import sys
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sample_209 import custom_pointplot


class TestCustomPointplot(unittest.TestCase):
    """Test cases for the custom_pointplot function in sample_209.py."""

    def setUp(self):
        """Set up test data."""
        # Create a sample DataFrame for testing
        self.test_data = pd.DataFrame({
            'x': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
            'y': [1, 2, 3, 4, 5, 6, 7, 8, 9]
        })
        
        # Close any existing plots to avoid interference
        plt.close('all')

    def tearDown(self):
        """Clean up after tests."""
        plt.close('all')

    def test_return_type(self):
        """Test that custom_pointplot returns a matplotlib Axes object."""
        result = custom_pointplot(self.test_data)
        self.assertIsInstance(result, Axes)

    def test_plot_properties(self):
        """Test that the plot has the expected properties."""
        ax = custom_pointplot(self.test_data)
        
        # Check that the plot has data
        self.assertTrue(len(ax.collections) > 0)
        
        # Check that the x-axis has the expected categories
        self.assertEqual(len(ax.get_xticks()), len(self.test_data['x'].unique()))

    def test_error_bars(self):
        """Test that error bars are present with the specified linewidth."""
        ax = custom_pointplot(self.test_data)
        
        # Find error bars in the plot
        error_bars = None
        for collection in ax.collections:
            if hasattr(collection, 'get_linewidth'):
                error_bars = collection
                break
        
        # Check that error bars exist
        self.assertIsNotNone(error_bars)

    def test_with_empty_dataframe(self):
        """Test behavior with an empty DataFrame."""
        empty_df = pd.DataFrame({'x': [], 'y': []})
        
        # Should raise ValueError because seaborn can't plot empty data
        with self.assertRaises(ValueError):
            custom_pointplot(empty_df)

    def test_with_missing_columns(self):
        """Test behavior with DataFrame missing required columns."""
        # DataFrame missing 'y' column
        df_missing_y = pd.DataFrame({'x': ['A', 'B', 'C']})
        
        # Should raise KeyError
        with self.assertRaises(KeyError):
            custom_pointplot(df_missing_y)
        
        # DataFrame missing 'x' column
        df_missing_x = pd.DataFrame({'y': [1, 2, 3]})
        
        # Should raise KeyError
        with self.assertRaises(KeyError):
            custom_pointplot(df_missing_x)

    def test_with_different_column_types(self):
        """Test with different data types for x and y columns."""
        # Numeric x and y
        numeric_df = pd.DataFrame({
            'x': [1, 2, 3, 1, 2, 3],
            'y': [4, 5, 6, 7, 8, 9]
        })
        
        result = custom_pointplot(numeric_df)
        self.assertIsInstance(result, Axes)
        
        # String x and numeric y
        mixed_df = pd.DataFrame({
            'x': ['A', 'B', 'C', 'A', 'B', 'C'],
            'y': [4, 5, 6, 7, 8, 9]
        })
        
        result = custom_pointplot(mixed_df)
        self.assertIsInstance(result, Axes)

    @patch('seaborn.pointplot')
    def test_function_parameters(self, mock_pointplot):
        """Test that the function calls seaborn.pointplot with the correct parameters."""
        # Set up the mock
        mock_axes = MagicMock(spec=Axes)
        mock_pointplot.return_value = mock_axes
        
        # Call the function
        result = custom_pointplot(self.test_data)
        
        # Check that pointplot was called with the correct parameters
        mock_pointplot.assert_called_once()
        args, kwargs = mock_pointplot.call_args
        
        self.assertEqual(kwargs['x'], 'x')
        self.assertEqual(kwargs['y'], 'y')
        self.assertEqual(kwargs['data'], self.test_data)
        self.assertEqual(kwargs['err_kws']['linewidth'], 2)


if __name__ == '__main__':
    unittest.main()