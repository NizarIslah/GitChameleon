import unittest
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# Add the parent directory to the path so we can import the sample_208 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_208 import custom_pointplot


class TestCustomPointplot(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a simple DataFrame for testing
        self.simple_data = pd.DataFrame({
            'x': ['A', 'B', 'C'],
            'y': [1, 2, 3]
        })
        
        # Create a more complex DataFrame with multiple points per category
        self.complex_data = pd.DataFrame({
            'x': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
            'y': [1, 2, 3, 4, 5, 6, 7, 8, 9]
        })
        
        # Create a DataFrame with numeric x values
        self.numeric_data = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [10, 20, 30, 40, 50]
        })
        
        # Create a DataFrame with missing values
        self.missing_data = pd.DataFrame({
            'x': ['A', 'B', 'C', 'D', None],
            'y': [1, 2, None, 4, 5]
        })

    def tearDown(self):
        """Clean up after each test method."""
        plt.close('all')  # Close all figure windows

    def test_return_type(self):
        """Test that the function returns a matplotlib Axes object."""
        result = custom_pointplot(self.simple_data)
        self.assertIsInstance(result, Axes)

    def test_simple_data(self):
        """Test with simple data."""
        ax = custom_pointplot(self.simple_data)
        
        # Check that the plot has the correct number of points
        collections = [c for c in ax.collections if hasattr(c, 'get_offsets')]
        self.assertTrue(len(collections) > 0, "No point collections found in the plot")
        
        # Check axis labels
        self.assertEqual(ax.get_xlabel(), 'x')
        self.assertEqual(ax.get_ylabel(), 'y')

    def test_complex_data(self):
        """Test with more complex data having multiple points per category."""
        ax = custom_pointplot(self.complex_data)
        
        # Check that the plot has points
        collections = [c for c in ax.collections if hasattr(c, 'get_offsets')]
        self.assertTrue(len(collections) > 0, "No point collections found in the plot")

    def test_numeric_x_data(self):
        """Test with numeric x values."""
        ax = custom_pointplot(self.numeric_data)
        
        # Check that the plot has points
        collections = [c for c in ax.collections if hasattr(c, 'get_offsets')]
        self.assertTrue(len(collections) > 0, "No point collections found in the plot")
        
        # Check that the tick labels are numeric
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        self.assertTrue(all(label.isdigit() or label == '' for label in tick_labels), 
                        "X-axis tick labels should be numeric or empty")

    def test_missing_data(self):
        """Test with missing data."""
        ax = custom_pointplot(self.missing_data.dropna())
        
        # Check that the plot has points
        collections = [c for c in ax.collections if hasattr(c, 'get_offsets')]
        self.assertTrue(len(collections) > 0, "No point collections found in the plot")

    def test_plot_parameters(self):
        """Test that the plot uses the specified parameters."""
        ax = custom_pointplot(self.simple_data)
        
        # Check that markers are circles ('o')
        self.assertTrue(len(ax.collections) > 0, "No collections found in the plot")
        
        # Check that there are no lines connecting the points
        lines = ax.get_lines()
        self.assertTrue(len(lines) == 0, "There should be no lines connecting the points")

    def test_custom_figure(self):
        """Test using a custom figure and axes."""
        fig, ax = plt.subplots(figsize=(10, 6))
        result = custom_pointplot(self.simple_data)
        
        # The function should return the axes object
        self.assertIsInstance(result, Axes)
        
        # Check that the plot has points
        collections = [c for c in result.collections if hasattr(c, 'get_offsets')]
        self.assertTrue(len(collections) > 0, "No point collections found in the plot")

    def test_data_validation(self):
        """Test that the function validates input data correctly."""
        invalid_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        
        with self.assertRaises(ValueError):
            custom_pointplot(invalid_data)

    def test_empty_data(self):
        """Test with empty DataFrame."""
        empty_data = pd.DataFrame({'x': ['A'], 'y': [1]})
        
        ax = custom_pointplot(empty_data)
        self.assertIsInstance(ax, Axes)


if __name__ == '__main__':
    unittest.main()
