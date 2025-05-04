# Add the parent directory to import sys
import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import geopandas as gpd
import sample_22
from shapely.geometry import Point

# Filter deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestCreateGeoseries(unittest.TestCase):
    """Test cases for the create_geoseries function in sample_22.py."""

    def test_basic_functionality_with_integers(self):
        """Test basic functionality with integer coordinates."""
        # Create lists of x and y coordinates
        x = [0, 1, 2, 3]
        y = [0, 1, 2, 3]

        # Create a GeoSeries using the function
        result = sample_22.create_geoseries(x, y)

        # Check that the result is a GeoSeries
        self.assertIsInstance(result, gpd.GeoSeries)

        # Check that the length is correct
        self.assertEqual(len(result), len(x))

        # Check that each point has the correct coordinates
        for i, point in enumerate(result):
            self.assertIsInstance(point, Point)
            self.assertEqual(point.x, x[i])
            self.assertEqual(point.y, y[i])

    def test_empty_lists_input(self):
        """Test with empty lists as input."""
        # Create empty lists
        x = []
        y = []

        # Create a GeoSeries using the function
        result = sample_22.create_geoseries(x, y)

        # Check that the result is a GeoSeries
        self.assertIsInstance(result, gpd.GeoSeries)

        # Check that the result is empty
        self.assertEqual(len(result), 0)

    def test_lists_of_different_lengths(self):
        """Test with lists of different lengths (should raise ValueError)."""
        # Create lists of different lengths
        x = [0, 1, 2]
        y = [0, 1]

        # This should raise a ValueError because the lists have different lengths
        with self.assertRaises(ValueError):
            sample_22.create_geoseries(x, y)

    def test_float_coordinates_input(self):
        """Test with float coordinates."""
        # Create lists of float coordinates
        x = [0.5, 1.5, 2.5]
        y = [0.5, 1.5, 2.5]

        # Create a GeoSeries using the function
        result = sample_22.create_geoseries(x, y)

        # Check that the result is a GeoSeries
        self.assertIsInstance(result, gpd.GeoSeries)

        # Check that each point has the correct coordinates
        for i, point in enumerate(result):
            self.assertIsInstance(point, Point)
            self.assertEqual(point.x, x[i])
            self.assertEqual(point.y, y[i])

    def test_single_point_input(self):
        """Test with a single point."""
        # Create lists with a single point
        x = [10]
        y = [20]

        # Create a GeoSeries using the function
        result = sample_22.create_geoseries(x, y)

        # Check that the result is a GeoSeries
        self.assertIsInstance(result, gpd.GeoSeries)

        # Check that the length is 1
        self.assertEqual(len(result), 1)

        # Check that the point has the correct coordinates
        self.assertEqual(result[0].x, 10)
        self.assertEqual(result[0].y, 20)

    def test_crs_specification(self):
        """Test that the function works with CRS specification."""
        # Create lists of coordinates
        x = [0, 1, 2]
        y = [0, 1, 2]

        # Create a GeoSeries using the function
        result = sample_22.create_geoseries(x, y)

        # Check that the result is a GeoSeries
        self.assertIsInstance(result, gpd.GeoSeries)

        # By default, the CRS should be None
        self.assertIsNone(result.crs)

        # We should be able to set the CRS
        result.crs = "EPSG:4326"
        self.assertEqual(result.crs, "EPSG:4326")

    def test_non_list_input_types(self):
        """Test with non-list input types (should raise appropriate exceptions)."""
        # Test with integer inputs
        with self.assertRaises(TypeError):
            sample_22.create_geoseries(1, 2)

        # Test with string inputs
        # This should raise a ValueError because it can't convert string to float
        with self.assertRaises(ValueError):
            sample_22.create_geoseries("1,2,3", "4,5,6")

        # Test with None inputs
        with self.assertRaises(TypeError):
            sample_22.create_geoseries(None, None)


if __name__ == "__main__":
    unittest.main()
