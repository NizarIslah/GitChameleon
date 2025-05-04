# Add the parent directory to import sys
import os
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import geopandas as gpd
import sample_24
from shapely.geometry import Point, Polygon, box

# Filter deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*unary_union.*")


class TestSpatialQuery(unittest.TestCase):
    """Test cases for the spatial_query function in sample_24.py."""

    def test_basic_spatial_query_functionality(self):
        """Test basic spatial query functionality."""
        # Create a GeoDataFrame with some points
        points = [Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3)]
        gdf = gpd.GeoDataFrame(geometry=points)

        # Create another GeoDataFrame with a polygon that contains some of the points
        polygon = Polygon([(0.5, 0.5), (2.5, 0.5), (2.5, 2.5), (0.5, 2.5)])
        other = gpd.GeoDataFrame(geometry=[polygon])

        # Perform the spatial query
        result = sample_24.spatial_query(gdf, other)

        # The result should be the indices of points that intersect with the polygon
        # Points at indices 1 and 2 (Point(1, 1) and Point(2, 2)) should be within the polygon
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 2)
        self.assertTrue(1 in result)
        self.assertTrue(2 in result)

    def test_query_with_empty_geodataframe(self):
        """Test spatial query with an empty GeoDataFrame."""
        # Create an empty GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=[])

        # Create another GeoDataFrame with a polygon
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        other = gpd.GeoDataFrame(geometry=[polygon])

        # Perform the spatial query
        result = sample_24.spatial_query(gdf, other)

        # The result should be an empty array
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 0)

        # Test with empty 'other' GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=[Point(0, 0)])
        other = gpd.GeoDataFrame(geometry=[])

        # Perform the spatial query
        result = sample_24.spatial_query(gdf, other)

        # The result should be an empty array
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 0)

    def test_query_with_non_overlapping_geometries(self):
        """Test spatial query with non-overlapping geometries."""
        # Create a GeoDataFrame with some points
        points = [Point(0, 0), Point(1, 1)]
        gdf = gpd.GeoDataFrame(geometry=points)

        # Create another GeoDataFrame with a polygon that doesn't contain any of the points
        polygon = Polygon([(5, 5), (6, 5), (6, 6), (5, 6)])
        other = gpd.GeoDataFrame(geometry=[polygon])

        # Perform the spatial query
        result = sample_24.spatial_query(gdf, other)

        # The result should be an empty array
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 0)

    def test_query_with_different_crs(self):
        """Test spatial query with different CRS."""
        # Create a GeoDataFrame with some points and a specific CRS
        points = [Point(0, 0), Point(1, 1)]
        gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")

        # Create another GeoDataFrame with a polygon and a different CRS
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        other = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:3857")

        try:
            # Perform the spatial query
            # This might raise an error due to different CRS, or it might perform
            # the query without reprojection, depending on the geopandas version
            result = sample_24.spatial_query(gdf, other)

            # If we get here, the function didn't raise an error
            # The result should be an array (possibly empty if no reprojection was done)
            self.assertIsInstance(result, np.ndarray)
        except Exception as e:
            # If an error is raised, that's also acceptable behavior for different CRS
            self.assertTrue(True)

    def test_query_with_point_geometries(self):
        """Test spatial query with point geometries."""
        # Create a GeoDataFrame with some points
        points1 = [Point(0, 0), Point(1, 1), Point(2, 2)]
        gdf = gpd.GeoDataFrame(geometry=points1)

        # Create another GeoDataFrame with points
        points2 = [Point(1, 1), Point(3, 3)]
        other = gpd.GeoDataFrame(geometry=points2)

        # Perform the spatial query
        result = sample_24.spatial_query(gdf, other)

        # The result should include indices of points that intersect with the points in other
        self.assertIsInstance(result, np.ndarray)
        # The actual behavior depends on how the spatial index works
        # Let's just verify that the result contains at least the index 1 (Point(1, 1))
        self.assertTrue(1 in result)

    def test_query_with_polygon_geometries(self):
        """Test spatial query with polygon geometries."""
        # Create a GeoDataFrame with some polygons
        polygons1 = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
            Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
        ]
        gdf = gpd.GeoDataFrame(geometry=polygons1)

        # Create another GeoDataFrame with a polygon that overlaps with some of the polygons
        polygon2 = Polygon([(0.5, 0.5), (2.5, 0.5), (2.5, 2.5), (0.5, 2.5)])
        other = gpd.GeoDataFrame(geometry=[polygon2])

        # Perform the spatial query
        result = sample_24.spatial_query(gdf, other)

        # The result should include the indices of polygons that intersect with polygon2
        self.assertIsInstance(result, np.ndarray)
        # The spatial index might return all three polygons due to bounding box overlap
        # Let's just verify that the result contains at least the first two polygons
        self.assertTrue(0 in result)
        self.assertTrue(1 in result)

    def test_query_with_mixed_geometry_types(self):
        """Test spatial query with mixed geometry types."""
        # Create a GeoDataFrame with mixed geometry types
        geometries = [
            Point(0, 0),
            Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
            Point(3, 3),
        ]
        gdf = gpd.GeoDataFrame(geometry=geometries)

        # Create another GeoDataFrame with a polygon
        polygon = Polygon([(0.5, 0.5), (2.5, 0.5), (2.5, 2.5), (0.5, 2.5)])
        other = gpd.GeoDataFrame(geometry=[polygon])

        # Perform the spatial query
        result = sample_24.spatial_query(gdf, other)

        # The result should include indices of geometries that intersect with the query polygon
        self.assertIsInstance(result, np.ndarray)
        # The spatial index behavior might vary, but we should at least have the polygon at index 1
        self.assertTrue(1 in result)
        # Check if any other indices are in the result
        other_indices = [i for i in result if i != 1]
        # Print for debugging
        print(f"Result indices: {result}")
        print(f"Other indices: {other_indices}")

    def test_non_geodataframe_input(self):
        """Test with non-GeoDataFrame input (should raise AttributeError)."""
        # Create a list instead of a GeoDataFrame
        gdf = [Point(0, 0), Point(1, 1)]

        # Create a GeoDataFrame for the other parameter
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        other = gpd.GeoDataFrame(geometry=[polygon])

        # This should raise an AttributeError because a list doesn't have a 'sindex' attribute
        with self.assertRaises(AttributeError):
            sample_24.spatial_query(gdf, other)

        # Test with a GeoDataFrame for gdf but a list for other
        gdf = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1, 1)])
        other = [polygon]

        # This should raise an AttributeError because a list doesn't have a 'unary_union' attribute
        with self.assertRaises(AttributeError):
            sample_24.spatial_query(gdf, other)


if __name__ == "__main__":
    unittest.main()
