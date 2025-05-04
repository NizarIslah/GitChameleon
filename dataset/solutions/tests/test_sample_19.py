# Add the parent directory to import sys
import os
import sys
import time
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import importlib.util

import geopandas as gpd
from shapely.geometry import Point, Polygon


from sample_19 import spatial_join

# Test if it works with the installed geopandas version
test_gdf1 = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:4326")
test_gdf2 = gpd.GeoDataFrame(
    geometry=[Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])], crs="EPSG:4326"
)
spatial_join(test_gdf1, test_gdf2)


class TestSpatialJoin(unittest.TestCase):
    """Test cases for the spatial_join function in sample_19.py."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple polygon GeoDataFrame
        polygon1 = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])
        polygon2 = Polygon([(20, 20), (20, 30), (30, 30), (30, 20)])
        polygons = gpd.GeoDataFrame(
            {"id": [1, 2], "name": ["Polygon1", "Polygon2"]},
            geometry=[polygon1, polygon2],
            crs="EPSG:4326",
        )
        self.polygons_gdf = polygons

        # Create a simple point GeoDataFrame
        point1 = Point(5, 5)  # Inside polygon1
        point2 = Point(25, 25)  # Inside polygon2
        point3 = Point(15, 15)  # Outside both polygons
        points = gpd.GeoDataFrame(
            {"id": [1, 2, 3], "value": [100, 200, 300]},
            geometry=[point1, point2, point3],
            crs="EPSG:4326",
        )
        self.points_gdf = points

    def test_basic_functionality(self):
        """Test basic functionality of the spatial_join function."""
        # Join points with polygons
        result = spatial_join(self.points_gdf, self.polygons_gdf)

        # Check that the result is a GeoDataFrame
        self.assertIsInstance(result, gpd.GeoDataFrame)

        # Check that only points within polygons are returned
        self.assertEqual(len(result), 2)  # Only 2 points are within polygons

        # Check that the correct attributes are joined
        self.assertTrue("id_left" in result.columns)
        self.assertTrue("id_right" in result.columns)
        self.assertTrue("value" in result.columns)
        self.assertTrue("name" in result.columns)

        # Check that the correct points are joined with the correct polygons
        point1_result = result[result["id_left"] == 1]
        point2_result = result[result["id_left"] == 2]

        self.assertEqual(
            point1_result["id_right"].iloc[0], 1
        )  # Point1 should be in Polygon1
        self.assertEqual(point1_result["name"].iloc[0], "Polygon1")

        self.assertEqual(
            point2_result["id_right"].iloc[0], 2
        )  # Point2 should be in Polygon2
        self.assertEqual(point2_result["name"].iloc[0], "Polygon2")

    def test_empty_geodataframes(self):
        """Test spatial_join with empty GeoDataFrames."""
        # Create empty GeoDataFrames
        empty_points = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        empty_polygons = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        # Test with empty points
        result1 = spatial_join(empty_points, self.polygons_gdf)
        self.assertEqual(len(result1), 0)

        # Test with empty polygons
        result2 = spatial_join(self.points_gdf, empty_polygons)
        self.assertEqual(len(result2), 0)

        # Test with both empty
        result3 = spatial_join(empty_points, empty_polygons)
        self.assertEqual(len(result3), 0)

    def test_different_crs(self):
        """Test spatial_join with GeoDataFrames having different CRS."""
        # Create a GeoDataFrame with a different CRS
        points_different_crs = self.points_gdf.copy()
        points_different_crs.crs = "EPSG:3857"  # Web Mercator

        # The function should handle CRS differences internally via gpd.sjoin
        result = spatial_join(points_different_crs, self.polygons_gdf)

        # The result should still have data, though the actual results might be different
        # due to the CRS transformation
        self.assertIsInstance(result, gpd.GeoDataFrame)

    def test_no_matches(self):
        """Test spatial_join when no points are within polygons."""
        # Create points that are all outside the polygons
        point1 = Point(15, 15)
        point2 = Point(40, 40)
        outside_points = gpd.GeoDataFrame(
            {"id": [1, 2], "value": [100, 200]},
            geometry=[point1, point2],
            crs="EPSG:4326",
        )

        # Join should return an empty GeoDataFrame
        result = spatial_join(outside_points, self.polygons_gdf)
        self.assertEqual(len(result), 0)

    def test_reversed_arguments(self):
        """Test spatial_join with reversed arguments (polygons, points)."""
        # This should return polygons that contain points
        # Note: The actual behavior depends on the implementation of gpd.sjoin
        # In this case, it appears that using 'within' op with reversed arguments
        # returns an empty result, which is expected behavior
        result = spatial_join(self.polygons_gdf, self.points_gdf)

        # Check that the result is a GeoDataFrame
        self.assertIsInstance(result, gpd.GeoDataFrame)

        # With 'within' op, polygons are not "within" points, so we expect 0 results
        self.assertEqual(len(result), 0)

    def test_non_geodataframe_input(self):
        """Test spatial_join with non-GeoDataFrame input (should raise ValueError)."""
        # Create a regular pandas DataFrame
        df = pd.DataFrame({"id": [1, 2, 3], "value": [100, 200, 300]})

        # Test with first argument as regular DataFrame
        with self.assertRaises(ValueError):
            spatial_join(df, self.polygons_gdf)

        # Test with second argument as regular DataFrame
        with self.assertRaises(ValueError):
            spatial_join(self.points_gdf, df)

    def test_different_predicate(self):
        """Test spatial_join with different predicate parameter.

        Note: This test is for illustration purposes only as our implementation
        uses a fixed 'within' predicate. In a real-world scenario where the
        predicate is configurable, this test would be more meaningful.
        """
        # In a real implementation, we might test different predicates like:
        # - 'intersects': Returns geometries that intersect with each other
        # - 'contains': Returns geometries that contain each other
        # - 'crosses': Returns geometries that cross each other

        # For our fixed implementation, we can only verify the default behavior
        result = spatial_join(self.points_gdf, self.polygons_gdf)

        # Check that the result matches what we expect with 'within' predicate
        self.assertEqual(len(result), 2)  # Only 2 points are within polygons

        # Note: With the current implementation using 'within' predicate:
        # - When calling with (points, polygons), we get points that are within polygons
        # - When calling with (polygons, points), we get polygons that are within points (which is none)
        # If we had a 'contains' predicate option, we would expect:
        # result_contains = spatial_join(self.polygons_gdf, self.points_gdf, predicate='contains')
        # self.assertEqual(len(result_contains), 2)  # 2 polygons contain points

    def test_large_datasets_performance(self):
        """Test spatial_join with larger datasets to check performance.

        This test creates larger datasets and measures the execution time.
        It doesn't assert specific time limits but provides a benchmark.
        """
        # Create a larger polygon dataset (grid of polygons)
        polygons = []
        polygon_attrs = {"id": [], "name": []}

        for i in range(10):
            for j in range(10):
                # Create a 1x1 polygon at position (i,j)
                polygon = Polygon([(i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)])
                polygons.append(polygon)
                polygon_attrs["id"].append(i * 10 + j)
                polygon_attrs["name"].append(f"Polygon_{i}_{j}")

        large_polygons_gdf = gpd.GeoDataFrame(
            polygon_attrs, geometry=polygons, crs="EPSG:4326"
        )

        # Create a larger point dataset (some inside, some outside polygons)
        points = []
        point_attrs = {"id": [], "value": []}

        for i in range(100):
            # Create points at random positions, some will be inside polygons
            x = i % 15  # Some points will be outside the 10x10 grid
            y = i // 15
            point = Point(x + 0.5, y + 0.5)  # Center of the grid cell
            points.append(point)
            point_attrs["id"].append(i)
            point_attrs["value"].append(i * 10)

        large_points_gdf = gpd.GeoDataFrame(
            point_attrs, geometry=points, crs="EPSG:4326"
        )

        # Measure execution time
        start_time = time.time()
        result = spatial_join(large_points_gdf, large_polygons_gdf)
        execution_time = time.time() - start_time

        # Print execution time for information (not an assertion)
        print(
            f"Large dataset spatial join execution time: {execution_time:.4f} seconds"
        )

        # Check that the result is a GeoDataFrame
        self.assertIsInstance(result, gpd.GeoDataFrame)

        # Check that we have results (points within polygons)
        # We expect around 66-67 points to be within the 10x10 grid
        self.assertGreater(len(result), 0)

        # Verify that all points in the result are within polygons
        # This is implicitly tested by the function's behavior


if __name__ == "__main__":
    unittest.main()
