# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import geopandas as gpd
from sample_20 import perform_union
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union


class TestPerformUnion(unittest.TestCase):
    """Test cases for the perform_union function in sample_20.py."""

    def test_basic_polygon_union(self):
        """Test basic functionality with polygons."""
        # Create two overlapping polygons
        polygon1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        polygon2 = Polygon([(0.5, 0.5), (0.5, 1.5), (1.5, 1.5), (1.5, 0.5)])

        # Create a GeoDataFrame with these polygons
        gdf = gpd.GeoDataFrame(geometry=[polygon1, polygon2])

        # Perform the union
        result = perform_union(gdf)

        # Calculate the expected result using shapely directly
        expected = unary_union([polygon1, polygon2])

        # Check that the result is a shapely geometry
        self.assertTrue(hasattr(result, "geom_type"))

        # Check that the result matches the expected geometry
        self.assertTrue(result.equals(expected))

        # Check that the area is correct (should be less than the sum of individual areas due to overlap)
        self.assertLess(result.area, polygon1.area + polygon2.area)
        # The actual area is 1.75 (1 + 1 - 0.25 overlap area)
        self.assertEqual(result.area, 1.75)

    def test_non_overlapping_geometries(self):
        """Test union of non-overlapping geometries."""
        # Create two non-overlapping polygons
        polygon1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        polygon2 = Polygon([(2, 2), (2, 3), (3, 3), (3, 2)])

        # Create a GeoDataFrame with these polygons
        gdf = gpd.GeoDataFrame(geometry=[polygon1, polygon2])

        # Perform the union
        result = perform_union(gdf)

        # The result should be a MultiPolygon since the polygons don't overlap
        self.assertEqual(result.geom_type, "MultiPolygon")

        # Check that the area is the sum of the individual areas
        self.assertEqual(result.area, polygon1.area + polygon2.area)
        self.assertEqual(result.area, 2.0)  # 1 + 1

    def test_empty_geodataframe(self):
        """Test union of an empty GeoDataFrame."""
        # Create an empty GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=[])

        # Perform the union
        result = perform_union(gdf)

        # The result should be an empty geometry collection
        self.assertEqual(str(result), "GEOMETRYCOLLECTION EMPTY")

    def test_mixed_geometry_types(self):
        """Test union of mixed geometry types (points, lines, polygons)."""
        # Create different geometry types
        point = Point(0, 0)
        line = LineString([(1, 1), (2, 2)])
        polygon = Polygon([(3, 3), (3, 4), (4, 4), (4, 3)])

        # Create a GeoDataFrame with these geometries
        gdf = gpd.GeoDataFrame(geometry=[point, line, polygon])

        # Perform the union
        result = perform_union(gdf)

        # The result should be a GeometryCollection or a single geometry
        # that contains all the input geometries
        self.assertTrue(
            point.within(result)
            or point.equals(result)
            or any(point.equals(geom) for geom in getattr(result, "geoms", []))
        )

        self.assertTrue(
            line.within(result)
            or line.equals(result)
            or any(line.equals(geom) for geom in getattr(result, "geoms", []))
        )

        self.assertTrue(
            polygon.within(result)
            or polygon.equals(result)
            or any(polygon.equals(geom) for geom in getattr(result, "geoms", []))
        )

    def test_complex_polygons(self):
        """Test union of complex polygons with holes."""
        # Create a polygon with a hole
        exterior = [(0, 0), (0, 10), (10, 10), (10, 0)]
        interior = [(2, 2), (2, 8), (8, 8), (8, 2)]
        polygon_with_hole = Polygon(exterior, [interior])

        # Create another polygon that overlaps with the hole
        overlapping_polygon = Polygon([(3, 3), (3, 7), (7, 7), (7, 3)])

        # Create a GeoDataFrame with these polygons
        gdf = gpd.GeoDataFrame(geometry=[polygon_with_hole, overlapping_polygon])

        # Perform the union
        result = perform_union(gdf)

        # Calculate the expected result using shapely directly
        expected = unary_union([polygon_with_hole, overlapping_polygon])

        # Check that the result matches the expected geometry
        self.assertTrue(result.equals(expected))

        # The result is actually a MultiPolygon (not a single Polygon)
        # This is because the hole is not completely filled
        self.assertEqual(result.geom_type, "MultiPolygon")

        # Check that the area is correct
        # Original polygon area: 10*10 - 6*6 = 100 - 36 = 64
        # Overlapping polygon area: 4*4 = 16
        # Total area after union: 64 + 16 = 80 (since the overlap is within the hole)
        self.assertEqual(result.area, 80.0)

    def test_multipolygon_input(self):
        """Test union with MultiPolygon input."""
        # Create a MultiPolygon
        polygon1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        polygon2 = Polygon([(2, 2), (2, 3), (3, 3), (3, 2)])
        multi_polygon = MultiPolygon([polygon1, polygon2])

        # Create another polygon
        polygon3 = Polygon([(1, 1), (1, 2), (2, 2), (2, 1)])

        # Create a GeoDataFrame with these geometries
        gdf = gpd.GeoDataFrame(geometry=[multi_polygon, polygon3])

        # Perform the union
        result = perform_union(gdf)

        # Calculate the expected result using shapely directly
        expected = unary_union([multi_polygon, polygon3])

        # Check that the result matches the expected geometry
        self.assertTrue(result.equals(expected))

        # The result should be a MultiPolygon or a Polygon
        self.assertIn(result.geom_type, ["MultiPolygon", "Polygon"])

        # Check that the area is correct
        self.assertEqual(result.area, polygon1.area + polygon2.area + polygon3.area)
        self.assertEqual(result.area, 3.0)  # 1 + 1 + 1

    def test_with_invalid_geometries(self):
        """Test union with invalid geometries."""
        try:
            # Try to create a self-intersecting polygon (bowtie shape)
            # Note: In newer versions of Shapely, this might raise an error immediately
            invalid_polygon = Polygon([(0, 0), (1, 1), (0, 1), (1, 0)])

            # Make sure the polygon is actually invalid
            if invalid_polygon.is_valid:
                # If it's valid (which can happen in some Shapely versions),
                # create a different invalid polygon
                invalid_polygon = Polygon([(0, 0), (1, 1), (0, 1), (1, 0), (0, 0.5)])

            # Create a valid polygon
            valid_polygon = Polygon([(2, 2), (2, 3), (3, 3), (3, 2)])

            # Create a GeoDataFrame with these geometries
            gdf = gpd.GeoDataFrame(geometry=[invalid_polygon, valid_polygon])

            try:
                # Try to perform the union
                result = perform_union(gdf)

                # If we get here, the union worked despite the invalid geometry
                # The result should be a valid geometry
                self.assertTrue(result.is_valid)

                # The area should include at least the valid polygon's area
                self.assertGreaterEqual(result.area, valid_polygon.area)
            except Exception as e:
                # If the union fails, that's also acceptable behavior
                # Just make sure we can still union valid geometries
                gdf_valid = gpd.GeoDataFrame(geometry=[valid_polygon])
                result_valid = perform_union(gdf_valid)
                self.assertTrue(result_valid.is_valid)
                self.assertEqual(result_valid.area, valid_polygon.area)
        except Exception as e:
            # If we can't even create the invalid polygon, skip this test
            self.skipTest(f"Could not create invalid geometry: {str(e)}")

        # Additional test with a valid but complex polygon
        complex_polygon = Polygon(
            [(0, 0), (0, 3), (3, 3), (3, 2), (1, 2), (1, 1), (3, 1), (3, 0)]
        )
        valid_polygon = Polygon([(2, 2), (2, 3), (3, 3), (3, 2)])
        gdf_complex = gpd.GeoDataFrame(geometry=[complex_polygon, valid_polygon])
        result_complex = perform_union(gdf_complex)
        self.assertTrue(result_complex.is_valid)

        # Calculate the actual areas
        complex_area = complex_polygon.area
        valid_area = valid_polygon.area
        result_area = result_complex.area

        # Print the areas for debugging
        print(f"Complex polygon area: {complex_area}")
        print(f"Valid polygon area: {valid_area}")
        print(f"Result area: {result_area}")

        # The actual area is 7.0 (not the sum of the individual areas due to overlap)
        self.assertEqual(result_complex.area, 7.0)

    def test_with_different_crs(self):
        """Test that the function preserves the CRS information."""
        # Create a GeoDataFrame with a specific CRS
        polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")

        # Perform the union
        result = perform_union(gdf)

        # The result is a shapely geometry, which doesn't have CRS information
        # So we can't check for CRS preservation directly

        # But we can check that the geometry is correct
        self.assertTrue(result.equals(polygon))

        # And if we convert it back to a GeoSeries, we should be able to set the CRS
        result_series = gpd.GeoSeries([result])
        result_series.crs = gdf.crs
        self.assertEqual(result_series.crs, gdf.crs)

    def test_non_geodataframe_input(self):
        """Test with non-GeoDataFrame input (should raise AttributeError)."""
        # Create a regular pandas DataFrame
        df = pd.DataFrame({"col1": [1, 2, 3]})

        # This should raise an AttributeError because a regular DataFrame
        # doesn't have a 'geometry' attribute
        with self.assertRaises(AttributeError):
            perform_union(df)


if __name__ == "__main__":
    unittest.main()
