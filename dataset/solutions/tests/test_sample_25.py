# Add the parent directory to import sys
import os
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import geopandas as gpd
import sample_25
from shapely.geometry import Point, Polygon

# Filter deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Check geopandas version
gpd_version = gpd.__version__
print(f"Using geopandas version: {gpd_version}")


class TestSpatialQuery(unittest.TestCase):
    """Tests for sample_25.spatial_query, verifying returned geometry indices."""

    def _get_geom_indices(self, result: np.ndarray) -> np.ndarray:
        # If result is Nx2, take first column; else flatten.
        if result.ndim == 2 and result.shape[1] >= 1:
            return result[:, 0]
        return result.flatten()

    def _expected_indices(
        self, gdf: gpd.GeoDataFrame, other: gpd.GeoSeries
    ) -> np.ndarray:
        # Using GeoPandas to compute which geometries intersect
        mask = gdf.geometry.intersects(other.unary_union)
        return np.where(mask)[0]

    def _assert_spatial_query(self, gdf: gpd.GeoDataFrame, other: gpd.GeoSeries):
        result = sample_25.spatial_query(gdf, other)
        self.assertIsInstance(result, np.ndarray)

        geom_indices = self._get_geom_indices(result)
        expected = self._expected_indices(gdf, other)

        # Compare as sorted arrays (order doesn't matter)
        np.testing.assert_array_equal(np.sort(geom_indices), expected)

    def test_query_with_empty_geodataframe(self):
        """Test spatial query on an empty GeoDataFrame."""
        gdf = gpd.GeoDataFrame(geometry=[])
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        other = gpd.GeoSeries([polygon])
        self._assert_spatial_query(gdf, other)

    def test_query_with_non_overlapping_geometries(self):
        """Test spatial query when no geometries overlap."""
        points = [Point(0, 0), Point(1, 1)]
        gdf = gpd.GeoDataFrame(geometry=points)
        polygon = Polygon([(5, 5), (6, 5), (6, 6), (5, 6)])
        other = gpd.GeoSeries([polygon])
        self._assert_spatial_query(gdf, other)


if __name__ == "__main__":
    unittest.main()
