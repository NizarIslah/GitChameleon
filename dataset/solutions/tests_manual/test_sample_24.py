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


gdf = gpd.GeoDataFrame({"geometry": [Point(1, 2)]})
other = gpd.GeoDataFrame({"geometry": [Point(1, 1)]})
result = spatial_query(gdf, other)
expected_result = gdf.sindex.query(other.unary_union)
assert (result == expected_result).all()
