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


gdf = gpd.GeoDataFrame({"geometry": [Point(1, 1), Point(2, 2), Point(3, 3)]})
other = gpd.GeoSeries([Polygon([(0, 0), (0, 4), (4, 4), (4, 0)])])
result = spatial_query(gdf, other)
expected_result = gdf.sindex.query_bulk(other)
assert (result == expected_result).all()
