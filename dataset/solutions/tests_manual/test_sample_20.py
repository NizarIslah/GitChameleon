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


gdf = gpd.GeoDataFrame({"geometry": [box(0, 0, 2, 5), box(0, 0, 2, 1)]})
expected_result = gdf.geometry.unary_union
assert perform_union(gdf).equals(expected_result)
