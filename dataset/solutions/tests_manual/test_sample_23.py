# Add the parent directory to import sys
import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import geopandas as gpd
import sample_23
from shapely.geometry import Point


x, y = [1, 2], [3, 4]
print(create_geoseries(x, y))
expected_result = gpd.points_from_xy(x, y)
assert create_geoseries(x, y).equals(expected_result)
