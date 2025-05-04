# Add the parent directory to import sys
import os
import sys
import time
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import geopandas as gpd
from sample_18 import spatial_join
from shapely.geometry import Point, Polygon


gdf1 = gpd.GeoDataFrame({"geometry": [Point(1, 1), Point(2, 2), Point(3, 3)]})
polygons = [
    Polygon([(0, 0), (0, 4), (4, 4), (4, 0)]),
    Polygon([(4, 4), (4, 8), (8, 8), (8, 4)]),
]
gdf2 = gpd.GeoDataFrame({"geometry": polygons})

assert spatial_join(gdf1, gdf2).equals(gpd.sjoin(gdf1, gdf2, predicate="within"))
