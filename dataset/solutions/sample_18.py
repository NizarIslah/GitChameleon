# library: geopandas
# version: 0.10.0
# extra_dependencies: ['rtree==0.9.3']
import geopandas as gpd
from shapely.geometry import Point, Polygon


def spatial_join(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gpd.sjoin(gdf1, gdf2, predicate="within")
