# library: geopandas
# version: 0.10.0
# extra_dependencies: ['rtree==0.9.3']
import geopandas as gpd
from shapely.geometry import Point, Polygon


def spatial_query(gdf: gpd.GeoDataFrame, other: gpd.GeoSeries) -> gpd.GeoDataFrame:
    return gdf.sindex.query_bulk(other)
