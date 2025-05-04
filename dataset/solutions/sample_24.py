# library: geopandas
# version: 0.13.0
# extra_dependencies: ['rtree==1.0.1']
import geopandas as gpd
from shapely.geometry import Point, Polygon, box


def spatial_query(gdf: gpd.GeoDataFrame, other: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    combined_geometry = other.unary_union
    return gdf.sindex.query(combined_geometry)
