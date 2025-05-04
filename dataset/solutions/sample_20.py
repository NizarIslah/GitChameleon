# library: geopandas
# version: 0.10.0
# extra_dependencies: []
import geopandas as gpd
from shapely.geometry import box


def perform_union(gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
    return gdf.geometry.unary_union
