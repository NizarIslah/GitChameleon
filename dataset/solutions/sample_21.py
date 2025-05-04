# library: geopandas
# version: 0.9.0
# extra_dependencies: ['shapely==1.8.5']
import geopandas as gpd
from shapely.geometry import box


def perform_union(gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
    return gdf.geometry.cascaded_union
