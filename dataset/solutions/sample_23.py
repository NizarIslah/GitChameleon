# library: geopandas
# version: 0.9.0
# extra_dependencies: ['shapely==1.8.5']
import geopandas as gpd


def create_geoseries(x: list[int], y: list[int]) -> gpd.GeoSeries:
    return gpd.points_from_xy(x, y)
