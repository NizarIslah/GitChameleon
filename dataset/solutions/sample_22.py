# library: geopandas
# version: 0.10.0
# extra_dependencies: []
import geopandas as gpd


def create_geoseries(x: list[int], y: list[int]) -> gpd.GeoSeries:
    return gpd.GeoSeries.from_xy(x, y)
