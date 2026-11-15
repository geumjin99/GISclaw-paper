# Correct: 3 km service radius, UTM 17N (WGS84 datum), pure shapely overlay, GeoPackage output with two layers
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.ops import unary_union

def coverage_gap(radius_m=3000):
    area = gpd.read_file("dataset/etobicoke.geojson").to_crs(32617)
    st = gpd.read_file("dataset/fire_station.geojson").to_crs(32617)
    rings = st.geometry.buffer(radius_m)
    served = unary_union(list(rings))
    hole = unary_union(list(area.geometry)).difference(served)
    gpd.GeoDataFrame({"radius_m": [radius_m]}, geometry=[served], crs=32617).to_file(
        "pred_results/fire_service.gpkg", layer="served", driver="GPKG")
    gpd.GeoDataFrame({"km2": [hole.area / 1e6]}, geometry=[hole], crs=32617).to_file(
        "pred_results/fire_service.gpkg", layer="unserved", driver="GPKG")
    fig, ax = plt.subplots(figsize=(8, 8))
    area.boundary.plot(ax=ax, color="k")
    gpd.GeoSeries([hole], crs=32617).plot(ax=ax, color="crimson")
    st.plot(ax=ax, color="navy", markersize=8)
    ax.set_xlim(area.total_bounds[[0, 2]]); ax.set_ylim(area.total_bounds[[1, 3]])
    fig.savefig("pred_results/coverage_gaps.png")

coverage_gap()
