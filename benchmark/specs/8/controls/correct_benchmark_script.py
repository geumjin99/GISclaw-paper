# Correct: UTM 17N, 2.5 km buffers, overlay difference per section.
# Adapted from the reference script of GeoAnalystBench (Zhang et al. 2025, Apache-2.0).
import geopandas as gpd, matplotlib.pyplot as plt
fire_stations = gpd.read_file("dataset/fire_station.geojson").to_crs('EPSG:32617')
fire_buffer_gdf = gpd.GeoDataFrame(geometry=fire_stations.buffer(2500), crs='EPSG:32617')
etobicoke = gpd.read_file("dataset/etobicoke.geojson").to_crs('EPSG:32617')
no_service = gpd.overlay(etobicoke, fire_buffer_gdf, how='difference')
fig, ax = plt.subplots(figsize=(10, 10))
etobicoke.plot(ax=ax, color='lightgrey', edgecolor='black')
no_service.plot(ax=ax, color='blue', alpha=0.5, edgecolor='black')
plt.savefig("pred_results/Fire_Service_Analysis.png")
