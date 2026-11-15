# Wrong: UTM 23S, 5.5 km buffer minus deforested area; the deforested area layer requested by the task is
# not drawn (only protected forest, roads, risky area, planned road).
# Adapted from the reference script of GeoAnalystBench (Zhang et al. 2025, Apache-2.0).
import geopandas as gpd, matplotlib.pyplot as plt
roads = gpd.read_file('dataset/roads.geojson').to_crs(epsg=32723)
deforestedArea = gpd.read_file('dataset/deforestedArea.geojson').to_crs(epsg=32723)
deforestedArea['geometry'] = deforestedArea.make_valid()
protectedForest = gpd.read_file('dataset/protectedForest.geojson').to_crs(epsg=32723)
proposed_road = gpd.read_file('dataset/planned_road.geojson').to_crs(epsg=32723)
proposed_road_buffer_dissolve = gpd.GeoDataFrame(geometry=proposed_road.buffer(5500)).dissolve()
proposed_road_buffer_dissolve = proposed_road_buffer_dissolve.set_crs(32723)
riskyArea = gpd.overlay(proposed_road_buffer_dissolve, deforestedArea, how='difference')
fig, ax = plt.subplots(figsize=(10, 10))
protectedForest.plot(ax=ax, color='green')
roads.plot(ax=ax, color='grey')
riskyArea.plot(ax=ax, color='red')
proposed_road.plot(ax=ax, color='yellow')
plt.savefig("pred_results/predictedRiskyArea.png")
