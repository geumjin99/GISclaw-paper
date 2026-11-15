# Wrong: "risk" = protected forest inside the planned-road buffer; the deforested layer is drawn but not overlaid.
import geopandas as gpd, shapely, matplotlib.pyplot as plt
C = 32720
roads = gpd.read_file("dataset/roads.geojson").to_crs(C)
deforested = gpd.read_file("dataset/deforestedArea.geojson").to_crs(C)
protected = gpd.read_file("dataset/protectedForest.geojson").to_crs(C)
planned = gpd.read_file("dataset/planned_road.geojson").to_crs(C)
buf = planned.geometry.union_all().buffer(5500)
defo = shapely.make_valid(shapely.clip_by_rect(deforested.geometry.iloc[0], *buf.bounds))

risk = gpd.overlay(protected, gpd.GeoDataFrame(geometry=[buf], crs=C), how="intersection")
fig, ax = plt.subplots(figsize=(10, 10))
protected.plot(ax=ax, color="green", alpha=0.5); deforested.plot(ax=ax, color="tan", alpha=0.5)
roads.plot(ax=ax, color="grey", linewidth=0.2); planned.plot(ax=ax, color="blue")
risk.plot(ax=ax, color="red")
plt.savefig("pred_results/predictedRiskyArea.png")
