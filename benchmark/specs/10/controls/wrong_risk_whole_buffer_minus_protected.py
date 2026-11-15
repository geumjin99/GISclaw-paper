# Wrong: "risk" = planned-road buffer minus protected areas (assumes protected land is safe, ignores existing
# deforestation, so already-cleared land is shown as newly at risk).
import geopandas as gpd, shapely, matplotlib.pyplot as plt
C = 32720
roads = gpd.read_file("dataset/roads.geojson").to_crs(C)
deforested = gpd.read_file("dataset/deforestedArea.geojson").to_crs(C)
protected = gpd.read_file("dataset/protectedForest.geojson").to_crs(C)
planned = gpd.read_file("dataset/planned_road.geojson").to_crs(C)
buf = planned.geometry.union_all().buffer(5500)
defo = shapely.make_valid(shapely.clip_by_rect(deforested.geometry.iloc[0], *buf.bounds))

risk = gpd.GeoDataFrame(geometry=[buf.difference(shapely.make_valid(protected.union_all()))], crs=C)
fig, ax = plt.subplots(figsize=(10, 10))
protected.plot(ax=ax, color="green", alpha=0.5); deforested.plot(ax=ax, color="tan", alpha=0.5)
roads.plot(ax=ax, color="grey", linewidth=0.2); planned.plot(ax=ax, color="blue")
risk.plot(ax=ax, color="red", alpha=0.6)
plt.savefig("pred_results/predictedRiskyArea.png")
