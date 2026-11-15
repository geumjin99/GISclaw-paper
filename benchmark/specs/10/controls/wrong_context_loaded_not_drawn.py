# Wrong: risky area computed correctly; existing roads, protected forest and deforested area are loaded but
# never drawn - the map shows only the planned road and the risky area.
import geopandas as gpd, shapely, matplotlib.pyplot as plt
C = 32720
roads = gpd.read_file("dataset/roads.geojson").to_crs(C)
deforested = gpd.read_file("dataset/deforestedArea.geojson").to_crs(C)
protected = gpd.read_file("dataset/protectedForest.geojson").to_crs(C)
planned = gpd.read_file("dataset/planned_road.geojson").to_crs(C)
buf = planned.geometry.union_all().buffer(5500)
defo = shapely.make_valid(shapely.clip_by_rect(deforested.geometry.iloc[0], *buf.bounds))

risk = gpd.GeoDataFrame(geometry=[buf.difference(defo)], crs=C)
fig, ax = plt.subplots(figsize=(10, 10))
risk.plot(ax=ax, color="red")
planned.plot(ax=ax, color="blue")
ax.set_title("Forest at risk from the planned road")
plt.savefig("pred_results/predictedRiskyArea.png")
