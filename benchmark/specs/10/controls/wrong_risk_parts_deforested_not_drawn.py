# Wrong: the deforested layer enters only the risk
# product (buffer minus union_all(deforested)), whose parts are drawn; the deforested area itself is not drawn.
import geopandas as gpd, shapely, matplotlib.pyplot as plt
C = 32720
roads = gpd.read_file("dataset/roads.geojson").to_crs(C)
defo = gpd.read_file("dataset/deforestedArea.geojson").to_crs(C)
defo["geometry"] = defo.geometry.make_valid()
protected = gpd.read_file("dataset/protectedForest.geojson").to_crs(C)
planned = gpd.read_file("dataset/planned_road.geojson").to_crs(C)
buf = shapely.union_all(planned.geometry).buffer(5500)
risk = shapely.difference(buf, shapely.union_all(defo.geometry))
rg = gpd.GeoDataFrame(geometry=list(shapely.get_parts(risk)), crs=C)
fig, ax = plt.subplots(figsize=(10, 10))
protected.plot(ax=ax, color="#a6d96a"); roads.plot(ax=ax, color="grey", lw=0.2)
rg.plot(ax=ax, color="red"); planned.plot(ax=ax, color="blue")
plt.savefig("pred_results/predictedRiskyArea.png")
