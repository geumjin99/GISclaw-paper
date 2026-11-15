# Wrong: buffer of 5.5 "units" interpreted as 0.055 degrees in EPSG:4326 and the overlay keeps the
# already-deforested part (intersection) instead of the remaining forest.
import geopandas as gpd, matplotlib.pyplot as plt
roads = gpd.read_file('dataset/roads.geojson')
defo = gpd.read_file('dataset/deforestedArea.geojson'); defo['geometry'] = defo.make_valid()
prot = gpd.read_file('dataset/protectedForest.geojson')
plan = gpd.read_file('dataset/planned_road.geojson').to_crs(4326)
buf = gpd.GeoDataFrame(geometry=plan.buffer(0.055), crs=4326)
risk = gpd.overlay(buf, gpd.clip(defo, buf.total_bounds), how='intersection')
ax = prot.plot(color='green'); defo.plot(ax=ax, color='orange'); roads.plot(ax=ax, color='grey', linewidth=0.2)
risk.plot(ax=ax, color='red'); plan.plot(ax=ax, color='k')
plt.savefig('pred_results/predictedRiskyArea.png')
