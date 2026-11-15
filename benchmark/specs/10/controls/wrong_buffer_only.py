# Wrong: maps the planned-road buffer as the risky area (no overlay with the deforested area).
import geopandas as gpd, matplotlib.pyplot as plt
roads = gpd.read_file('dataset/roads.geojson').to_crs(32720)
defo = gpd.read_file('dataset/deforestedArea.geojson').to_crs(32720)
prot = gpd.read_file('dataset/protectedForest.geojson').to_crs(32720)
plan = gpd.read_file('dataset/planned_road.geojson').to_crs(32720)
risk = gpd.GeoDataFrame(geometry=plan.buffer(5500), crs=32720)
ax = prot.plot(color='green'); defo.plot(ax=ax, color='orange'); roads.plot(ax=ax, color='grey', linewidth=0.2)
risk.plot(ax=ax, color='red'); plan.plot(ax=ax, color='k')
plt.savefig('pred_results/predictedRiskyArea.png')
