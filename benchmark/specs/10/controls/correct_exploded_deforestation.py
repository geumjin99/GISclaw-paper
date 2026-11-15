# Correct: deforested multipolygon exploded into parts and repaired (common way to handle the
# invalid geometry); risky = buffer minus deforested parts; all context layers drawn.
import geopandas as gpd, matplotlib.pyplot as plt
C = 32720
roads = gpd.read_file('dataset/roads.geojson').to_crs(C)
deforested = gpd.read_file('dataset/deforestedArea.geojson').explode(index_parts=False).to_crs(C)
deforested['geometry'] = deforested.geometry.make_valid()
protected = gpd.read_file('dataset/protectedForest.geojson').to_crs(C)
planned = gpd.read_file('dataset/planned_road.geojson').to_crs(C)
zone = gpd.GeoDataFrame(geometry=planned.buffer(5500), crs=C).dissolve()
near = deforested[deforested.intersects(zone.geometry.iloc[0])]
risky = gpd.overlay(zone, near[['geometry']], how='difference')
fig, ax = plt.subplots(figsize=(11, 11))
protected.plot(ax=ax, color='#a6d96a'); deforested.plot(ax=ax, color='#d8b365', alpha=0.6)
roads.plot(ax=ax, color='grey', linewidth=0.2); risky.plot(ax=ax, color='red', alpha=0.7)
planned.plot(ax=ax, color='blue', linewidth=1.5)
ax.set_title('Forest within 5.5 km of the planned road (deforestation risk)')
plt.savefig('pred_results/predictedRiskyArea.png', dpi=130)
