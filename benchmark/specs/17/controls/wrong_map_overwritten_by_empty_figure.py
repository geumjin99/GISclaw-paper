# Wrong: a correct map (boroughs + all 72 crashes) is saved, then an empty current figure is
# saved to the SAME file - the delivered collisions_map_vis.png is an empty axes frame.
import geopandas as gpd, matplotlib.pyplot as plt
gdf = gpd.read_file('dataset/fatal_collisions.geojson')
boroughs = gpd.read_file('dataset/nyc_boroughs.geojson')
ax = boroughs.plot(color='white', edgecolor='black')
gdf.plot(ax=ax, column='NUMBER OF PERSONS KILLED', legend=True)
plt.title('Collisions Map Visualization')
plt.savefig('pred_results/collisions_map_vis.png', dpi=150, bbox_inches='tight')
plt.close()
fig, ax = plt.subplots()
plt.savefig('pred_results/collisions_map_vis.png', dpi=150, bbox_inches='tight')
plt.close()
