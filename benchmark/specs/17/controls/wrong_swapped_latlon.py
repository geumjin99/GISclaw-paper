# Wrong: crash points rebuilt with x and y swapped (latitude as x), so they are not at their positions.
import geopandas as gpd, matplotlib.pyplot as plt
b = gpd.read_file('dataset/nyc_boroughs.geojson'); c = gpd.read_file('dataset/fatal_collisions.geojson')
pts = gpd.GeoDataFrame(c.drop(columns='geometry'), geometry=gpd.points_from_xy(c.geometry.y, c.geometry.x), crs=4326)  # x/y swapped
ax = b.plot(color='lightgrey'); pts.plot(ax=ax, color='red', markersize=10)
plt.savefig('pred_results/collisions_map_vis.png')
