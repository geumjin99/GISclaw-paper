# Correct: hull via shapely, KDE drawn with seaborn.kdeplot (filled contours, bw_adjust 0.3),
# DBSCAN eps 400 m / 80 fixes in UTM 11N; overlay in one figure.
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.cluster import DBSCAN
elk = gpd.read_file('dataset/Elk_in_Southwestern_Alberta_2009.geojson').to_crs(32611)
elk['x'], elk['y'] = elk.geometry.x, elk.geometry.y
hull = gpd.GeoSeries([elk.union_all().convex_hull], crs=32611)
hull_gdf = gpd.GeoDataFrame({'what': ['home range']}, geometry=hull)
elk['dbscan'] = DBSCAN(eps=400, min_samples=80).fit_predict(elk[['x', 'y']].to_numpy())
fig, ax = plt.subplots(figsize=(11, 11))
sns.kdeplot(data=elk.sample(30000, random_state=0), x='x', y='y', fill=True, cmap='YlOrBr', levels=12, bw_adjust=0.3, thresh=0.02, ax=ax)
hull_gdf.boundary.plot(ax=ax, color='red', linewidth=2)
cl = elk[elk.dbscan >= 0]
ax.scatter(cl.x, cl.y, c=cl.dbscan, cmap='tab20', s=0.4)
ax.set_title('Elk home range, kernel density and DBSCAN clusters')
plt.savefig('pred_results/Elk_Analysis.png', dpi=130)
