# Wrong: hull, KDE (1 km, mgrid) and DBSCAN are all computed and a first overlay is saved, but
# Elk_Analysis.png is then overwritten by a figure that shows only the clustered fixes
# (gdf.plot without ax= opens a new figure), so the saved map does not overlay the three results.
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.cluster import DBSCAN

gdf = gpd.read_file('dataset/Elk_in_Southwestern_Alberta_2009.geojson')
gdf_proj = gdf.to_crs(epsg=32612)
convex_hulls = gdf_proj.groupby('study_name')['geometry'].apply(lambda x: gpd.GeoSeries(x).unary_union.convex_hull)
x, y = gdf_proj.geometry.x.values, gdf_proj.geometry.y.values
clusters = DBSCAN(eps=1000, min_samples=10).fit_predict(np.c_[x, y])
gdf_proj['cluster'] = clusters
xx, yy = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
kde = KernelDensity(bandwidth=1000).fit(np.c_[x[::5], y[::5]])
density = np.exp(kde.score_samples(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
fig, ax = plt.subplots(figsize=(10, 10))
gpd.GeoSeries(convex_hulls.values).plot(ax=ax, color='red', alpha=0.5)
ax.imshow(density.T, cmap='Blues', alpha=0.5, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
gdf_proj.plot(ax=ax, column='cluster', cmap='viridis', markersize=1)
plt.savefig('pred_results/Elk_Analysis.png', dpi=100, bbox_inches='tight')
plt.close()
convex_hulls.plot(color='green', alpha=0.5)
gdf_proj.plot(column='cluster', cmap='viridis', markersize=1)
plt.savefig('pred_results/Elk_Analysis.png', dpi=100, bbox_inches='tight')
plt.close()
