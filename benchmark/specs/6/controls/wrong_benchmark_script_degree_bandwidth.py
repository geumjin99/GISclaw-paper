# Wrong: KDE bandwidth 468.8 (metres in ArcGIS) applied to lon/lat degrees, so the density surface is flat.
# Adapted from the reference script of GeoAnalystBench (Zhang et al. 2025, Apache-2.0), with DBSCAN
# in place of its HDBSCAN.
import geopandas as gpd, numpy as np, matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
gdf = gpd.read_file("dataset/Elk_in_Southwestern_Alberta_2009.geojson")
convex_hull = gdf.unary_union.convex_hull
gdf_hull = gpd.GeoDataFrame([1], geometry=[convex_hull], crs=gdf.crs)
coords = np.array(list(zip(gdf.geometry.x, gdf.geometry.y)))
kde = KernelDensity(bandwidth=468.805393200003, metric='euclidean').fit(coords)
x_min, y_min, x_max, y_max = gdf.total_bounds
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
z = np.exp(kde.score_samples(np.vstack([xx.ravel(), yy.ravel()]).T)).reshape(xx.shape)
plt.figure(figsize=(10, 10))
plt.imshow(z, origin='lower', extent=[x_min, x_max, y_min, y_max], alpha=0.5)
gdf_hull.boundary.plot(ax=plt.gca(), edgecolor='red')
from sklearn.cluster import DBSCAN
gdf['Cluster'] = DBSCAN(eps=0.005, min_samples=100).fit_predict(coords)
gdf[gdf.Cluster >= 0].plot(column='Cluster', ax=plt.gca(), markersize=1)
plt.savefig("pred_results/Elk_Analysis.png")
