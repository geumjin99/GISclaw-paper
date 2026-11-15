# Wrong: bounding rectangle (envelope) instead of a convex hull, k-means instead of a
# density-based clustering (KDE itself is fine).
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
elk = gpd.read_file('dataset/Elk_in_Southwestern_Alberta_2009.geojson').to_crs(32611)
xy = np.c_[elk.geometry.x, elk.geometry.y]
home = gpd.GeoDataFrame(geometry=[elk.geometry.union_all().envelope], crs=elk.crs)
x0, y0, x1, y1 = elk.total_bounds
H, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=[np.arange(x0, x1, 250), np.arange(y0, y1, 250)])
dens = gaussian_filter(H.T, 2)[::-1]
elk['cluster'] = KMeans(n_clusters=8, n_init=3, random_state=0).fit_predict(xy)
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(dens, extent=[x0, x1, y0, y1])
home.boundary.plot(ax=ax, color='r')
ax.scatter(xy[::10, 0], xy[::10, 1], c=elk.cluster.values[::10], s=0.3)
plt.savefig('pred_results/Elk_Analysis.png')
