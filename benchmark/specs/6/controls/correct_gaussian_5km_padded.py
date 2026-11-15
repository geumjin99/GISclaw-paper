# Correct: wide KDE bandwidth (Gaussian sigma 5 km, binned on a 500 m grid padded by
# 3 bandwidths), population convex hull, DBSCAN 500 m / 100. A 5 km bandwidth is a legitimate
# home-range smoothing choice.
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN

elk = gpd.read_file('dataset/Elk_in_Southwestern_Alberta_2009.geojson').to_crs(32611)
xy = np.c_[elk.geometry.x, elk.geometry.y]
hull = gpd.GeoDataFrame(geometry=[elk.union_all().convex_hull], crs=elk.crs)
bw, cell = 5000.0, 500.0
x0, y0, x1, y1 = elk.total_bounds
xe = np.arange(x0 - 3 * bw, x1 + 3 * bw + cell, cell)
ye = np.arange(y0 - 3 * bw, y1 + 3 * bw + cell, cell)
H, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[xe, ye])
density = gaussian_filter(H.T, sigma=bw / cell, mode='constant')[::-1] / (cell * cell / 1e6)
elk['cluster'] = DBSCAN(eps=500, min_samples=100).fit_predict(xy)
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(density, extent=(xe[0], xe[-1], ye[0], ye[-1]), cmap='YlOrBr')
hull.boundary.plot(ax=ax, color='red')
c = elk[elk.cluster >= 0]
ax.scatter(c.geometry.x, c.geometry.y, c=c.cluster, s=0.3, cmap='tab20')
plt.savefig('pred_results/Elk_Analysis.png')
