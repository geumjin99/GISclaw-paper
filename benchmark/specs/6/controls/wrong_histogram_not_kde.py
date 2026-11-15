# Wrong: "kernel density" replaced by a plain 2-D histogram of fixes per 1 km cell (no kernel smoothing);
# hull and DBSCAN are done correctly.
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
elk = gpd.read_file('dataset/Elk_in_Southwestern_Alberta_2009.geojson').to_crs(32611)
xy = np.c_[elk.geometry.x, elk.geometry.y]
home = gpd.GeoDataFrame(geometry=[elk.union_all().convex_hull], crs=32611)
xmin, ymin, xmax, ymax = elk.total_bounds
H, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=[np.arange(xmin, xmax + 1000, 1000), np.arange(ymin, ymax + 1000, 1000)])
density = H.T[::-1]
elk['cl'] = DBSCAN(eps=500, min_samples=100).fit_predict(xy)
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(np.log1p(density), extent=(xe[0], xe[-1], ye[0], ye[-1]), cmap='YlOrBr')
home.boundary.plot(ax=ax, color='red')
c = elk[elk.cl >= 0]
ax.scatter(c.geometry.x, c.geometry.y, c=c.cl, s=0.5, cmap='tab20')
plt.savefig('pred_results/Elk_Analysis.png')
