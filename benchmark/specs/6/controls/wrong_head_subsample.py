# Wrong: every analysis runs on the first 20 000 fixes only (elk.head(20000)) instead of the full dataset.
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
elk_all = gpd.read_file('dataset/Elk_in_Southwestern_Alberta_2009.geojson').to_crs(32611)
elk = elk_all.head(20000).copy()
xy = np.c_[elk.geometry.x, elk.geometry.y]
home_range = gpd.GeoDataFrame(geometry=[elk.union_all().convex_hull], crs=32611)
kde = gaussian_kde(xy.T)
xmin, ymin, xmax, ymax = elk.total_bounds
X, Y = np.meshgrid(np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200))
Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
elk['cluster'] = DBSCAN(eps=500, min_samples=50).fit_predict(xy)
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(Z, origin='lower', extent=(xmin, xmax, ymin, ymax), cmap='viridis')
home_range.boundary.plot(ax=ax, color='red')
c = elk[elk.cluster >= 0]
ax.scatter(c.geometry.x, c.geometry.y, c=c.cluster, cmap='tab20', s=1)
plt.savefig('pred_results/Elk_Analysis.png')
