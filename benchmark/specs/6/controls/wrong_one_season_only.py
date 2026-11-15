# Wrong: filters to summer_indicator == 1 ("summer") before all analyses; the task asks for the 2009 movements.
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
elk = gpd.read_file('dataset/Elk_in_Southwestern_Alberta_2009.geojson')
elk = elk[elk.summer_indicator == 1].to_crs(32611)
xy = np.c_[elk.geometry.x, elk.geometry.y]
hull = gpd.GeoDataFrame(geometry=[elk.union_all().convex_hull], crs=32611)
xmin, ymin, xmax, ymax = elk.total_bounds
X, Y = np.meshgrid(np.linspace(xmin, xmax, 200), np.linspace(ymax, ymin, 200))
dens = gaussian_kde(xy[::5].T, bw_method=0.05)(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
elk['cluster'] = DBSCAN(eps=500, min_samples=100).fit_predict(xy)
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(dens, extent=(xmin, xmax, ymin, ymax), cmap='magma')
hull.boundary.plot(ax=ax, color='cyan')
c = elk[elk.cluster >= 0]
ax.scatter(c.geometry.x, c.geometry.y, c=c.cluster, cmap='tab20', s=1)
plt.savefig('pred_results/Elk_Analysis.png')
