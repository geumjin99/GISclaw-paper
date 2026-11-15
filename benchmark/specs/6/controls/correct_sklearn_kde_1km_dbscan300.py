# Correct: UTM 12N, sklearn KernelDensity (bandwidth 1 km) on a 120 x 120 grid over the fixes,
# population convex hull via MultiPoint, DBSCAN eps 300 m / min_samples 50.
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from shapely.geometry import MultiPoint
from sklearn.neighbors import KernelDensity
from sklearn.cluster import DBSCAN

elk = gpd.read_file('dataset/Elk_in_Southwestern_Alberta_2009.geojson').to_crs(32612)
coords = np.c_[elk.geometry.x, elk.geometry.y]
hull = gpd.GeoDataFrame(geometry=[MultiPoint(coords).convex_hull], crs=elk.crs)
kd = KernelDensity(bandwidth=1000).fit(coords[::5])
x0, y0, x1, y1 = elk.total_bounds
gx, gy = np.meshgrid(np.linspace(x0, x1, 120), np.linspace(y1, y0, 120))
dens = np.exp(kd.score_samples(np.c_[gx.ravel(), gy.ravel()])).reshape(gx.shape)
labels = DBSCAN(eps=300, min_samples=50).fit_predict(coords)
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(dens, extent=[x0, x1, y0, y1], cmap='magma')
hull.boundary.plot(ax=ax, color='cyan')
m = labels >= 0
ax.scatter(coords[m, 0], coords[m, 1], c=labels[m], s=0.3, cmap='tab20')
plt.savefig('pred_results/Elk_Analysis.png')
