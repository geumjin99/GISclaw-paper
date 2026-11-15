# Wrong: "home ranges" as hulls of the 7 'comments' groups (regions/herds, not animals): IoU 0.82
# with the population hull, 0.55 / 0.60 with the per-animal / per-collar unions, so this is not a
# minimum bounding geometry of the animals' ranges. KDE and DBSCAN are fine.
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.cluster import DBSCAN

elk = gpd.read_file('dataset/Elk_in_Southwestern_Alberta_2009.geojson').to_crs(32611)
coords = np.c_[elk.geometry.x, elk.geometry.y]
ranges = elk.dissolve(by='comments').convex_hull.reset_index(name='geometry')
ranges = gpd.GeoDataFrame(ranges, geometry='geometry', crs=elk.crs)
kd = KernelDensity(bandwidth=1000).fit(coords[::5])
x0, y0, x1, y1 = elk.total_bounds
gx, gy = np.meshgrid(np.linspace(x0, x1, 120), np.linspace(y1, y0, 120))
dens = np.exp(kd.score_samples(np.c_[gx.ravel(), gy.ravel()])).reshape(gx.shape)
elk['cluster'] = DBSCAN(eps=500, min_samples=100).fit_predict(coords)
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(dens, extent=[x0, x1, y0, y1], cmap='magma')
ranges.boundary.plot(ax=ax, color='cyan')
c = elk[elk.cluster >= 0]
ax.scatter(c.geometry.x, c.geometry.y, c=c.cluster, s=0.3, cmap='tab20')
plt.savefig('pred_results/Elk_Analysis.png')
