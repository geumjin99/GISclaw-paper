# Correct: scipy gaussian_kde (Scott bandwidth) in lon/lat on a 150 x 150 grid, per-animal
# convex hulls, HDBSCAN on a 20 000-fix random sample (projected coordinates).
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.cluster import HDBSCAN

gdf = gpd.read_file('dataset/Elk_in_Southwestern_Alberta_2009.geojson')
h = gdf.dissolve(by='ind_ident').convex_hull
hulls = gpd.GeoDataFrame({'ind_ident': h.index}, geometry=h.values, crs=gdf.crs)
x0, y0, x1, y1 = gdf.total_bounds
xx, yy = np.meshgrid(np.linspace(x0, x1, 150), np.linspace(y0, y1, 150))
sub = gdf.sample(20000, random_state=0)
kde = gaussian_kde(np.vstack([sub.geometry.x, sub.geometry.y]), bw_method=0.05)
density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
subm = sub.to_crs(32612)
subm['cluster'] = HDBSCAN(min_cluster_size=50).fit_predict(np.c_[subm.geometry.x, subm.geometry.y])
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(density, origin='lower', extent=[x0, x1, y0, y1], cmap='viridis')
hulls.boundary.plot(ax=ax, color='white', linewidth=0.5)
sub.assign(cluster=subm['cluster'].values).query('cluster >= 0').plot(ax=ax, column='cluster', markersize=1, cmap='tab20')
plt.savefig('pred_results/Elk_Analysis.png')
