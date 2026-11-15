# Correct: convex hull as a shapely geometry (no CRS) drawn with plt.plot; scipy gaussian_kde
# (Scott bandwidth) in degrees evaluated on np.mgrid[x, y] (array indexed [x, y], i.e. transposed)
# over a fixed lon/lat window and drawn with pcolor; DBSCAN eps 0.01 deg.
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN

elk = gpd.read_file('dataset/Elk_in_Southwestern_Alberta_2009.geojson')
hull = elk.unary_union.convex_hull
kde = gaussian_kde(np.vstack([elk['long'].values, elk['lat'].values]))
xs, ys = np.mgrid[-115.2:-113:100j, 48.5:50.5:100j]
density = kde(np.vstack([xs.flatten(), ys.flatten()]))
elk['cluster'] = DBSCAN(eps=0.01, min_samples=10).fit_predict(elk[['long', 'lat']])
plt.figure(figsize=(10, 8))
plt.pcolor(xs, ys, density.reshape(xs.shape), cmap='viridis', alpha=0.6)
plt.plot(*hull.exterior.xy, 'k-')
plt.scatter(elk['long'], elk['lat'], c=elk['cluster'], cmap='tab20', s=1, alpha=0.5)
plt.title('Elk home range, kernel density and DBSCAN clusters')
plt.savefig('pred_results/Elk_Analysis.png', dpi=100, bbox_inches='tight')
plt.close()
