# Wrong: k-means instead of density-based clustering; KDE only plotted (never stored); hull correct.
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde
elk = gpd.read_file('dataset/Elk_in_Southwestern_Alberta_2009.geojson').to_crs(3400)
gpd.GeoDataFrame(geometry=[elk.geometry.union_all().convex_hull], crs=3400).to_file('pred_results/home_range.geojson')
xy = np.c_[elk.geometry.x, elk.geometry.y]
elk['cluster'] = KMeans(n_clusters=8, n_init=3, random_state=0).fit_predict(xy)
elk[['cluster', 'geometry']].to_file('pred_results/clusters.geojson')
sub = xy[::50]
kde = gaussian_kde(sub.T)
gx, gy = np.meshgrid(np.linspace(xy[:, 0].min(), xy[:, 0].max(), 150), np.linspace(xy[:, 1].min(), xy[:, 1].max(), 150))
z = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
plt.imshow(z, origin='lower'); plt.savefig('pred_results/kde.png')
