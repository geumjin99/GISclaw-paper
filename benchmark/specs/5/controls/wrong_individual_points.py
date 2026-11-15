# Wrong: kernel density at the case points and a map of the individual points (no hexagons,
# privacy not protected, no hot spot statistic).
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
pts = gpd.read_file('dataset/High_Blood_Level_Results.shp').to_crs(32610)
xy = np.vstack([pts.geometry.x, pts.geometry.y])
pts['density'] = gaussian_kde(xy)(xy)
ax = pts.plot(column='density', cmap='Reds', markersize=8, legend=True)
plt.savefig('pred_results/transit_access.png')
