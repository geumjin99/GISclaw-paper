# Correct: IDW straight on all 26k samples (no station aggregation), 150x150 mgrid as in the
# benchmark's reference script.
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from scipy.spatial import cKDTree
pts = gpd.read_file('dataset/DissolvedO2.geojson'); bay = gpd.read_file('dataset/Bay.geojson')
xy = np.c_[pts.geometry.x, pts.geometry.y]; val = pts['MeasureValue'].values
minx, miny, maxx, maxy = bay.total_bounds
x, y = np.mgrid[minx:maxx:150j, miny:maxy:150j]
q = np.c_[x.ravel(), y.ravel()]
d, i = cKDTree(xy).query(q, k=200)
w = 1 / (d + 1) ** 2
z = ((w * val[i]).sum(1) / w.sum(1)).reshape(x.shape)
inside = gpd.points_from_xy(q[:, 0], q[:, 1]).within(bay.geometry.union_all()).reshape(x.shape)
z_masked = np.where(inside, z, np.nan)
plt.figure(figsize=(8, 10))
plt.imshow(z_masked.T, origin='lower', extent=(minx, maxx, miny, maxy), cmap='Spectral')
bay.boundary.plot(ax=plt.gca(), color='k'); plt.colorbar(label='DO (mg/L)')
plt.savefig('pred_results/interploated_water_quality.png')
