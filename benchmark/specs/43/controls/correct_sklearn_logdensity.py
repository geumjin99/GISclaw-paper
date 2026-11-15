# Correct: sklearn KernelDensity (Gaussian, 1.5 km) evaluated on a
# 500 m grid and written as returned by score_samples, i.e. LOG density; hull of all fixes;
# DBSCAN (eps 300 m). A log-density surface is a monotone rendering of the same KDE.
import numpy as np, geopandas as gpd, rasterio
from rasterio.transform import from_origin
from sklearn.neighbors import KernelDensity
from sklearn.cluster import DBSCAN
elk = gpd.read_file('dataset/Elk_in_Southwestern_Alberta_2009.geojson').to_crs(32611)
gpd.GeoDataFrame(geometry=[elk.geometry.union_all().convex_hull], crs=32611).to_file('pred_results/home_range.geojson')
x, y = elk.geometry.x.values, elk.geometry.y.values
res = 500
x0, y1 = x.min() - 5000, y.max() + 5000
nx, ny = int((x.max() + 5000 - x0) / res), int((y1 - (y.min() - 5000)) / res)
XX, YY = np.meshgrid(x0 + (np.arange(nx) + .5) * res, y1 - (np.arange(ny) + .5) * res)
kd = KernelDensity(bandwidth=1500, rtol=1e-2).fit(np.c_[x, y])
logdens = kd.score_samples(np.c_[XX.ravel(), YY.ravel()]).reshape(ny, nx)
with rasterio.open('pred_results/elk_kde_log.tif', 'w', driver='GTiff', width=nx, height=ny, count=1, dtype='float32',
                   crs='EPSG:32611', transform=from_origin(x0, y1, res, res)) as d:
    d.write(logdens.astype('float32'), 1)
elk['cluster'] = DBSCAN(eps=300, min_samples=50).fit_predict(np.c_[x, y])
elk[['ind_ident', 'cluster', 'geometry']].to_file('pred_results/elk_dbscan.gpkg')
