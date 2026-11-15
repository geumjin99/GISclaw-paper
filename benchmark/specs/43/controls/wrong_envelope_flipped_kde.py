# Wrong: 'minimum bounding geometry' as an envelope (bounding box) instead of the convex hull, and
# a KDE grid from np.histogram2d written without transposing/flipping (rows = x, origin at the bottom).
import numpy as np, geopandas as gpd, rasterio
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter
from sklearn.cluster import HDBSCAN
elk = gpd.read_file('dataset/Elk_in_Southwestern_Alberta_2009.geojson').to_crs(3400)
gpd.GeoDataFrame(geometry=[elk.geometry.union_all().envelope], crs=3400).to_file('pred_results/home_range.geojson')
x, y = elk.geometry.x.values, elk.geometry.y.values
res = 200
H, xe, ye = np.histogram2d(x, y, bins=[int(np.ptp(x) / res), int(np.ptp(y) / res)])
kd = gaussian_filter(H, 3)
with rasterio.open('pred_results/kernel_density.tif', 'w', driver='GTiff', width=kd.shape[1], height=kd.shape[0],
                   count=1, dtype='float32', crs='EPSG:3400', transform=from_origin(x.min(), y.max(), res, res)) as d:
    d.write(kd.astype('float32'), 1)
elk['cluster'] = HDBSCAN(min_cluster_size=100).fit_predict(np.c_[x, y])
elk[['cluster', 'geometry']].to_file('pred_results/clusters.geojson')
