# Correct: UTM 11N, per-collar convex hulls (tag_ident), Gaussian-smoothed
# histogram KDE (sigma 1 km, 250 m cells), DBSCAN (eps 300 m, min_samples 50); shapefile/tif outputs.
import numpy as np, geopandas as gpd, rasterio
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN
elk = gpd.read_file('dataset/Elk_in_Southwestern_Alberta_2009.geojson').to_crs(32611)
hulls = elk.dissolve(by='tag_ident').convex_hull.reset_index()
hulls = gpd.GeoDataFrame(hulls[['tag_ident']], geometry=hulls[0], crs=32611)
hulls.to_file('pred_results/home_ranges_per_collar.shp')
x, y = elk.geometry.x.values, elk.geometry.y.values
res = 250
x0, y1 = x.min() - 5000, y.max() + 5000
nx, ny = int((x.max() + 5000 - x0) / res), int((y1 - (y.min() - 5000)) / res)
H, _, _ = np.histogram2d(y1 - y, x - x0, bins=[ny, nx], range=[[0, ny * res], [0, nx * res]])
kde = gaussian_filter(H, 1000 / res) / (res * res / 1e6)
with rasterio.open('pred_results/kde.tif', 'w', driver='GTiff', width=nx, height=ny, count=1, dtype='float32',
                   crs='EPSG:32611', transform=from_origin(x0, y1, res, res)) as d:
    d.write(kde.astype('float32'), 1)
elk['dbscan'] = DBSCAN(eps=300, min_samples=50).fit_predict(np.c_[x, y])
elk[['ind_ident', 'dbscan', 'geometry']].to_file('pred_results/elk_dbscan.shp')
