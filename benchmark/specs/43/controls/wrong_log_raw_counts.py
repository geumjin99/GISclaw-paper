# Wrong: "density" = log of raw fix counts per 250 m cell (no kernel), stored as
# log share (-inf -> very negative); a log of counts is not a kernel density surface.
import numpy as np, geopandas as gpd, rasterio
from rasterio.transform import from_origin
from sklearn.cluster import DBSCAN
elk = gpd.read_file('dataset/Elk_in_Southwestern_Alberta_2009.geojson').to_crs(32611)
gpd.GeoDataFrame(geometry=[elk.geometry.union_all().convex_hull], crs=32611).to_file('pred_results/home_range.geojson')
x, y = elk.geometry.x.values, elk.geometry.y.values
res = 250
x0, y1 = x.min() - 5000, y.max() + 5000
nx, ny = int((x.max() + 5000 - x0) / res), int((y1 - (y.min() - 5000)) / res)
H, _, _ = np.histogram2d(y1 - y, x - x0, bins=[ny, nx], range=[[0, ny * res], [0, nx * res]])
L = np.log(np.maximum(H, 1e-12) / H.sum())
with rasterio.open('pred_results/density.tif', 'w', driver='GTiff', width=nx, height=ny, count=1, dtype='float32',
                   crs='EPSG:32611', transform=from_origin(x0, y1, res, res)) as d:
    d.write(L.astype('float32'), 1)
elk['cluster'] = DBSCAN(eps=300, min_samples=50).fit_predict(np.c_[x, y])
elk[['ind_ident', 'cluster', 'geometry']].to_file('pred_results/elk_dbscan.gpkg')
