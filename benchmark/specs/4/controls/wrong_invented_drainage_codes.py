# Wrong: an invented drainage coding ('Moderately well drained' = 3, the 'Excessively drained' class
# dropped to 0) instead of the coding stated in the task, then the full workflow.
import numpy as np, geopandas as gpd, rasterio
from rasterio.features import rasterize
soils = gpd.read_file('dataset/mc_soils.shp').to_crs('EPSG:32126')
with rasterio.open('dataset/land_cover.tif') as src:
    lc = src.read(1); tr = src.transform; prof = src.profile
drain_map = {"Somewhat excessively drained": 1, "Well drained": 2, "Moderately well drained": 3,
             "Somewhat poorly drained": 4, "Poorly drained": 5}
soils['dv'] = soils['drclassdcd'].map(drain_map).fillna(0).astype(int)
drain = rasterize(zip(soils.geometry, soils.dv), out_shape=lc.shape, transform=tr, fill=0, dtype='uint8')
rd = np.select([drain == k for k in (1, 2, 3, 4, 5)], [3, 1, 4, 5, 2], 0).astype(float)
depth = rasterize(zip(soils.geometry, soils.wdepannmin.fillna(0).astype(float)), out_shape=lc.shape, transform=tr, fill=0, dtype='float32')
rw = np.select([depth <= 10, (depth > 10) & (depth <= 33), (depth > 33) & (depth <= 61), (depth > 61) & (depth <= 80), depth > 80], [5, 4, 3, 2, 1], 0).astype(float)
v = rd * 5 + rw * 4
vmin, vmax = np.nanmin(v[v > 0]), np.nanmax(v)
rv = 1 + 9 * (v - vmin) / (vmax - vmin)
lcmap = {11: 1, 21: 6, 22: 7, 23: 8, 24: 10, 31: 4, 41: 3, 42: 1, 43: 3, 52: 3, 71: 2, 81: 5, 82: 9, 90: 1, 95: 1}
rl = np.zeros(lc.shape)
for k, val in lcmap.items():
    rl[lc == k] = val
risk = rv * 8 + rl * 10
out = np.where((risk > 100) & np.isin(lc, [41, 95, 81, 71, 43, 52, 90, 31]), lc, 0).astype('uint8')
prof.update(nodata=0, crs='EPSG:32126')
with rasterio.open('pred_results/undeveloped_areas.tif', 'w', **prof) as d:
    d.write(out, 1)
