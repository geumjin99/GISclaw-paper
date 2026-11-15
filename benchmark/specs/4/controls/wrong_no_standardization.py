# Wrong: skips the [1, 10] standardisation of vulnerable_areas (raw 14-37 scores enter the risk sum).
import numpy as np, geopandas as gpd, rasterio
from rasterio.features import rasterize
soils = gpd.read_file('dataset/mc_soils.shp').to_crs('EPSG:32126')
with rasterio.open('dataset/land_cover.tif') as src:
    lc = src.read(1); tr = src.transform; prof = src.profile
order = list(dict.fromkeys(soils['drclassdcd']))
soils['dcode'] = soils['drclassdcd'].map({s: i + 1 for i, s in enumerate(order)})
drain = rasterize(zip(soils.geometry, soils.dcode), out_shape=lc.shape, transform=tr, fill=0, dtype='int16')
depth = rasterize(zip(soils.geometry, soils.wdepannmin.astype(float)), out_shape=lc.shape, transform=tr, fill=-1, dtype='float32')
rd = np.select([drain == k for k in (1, 2, 3, 4, 5)], [3, 1, 4, 5, 2], np.nan)
rw = np.select([(depth >= 0) & (depth <= 10), (depth > 10) & (depth <= 33), (depth > 33) & (depth <= 61),
                (depth > 61) & (depth <= 80), depth > 80], [5, 4, 3, 2, 1], np.nan)
v = rd * 5 + rw * 4
lcmap = {11: 1, 21: 6, 22: 7, 23: 8, 24: 10, 31: 4, 41: 3, 42: 1, 43: 3, 52: 3, 71: 2, 81: 5, 82: 9, 90: 1, 95: 1}
rl = np.vectorize(lambda x: lcmap.get(int(x), np.nan))(lc)
risk = v * 8 + rl * 10
out = np.where((risk > 100) & np.isin(lc, [41, 95, 81, 71, 43, 52, 90, 31]), lc, 0).astype('uint8')
prof.update(nodata=0, crs='EPSG:32126')
with rasterio.open('pred_results/undeveloped_areas.tif', 'w', **prof) as d:
    d.write(out, 1)
