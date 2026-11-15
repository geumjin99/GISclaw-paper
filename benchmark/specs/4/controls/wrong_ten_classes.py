# Wrong: the task says continuous linear rescaling - stated coding, upper-inclusive depth bins,
# vulnerability split into 10 equal-interval classes instead of a linear rescale.
import numpy as np, geopandas as gpd, rasterio
from rasterio.features import rasterize

soils = gpd.read_file('dataset/mc_soils.shp').to_crs('EPSG:32126')
with rasterio.open('dataset/land_cover.tif') as src:
    lc = src.read(1); tr = src.transform; prof = src.profile
code = {'Well drained': 1, 'Poorly drained': 2, 'Somewhat excessively drained': 3, 'Excessively drained': 4,
        'Somewhat poorly drained': 5}   # coding stated in the task
soils['dcode'] = soils['drclassdcd'].map(code)
drain = rasterize(zip(soils.geometry, soils.dcode), out_shape=lc.shape, transform=tr, fill=0, dtype='int16')
depth = rasterize(zip(soils.geometry, soils.wdepannmin.astype(float)), out_shape=lc.shape, transform=tr, fill=-1, dtype='float32')
rd = np.select([drain == 1, drain == 2, drain == 3, drain == 4, drain == 5], [3, 1, 4, 5, 2], np.nan)
rw = np.select([(depth >= 0) & (depth <= 10), (depth > 10) & (depth <= 33), (depth > 33) & (depth <= 61),
                (depth > 61) & (depth <= 80), (depth > 80) & (depth <= 92)], [5, 4, 3, 2, 1], np.nan)
v = rd * 5 + rw * 4
v = np.clip(np.floor((v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v)) * 10) + 1, 1, 10)   # 10 classes
lcmap = {11: 1, 21: 6, 22: 7, 23: 8, 24: 10, 31: 4, 41: 3, 42: 1, 43: 3, 52: 3, 71: 2, 81: 5, 82: 9, 90: 1, 95: 1}
rl = np.vectorize(lambda x: lcmap.get(int(x), np.nan))(lc)
risk = v * 8 + rl * 10
mask = (risk > 100) & np.isin(lc, [41, 95, 81, 71, 43, 52, 90, 31])
prof.update(dtype='uint8', nodata=255, crs='EPSG:32126', compress='lzw')
with rasterio.open('pred_results/undeveloped_areas.tif', 'w', **prof) as dst:
    dst.write(mask.astype('uint8'), 1)
