# Wrong: drainage codes in alphabetical order (np.unique) although the task states the coding;
# depth bins lower-bound inclusive (np.digitize), output a 0/1 mask.
import numpy as np, geopandas as gpd, rasterio
from rasterio.features import rasterize

soils = gpd.read_file('dataset/mc_soils.shp').to_crs('EPSG:32126')
with rasterio.open('dataset/land_cover.tif') as src:
    lc = src.read(1); tr = src.transform; prof = src.profile
names = np.unique(soils['drclassdcd'])
soils['dcode'] = np.searchsorted(names, soils['drclassdcd']) + 1
drain = rasterize(zip(soils.geometry, soils.dcode), out_shape=lc.shape, transform=tr, fill=0, dtype='int16')
depth = rasterize(zip(soils.geometry, soils.wdepannmin.astype(float)), out_shape=lc.shape, transform=tr, fill=-1, dtype='float32')
rd = np.select([drain == 1, drain == 2, drain == 3, drain == 4, drain == 5], [3, 1, 4, 5, 2], np.nan)
rw = np.array([np.nan, 5, 4, 3, 2, 1, 1])[np.digitize(depth, [0, 10, 33, 61, 80, 92])]
rw[depth < 0] = np.nan
v = rd * 5 + rw * 4
v = 1 + 9 * (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v))
lcmap = {11: 1, 21: 6, 22: 7, 23: 8, 24: 10, 31: 4, 41: 3, 42: 1, 43: 3, 52: 3, 71: 2, 81: 5, 82: 9, 90: 1, 95: 1}
rl = np.vectorize(lambda x: lcmap.get(int(x), np.nan))(lc)
risk = v * 8 + rl * 10
mask = (risk > 100) & np.isin(lc, [41, 95, 81, 71, 43, 52, 90, 31])
prof.update(dtype='uint8', nodata=255, crs='EPSG:32126', compress='lzw')
with rasterio.open('pred_results/undeveloped_areas.tif', 'w', **prof) as dst:
    dst.write(mask.astype('uint8'), 1)
