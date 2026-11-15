# Correct: same interpretation as the reference (codes in order of appearance via pd.factorize, ArcGIS
# upper-inclusive bins, linear rescale) but the result is reprojected to UTM 11N and saved .tiff.
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.features import rasterize
from rasterio.warp import calculate_default_transform, reproject, Resampling

soils = gpd.read_file('dataset/mc_soils.shp').to_crs('EPSG:32126')
src = rasterio.open('dataset/land_cover.tif'); lc = src.read(1); tr = src.transform
codes, uniq = pd.factorize(soils['drclassdcd'])
drain = rasterize(zip(soils.geometry, codes + 1), out_shape=lc.shape, transform=tr, fill=0, dtype='int16')
depth = rasterize(zip(soils.geometry, soils.wdepannmin.astype(float)), out_shape=lc.shape, transform=tr, fill=-1, dtype='float32')
rd = pd.Series(drain.ravel()).map({1: 3, 2: 1, 3: 4, 4: 5, 5: 2}).to_numpy().reshape(lc.shape)
rw = np.select([(depth >= 0) & (depth <= 10), (depth > 10) & (depth <= 33), (depth > 33) & (depth <= 61),
                (depth > 61) & (depth <= 80), (depth > 80) & (depth <= 92)], [5, 4, 3, 2, 1], np.nan)
v = rd * 5 + rw * 4
v = 1 + 9 * (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v))
lcmap = {11: 1, 21: 6, 22: 7, 23: 8, 24: 10, 31: 4, 41: 3, 42: 1, 43: 3, 52: 3, 71: 2, 81: 5, 82: 9, 90: 1, 95: 1}
rl = pd.Series(lc.ravel()).map(lcmap).to_numpy().reshape(lc.shape)
risk = v * 8 + rl * 10
out = np.where((risk > 100) & np.isin(lc, [41, 95, 81, 71, 43, 52, 90, 31]), lc, 0).astype('uint8')
dst_crs = 'EPSG:32611'
t2, w2, h2 = calculate_default_transform('EPSG:32126', dst_crs, src.width, src.height, *src.bounds, resolution=30)
o2 = np.zeros((h2, w2), 'uint8')
reproject(out, o2, src_transform=tr, src_crs='EPSG:32126', dst_transform=t2, dst_crs=dst_crs, resampling=Resampling.nearest)
with rasterio.open('pred_results/undeveloped_areas.tiff', 'w', driver='GTiff', width=w2, height=h2, count=1,
                   dtype='uint8', crs=dst_crs, transform=t2, nodata=0) as d:
    d.write(o2, 1)
