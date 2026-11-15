"""Correct: works on the land-cover grid (metres), with the vulnerable areas resampled to it by
rasterio.warp; the output is a 0/1 mask written to output/ as the instruction says.
"""
import os, numpy as np, rasterio
from rasterio.warp import reproject, Resampling
lc_ds = rasterio.open('dataset/land_cover.tif'); lc = lc_ds.read(1)
va_ds = rasterio.open('dataset/vulnerable_areas.tif')
va = np.full(lc.shape, -32768, 'int16')
reproject(va_ds.read(1), va, src_transform=va_ds.transform, src_crs=va_ds.crs, dst_transform=lc_ds.transform,
          dst_crs=lc_ds.crs, src_nodata=-32768, dst_nodata=-32768, resampling=Resampling.nearest)
va = np.where(va == -32768, np.nan, va.astype(float))
std = 1 + 9 * (va - np.nanmin(va)) / (np.nanmax(va) - np.nanmin(va))
remap = {11: 1, 21: 6, 22: 7, 23: 8, 24: 10, 31: 4, 41: 3, 42: 1, 43: 3, 52: 3, 71: 2, 81: 5, 82: 9, 90: 1, 95: 1}
lcs = np.vectorize(lambda v: remap.get(int(v), np.nan), otypes=[float])(lc)
risk = std * 8 + lcs * 10
mask = (risk > 100) & np.isin(lc, [41, 95, 81, 71, 43, 52, 90, 31])
os.makedirs('output', exist_ok=True)
p = lc_ds.profile; p.update(dtype='uint8', nodata=255)
with rasterio.open('output/undeveloped_areas.tif', 'w', **p) as d:
    d.write(mask.astype('uint8'), 1)
