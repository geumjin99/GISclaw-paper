"""Wrong: stops at the high-risk zones (no undeveloped land-cover filter)."""
import numpy as np, rasterio
from rasterio.warp import reproject, Resampling
v = rasterio.open('dataset/vulnerable_areas.tif'); va = v.read(1).astype(float); va[va == v.nodata] = np.nan
l = rasterio.open('dataset/land_cover.tif'); lc = np.zeros(va.shape, 'uint8')
reproject(l.read(1), lc, src_transform=l.transform, src_crs=l.crs, dst_transform=v.transform, dst_crs=v.crs,
          resampling=Resampling.nearest)
std = 9 * (va - np.nanmin(va)) / (np.nanmax(va) - np.nanmin(va)) + 1
remap = {11: 1, 21: 6, 22: 7, 23: 8, 24: 10, 31: 4, 41: 3, 42: 1, 43: 3, 52: 3, 71: 2, 81: 5, 82: 9, 90: 1, 95: 1}
lcs = np.full(va.shape, np.nan)
for k, s in remap.items():
    lcs[lc == k] = s
risk = std * 8 + lcs * 10
p = v.profile; p.update(dtype='uint8', nodata=0)
with rasterio.open('pred_results/undeveloped_areas.tif', 'w', **p) as d:
    d.write((risk > 100).astype('uint8'), 1)
