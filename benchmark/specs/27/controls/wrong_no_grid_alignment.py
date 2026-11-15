"""Wrong: ignores the different grids and crops the land-cover array to the vulnerable-areas shape
by index.
"""
import numpy as np, rasterio
v = rasterio.open('dataset/vulnerable_areas.tif'); va = v.read(1).astype(float); va[va == v.nodata] = np.nan
lc = rasterio.open('dataset/land_cover.tif').read(1)[:va.shape[0], :va.shape[1]]
std = 9 * (va - np.nanmin(va)) / (np.nanmax(va) - np.nanmin(va)) + 1
remap = {11: 1, 21: 6, 22: 7, 23: 8, 24: 10, 31: 4, 41: 3, 42: 1, 43: 3, 52: 3, 71: 2, 81: 5, 82: 9, 90: 1, 95: 1}
lcs = np.full(va.shape, np.nan)
for k, s in remap.items():
    lcs[lc == k] = s
risk = std * 8 + lcs * 10
out = np.where((risk > 100) & np.isin(lc, [31, 41, 43, 52, 71, 81, 90, 95]), lc, 0).astype('uint8')
p = v.profile; p.update(dtype='uint8', nodata=0)
with rasterio.open('pred_results/undeveloped_areas.tif', 'w', **p) as d:
    d.write(out, 1)
