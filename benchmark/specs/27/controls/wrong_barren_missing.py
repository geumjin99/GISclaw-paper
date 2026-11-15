"""Wrong: Barren Land (31) is missing from the undeveloped class list."""
import numpy as np, rasterio
from rasterio.warp import reproject, Resampling
REMAP = {11: 1, 21: 6, 22: 7, 23: 8, 24: 10, 31: 4, 41: 3, 42: 1, 43: 3, 52: 3, 71: 2, 81: 5, 82: 9, 90: 1, 95: 1}
UNDEV = [41, 43, 52, 71, 81, 90, 95]
with rasterio.open('dataset/vulnerable_areas.tif') as v:
    vul = v.read(1).astype(float); vul[vul == v.nodata] = np.nan; prof = v.profile; T = v.transform; C = v.crs
with rasterio.open('dataset/land_cover.tif') as l:
    lc = np.zeros(vul.shape, 'uint8')
    reproject(l.read(1), lc, src_transform=l.transform, src_crs=l.crs, dst_transform=T, dst_crs=C, resampling=Resampling.nearest)
vmin, vmax = np.nanmin(vul), np.nanmax(vul)
s = 9 * (vul - vmin) / (vmax - vmin) + 1
lcs = np.full(vul.shape, np.nan)
for k, x in REMAP.items():
    lcs[lc == k] = x
risk = s * 8 + lcs * 10
sel = (risk > 100) & np.isin(lc, UNDEV)
print('selected', sel.sum())
prof.update(dtype='uint8', nodata=0)
with rasterio.open('pred_results/undeveloped_areas.tif', 'w', **prof) as d:
    d.write(np.where(sel, lc, 0).astype('uint8'), 1)
