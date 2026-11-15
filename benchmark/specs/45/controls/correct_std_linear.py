# Correct: 3x3 focal standard deviation, linear rescale to 1 (flat) .. 10 (rugged).
import numpy as np, rasterio
from scipy.ndimage import uniform_filter
with rasterio.open('dataset/Elevation.tif') as s:
    z = s.read(1, masked=True).astype(float).filled(np.nan); prof = s.profile
ok = np.isfinite(z) & (np.abs(z) < 1e30)
zf = np.where(ok, z, np.nanmean(z[ok]))
m = uniform_filter(zf, 3); m2 = uniform_filter(zf * zf, 3)
rug = np.where(ok, np.sqrt(np.clip(m2 - m * m, 0, None)), np.nan)
cost = 1 + 9 * (rug - np.nanmin(rug)) / (np.nanmax(rug) - np.nanmin(rug))
prof.update(dtype='float32', nodata=np.nan)
with rasterio.open('pred_results/ruggedness_cost.tif', 'w', **prof) as d:
    d.write(cost.astype('float32'), 1)
