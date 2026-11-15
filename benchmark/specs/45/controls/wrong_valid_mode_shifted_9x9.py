# Wrong: 9x9 focal range computed in 'valid' mode (8 cells shorter per dimension) instead of the
# 3x3 window the instruction asks for.
import numpy as np, rasterio
from rasterio.transform import Affine
from scipy.ndimage import maximum_filter, minimum_filter
with rasterio.open('dataset/Elevation.tif') as s:
    z = s.read(1, masked=True).astype(float).filled(np.nan); prof = s.profile; T = s.transform
z[np.abs(z) > 1e30] = np.nan
zf = np.where(np.isfinite(z), z, np.nanmean(z))
rug = maximum_filter(zf, 9) - minimum_filter(zf, 9)
rug[~np.isfinite(z)] = np.nan
rug = rug[4:-4, 4:-4]
cost = 1 + 9 * (rug - np.nanmin(rug)) / (np.nanmax(rug) - np.nanmin(rug))
prof.update(dtype='float32', nodata=np.nan, height=rug.shape[0], width=rug.shape[1], transform=T * Affine.translation(4, 4))
with rasterio.open('pred_results/rugged_cost.tif', 'w', **prof) as d:
    d.write(cost.astype('float32'), 1)
