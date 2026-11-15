# Wrong: focal statistics with the default MEAN statistic (smoothing, not terrain variation), then rescaled.
import numpy as np, rasterio
from scipy.ndimage import uniform_filter
with rasterio.open('dataset/Elevation.tif') as s:
    z = s.read(1).astype(float); prof = s.profile
z[np.abs(z) > 1e30] = np.nan
foc = uniform_filter(np.nan_to_num(z, nan=0), 3)
foc[np.isnan(z)] = np.nan
cost = 1 + 9 * (foc - np.nanmin(foc)) / (np.nanmax(foc) - np.nanmin(foc))
prof.update(dtype='float32', nodata=np.nan)
with rasterio.open('pred_results/ruggedness_cost.tif', 'w', **prof) as d:
    d.write(cost.astype('float32'), 1)
