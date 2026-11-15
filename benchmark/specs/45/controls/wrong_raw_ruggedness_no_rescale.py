# Wrong: skips the rescale step - raw 3x3 range (metres) saved as the 'cost' layer.
import numpy as np, rasterio
from scipy.ndimage import maximum_filter, minimum_filter
with rasterio.open('dataset/Elevation.tif') as s:
    z = s.read(1).astype(float); prof = s.profile
z[np.abs(z) > 1e30] = np.nan
rug = maximum_filter(np.nan_to_num(z, nan=-1e9), 3) - minimum_filter(np.nan_to_num(z, nan=1e9), 3)
rug[np.isnan(z) | (rug > 1e5)] = np.nan
prof.update(dtype='float32', nodata=np.nan)
with rasterio.open('pred_results/ruggedness_cost.tif', 'w', **prof) as d:
    d.write(rug.astype('float32'), 1)
