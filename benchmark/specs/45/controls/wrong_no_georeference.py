# Wrong: correct range + rescale, but the cost raster is written without CRS/transform (not georeferenced).
import numpy as np, rasterio
from scipy.ndimage import maximum_filter, minimum_filter
with rasterio.open('dataset/Elevation.tif') as s:
    z = s.read(1).astype(float)
z[np.abs(z) > 1e30] = np.nan
rug = maximum_filter(np.nan_to_num(z, nan=-1e9), 3) - minimum_filter(np.nan_to_num(z, nan=1e9), 3)
rug[np.isnan(z)] = np.nan
cost = 10 - 9 * (rug - np.nanmin(rug)) / (np.nanmax(rug) - np.nanmin(rug))
with rasterio.open('pred_results/cost.tif', 'w', driver='GTiff', width=cost.shape[1], height=cost.shape[0],
                   count=1, dtype='float32') as d:
    d.write(cost.astype('float32'), 1)
