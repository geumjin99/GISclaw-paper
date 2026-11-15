# Correct: ruggedness as Horn slope computed by array slicing ('valid' mode: the 3x3 window needs
# all neighbours, so the result is 2 cells shorter in each dimension); written with the geotransform
# moved by one cell, inverse 0-1 rescale (rugged = low cost), NoData kept.
import numpy as np, rasterio
from rasterio.transform import Affine
with rasterio.open('dataset/Elevation.tif') as s:
    z = s.read(1, masked=True).astype(float).filled(np.nan); prof = s.profile; T = s.transform
z[np.abs(z) > 1e30] = np.nan
H, W = z.shape
w = lambda i, j: z[1 + i:H - 1 + i, 1 + j:W - 1 + j]
res = T.a
dzdx = ((w(-1, 1) + 2 * w(0, 1) + w(1, 1)) - (w(-1, -1) + 2 * w(0, -1) + w(1, -1))) / (8 * res)
dzdy = ((w(1, -1) + 2 * w(1, 0) + w(1, 1)) - (w(-1, -1) + 2 * w(-1, 0) + w(-1, 1))) / (8 * res)
slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
cost = 1 - (slope - np.nanmin(slope)) / (np.nanmax(slope) - np.nanmin(slope))
prof.update(dtype='float32', nodata=np.nan, height=H - 2, width=W - 2, transform=T * Affine.translation(1, 1))
with rasterio.open('pred_results/rugged_cost.tif', 'w', **prof) as d:
    d.write(cost.astype('float32'), 1)
