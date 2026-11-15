# Correct: "saved in the geodatabase" read as a GeoPackage - ruggedness (3x3 range) and its linear 1..10 cost
# written as two raster tables of one GeoPackage; NoData written as NaN (a -9999 nodata tag is not kept by
# the GPKG float tiles, so -9999 would read back as a value).
import numpy as np, rasterio
from scipy.ndimage import maximum_filter, minimum_filter
with rasterio.open('dataset/Elevation.tif') as s:
    z = s.read(1, masked=True).astype(float).filled(np.nan); prof = s.profile
ok = np.isfinite(z) & (np.abs(z) < 1e30)
hi = maximum_filter(np.where(ok, z, -np.inf), 3); lo = minimum_filter(np.where(ok, z, np.inf), 3)
rug = np.where(ok, hi - lo, np.nan)
cost = 10 - 9 * (rug - np.nanmin(rug)) / (np.nanmax(rug) - np.nanmin(rug))   # rugged = low cost
gp = dict(driver='GPKG', width=prof['width'], height=prof['height'], count=1, dtype='float32',
          crs=prof['crs'], transform=prof['transform'])
with rasterio.open('pred_results/mountain_lion.gpkg', 'w', RASTER_TABLE='ruggedness', **gp) as d:
    d.write(np.where(ok, rug, np.nan).astype('float32'), 1)
with rasterio.open('pred_results/mountain_lion.gpkg', 'w', RASTER_TABLE='ruggedness_cost',
                   APPEND_SUBDATASET='YES', **gp) as d:
    d.write(np.where(ok, cost, np.nan).astype('float32'), 1)
