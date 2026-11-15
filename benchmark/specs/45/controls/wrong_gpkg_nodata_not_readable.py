# Wrong: cost in a GeoPackage raster with DEM NoData written as -9999, but the nodata tag does not
# survive in the GPKG float tiles, so -9999 reads back as a cost value (NoData lost).
import numpy as np, rasterio
from scipy.ndimage import maximum_filter, minimum_filter
with rasterio.open('dataset/Elevation.tif') as s:
    z = s.read(1, masked=True).astype(float).filled(np.nan); prof = s.profile
ok = np.isfinite(z) & (np.abs(z) < 1e30)
hi = maximum_filter(np.where(ok, z, -np.inf), 3); lo = minimum_filter(np.where(ok, z, np.inf), 3)
rug = np.where(ok, hi - lo, np.nan)
cost = 10 - 9 * (rug - np.nanmin(rug)) / (np.nanmax(rug) - np.nanmin(rug))   # rugged = low cost
gp = dict(driver='GPKG', width=prof['width'], height=prof['height'], count=1, dtype='float32',
          crs=prof['crs'], transform=prof['transform'], nodata=-9999.0)
with rasterio.open('pred_results/mountain_lion.gpkg', 'w', RASTER_TABLE='ruggedness', **gp) as d:
    d.write(np.where(ok, rug, -9999).astype('float32'), 1)
with rasterio.open('pred_results/mountain_lion.gpkg', 'w', RASTER_TABLE='ruggedness_cost',
                   APPEND_SUBDATASET='YES', **gp) as d:
    d.write(np.where(ok, cost, -9999).astype('float32'), 1)
