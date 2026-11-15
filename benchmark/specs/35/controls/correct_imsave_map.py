"""Correct: weighted overlay with equal-interval burn and slope classes; the map is written
with plt.imsave (no Figure), which counts as the drawn image.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, calculate_default_transform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, MultiPolygon, box
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')
os.makedirs('pred_results', exist_ok=True)


_src = rasterio.open('dataset/Before_L8.tif')
before = _src.read().astype(float) if _src.count > 1 else _src.read(1).astype(float)
before_meta = dict(crs=_src.crs, transform=_src.transform, width=_src.width, height=_src.height,
               nodata=_src.nodata, dtype=_src.dtypes[0], count=_src.count, bounds=_src.bounds, res=_src.res)
if _src.nodata is not None:
    before[before == _src.nodata] = np.nan
print(f"Size: {_src.width} x {_src.height}, bands: {_src.count}, dtype: {_src.dtypes[0]}")
print(f"CRS: {_src.crs.to_string() if _src.crs else None}, resolution: {_src.res}")
print(f"Bounds: {tuple(round(float(_b), 4) for _b in _src.bounds)}")
print(f"NoData: {_src.nodata} (replaced by NaN in before)")
_valid = before[~np.isnan(before)]
print(f"Valid values: {_valid.size} of {before.size}" + (f", min={_valid.min():.4g}, max={_valid.max():.4g}, mean={_valid.mean():.4g}" if _valid.size else ""))
print("Variables: before = float array" + (" (bands, rows, cols)" if _src.count > 1 else " (rows, cols)") + ", before_meta = dict(crs, transform, nodata, ...)")
_src.close()


_src = rasterio.open('dataset/After_L8.tif')
after = _src.read().astype(float) if _src.count > 1 else _src.read(1).astype(float)
after_meta = dict(crs=_src.crs, transform=_src.transform, width=_src.width, height=_src.height,
               nodata=_src.nodata, dtype=_src.dtypes[0], count=_src.count, bounds=_src.bounds, res=_src.res)
if _src.nodata is not None:
    after[after == _src.nodata] = np.nan
print(f"Size: {_src.width} x {_src.height}, bands: {_src.count}, dtype: {_src.dtypes[0]}")
print(f"CRS: {_src.crs.to_string() if _src.crs else None}, resolution: {_src.res}")
print(f"Bounds: {tuple(round(float(_b), 4) for _b in _src.bounds)}")
print(f"NoData: {_src.nodata} (replaced by NaN in after)")
_valid = after[~np.isnan(after)]
print(f"Valid values: {_valid.size} of {after.size}" + (f", min={_valid.min():.4g}, max={_valid.max():.4g}, mean={_valid.mean():.4g}" if _valid.size else ""))
print("Variables: after = float array" + (" (bands, rows, cols)" if _src.count > 1 else " (rows, cols)") + ", after_meta = dict(crs, transform, nodata, ...)")
_src.close()


import numpy as np, rasterio, os, warnings, sys, math, json, pathlib, matplotlib.pyplot as plt
# Compute NBR for before and after
# Band indices: 5-> index 4, 7-> index 6
b5_before = before[4,:,:]
b7_before = before[6,:,:]
nbr_before = (b5_before - b7_before) / (b5_before + b7_before + 1e-10)  # avoid zero division

b5_after = after[4,:,:]
b7_after = after[6,:,:]
nbr_after = (b5_after - b7_after) / (b5_after + b7_after + 1e-10)

burn_severity = nbr_before - nbr_after

# Save rasters
out_dir = "pred_results"
os.makedirs(out_dir, exist_ok=True)

def save_raster(arr, meta, filename):
    # arr should be 2D; set dtype float32
    profile = meta.copy()
    profile.update(dtype=rasterio.float32, count=1, compress='lzw')
    with rasterio.open(filename, 'w', **profile) as dst:
        dst.write(arr.astype(rasterio.float32), 1)

save_raster(nbr_before, before_meta, os.path.join(out_dir, "nbr_before.tif"))
save_raster(nbr_after, after_meta, os.path.join(out_dir, "nbr_after.tif"))
save_raster(burn_severity, before_meta, os.path.join(out_dir, "burn_severity.tif"))

# quick stats
print("NBR before min/max:", np.nanmin(nbr_before), np.nanmax(nbr_before))
print("NBR after min/max:", np.nanmin(nbr_after), np.nanmax(nbr_after))
print("Burn severity min/max:", np.nanmin(burn_severity), np.nanmax(burn_severity))


import rasterio
import numpy as np, os
from rasterio.warp import calculate_default_transform, reproject, Resampling

# Load DEM
with rasterio.open('dataset/DEM_30m.tif') as src:
    dem = src.read(1).astype('float32')
    dem_meta = src.meta.copy()
    dem_crs = src.crs
    dem_transform = src.transform
    print('DEM shape', dem.shape, 'CRS', dem_crs)

# Load target metadata from Landsat (use nbr_before shape)
with rasterio.open('dataset/Before_L8.tif') as src_l8:
    target_transform = src_l8.transform
    target_crs = src_l8.crs
    target_width = src_l8.width
    target_height = src_l8.height
    print('Target shape', target_height, target_width, 'CRS', target_crs)

# Prepare destination array
dem_reproj = np.empty((target_height, target_width), dtype='float32')

# Reproject
reproject(
    source=dem,
    destination=dem_reproj,
    src_transform=dem_transform,
    src_crs=dem_crs,
    dst_transform=target_transform,
    dst_crs=target_crs,
    resampling=Resampling.bilinear,
    dst_nodata=np.nan,
    src_nodata=None
)
print('Reprojected DEM stats: min', np.nanmin(dem_reproj), 'max', np.nanmax(dem_reproj))

# Compute slope in degrees
pixel_width = target_transform.a  # pixel size in x direction (positive)
pixel_height = -target_transform.e  # pixel size in y (positive)
# gradient
dzdx = np.gradient(dem_reproj, axis=1) / pixel_width
dzdy = np.gradient(dem_reproj, axis=0) / pixel_height
slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
slope_deg = np.degrees(slope_rad)
print('Slope stats: min', np.nanmin(slope_deg), 'max', np.nanmax(slope_deg))

# Save slope raster
slope_meta = dem_meta.copy()
slope_meta.update({
    'driver': 'GTiff',
    'height': target_height,
    'width': target_width,
    'transform': target_transform,
    'crs': target_crs,
    'count': 1,
    'dtype': 'float32',
    'nodata': np.nan
})
out_path = os.path.join('pred_results', 'slope.tif')
with rasterio.open(out_path, 'w', **slope_meta) as dst:
    dst.write(slope_deg, 1)
print('Saved slope to', out_path)


slope = slope_deg
print('Created variable slope with shape', slope.shape, 'dtype', slope.dtype)

_src = rasterio.open('dataset/Sonoma_NLCD2011.tif')
landcover = _src.read().astype(float) if _src.count > 1 else _src.read(1).astype(float)
landcover_meta = dict(crs=_src.crs, transform=_src.transform, width=_src.width, height=_src.height,
               nodata=_src.nodata, dtype=_src.dtypes[0], count=_src.count, bounds=_src.bounds, res=_src.res)
if _src.nodata is not None:
    landcover[landcover == _src.nodata] = np.nan
print(f"Size: {_src.width} x {_src.height}, bands: {_src.count}, dtype: {_src.dtypes[0]}")
print(f"CRS: {_src.crs.to_string() if _src.crs else None}, resolution: {_src.res}")
print(f"Bounds: {tuple(round(float(_b), 4) for _b in _src.bounds)}")
print(f"NoData: {_src.nodata} (replaced by NaN in landcover)")
_valid = landcover[~np.isnan(landcover)]
print(f"Valid values: {_valid.size} of {landcover.size}" + (f", min={_valid.min():.4g}, max={_valid.max():.4g}, mean={_valid.mean():.4g}" if _valid.size else ""))
print("Variables: landcover = float array" + (" (bands, rows, cols)" if _src.count > 1 else " (rows, cols)") + ", landcover_meta = dict(crs, transform, nodata, ...)")
_src.close()


import numpy as np, rasterio
from rasterio.warp import reproject, Resampling

# target parameters from earlier steps
target_transform = target_transform  # should be defined earlier
target_crs = target_crs
target_width = target_width
target_height = target_height

# Prepare array for reprojected landcover
landcover_reproj = np.empty((target_height, target_width), dtype=np.float32)

# perform reprojection using nearest neighbor
reproject(
    source=landcover,
    destination=landcover_reproj,
    src_transform=landcover_meta['transform'],
    src_crs=landcover_meta['crs'],
    dst_transform=target_transform,
    dst_crs=target_crs,
    resampling=Resampling.nearest,
    dst_nodata=np.nan,
    src_nodata=np.nan
)

# Check some stats
print("Reprojected landcover stats:", np.nanmin(landcover_reproj), np.nanmax(landcover_reproj), np.isnan(landcover_reproj).sum())

# Reclassify according to mapping
landcover_class = np.full_like(landcover_reproj, fill_value=np.nan, dtype=np.float32)

# rule: <12 -> 1
landcover_class[(landcover_reproj < 12) & (~np.isnan(landcover_reproj))] = 1

# 21-22 ->3
landcover_class[(landcover_reproj >= 21) & (landcover_reproj <= 22)] = 3

# 23 ->4
landcover_class[landcover_reproj == 23] = 4

# 24-31 ->5
landcover_class[(landcover_reproj >= 24) & (landcover_reproj <= 31)] = 5

# 41-44 ->1
landcover_class[(landcover_reproj >= 41) & (landcover_reproj <= 44)] = 1

# 52 ->3
landcover_class[landcover_reproj == 52] = 3

# 71 ->4
landcover_class[landcover_reproj == 71] = 4

# 81 ->3
landcover_class[landcover_reproj == 81] = 3

# 82 ->4
landcover_class[landcover_reproj == 82] = 4

# 90-95 ->2
landcover_class[(landcover_reproj >= 90) & (landcover_reproj <= 95)] = 2

# Optionally, set any remaining nan to 0 or keep nan
# We'll set nodata = 0 for output raster
nodata_val = 0
landcover_class_filled = np.where(np.isnan(landcover_class), nodata_val, landcover_class).astype(np.uint8)

# Save to file
out_path = f"{out_dir}/landcover_class.tif"
profile = {
    'driver': 'GTiff',
    'height': target_height,
    'width': target_width,
    'count': 1,
    'dtype': rasterio.uint8,
    'crs': target_crs,
    'transform': target_transform,
    'nodata': nodata_val,
    'compress': 'deflate'
}
with rasterio.open(out_path, 'w', **profile) as dst:
    dst.write(landcover_class_filled, 1)

print("Saved reprojected and classified landcover to", out_path)


import numpy as np, rasterio, os, json, math, warnings, sys, pathlib, matplotlib.pyplot as plt
# Reclassify burn severity into 5 equal interval categories (1-5)
bs = burn_severity
bs_min, bs_max = np.nanmin(bs), np.nanmax(bs)
interval_bs = (bs_max - bs_min) / 5.0
# Create classification: values <= min+interval =>1 etc.
burn_severity_class = np.digitize(bs, bins=[bs_min + i*interval_bs for i in range(1,5)], right=False) + 1
# bins has length 4, so digitize returns 0-4; +1 gives 1-5.
# Ensure max goes to 5
burn_severity_class = np.clip(burn_severity_class, 1, 5).astype(np.uint8)

# Reclassify slope (degrees) into 5 equal interval categories
sl = slope_deg
sl_min, sl_max = float(np.nanmin(sl)), float(np.nanmax(sl))
interval_sl = (sl_max - sl_min) / 5.0
slope_class = np.digitize(sl, bins=[sl_min + i*interval_sl for i in range(1,5)], right=False) + 1
slope_class = np.clip(slope_class, 1, 5).astype(np.uint8)

# Save rasters
profile = before_meta.copy()
profile.update({
    'dtype': 'uint8',
    'count': 1,
    'nodata': 0,
    'compress': 'lzw'
})

out_dir = out_dir if 'out_dir' in globals() else 'pred_results'
os.makedirs(out_dir, exist_ok=True)

burn_path = os.path.join(out_dir, 'burn_severity_class.tif')
with rasterio.open(burn_path, 'w', **profile) as dst:
    dst.write(burn_severity_class, 1)

slope_path = os.path.join(out_dir, 'slope_class.tif')
with rasterio.open(slope_path, 'w', **profile) as dst:
    dst.write(slope_class, 1)

print("Burn severity class stats:", np.unique(burn_severity_class, return_counts=True))
print("Slope class stats:", np.unique(slope_class, return_counts=True))


import numpy as np, matplotlib.pyplot as plt, os
# Ensure arrays exist
print('Shapes:', landcover_class.shape, burn_severity_class.shape, slope_class.shape)
# Compute weighted overlay
risk = 0.15 * landcover_class.astype(np.float32) + 0.30 * burn_severity_class.astype(np.float32) + 0.55 * slope_class.astype(np.float32)
print('Risk stats before scaling: min', risk.min(), 'max', risk.max())
# Rescale to 0-255
risk_rescaled = ((risk - 1) / (5 - 1) * 255).astype(np.uint8)
print('Rescaled stats: min', risk_rescaled.min(), 'max', risk_rescaled.max())
# Save as PNG
out_png = os.path.join('pred_results', 'landslide_map.png')
# Use plt.imsave with a colormap, e.g., 'viridis'
plt.imsave(out_png, risk_rescaled, cmap='viridis')
print('Saved to', out_png)
# Also optional: save array as .npy for verification
np.save(os.path.join('pred_results','landslide_risk.npy'), risk_rescaled)
