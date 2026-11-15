"""Wrong: drops the 0.01 (cm -> m) factor inside the log of the damage function."""
import numpy as np, geopandas as gpd, rasterio, matplotlib.pyplot as plt
from rasterstats import zonal_stats
b = gpd.read_file('dataset/StudyAreaBuildings.shp')
with rasterio.open('dataset/Elevation_2050.tif') as src:
    elev = src.read(1).astype(float); tr = src.transform
depth = np.where((elev != 0) & (elev <= -200), -200 - elev, np.nan)
b['mean_depth'] = [z['mean'] or 0 for z in zonal_stats(b, np.nan_to_num(depth, nan=-1), affine=tr, nodata=-1, stats=['mean'], all_touched=True)]
b['damage'] = np.where(b.mean_depth > 0, (0.298 * np.log(b.mean_depth.clip(lower=1e-9)) + 1.4502) * 271 * b.Shape_Area, 0)
fig, ax = plt.subplots()
ax.imshow(depth, cmap='Blues')
plt.savefig('pred_results/flooding_analysis.png')
