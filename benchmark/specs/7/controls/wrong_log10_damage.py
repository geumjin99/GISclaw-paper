"""Wrong: the same analysis as the reference solution, but the 'Log' of the damage function is read
as log10.
"""
import numpy as np, geopandas as gpd, rasterio, rasterio.plot, matplotlib.pyplot as plt
from rasterstats import zonal_stats
b = gpd.read_file('dataset/StudyAreaBuildings.shp')
with rasterio.open('dataset/Elevation_2050.tif') as s:
    elev = s.read(1).astype(float); tr = s.transform; ext = rasterio.plot.plotting_extent(s)
flooded = (elev != 0) & (elev < -200)
flood_depth = np.where(flooded, -200 - elev, np.nan)
zs = zonal_stats(b, np.where(flooded, flood_depth, -9999.0), affine=tr, nodata=-9999.0, stats=['mean'], all_touched=True)
b['mean_depth'] = [z['mean'] or 0 for z in zs]
with np.errstate(divide='ignore'):
    b['damage'] = np.where(b.mean_depth > 0, np.clip((0.298 * np.log10(0.01 * b.mean_depth) + 1.4502) * 271 * b.Shape_Area, 0, None), 0)
fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(flood_depth, extent=ext, cmap='Blues')
b[b.damage > 0].plot(ax=ax, column='damage', cmap='OrRd', legend=True)
plt.savefig('pred_results/flooding_analysis.png')
print(b.damage.sum())
