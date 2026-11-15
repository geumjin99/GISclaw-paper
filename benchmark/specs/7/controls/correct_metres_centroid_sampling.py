"""Correct: depth in metres, building depth sampled at the footprint centroid, damage computed with
the depth already in metres (so no 0.01 factor).
"""
import numpy as np, geopandas as gpd, rasterio, matplotlib.pyplot as plt
from rasterio.plot import plotting_extent
b = gpd.read_file('dataset/StudyAreaBuildings.shp')
with rasterio.open('dataset/Elevation_2050.tif') as src:
    elev = src.read(1).astype(float); ext = plotting_extent(src); inv = ~src.transform
elev[elev == 0] = np.nan
depth_m = np.where(elev < -200, (-200 - elev) / 100.0, 0.0)
c = b.geometry.centroid
cols, rows = inv * (c.x.values, c.y.values)
rows, cols = rows.astype(int), cols.astype(int)
ok = (rows >= 0) & (rows < depth_m.shape[0]) & (cols >= 0) & (cols < depth_m.shape[1])
b['depth_m'] = 0.0
b.loc[ok, 'depth_m'] = depth_m[rows[ok], cols[ok]]
b['damage'] = np.where(b.depth_m > 0, np.clip((0.298 * np.log(b.depth_m.clip(lower=1e-9)) + 1.4502) * 271 * b.Shape_Area, 0, None), 0)
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(np.where(depth_m > 0, depth_m, np.nan), extent=ext, cmap='Blues')
b[b.damage > 0].plot(ax=ax, column='damage', cmap='Reds')
plt.savefig('pred_results/flooding_analysis.png')
