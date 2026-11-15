"""Correct: the workflow as in ArcGIS - raster calculation, raster -> polygon, intersection with the
buildings, area-weighted mean depth per building, damage function.
"""
import numpy as np, pandas as pd, geopandas as gpd, rasterio, matplotlib.pyplot as plt
from rasterio.features import shapes
from shapely.geometry import shape
b = gpd.read_file('dataset/StudyAreaBuildings.shp')
b['bid'] = np.arange(len(b))
with rasterio.open('dataset/Elevation_2050.tif') as src:
    elev = src.read(1); tr = src.transform
flood = (elev != 0) & (elev <= -200)
depth = np.where(flood, -200 - elev, 0).astype('float32')
polys = gpd.GeoDataFrame([{'depth': v, 'geometry': shape(g)} for g, v in shapes(np.round(depth).astype('int32'), mask=flood, transform=tr)], crs=b.crs)
inter = gpd.overlay(b[['bid', 'Shape_Area', 'geometry']], polys, how='intersection')
inter['a'] = inter.area
md = inter.groupby('bid').apply(lambda d: np.average(d['depth'], weights=d['a']))
b['mean_depth'] = b['bid'].map(md).fillna(0)
cost = (0.298 * np.log(0.01 * b['mean_depth'].clip(lower=1e-9)) + 1.4502) * 271 * b['Shape_Area']
b['cost'] = np.where(b['mean_depth'] > 0, cost.clip(lower=0), 0)
ax = polys.plot(column='depth', cmap='Blues', figsize=(10, 8))
b.plot(ax=ax, column='cost', cmap='OrRd')
plt.savefig('pred_results/flooding_analysis.png')
