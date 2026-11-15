"""Correct: mean depth per building from rasterstats zonal statistics with all_touched=True."""
import numpy as np, rasterio, geopandas as gpd, matplotlib.pyplot as plt
from rasterstats import zonal_stats
src = rasterio.open('dataset/Elevation_2050.tif')
elev = src.read(1)
depth = np.where(elev <= -200, -(elev + 200), np.nan)          # cm
bld = gpd.read_file('dataset/StudyAreaBuildings.shp')
zs = zonal_stats(bld, depth, affine=src.transform, stats=['mean'], all_touched=True, nodata=np.nan)
bld['depth_cm'] = [z['mean'] for z in zs]
fl = bld[bld.depth_cm.notna()].copy()
d = fl.depth_cm
fl['damage'] = np.where(d > 1, (0.298 * np.log(0.01 * d.clip(lower=0.01)) + 1.4502) * 271 * fl.Shape_Area, 0)
total = fl.damage.sum()
fl.to_file('pred_results/flooded_buildings.gpkg')
fl.plot(column='damage', legend=True); plt.savefig('pred_results/damage.png')
