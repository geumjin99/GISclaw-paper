"""Wrong: the Arcade Log is read as a base-10 logarithm; Arcade Log is the natural logarithm."""
import numpy as np, rasterio, geopandas as gpd, matplotlib.pyplot as plt
from rasterstats import zonal_stats
src = rasterio.open('dataset/Elevation_2050.tif'); elev = src.read(1)
depth = np.where(elev <= -200, -(elev + 200), np.nan)
bld = gpd.read_file('dataset/StudyAreaBuildings.shp')
bld['depth_cm'] = [z['mean'] for z in zonal_stats(bld, depth, affine=src.transform, stats=['mean'], all_touched=True, nodata=np.nan)]
fl = bld[bld.depth_cm.notna()].copy(); d = fl.depth_cm
fl['damage'] = np.where(d > 1, (0.298 * np.log10(0.01 * d.clip(lower=0.01)) + 1.4502) * 271 * fl.Shape_Area, 0)
total_damage = fl.damage.sum(); print('total damage', total_damage)
fl.to_file('pred_results/flooded_buildings.gpkg')
fl.plot(column='damage', legend=True); plt.savefig('pred_results/damage.png')
