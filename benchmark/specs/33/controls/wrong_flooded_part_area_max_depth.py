"""Wrong: uses the maximum depth per building instead of the mean, and omits the '> 1 cm' rule."""
import numpy as np, rasterio, geopandas as gpd, matplotlib.pyplot as plt
from rasterstats import zonal_stats
src = rasterio.open('dataset/Elevation_2050.tif'); elev = src.read(1)
depth = np.where(elev <= -200, -(elev + 200), np.nan)
bld = gpd.read_file('dataset/StudyAreaBuildings.shp')
bld['depth_cm'] = [z['max'] for z in zonal_stats(bld, depth, affine=src.transform, stats=['max'], all_touched=True, nodata=np.nan)]
fl = bld[bld.depth_cm.notna()].copy(); d = fl.depth_cm.clip(lower=0.5)
fl['damage'] = (0.298 * np.log(0.01 * d) + 1.4502) * 271 * fl.Shape_Area
total_damage = fl.damage.sum(); print('total damage', total_damage)
fl.to_file('pred_results/flooded_buildings.gpkg')
fl.plot(column='damage', legend=True); plt.savefig('pred_results/damage.png')
