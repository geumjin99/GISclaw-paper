"""Correct: the same analysis inside a function; the result is written as a shapefile and the total
to a text file, without a figure.
"""
import numpy as np, rasterio, geopandas as gpd
from rasterstats import zonal_stats
def run():
    src = rasterio.open('dataset/Elevation_2050.tif'); elev = src.read(1)
    depth = np.where(elev <= -200, -(elev + 200), np.nan)
    bld = gpd.read_file('dataset/StudyAreaBuildings.shp')
    bld['mean_depth'] = [z['mean'] for z in zonal_stats(bld, depth, affine=src.transform, stats=['mean'], all_touched=True, nodata=np.nan)]
    fl = bld[bld.mean_depth.notna()].copy(); d = fl.mean_depth
    fl['DamageCost'] = np.where(d > 1, (0.298 * np.log(0.01 * d.clip(lower=0.01)) + 1.4502) * 271 * fl.Shape_Area, 0)
    fl.to_file('pred_results/flood_damage_buildings.shp')
    open('pred_results/summary.txt', 'w').write(f'flooded buildings {len(fl)}; total damage {fl.DamageCost.sum():.0f}\n')
run()
