"""Correct: building ids kept in a column; the result is a plain table (no geometry) of flooded
buildings written as CSV.
"""
import numpy as np, pandas as pd, rasterio, geopandas as gpd
from rasterstats import zonal_stats
src = rasterio.open('dataset/Elevation_2050.tif'); elev = src.read(1)
depth = np.where(elev <= -200, -(elev + 200), np.nan)
bld = gpd.read_file('dataset/StudyAreaBuildings.shp')
bld['bldg_id'] = np.arange(len(bld))
zs = zonal_stats(bld, depth, affine=src.transform, stats=['mean'], all_touched=True, nodata=np.nan)
res = pd.DataFrame({'bldg_id': bld.bldg_id, 'area_m2': bld.Shape_Area, 'mean_depth_cm': [z['mean'] for z in zs]})
res = res.dropna(subset=['mean_depth_cm']).reset_index(drop=True)
d = res.mean_depth_cm
res['damage'] = np.where(d > 1, (0.298 * np.log(0.01 * d.clip(lower=0.01)) + 1.4502) * 271 * res.area_m2, 0)
res.to_csv('pred_results/building_flood_damage.csv', index=False)
print('total', res.damage.sum())
