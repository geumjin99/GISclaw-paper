"""Wrong: the depth in cm is put into ln() without the 0.01 conversion to metres."""
import numpy as np, rasterio, geopandas as gpd
from rasterstats import zonal_stats
src = rasterio.open('dataset/Elevation_2050.tif')
elev = src.read(1)
depth = np.where(elev <= -200, -(elev + 200), np.nan)
bld = gpd.read_file('dataset/StudyAreaBuildings.shp')
bld['depth_cm'] = [z['mean'] for z in zonal_stats(bld, depth, affine=src.transform, stats=['mean'], all_touched=True, nodata=np.nan)]
fl = bld[bld.depth_cm.notna()].copy()
d = fl.depth_cm
fl['damage'] = np.where(d > 1, (0.298 * np.log(d) + 1.4502) * 271 * fl.Shape_Area, 0)   # cm put into ln() without 0.01
total = fl.damage.sum()
fl.to_file('pred_results/flooded_buildings.gpkg')
