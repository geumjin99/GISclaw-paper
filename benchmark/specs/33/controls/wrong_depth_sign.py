"""Wrong: sign error in the flood depth (depths are negative)."""
import numpy as np, rasterio, geopandas as gpd
from rasterstats import zonal_stats
src = rasterio.open('dataset/Elevation_2050.tif')
elev = src.read(1)
depth = np.where(elev <= -200, elev + 200, np.nan)               # sign error: depths negative
bld = gpd.read_file('dataset/StudyAreaBuildings.shp')
bld['depth_cm'] = [z['mean'] for z in zonal_stats(bld, depth, affine=src.transform, stats=['mean'], all_touched=True, nodata=np.nan)]
fl = bld[bld.depth_cm.notna()].copy()
d = fl.depth_cm
fl['damage'] = np.where(d > 1, (0.298 * np.log(0.01 * d.clip(lower=0.01)) + 1.4502) * 271 * fl.Shape_Area, 0)
total = fl.damage.sum()
fl.to_file('pred_results/flooded_buildings.gpkg')
