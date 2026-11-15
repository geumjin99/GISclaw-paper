"""Wrong: skips the raster calculation and uses the (negative) elevation itself as the flood depth for
cells below -200 cm, and the perimeter (Shape_Leng) as the building size.
"""
import numpy as np, geopandas as gpd, rasterio, matplotlib.pyplot as plt
from rasterstats import zonal_stats
b = gpd.read_file('dataset/StudyAreaBuildings.shp')
with rasterio.open('dataset/Elevation_2050.tif') as src:
    elev = src.read(1).astype(float); tr = src.transform
depth = np.where(elev <= -200, -elev, 0.0)
b['mean_depth'] = [z['mean'] or 0 for z in zonal_stats(b, depth, affine=tr, nodata=0, stats=['mean'])]
b['damage'] = np.where(b.mean_depth > 0, (0.298 * np.log(0.01 * b.mean_depth.clip(lower=1e-9)) + 1.4502) * 271 * b.Shape_Leng, 0)
fig, ax = plt.subplots()
ax.imshow(depth, cmap='Blues')
plt.savefig('pred_results/flooding_analysis.png')
