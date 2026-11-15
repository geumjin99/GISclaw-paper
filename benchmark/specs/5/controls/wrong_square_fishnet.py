# Wrong: square fishnet instead of the requested hexagons (counts and Gi* otherwise fine).
import numpy as np, geopandas as gpd, libpysal, matplotlib.pyplot as plt
from shapely.geometry import box
from esda.getisord import G_Local
pts = gpd.read_file('dataset/High_Blood_Level_Results.shp').to_crs(32610)
zips = gpd.read_file('dataset/Sacramento_ZIP_Codes.shp').to_crs(32610)
x0, y0, x1, y1 = zips.total_bounds
grid = gpd.GeoDataFrame(geometry=[box(x, y, x + 1000, y + 1000) for x in np.arange(x0, x1, 1000) for y in np.arange(y0, y1, 1000)], crs=32610)
grid = grid[grid.intersects(zips.union_all())].reset_index(drop=True)
grid['count'] = gpd.sjoin(pts, grid, predicate='within').groupby('index_right').size().reindex(grid.index, fill_value=0)
w = libpysal.weights.Queen.from_dataframe(grid, use_index=False)
grid['gi_z'] = G_Local(grid['count'].values.astype(float), w, star=True, permutations=0).Zs
grid.plot(column='gi_z', cmap='RdBu_r', legend=True)
plt.savefig('pred_results/transit_access.png')
