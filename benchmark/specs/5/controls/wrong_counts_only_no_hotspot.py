# Wrong: hexagon counts only - the hot spot analysis step is skipped.
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from shapely.geometry import Polygon
pts = gpd.read_file('dataset/High_Blood_Level_Results.shp').to_crs(32610)
zips = gpd.read_file('dataset/Sacramento_ZIP_Codes.shp').to_crs(32610)
s = 620.4
xmin, ymin, xmax, ymax = zips.total_bounds
polys = [Polygon([(x + s * np.cos(np.radians(a)), y + s * np.sin(np.radians(a))) for a in range(0, 360, 60)])
         for c, x in enumerate(np.arange(xmin, xmax + 2 * s, 1.5 * s))
         for y in np.arange(ymin - (np.sqrt(3) * s / 2 if c % 2 else 0), ymax + 2 * s, np.sqrt(3) * s)]
hx = gpd.GeoDataFrame(geometry=polys, crs=32610)
hx = hx[hx.intersects(zips.union_all())].reset_index(drop=True)
hx['cases'] = gpd.sjoin(pts, hx, predicate='within').groupby('index_right').size().reindex(hx.index, fill_value=0)
hx['density'] = hx['cases'] / (hx.area / 1e6)
hx.plot(column='density', cmap='Reds', legend=True)
plt.savefig('pred_results/transit_access.png')
