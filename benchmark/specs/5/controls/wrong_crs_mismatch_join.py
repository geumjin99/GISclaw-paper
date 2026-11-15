# Wrong: hexagons built in UTM but joined with the points still in Web Mercator (CRS mismatch
# ignored) -> almost no cases land in the hexagons; Gi* on the resulting counts.
import numpy as np, geopandas as gpd, libpysal, matplotlib.pyplot as plt
from shapely.geometry import Polygon
from esda.getisord import G_Local
pts = gpd.read_file('dataset/High_Blood_Level_Results.shp')            # EPSG:3857
zips = gpd.read_file('dataset/Sacramento_ZIP_Codes.shp').to_crs(32610)
s = 620.4
xmin, ymin, xmax, ymax = zips.total_bounds
polys = [Polygon([(x + s * np.cos(np.radians(a)), y + s * np.sin(np.radians(a))) for a in range(0, 360, 60)])
         for c, x in enumerate(np.arange(xmin, xmax + 2 * s, 1.5 * s))
         for y in np.arange(ymin - (np.sqrt(3) * s / 2 if c % 2 else 0), ymax + 2 * s, np.sqrt(3) * s)]
hx = gpd.GeoDataFrame(geometry=polys, crs=32610)
hx = hx[hx.intersects(zips.union_all())].reset_index(drop=True)
pts = pts.set_crs(32610, allow_override=True)
hx['cases'] = gpd.sjoin(pts, hx, predicate='within').groupby('index_right').size().reindex(hx.index, fill_value=0)
w = libpysal.weights.Queen.from_dataframe(hx, use_index=False)
hx['gi_z'] = G_Local(hx['cases'].values.astype(float) + np.random.rand(len(hx)) * 1e-6, w, star=True, permutations=0).Zs
hx.plot(column='gi_z', cmap='RdBu_r', legend=True)
plt.savefig('pred_results/transit_access.png')
