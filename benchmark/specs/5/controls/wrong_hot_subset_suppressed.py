# Wrong: Gi* on small (~440 m) hexagons is correct, but the map draws only the hexagons with cases: those
# with < 5 cases as one grey hatched "suppressed" layer, the few with >= 8 cases coloured by count, and
# outlines the significant hot hexagons only among the latter (2 of 14 significant hot hexagons shown), so
# the hot spot result is not drawn.
import numpy as np, geopandas as gpd, libpysal, matplotlib.pyplot as plt
from shapely.geometry import Polygon
from esda.getisord import G_Local

pts = gpd.read_file('dataset/High_Blood_Level_Results.shp').to_crs(32610)
zips = gpd.read_file('dataset/Sacramento_ZIP_Codes.shp').to_crs(32610)
s = 440.0
xmin, ymin, xmax, ymax = zips.total_bounds
polys = []
for c, x in enumerate(np.arange(xmin, xmax + 2 * s, 1.5 * s)):
    for y in np.arange(ymin - (np.sqrt(3) * s / 2 if c % 2 else 0), ymax + 2 * s, np.sqrt(3) * s):
        polys.append(Polygon([(x + s * np.cos(np.radians(a)), y + s * np.sin(np.radians(a))) for a in range(0, 360, 60)]))
hex_grid = gpd.GeoDataFrame(geometry=polys, crs=32610)
hex_grid = hex_grid[hex_grid.intersects(zips.union_all())].reset_index(drop=True)
hex_grid['case_count'] = gpd.sjoin(pts, hex_grid, predicate='within').groupby('index_right').size() \
    .reindex(hex_grid.index, fill_value=0)
cent = np.c_[hex_grid.centroid.x, hex_grid.centroid.y]
w = libpysal.weights.DistanceBand(cent, threshold=2.1 * np.sqrt(3) * s, binary=True, silence_warnings=True)
hex_grid['Gi_z'] = G_Local(hex_grid['case_count'].values.astype(float), w, star=True, permutations=0).Zs
hex_grid['hot_significant'] = hex_grid['Gi_z'] >= 1.645
hx_vis = hex_grid[hex_grid.case_count > 0].copy()
supp = hx_vis[hx_vis.case_count < 8]
shown = hx_vis[hx_vis.case_count >= 8]
hot_out = shown[shown.hot_significant]
fig, ax = plt.subplots(figsize=(10, 10))
zips.plot(ax=ax, facecolor='#f4f4f4', edgecolor='#444444', linewidth=0.8)
supp.plot(ax=ax, facecolor='#d9d9d9', edgecolor='#999999', hatch='///', linewidth=0.3)
if len(shown):
    shown.plot(column='case_count', ax=ax, cmap='YlOrRd', vmin=8, edgecolor='white', linewidth=0.3)
if len(hot_out):
    hot_out.plot(ax=ax, facecolor='none', edgecolor='#00316e', linewidth=2.2)
print('drawn hexagons:', len(hx_vis), '| suppressed:', len(supp), '| shaded:', len(shown),
      '| significant hot:', int(hex_grid.hot_significant.sum()), '| hot outlined:', len(hot_out))
plt.savefig('pred_results/blood_lead_hotspots.png')
