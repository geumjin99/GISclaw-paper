# Wrong: the task prescribes a "quadtree format map"; this aggregates by a regular grid / hexagons, not quadtree cells.
# Grid aggregation, null share computed inline from Species (no per-tree flag column stored).
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from shapely.geometry import box
trees = gpd.read_file('dataset/street_trees_sample.geojson').to_crs(26910)
sf = gpd.read_file('dataset/sfo_boroughs.geojson').to_crs(26910)
x0, y0, x1, y1 = sf.total_bounds
cells = gpd.GeoDataFrame(geometry=[box(x, y, x + 1000, y + 1000) for x in np.arange(x0, x1, 1000) for y in np.arange(y0, y1, 1000)], crs=26910)
j = gpd.sjoin(trees, cells, predicate='within')
cells['null_pct'] = j.groupby('index_right')['Species'].apply(lambda s: s.isna().mean() * 100)
cells['n'] = j.groupby('index_right').size()
grid = cells[cells.n >= 10]
ax = grid.plot(column='null_pct', cmap='Reds', legend=True); sf.boundary.plot(ax=ax, color='k')
plt.savefig('pred_results/trees_count_vis.png')
