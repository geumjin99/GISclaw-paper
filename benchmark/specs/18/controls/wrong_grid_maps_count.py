# Wrong: computes the null share per grid cell correctly, but the map shows the tree count.
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from shapely.geometry import box
trees = gpd.read_file('dataset/street_trees_sample.geojson').to_crs(26910)
sf = gpd.read_file('dataset/sfo_boroughs.geojson').to_crs(26910)
trees['is_null'] = trees['Species'].isna().astype(int)
x0, y0, x1, y1 = sf.total_bounds
cells = gpd.GeoDataFrame(geometry=[box(x, y, x + 800, y + 800) for x in np.arange(x0, x1, 800) for y in np.arange(y0, y1, 800)], crs=26910)
j = gpd.sjoin(trees, cells, predicate='within')
stats = j.groupby('index_right')['is_null'].agg(['mean', 'size'])
cells['null_share'] = stats['mean']
cells['tree_count'] = stats['size']
cells = cells.dropna(subset=['tree_count'])
ax = cells.plot(column='tree_count', cmap='Greens', legend=True)
sf.boundary.plot(ax=ax, color='k')
ax.set_title('Street trees per cell')
plt.savefig('pred_results/trees_count_vis.png')
