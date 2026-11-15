# Correct: hand-written quadtree (as correct_custom_quadtree_lonlat), each leaf drawn with
# ax.add_patch(matplotlib Polygon), facecolor = ScalarMappable.to_rgba(NULL %).
import numpy as np, pandas as pd, geopandas as gpd, matplotlib.pyplot as plt
from shapely.geometry import box
trees = gpd.read_file('dataset/street_trees_sample.geojson').to_crs(4326)
sf = gpd.read_file('dataset/sfo_boroughs.geojson').to_crs(4326)
x = trees.geometry.x.to_numpy(); y = trees.geometry.y.to_numpy()
isnull = trees['Species'].isna().to_numpy()
leaves = []
def build(x0, y0, x1, y1, idx, depth=0):
    if len(idx) > 300 and depth < 8:
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        for a, b, c, d in ((x0, y0, xm, ym), (xm, y0, x1, ym), (x0, ym, xm, y1), (xm, ym, x1, y1)):
            m = (x[idx] >= a) & (x[idx] < c) & (y[idx] >= b) & (y[idx] < d)
            build(a, b, c, d, idx[m], depth + 1)
    elif len(idx):
        leaves.append({'geometry': box(x0, y0, x1, y1), 'n': len(idx), 'pct_null_species': 100 * isnull[idx].mean()})
bx0, by0, bx1, by1 = trees.total_bounds
build(bx0 - 1e-6, by0 - 1e-6, bx1 + 1e-6, by1 + 1e-6, np.arange(len(trees)))
qt = gpd.GeoDataFrame(pd.DataFrame(leaves), geometry='geometry', crs=4326)
ax = qt.plot(column='pct_null_species', cmap='OrRd', edgecolor='grey', linewidth=0.3, legend=True, figsize=(8, 8))
import matplotlib.cm as cm, matplotlib.colors as mcolors
from matplotlib.patches import Polygon as MplPolygon
norm = mcolors.Normalize(qt['pct_null_species'].min(), qt['pct_null_species'].max())
mappable = cm.ScalarMappable(cmap='OrRd', norm=norm)
fig, ax = plt.subplots(figsize=(8, 8))
for _, row in qt.iterrows():
    ax.add_patch(MplPolygon(np.asarray(row.geometry.exterior.coords), closed=True,
                            facecolor=mappable.to_rgba(row['pct_null_species']), edgecolor='grey', linewidth=0.3))
sf.boundary.plot(ax=ax, color='k', linewidth=0.8)
fig.colorbar(mappable, ax=ax, label='NULL species %')
ax.set_xlim(bx0, bx1); ax.set_ylim(by0, by1)
plt.savefig('pred_results/trees_count_vis.png', dpi=120)
