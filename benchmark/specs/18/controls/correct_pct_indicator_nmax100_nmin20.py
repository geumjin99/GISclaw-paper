# Correct: geoplot quadtree with hue = Species.isnull()*100 (a 0/100 percent indicator) and mean
# aggregation; the final map uses nmax=100, nmin=20.
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, calculate_default_transform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, MultiPolygon, box
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')
os.makedirs('pred_results', exist_ok=True)


trees = gpd.read_file('dataset/street_trees_sample.geojson')
print(f"Rows: {len(trees)}")
print(f"CRS: {trees.crs.to_string() if trees.crs is not None else None}")
print(f"Geometry types: { {k: int(c) for k, c in trees.geom_type.value_counts().items()} }")
print(f"Bounds (minx, miny, maxx, maxy): {[round(float(_b), 4) for _b in trees.total_bounds]}")
_cols = [c for c in trees.columns if c != trees.geometry.name]
print(f"Columns ({len(_cols)}): {_cols}")
import pandas as _pd
with _pd.option_context('display.max_columns', 30, 'display.width', 200, 'display.max_colwidth', 40):
    print(trees[_cols].head(3).to_string())


import geopandas as gpd, geoplot as gplt, geoplot.crs as gcrs, matplotlib.pyplot as plt
boroughs = gpd.read_file('dataset/sfo_boroughs.geojson')
print(boroughs.shape, boroughs.crs)
print(trees.Species.isnull().mean(), trees.Species.value_counts().head())

import numpy as np, os
os.makedirs('pred_results', exist_ok=True)
t = trees.assign(nullity=trees['Species'].isnull().astype(int)*100)
ax = gplt.quadtree(t, nmax=1, projection=gcrs.AlbersEqualArea(), clip=boroughs.geometry,
                   hue='nullity', cmap='Greens', agg=np.mean, edgecolor='white',
                   linewidth=0.5, legend=True, figsize=(10,10))
gplt.polyplot(boroughs, ax=ax, edgecolor='black', linewidth=1, zorder=2)
ax.set_title('San Francisco street trees: % with NULL species (quadtree aggregation)', fontsize=13)
plt.savefig('pred_results/trees_count_vis.png', dpi=150, bbox_inches='tight')
plt.close()
print(os.path.getsize('pred_results/trees_count_vis.png'))

ax = gplt.quadtree(t, nmax=100, nmin=20, projection=gcrs.AlbersEqualArea(), clip=boroughs.geometry,
                   hue='nullity', cmap='Greens', agg=np.mean, edgecolor='white',
                   linewidth=0.5, legend=True, legend_kwargs={'label':'% of trees with NULL species','shrink':0.6}, figsize=(10,10))
gplt.polyplot(boroughs, ax=ax, edgecolor='black', linewidth=1, zorder=2)
ax.set_title('San Francisco street trees: % with NULL species (quadtree aggregation)', fontsize=13)
vals=[c.get_array() for c in ax.collections if c.get_array() is not None]
print([ (np.nanmin(v),np.nanmax(v),len(v)) for v in vals])
plt.savefig('pred_results/trees_count_vis.png', dpi=150, bbox_inches='tight')
plt.close()

ax = gplt.quadtree(t, nmax=100, nmin=20, projection=gcrs.AlbersEqualArea(), clip=boroughs.geometry,
                   hue='nullity', cmap='Greens', agg=np.mean, edgecolor='white', linewidth=0.5, legend=True,
                   legend_kwargs={'label':'% of trees with NULL species','shrink':0.6}, figsize=(10,10))
print(len(ax.patches), len(ax.collections), [type(c).__name__ for c in ax.collections][:5])
fig=ax.figure; print([a.get_ylabel() for a in fig.axes])
plt.close()