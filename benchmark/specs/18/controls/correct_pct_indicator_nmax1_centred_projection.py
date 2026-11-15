# Correct: geoplot quadtree with hue = Species.isnull()*100 (a 0/100 percent indicator) and mean
# aggregation; nmax=1, Albers projection centred on the city.
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
print(trees['Species'].isnull().sum(), trees['Species'].isnull().mean())

import numpy as np, os
os.makedirs('pred_results', exist_ok=True)
data = trees.assign(nullity=trees['Species'].isnull().astype(int)*100)
proj = gcrs.AlbersEqualArea(central_latitude=37.7749, central_longitude=-122.4194)
ax = gplt.quadtree(data, projection=proj, hue='nullity', nmax=1, agg=np.mean,
                   clip=boroughs.geometry, cmap='Reds', edgecolor='white', linewidth=0.3,
                   legend=True, figsize=(12,10))
gplt.polyplot(boroughs, ax=ax, edgecolor='black', linewidth=1, zorder=2)
ax.set_title('San Francisco Street Trees: % of Trees with NULL Species (quadtree)', fontsize=14)
plt.savefig('pred_results/trees_count_vis.png', dpi=150, bbox_inches='tight')
plt.close()
print(os.path.getsize('pred_results/trees_count_vis.png'))