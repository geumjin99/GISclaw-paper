# Correct: geoplot quadtree with hue = Species.isnull()*100 (a 0/100 percent indicator) and mean
# aggregation; nmax=1, the call wrapped in try/except.
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


import geopandas as gpd, geoplot as gplt, geoplot.crs as gcrs, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, os
os.makedirs('pred_results', exist_ok=True)
trees = gpd.read_file('dataset/street_trees_sample.geojson')
boroughs = gpd.read_file('dataset/sfo_boroughs.geojson')
print(trees.shape, trees.crs, boroughs.shape, boroughs.crs)
print(trees['Species'].isnull().mean(), trees.total_bounds, boroughs.total_bounds)

import numpy as np
t = trees.assign(nullity=trees['Species'].isnull().astype(int) * 100)
try:
    ax = gplt.quadtree(t, nmax=1, projection=gcrs.AlbersEqualArea(), clip=boroughs.geometry,
                       hue='nullity', cmap='Reds', agg=np.mean, edgecolor='white',
                       legend=True, figsize=(10, 10))
    gplt.polyplot(boroughs, ax=ax, edgecolor='black', linewidth=0.5)
except Exception as e:
    print('err', repr(e)); raise
plt.title('San Francisco Street Trees: % of Trees with NULL Species (quadtree)')
plt.savefig('pred_results/trees_count_vis.png', dpi=150, bbox_inches='tight')
print(os.path.getsize('pred_results/trees_count_vis.png'))