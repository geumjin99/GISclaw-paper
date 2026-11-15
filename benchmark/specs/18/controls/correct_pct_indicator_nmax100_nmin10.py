# Correct: geoplot quadtree with hue = Species.isnull()*100 (a 0/100 percent indicator) and mean
# aggregation; the final map uses nmax=100, nmin=10.
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
import matplotlib.pyplot as plt
trees=gpd.read_file('dataset/street_trees_sample.geojson'); b=gpd.read_file('dataset/sfo_boroughs.geojson')
print(trees.shape, trees.crs, b.shape, b.crs, trees.Species.isnull().mean())

import os; os.makedirs('pred_results',exist_ok=True)
t=trees.assign(nullity=trees['Species'].isnull().astype(int)*100)
ax=gplt.quadtree(t, projection=gcrs.AlbersEqualArea(), hue='nullity', nmax=1, cmap='Reds', clip=b.geometry, edgecolor='white', linewidth=0.5, legend=True, figsize=(10,10))
gplt.polyplot(b, ax=ax, edgecolor='black', linewidth=1, zorder=2)
ax.set_title('Percentage of San Francisco street trees with NULL species (quadtree)')
plt.savefig('pred_results/trees_count_vis.png', dpi=150, bbox_inches='tight')

import numpy as np
from PIL import Image
im=Image.open('pred_results/trees_count_vis.png'); print(im.size)
plt.close('all')
fig=plt.figure(figsize=(6,6)); plt.imshow(im); plt.axis('off'); plt.savefig('/tmp/check.png')
import inspect; print(inspect.signature(gplt.quadtree))

plt.close('all')
ax=gplt.quadtree(t, projection=gcrs.AlbersEqualArea(), hue='nullity', nmax=100, nmin=10, agg=np.mean, cmap='Reds', clip=b.geometry, edgecolor='white', linewidth=0.5, legend=True, legend_kwargs={'label':'% of trees with NULL species','shrink':0.7}, figsize=(10,10))
gplt.polyplot(b, ax=ax, edgecolor='black', linewidth=1, zorder=2)
ax.set_title('San Francisco street trees: percentage with NULL species\n(quadtree aggregation)', fontsize=14)
vals=[c.get_array() for c in ax.collections if c.get_array() is not None]
print([ (len(v), np.nanmin(v), np.nanmax(v)) for v in vals])
plt.savefig('pred_results/trees_count_vis.png', dpi=150, bbox_inches='tight')

print(len(ax.patches), len(ax.collections))
fc=set(tuple(np.round(p.get_facecolor(),2)) for p in ax.patches); print(len(fc))

fcs=[tuple(np.round(c.get_facecolor()[0],2)) for c in ax.collections if len(c.get_facecolor())]
print(len(set(fcs)))