# Correct: all hexagons drawn grey; only the p_sim < 0.05 hexagons drawn again, coloured by Gi z.
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


import h3, inspect, pkgutil, sys, json, pprint, importlib
print('h3 attributes:', [a for a in dir(h3) if not a.startswith('_')][:20])
print('has geo_to_h3?', hasattr(h3, 'geo_to_h3'))
print('dir h3:', [a for a in dir(h3) if 'geo' in a.lower()])


import h3, inspect, sys, pkgutil
attrs = [a for a in dir(h3) if 'h3' in a.lower()]
print(attrs)


import numpy as np, geopandas as gpd, math, shapely.geometry as geom, libpysal, warnings
warnings.filterwarnings('ignore')
from esda import G_Local

points = gpd.read_file('dataset/High_Blood_Level_Results.shp')
zip_codes = gpd.read_file('dataset/Sacramento_ZIP_Codes.shp')
zip_proj = zip_codes.to_crs(points.crs)

def create_hex_grid(gdf, edge_length):
    minx, miny, maxx, maxy = gdf.total_bounds
    width = edge_length * 2
    height = math.sqrt(3) * edge_length
    n_cols = int((maxx - minx) / (width * 0.75)) + 4
    n_rows = int((maxy - miny) / height) + 4
    polys = []
    for col in range(n_cols):
        for row in range(n_rows):
            x = minx + col * (width * 0.75)
            y = miny + row * height + ((col % 2) * (height/2))
            verts = [(x + edge_length * math.cos(math.radians(60*i - 30)),
                      y + edge_length * math.sin(math.radians(60*i - 30))) for i in range(6)]
            polys.append(geom.Polygon(verts))
    hg = gpd.GeoDataFrame(geometry=polys, crs=gdf.crs)
    hg = hg[hg.intersects(gdf.unary_union)].reset_index(drop=True)
    return hg

hex_grid = create_hex_grid(points, edge_length=2000)
joined = gpd.sjoin(points, hex_grid, how='left', predicate='within')
counts = joined.groupby('index_right').size()
hex_grid['count'] = 0
hex_grid.loc[counts.index, 'count'] = counts.values
hex_data = hex_grid[hex_grid['count']>0].copy()
# distance band weights 5000m
w = libpysal.weights.DistanceBand.from_dataframe(hex_data, threshold=5000, silence_warnings=True)
w.transform = 'R'
print('neighbors example for first:', list(w.neighbors.items())[:5])
# ensure float array
y = hex_data['count'].astype(float).values
print('y dtype', y.dtype, 'unique counts', np.unique(y)[:5])
local_g = G_Local(y, w, transform='R', permutations=99)
hex_data['GiZ'] = local_g.Zs
hex_data['p_val'] = local_g.p_sim
hex_data['significant'] = hex_data['p_val'] < 0.05
print('Significant count', hex_data['significant'].sum())


import matplotlib.pyplot as plt
# Ensure plot directory exists
import os
os.makedirs('pred_results', exist_ok=True)

fig, ax = plt.subplots(figsize=(10,10))
# Plot ZIP boundaries for context
zip_proj.boundary.plot(ax=ax, linewidth=0.5, edgecolor='black')
# Plot all hexes with light gray fill for context
hex_grid.plot(ax=ax, facecolor='lightgrey', edgecolor='white', linewidth=0.3)
# Plot significant hotspots with GiZ colors
sig = hex_data[hex_data['significant']]
sig.plot(column='GiZ', cmap='coolwarm', legend=True, ax=ax, edgecolor='none')
ax.set_title('Optimized Hot Spot Analysis (Getis‑Ord Gi*) of High Blood Lead Levels')
ax.set_axis_off()
plt.tight_layout()
plt.savefig('pred_results/blood_lead_hotspots.png', dpi=300)
plt.close()
print('Saved plot to pred_results/blood_lead_hotspots.png')
