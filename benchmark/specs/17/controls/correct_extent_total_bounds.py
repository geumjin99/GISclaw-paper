# Correct: geoplot pointplot whose map extent is given as the total_bounds of the boroughs layer,
# so all crashes are visible.
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


import geopandas as gpd
b=gpd.read_file('dataset/nyc_boroughs.geojson');c=gpd.read_file('dataset/fatal_collisions.geojson')
print(b.head(),b.crs,c.head(),c.crs,len(c),c['NUMBER OF PERSONS KILLED'].value_counts())

import geoplot as gplt, geoplot.crs as gcrs, matplotlib.pyplot as plt, os
os.makedirs('pred_results',exist_ok=True)
ax=gplt.polyplot(b,projection=gcrs.AlbersEqualArea(),facecolor='lightgray',edgecolor='white',figsize=(10,10))
gplt.pointplot(c,ax=ax,hue='BOROUGH',scale='NUMBER OF PERSONS KILLED',limits=(6,14),legend=True,legend_var='hue',cmap='Set1',edgecolor='black',linewidth=0.5,extent=b.total_bounds)
ax.set_title('Fatal Car Crashes in New York City, 2016',fontsize=16)
plt.savefig('pred_results/collisions_map_vis.png',dpi=150,bbox_inches='tight')