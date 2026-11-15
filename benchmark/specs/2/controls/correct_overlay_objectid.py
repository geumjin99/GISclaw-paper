# Correct: the overlay brings BusServiceArea's OBJECTID (=1 on every row) into the block-group table,
# so the drawn factors can only be aligned by geometry.
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


bus_gdf = gpd.read_file('dataset/BusServiceArea.geojson')
print(f"Rows: {len(bus_gdf)}")
print(f"CRS: {bus_gdf.crs.to_string() if bus_gdf.crs is not None else None}")
print(f"Geometry types: { {k: int(c) for k, c in bus_gdf.geom_type.value_counts().items()} }")
print(f"Bounds (minx, miny, maxx, maxy): {[round(float(_b), 4) for _b in bus_gdf.total_bounds]}")
_cols = [c for c in bus_gdf.columns if c != bus_gdf.geometry.name]
print(f"Columns ({len(_cols)}): {_cols}")
import pandas as _pd
with _pd.option_context('display.max_columns', 30, 'display.width', 200, 'display.max_colwidth', 40):
    print(bus_gdf[_cols].head(3).to_string())


demo_gdf = gpd.read_file('dataset/HamiltonDemographics.geojson')
print(f"Rows: {len(demo_gdf)}")
print(f"CRS: {demo_gdf.crs.to_string() if demo_gdf.crs is not None else None}")
print(f"Geometry types: { {k: int(c) for k, c in demo_gdf.geom_type.value_counts().items()} }")
print(f"Bounds (minx, miny, maxx, maxy): {[round(float(_b), 4) for _b in demo_gdf.total_bounds]}")
_cols = [c for c in demo_gdf.columns if c != demo_gdf.geometry.name]
print(f"Columns ({len(_cols)}): {_cols}")
import pandas as _pd
with _pd.option_context('display.max_columns', 30, 'display.width', 200, 'display.max_colwidth', 40):
    print(demo_gdf[_cols].head(3).to_string())


import geopandas as gpd, matplotlib.pyplot as plt
bus=bus_gdf.to_crs(epsg=4326)
demo=demo_gdf.to_crs(epsg=4326)
demo=demo.copy()
demo['pop_density']=demo['populationtotals_TOTPOP_CY']/(demo['AREALAND']/1e6)
cols=['households_ACSHHBPOV','AtRisk_ACSOVEH0','pop_density','geometry']
demo_subset=demo[cols]
intersect=gpd.overlay(demo_subset,bus,how='intersection')
print('Intersection rows:',len(intersect))
print('Poverty range:',intersect['households_ACSHHBPOV'].min(),intersect['households_ACSHHBPOV'].max())
print('Vehicle risk range:',intersect['AtRisk_ACSOVEH0'].min(),intersect['AtRisk_ACSOVEH0'].max())
print('Population density range:',intersect['pop_density'].min(),intersect['pop_density'].max())
fig,axes=plt.subplots(1,3,figsize=(18,6),constrained_layout=True)
intersect.plot(column='households_ACSHHBPOV',cmap='Reds',legend=True,ax=axes[0],alpha=0.6)
bus.boundary.plot(ax=axes[0],edgecolor='black',linewidth=1)
axes[0].set_title('Poverty')
axes[0].axis('off')
intersect.plot(column='AtRisk_ACSOVEH0',cmap='Blues',legend=True,ax=axes[1],alpha=0.6)
bus.boundary.plot(ax=axes[1],edgecolor='black',linewidth=1)
axes[1].set_title('Vehicle risk')
axes[1].axis('off')
intersect.plot(column='pop_density',cmap='Greens',legend=True,ax=axes[2],alpha=0.6)
bus.boundary.plot(ax=axes[2],edgecolor='black',linewidth=1)
axes[2].set_title('Population density')
axes[2].axis('off')
plt.savefig('pred_results/transit_access.png',dpi=300)
plt.close()
