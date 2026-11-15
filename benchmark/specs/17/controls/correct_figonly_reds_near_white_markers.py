# Correct analysis, presentation wrong: geoplot pointplot with hue by persons killed and cmap 'Reds'
# makes the 70 one-death crashes near-white on a white map. Fails only the kind="figure"
# marker-visibility check.
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
boroughs_gdf = gpd.read_file('dataset/nyc_boroughs.geojson')
print(f"Rows: {len(boroughs_gdf)}")
print(f"CRS: {boroughs_gdf.crs.to_string() if boroughs_gdf.crs is not None else None}")
print(f"Geometry types: { {k: int(c) for k, c in boroughs_gdf.geom_type.value_counts().items()} }")
print(f"Bounds (minx, miny, maxx, maxy): {[round(float(_b), 4) for _b in boroughs_gdf.total_bounds]}")
_cols = [c for c in boroughs_gdf.columns if c != boroughs_gdf.geometry.name]
print(f"Columns ({len(_cols)}): {_cols}")
import pandas as _pd
with _pd.option_context('display.max_columns', 30, 'display.width', 200, 'display.max_colwidth', 40):
    print(boroughs_gdf[_cols].head(3).to_string())


collisions_gdf = gpd.read_file('dataset/fatal_collisions.geojson')
print(f"Rows: {len(collisions_gdf)}")
print(f"CRS: {collisions_gdf.crs.to_string() if collisions_gdf.crs is not None else None}")
print(f"Geometry types: { {k: int(c) for k, c in collisions_gdf.geom_type.value_counts().items()} }")
print(f"Bounds (minx, miny, maxx, maxy): {[round(float(_b), 4) for _b in collisions_gdf.total_bounds]}")
_cols = [c for c in collisions_gdf.columns if c != collisions_gdf.geometry.name]
print(f"Columns ({len(_cols)}): {_cols}")
import pandas as _pd
with _pd.option_context('display.max_columns', 30, 'display.width', 200, 'display.max_colwidth', 40):
    print(collisions_gdf[_cols].head(3).to_string())


import geoplot
import geoplot.crs as gcrs
import matplotlib.pyplot as plt

# Ensure the coordinate reference system is consistent (both are EPSG:4326)
# Plot borough polygons as background
fig, ax = plt.subplots(figsize=(10,10), subplot_kw={'projection': gcrs.AlbersEqualArea()})
# Plot boroughs
geoplot.polyplot(boroughs_gdf, ax=ax, edgecolor='black', facecolor='none')
# Plot collisions, size based on number killed (scale factor)
# Use a column for size, e.g., 'NUMBER OF PERSONS KILLED'
geoplot.pointplot(collisions_gdf, ax=ax, hue='NUMBER OF PERSONS KILLED', cmap='Reds', s=30, legend=True)
ax.set_title('Fatal Car Collisions in NYC (2016)')
plt.savefig('pred_results/collisions_map_vis.png', dpi=300, bbox_inches='tight')
plt.close()
