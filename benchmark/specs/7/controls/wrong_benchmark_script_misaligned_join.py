"""Wrong: the per-building summary is assigned by position from a table indexed by flood polygon, so
the damage is attached to the wrong buildings.

Adapted from the reference script of GeoAnalystBench (Zhang et al. 2025, Apache-2.0).
"""
import geopandas as gpd, rasterio, numpy as np, math, matplotlib.pyplot as plt
from rasterio.features import shapes
buildings = gpd.read_file("dataset/StudyAreaBuildings.shp")
with rasterio.open("dataset/Elevation_2050.tif") as src:
    elevation = src.read(1)
    flooded_area = np.where(elevation <= -200, elevation, np.nan)
    flood_depth = np.where(flooded_area <= -200, (flooded_area + 200) * -1, 0).astype(np.float32)
    polygons = [{'properties': {'raster_value': v}, 'geometry': s} for s, v in shapes(flood_depth, mask=flood_depth != 0, transform=src.transform)]
flood_depth_polygons = gpd.GeoDataFrame.from_features(polygons, crs=buildings.crs)
buildings_in_flood = gpd.sjoin(buildings, flood_depth_polygons, how="inner", predicate='intersects')
summary = buildings_in_flood.groupby('index_right').agg({'raster_value': 'mean', 'geometry': 'size'}).reset_index()
summary.columns = ['index', 'mean_depth', 'num_buildings']
buildings['mean_depth'] = summary['mean_depth']
buildings['DamageCosts'] = buildings.apply(lambda r: (0.298 * math.log(0.01 * r['mean_depth']) + 1.4502) * 271 * r['geometry'].area if r['mean_depth'] > 1 else 0, axis=1)
fig, ax = plt.subplots(figsize=(10, 10))
flood_depth_polygons.plot(ax=ax, column='raster_value', cmap='Blues')
buildings.plot(ax=ax, facecolor='none', edgecolor='red')
plt.savefig("pred_results/flooding_analysis.png")
