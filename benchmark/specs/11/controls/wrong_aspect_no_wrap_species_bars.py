# Wrong: aspect converted to a compass bearing with 450-a without mod 360 (12.6 % of aspects fall in
# 360-450), so the per-taxon mean aspects are wrong; the bars draw those wrong means.
import os, json, warnings, math
import numpy as np, pandas as pd, geopandas as gpd
import rasterio
from rasterio import features
from rasterio.plot import show
from rasterio.warp import transform_geom
import matplotlib.pyplot as plt

# Ensure output directory
out_dir = 'pred_results'
os.makedirs(out_dir, exist_ok=True)

# Load raster
raster_path = 'dataset/CatalinaBathymetry.tif'
with rasterio.open(raster_path) as src:
    bathy = src.read(1, masked=True)  # mask nodata
    transform = src.transform
    crs = src.crs
    nodata = src.nodata
    # compute gradient using central differences
    dx = transform.a  # pixel width
    dy = -transform.e  # pixel height (negative)
    # compute gradient
    # Use numpy gradient returning per axis derivatives in spatial units
    grad_y, grad_x = np.gradient(bathy, dy, dx)  # grad_y corresponds to north-south (y), grad_x east-west (x)
    # Slope in degrees
    slope = np.degrees(np.arctan(np.sqrt(grad_x**2 + grad_y**2)))
    # Aspect calculation
    aspect = np.degrees(np.arctan2(grad_y, -grad_x))
    # Convert to compass bearing
    aspect = np.where(aspect < 0, 90.0 - aspect, 450.0 - aspect)
    # Set aspect to -1 where slope is 0 (flat)
    aspect = np.where(slope == 0, -1, aspect)
    # Mask where original bathy is masked
    slope = np.where(bathy.mask, np.nan, slope)
    aspect = np.where(bathy.mask, np.nan, aspect)
    # Save slope and aspect arrays for later sampling
    slope_arr = slope
    aspect_arr = aspect
    # Save raster stats for sampling
    raster_meta = src.meta.copy()

# Load points
points_path = 'dataset/CoralandSpongeCatalina.geojson'
gdf = gpd.read_file(points_path)
# Ensure same CRS
if gdf.crs != crs:
    gdf = gdf.to_crs(crs)

# Sample slope and aspect at point locations
coords = [(x,y) for x,y in zip(gdf.geometry.x, gdf.geometry.y)]
with rasterio.open(raster_path) as src:
    # Compute row, col for each point using src.index
    rows_cols = [src.index(x,y) for x,y in coords]
    slopes = []
    aspects = []
    for row, col in rows_cols:
        # Check bounds
        if (0 <= row < src.height) and (0 <= col < src.width):
            slopes.append(slope_arr[row, col])
            aspects.append(aspect_arr[row, col])
        else:
            slopes.append(np.nan)
            aspects.append(np.nan)

gdf['slope'] = slopes
gdf['aspect'] = aspects

# Save updated geojson
updated_path = os.path.join(out_dir, 'CoralandSpongeCatalina_updated.geojson')
gdf.to_file(updated_path, driver='GeoJSON')

# Compute mean slope and aspect per species (ScientificName)
species_stats = gdf.groupby('ScientificName').agg({'slope':'mean','aspect':'mean'}).reset_index()

# Plot
plt.figure(figsize=(10,6))
# Use two subplots side by side
fig, ax = plt.subplots(1,2, figsize=(14,6))
# sort by mean slope for consistent order
species_stats = species_stats.sort_values('slope')
ax[0].barh(species_stats['ScientificName'], species_stats['slope'], color='skyblue')
ax[0].set_xlabel('Mean Slope (degrees)')
ax[0].set_title('Mean Slope by Species')
ax[1].barh(species_stats['ScientificName'], species_stats['aspect'], color='salmon')
ax[1].set_xlabel('Mean Aspect (degrees)')
ax[1].set_title('Mean Aspect by Species')
plt.tight_layout()
plot_path = os.path.join(out_dir, 'CoralandSponge.png')
plt.savefig(plot_path, dpi=300)
print('Saved plot to', plot_path)
print('Updated geojson saved to', updated_path)
