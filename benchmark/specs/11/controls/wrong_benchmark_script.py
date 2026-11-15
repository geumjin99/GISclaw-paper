# Wrong: depth surface (elevation * -1), np.gradient spacing = array shape, and the raster georeferenced
# from the point bounding box instead of its own transform, so slope, aspect and sampling are all off.
# Adapted from the reference script of GeoAnalystBench (Zhang et al. 2025, Apache-2.0).
import geopandas as gpd, numpy as np, rasterio, matplotlib.pyplot as plt
from rasterio.transform import from_origin
raster = rasterio.open("dataset/CatalinaBathymetry.tif")
imarray = raster.read(1) * -1
coral_sponge_gdf = gpd.read_file("dataset/CoralandSpongeCatalina.geojson").to_crs(epsg=4326)
grad_y, grad_x = np.gradient(imarray, imarray.shape[0], imarray.shape[1])
slope = np.arctan(np.sqrt(grad_x**2 + grad_y**2)) * (180 / np.pi)
aspect = np.arctan2(-grad_y, grad_x) * (180 / np.pi)
aspect = np.where(aspect < 0, 360 + aspect, aspect)
xmin, ymin, xmax, ymax = coral_sponge_gdf.total_bounds
height, width = slope.shape
transform = from_origin(xmin, ymax, (xmax - xmin) / width, (ymax - ymin) / height)
sv, av = [], []
for geom in coral_sponge_gdf.geometry:
    col = int((geom.x - transform[2]) / transform[0]); row = int((geom.y - transform[5]) / transform[4])
    ok = 0 <= row < height and 0 <= col < width
    sv.append(slope[row, col] if ok else np.nan); av.append(aspect[row, col] if ok else np.nan)
coral_sponge_gdf['slope'] = sv; coral_sponge_gdf['aspect'] = av
mean_slope_aspect = coral_sponge_gdf.groupby('VernacularNameCategory')[['slope', 'aspect']].mean().reset_index()
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1); plt.bar(mean_slope_aspect['VernacularNameCategory'], mean_slope_aspect['slope'])
plt.subplot(1, 2, 2); plt.bar(mean_slope_aspect['VernacularNameCategory'], mean_slope_aspect['aspect'])
plt.savefig("pred_results/CoralandSponge.png")
