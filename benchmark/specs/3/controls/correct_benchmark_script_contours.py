# Correct: dNBR > 0.2, skimage contours -> polygons.
# Adapted from the reference script of GeoAnalystBench (Zhang et al. 2025, Apache-2.0).
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.plot import show
import geopandas as gpd
from shapely.geometry import Polygon
from skimage import measure

def calculate_nbr(nir_band, swir_band):
    return (nir_band.astype(float) - swir_band.astype(float)) / (nir_band.astype(float) + swir_band.astype(float))

landsat_2014 = rasterio.open('dataset/G_2014.tif')
landsat_2015 = rasterio.open('dataset/G_2015.tif')
nbr_2014 = calculate_nbr(landsat_2014.read(5), landsat_2014.read(7))
nbr_2015 = calculate_nbr(landsat_2015.read(5), landsat_2015.read(7))
delta_nbr = nbr_2014 - nbr_2015
burn_scar = delta_nbr > 0.2
burn_scar_mask = np.ma.masked_where(~burn_scar, delta_nbr)
contours = measure.find_contours(burn_scar, 0.1)
polygons = []
for contour in contours:
    if len(contour) < 4:
        continue
    contour[:, [0, 1]] = contour[:, [1, 0]]
    contour[:, 0] = landsat_2014.transform[2] + contour[:, 0] * landsat_2014.transform[0]
    contour[:, 1] = landsat_2014.transform[5] + contour[:, 1] * landsat_2014.transform[4]
    polygons.append(Polygon(contour))
burn_scar_gdf = gpd.GeoDataFrame(geometry=polygons, crs=landsat_2014.crs)
fig, ax = plt.subplots(figsize=(10, 10))
show(burn_scar_mask, transform=landsat_2014.transform, cmap='Reds', ax=ax)
burn_scar_gdf.boundary.plot(ax=ax, color='black', linewidth=1)
plt.savefig('pred_results/burn_scar_analysis.png')
