# Wrong: value-weighted KDE as hinted by the Domain Knowledge; it estimates sampling density, not DO.
# Adapted from the reference script of GeoAnalystBench (Zhang et al. 2025, Apache-2.0).
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from shapely.geometry import Point
points_gdf = gpd.read_file("dataset/DissolvedO2.geojson"); polygons_gdf = gpd.read_file("dataset/Bay.geojson")
coords = np.array([(p.x, p.y) for p in points_gdf.geometry]); values = points_gdf['MeasureValue'].values
kde = gaussian_kde(coords.T, weights=values, bw_method=0.6)
minx, miny, maxx, maxy = polygons_gdf.total_bounds
x, y = np.mgrid[minx:maxx:150j, miny:maxy:150j]
z = kde(np.vstack([x.ravel(), y.ravel()])).reshape(x.shape)
mask = np.array([polygons_gdf.contains(Point(px, py)).any() for px, py in zip(x.ravel(), y.ravel())]).reshape(x.shape)
z_masked = np.where(mask, z, np.nan)
plt.figure(figsize=(8, 10)); plt.imshow(z_masked.T, origin='lower', extent=(minx, maxx, miny, maxy), cmap='Spectral')
polygons_gdf.boundary.plot(ax=plt.gca(), edgecolor='white'); plt.colorbar()
plt.savefig("pred_results/interploated_water_quality.png")
