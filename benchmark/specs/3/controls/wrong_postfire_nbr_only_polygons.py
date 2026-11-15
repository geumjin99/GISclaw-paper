# Wrong: dNBR computed and shown, but the "damage" polygons come from low post-fire NBR (NBR_2015 < 0.1),
# i.e. bare ground + burned areas, not from the change.
import numpy as np, rasterio, geopandas as gpd, matplotlib.pyplot as plt
from rasterio import features
from shapely.geometry import shape
src14 = rasterio.open('dataset/G_2014.tif'); src15 = rasterio.open('dataset/G_2015.tif')
nbr14 = (src14.read(5).astype(float) - src14.read(7)) / (src14.read(5).astype(float) + src14.read(7))
nbr15 = (src15.read(5).astype(float) - src15.read(7)) / (src15.read(5).astype(float) + src15.read(7))
d_nbr = nbr14 - nbr15
low = (np.nan_to_num(nbr15, nan=1) < 0.1).astype('uint8')
low = features.sieve(low, 11)
burn_poly = gpd.GeoDataFrame(geometry=[shape(g) for g, v in features.shapes(low, mask=low == 1, transform=src15.transform)], crs=src15.crs)
fig, ax = plt.subplots(figsize=(10, 10))
b = src14.bounds
ax.imshow(d_nbr, extent=(b.left, b.right, b.bottom, b.top), cmap='RdYlGn_r', vmin=-0.3, vmax=0.7)
burn_poly.boundary.plot(ax=ax, color='k', linewidth=0.5)
plt.savefig('pred_results/burn_scar_analysis.png')
