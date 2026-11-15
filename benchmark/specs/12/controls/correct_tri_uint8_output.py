# Correct: the TRI exactly as defined (RMS over the
# 3x3 window incl. the zero centre difference, i.e. sqrt(8/9) x TRI), but generic_filter on the
# uint8 band writes into a uint8 output, truncating the TRI (median ~1.3) to integers.
import numpy as np, rasterio, matplotlib.pyplot as plt
from scipy import ndimage
src = rasterio.open('dataset/Elevation.tif')
elevation_raster = src.read(1)
def tri_filter(window):
    center = window[4]
    differences = (window - center) ** 2
    return np.sqrt(np.mean(differences))
ruggedness_raster = ndimage.generic_filter(elevation_raster, tri_filter, size=3)
b = src.bounds
plt.figure(figsize=(10, 8))
im = plt.imshow(ruggedness_raster, cmap='viridis', extent=[b.left, b.right, b.bottom, b.top])
plt.colorbar(im); plt.title('Ruggedness Raster')
plt.savefig('pred_results/ruggedness.png', dpi=150, bbox_inches='tight')
