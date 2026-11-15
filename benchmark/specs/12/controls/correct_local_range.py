# Correct: local range (max - min in the 3x3 window), a neighbour-difference ruggedness measure; it is the
# measure used by the reference script of GeoAnalystBench.
import numpy as np, rasterio, matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, minimum_filter
dem = rasterio.open('dataset/Elevation.tif').read(1).astype(float)
rugged = maximum_filter(dem, 3) - minimum_filter(dem, 3)
plt.imshow(rugged, cmap='viridis'); plt.colorbar(); plt.savefig('pred_results/ruggedness.png')
