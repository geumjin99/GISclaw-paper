# Wrong: the TRI is computed, but the saved figure (imshow) shows the elevation.
import numpy as np, rasterio, matplotlib.pyplot as plt
dem = rasterio.open('dataset/Elevation.tif').read(1).astype(float)
p = np.pad(dem, 1, mode='edge'); H, W = dem.shape
tri = np.sqrt(sum((p[1+dy:1+dy+H, 1+dx:1+dx+W] - dem) ** 2 for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 8)
plt.imshow(dem, cmap='viridis')          # plots the elevation, not the TRI just computed
plt.colorbar(label='Ruggedness'); plt.savefig('pred_results/ruggedness.png')
