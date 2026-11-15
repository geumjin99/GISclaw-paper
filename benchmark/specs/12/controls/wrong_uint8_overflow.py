# Wrong: the neighbour differences are computed on the uint8 band and wrap around, so the TRI is corrupted.
import numpy as np, rasterio, matplotlib.pyplot as plt
dem = rasterio.open('dataset/Elevation.tif').read(1)       # uint8: differences wrap around
p = np.pad(dem, 1, mode='edge'); H, W = dem.shape
sq = np.zeros((H, W))
for dy in (-1, 0, 1):
    for dx in (-1, 0, 1):
        if dy or dx:
            diff = p[1+dy:1+dy+H, 1+dx:1+dx+W] - dem          # uint8 arithmetic
            sq += diff.astype(float) ** 2
tri = np.sqrt(sq / 8)
plt.imshow(tri, cmap='viridis'); plt.colorbar(); plt.savefig('pred_results/ruggedness.png')
