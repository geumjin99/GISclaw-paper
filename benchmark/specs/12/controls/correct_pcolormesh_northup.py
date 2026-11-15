# Correct: full-resolution TRI (vectorised, as defined), drawn north-up with pcolormesh by
# flipping the rows (pcolormesh puts row 0 at the bottom).
import numpy as np, rasterio, matplotlib.pyplot as plt
with rasterio.open('dataset/Elevation.tif') as s:
    z = s.read(1).astype(float)
pad = np.pad(z, 1, mode='edge')
H, W = z.shape
sq = sum((pad[1 + dy:1 + dy + H, 1 + dx:1 + dx + W] - z) ** 2 for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
tri = np.sqrt(sq / 8.0)
fig, ax = plt.subplots(figsize=(11, 8))
m = ax.pcolormesh(tri[::-1], cmap='inferno', vmax=np.percentile(tri, 99))
ax.set_aspect('equal'); fig.colorbar(m, label='TRI')
fig.savefig('pred_results/ruggedness.png', dpi=100)
