# Correct: TRI as defined, mapped with filled contours (contourf keeps only its levels, so the
# drawn variable is identified from the plotting call).
import numpy as np, rasterio, matplotlib.pyplot as plt
with rasterio.open('dataset/Elevation.tif') as s:
    dem = s.read(1).astype(float)
    b = s.bounds
H, W = dem.shape
p = np.pad(dem, 1, mode='edge')
tri = np.sqrt(sum((p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W] - dem) ** 2
                  for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx) / 8.0)
x = np.linspace(b.left, b.right, W)
y = np.linspace(b.top, b.bottom, H)
fig, ax = plt.subplots(figsize=(11, 8))
cs = ax.contourf(x, y, tri, levels=np.linspace(0, np.percentile(tri, 99), 12), extend='max', cmap='magma')
fig.colorbar(cs, label='TRI')
ax.set_aspect('equal')
fig.savefig('pred_results/ruggedness.png', dpi=100)
