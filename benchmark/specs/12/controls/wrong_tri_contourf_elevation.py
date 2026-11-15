# Wrong: TRI computed correctly, but the saved figure is a filled-contour map of the elevation.
import numpy as np, rasterio, matplotlib.pyplot as plt
with rasterio.open('dataset/Elevation.tif') as s:
    dem = s.read(1).astype(float)
p = np.pad(dem, 1, mode='edge')
acc = np.zeros_like(dem)
for dy in (-1, 0, 1):
    for dx in (-1, 0, 1):
        if dy or dx:
            acc += (p[1 + dy:1 + dy + dem.shape[0], 1 + dx:1 + dx + dem.shape[1]] - dem) ** 2
tri = np.sqrt(acc / 8)
fig, ax = plt.subplots(figsize=(11, 8))
cs = ax.contourf(dem[::-1], levels=15, cmap='terrain')
fig.colorbar(cs, label='Ruggedness')
ax.set_title('Terrain ruggedness')
plt.savefig('pred_results/ruggedness.png')
