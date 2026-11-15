# Wrong: slope (gradient magnitude) presented as the terrain ruggedness index.
import numpy as np, rasterio, matplotlib.pyplot as plt
with rasterio.open('dataset/Elevation.tif') as s:
    dem = s.read(1).astype(float); res = s.res[0]
gy, gx = np.gradient(dem, res)
ruggedness = np.degrees(np.arctan(np.hypot(gx, gy)))
plt.figure(figsize=(11, 8))
plt.imshow(ruggedness, cmap='magma', vmax=np.percentile(ruggedness, 99))
plt.colorbar(label='TRI')
plt.title('Terrain Ruggedness')
plt.savefig('pred_results/ruggedness.png')
