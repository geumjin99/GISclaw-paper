# Wrong: slope (not ruggedness) stored as integer degrees; slope is a different variable from the TRI the
# task defines.
import numpy as np, rasterio, matplotlib.pyplot as plt
with rasterio.open('dataset/Elevation.tif') as s:
    dem = s.read(1).astype(float); res = s.res[0]
gy, gx = np.gradient(dem, res)
ruggedness = np.floor(np.degrees(np.arctan(np.hypot(gx, gy))))
plt.figure(figsize=(11, 8)); plt.imshow(ruggedness, cmap='magma'); plt.colorbar(label='TRI')
plt.savefig('pred_results/ruggedness.png')
