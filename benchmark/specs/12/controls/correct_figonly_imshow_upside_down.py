# Correct analysis (TRI as defined) with a presentation error only: the map is drawn upside down
# (origin='lower').
import numpy as np, rasterio, matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
with rasterio.open('dataset/Elevation.tif') as src:
    dem = src.read(1).astype('float32')
def tri(v):
    c = v[4]
    d = np.delete(v, 4) - c
    return np.sqrt(np.mean(d * d))
rug = generic_filter(dem, tri, size=3, mode='nearest')
plt.figure(figsize=(10, 8)); plt.imshow(rug, cmap='viridis', origin='lower'); plt.colorbar(label='TRI')
plt.title('Terrain Ruggedness Index'); plt.savefig('pred_results/ruggedness.png')
