# Wrong: skips the land-cover criterion (only three layers combined).
import rasterio, matplotlib.pyplot as plt
d = rasterio.open('dataset/distance.tif').read(1).astype(float)
r = rasterio.open('dataset/ruggedness.tif').read(1).astype(float)
p = rasterio.open('dataset/protected_status_reclassified.tif').read(1).astype(float)
cost = 1 * d + 1.25 * r + 1 * p
plt.imshow(cost); plt.colorbar(); plt.savefig('pred_results/mountainLionCorridor.png')
