# Correct: rioxarray Minus in percentage points, 3-class change array, GeoTIFF written,
# diverging map.
import rioxarray, numpy as np, matplotlib.pyplot as plt
a = rioxarray.open_rasterio('dataset/2015_Nature_Africa_PR.2000.tif', masked=True).squeeze()
b = rioxarray.open_rasterio('dataset/2015_Nature_Africa_PR.2015.tif', masked=True).squeeze()
diff = (b - a) * 100            # percentage points
diff_np = diff.values
cls = np.full(diff_np.shape, np.nan)
cls[diff_np < -5] = -1; cls[(diff_np >= -5) & (diff_np <= 5)] = 0; cls[diff_np > 5] = 1
diff.rio.to_raster('pred_results/prevalence_change_pp.tif')
fig, ax = plt.subplots(figsize=(8, 8))
diff.plot(ax=ax, cmap='PuOr_r', vmin=-60, vmax=60)
plt.savefig('pred_results/malaria_prevalence_change_map.png')
