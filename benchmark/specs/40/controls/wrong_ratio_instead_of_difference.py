# Wrong: maps the relative change (2015 / 2000 - 1) instead of the Minus difference the task asks for.
import rasterio, numpy as np, matplotlib.pyplot as plt
a = rasterio.open('dataset/2015_Nature_Africa_PR.2000.tif').read(1, masked=True).filled(np.nan)
b = rasterio.open('dataset/2015_Nature_Africa_PR.2015.tif').read(1, masked=True).filled(np.nan)
change = b / a - 1                       # relative change, not the Minus difference
change[~np.isfinite(change)] = np.nan
plt.imshow(np.clip(change, -1, 1), cmap='RdBu_r'); plt.colorbar()
plt.savefig('pred_results/malaria_change.png')
