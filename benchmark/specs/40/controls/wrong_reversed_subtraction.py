# Wrong: reversed subtraction (2000 minus 2015), so decreases are shown as increases.
import rasterio, numpy as np, matplotlib.pyplot as plt
a = rasterio.open('dataset/2015_Nature_Africa_PR.2000.tif').read(1, masked=True)
b = rasterio.open('dataset/2015_Nature_Africa_PR.2015.tif').read(1, masked=True)
change = (a - b).filled(np.nan)      # 2000 minus 2015: decreases shown as increases
plt.imshow(change, cmap='RdBu_r', vmin=-0.6, vmax=0.6); plt.colorbar(label='change (red = increase)')
plt.savefig('pred_results/malaria_change.png')
