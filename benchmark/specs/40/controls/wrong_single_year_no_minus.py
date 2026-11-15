# Wrong: no Minus; maps the 2015 prevalence centred on its median instead of the 2000-2015 change.
import rasterio, numpy as np, matplotlib.pyplot as plt
pr15 = rasterio.open('dataset/2015_Nature_Africa_PR.2015.tif').read(1, masked=True).filled(np.nan)
# maps 2015 prevalence centred on its median instead of computing the 2000-2015 change
change = pr15 - np.nanmedian(pr15)
plt.imshow(change, cmap='RdBu_r'); plt.colorbar()
plt.savefig('pred_results/malaria_change.png')
