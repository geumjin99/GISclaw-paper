# Wrong: no model at all - a rescaled gravity layer is shown as the 'probability' map.
import rasterio, numpy as np, matplotlib.pyplot as plt
with rasterio.open('dataset/tasgrav_IR.tif') as s:
    g = s.read(1).astype(float); g[g == s.nodata] = np.nan
norm = (g - np.nanmin(g)) / (np.nanmax(g) - np.nanmin(g))     # rescaled gravity shown as 'probability'
plt.imshow(norm, vmin=0, vmax=1); plt.colorbar(label='probability'); plt.savefig('pred_results/mineral_prospectivity.png')
