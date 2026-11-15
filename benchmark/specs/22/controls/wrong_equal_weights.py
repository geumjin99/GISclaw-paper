# Wrong: ignores the given weights (simple average of the four criteria).
import numpy as np, rasterio, matplotlib.pyplot as plt
arrs = [rasterio.open(f'dataset/{n}.tif').read(1).astype(float) for n in
        ['distance', 'ruggedness', 'protected_status_reclassified', 'landcover_reclassified']]
arrs[3][arrs[3] == 255] = np.nan
cost = sum(arrs) / 4
plt.imshow(cost); plt.colorbar(); plt.savefig('pred_results/mountainLionCorridor.png')
