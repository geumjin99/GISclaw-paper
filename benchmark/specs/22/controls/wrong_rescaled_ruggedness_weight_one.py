# Wrong: ruggedness rescaled to 0-10 but weighted 1.0 instead of 1.25 (ratio 10/36 = 0.278, close to
# the accepted 1-10 rescale form 0.3125).
import numpy as np, rasterio, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
L = {}
for n in ['distance', 'ruggedness', 'protected_status_reclassified', 'landcover_reclassified']:
    with rasterio.open(f'dataset/{n}.tif') as s:
        L[n] = s.read(1).astype(float)
L['landcover_reclassified'][L['landcover_reclassified'] == 255] = np.nan
r = L['ruggedness']; r = (r - r.min()) / (r.max() - r.min()) * 10
cost = (1.0 * L['distance'] + 1.0 * r + 1.0 * L['protected_status_reclassified'] + 1.25 * L['landcover_reclassified']) / 4.25
plt.figure(figsize=(10, 8)); plt.imshow(cost, cmap='viridis'); plt.colorbar(label='cost')
plt.savefig('pred_results/mountainLionCorridor.png')
