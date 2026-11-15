# Wrong: weights mis-assigned - ruggedness gets 1.0 instead of 1.25.
import numpy as np, rasterio, matplotlib.pyplot as plt
L = {}
for n in ['distance', 'ruggedness', 'protected_status_reclassified', 'landcover_reclassified']:
    with rasterio.open(f'dataset/{n}.tif') as s:
        L[n] = s.read(1).astype(float)
L['landcover_reclassified'][L['landcover_reclassified'] == 255] = np.nan
cost = 1.0 * L['distance'] + 1.0 * L['ruggedness'] + 1.0 * L['protected_status_reclassified'] + 1.25 * L['landcover_reclassified']
plt.figure(figsize=(10, 8)); plt.imshow(cost, cmap='viridis'); plt.colorbar(label='cost')
plt.savefig('pred_results/mountainLionCorridor.png')
