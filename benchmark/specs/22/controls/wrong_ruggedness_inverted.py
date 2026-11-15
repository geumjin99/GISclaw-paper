# Wrong: ruggedness treated as a suitability (rugged = good for lions) and inverted before weighting: 36 - rug.
import numpy as np, rasterio, matplotlib.pyplot as plt
L = {}
for n in ['distance', 'ruggedness', 'protected_status_reclassified', 'landcover_reclassified']:
    with rasterio.open(f'dataset/{n}.tif') as s:
        L[n] = s.read(1).astype(float)
L['landcover_reclassified'][L['landcover_reclassified'] == 255] = np.nan
rug_cost = (L['ruggedness'].max() - L['ruggedness']) / L['ruggedness'].max() * 10
cost = (1.0 * L['distance'] + 1.25 * rug_cost + 1.0 * L['protected_status_reclassified'] + 1.25 * L['landcover_reclassified']) / 4.5
plt.figure(figsize=(10, 8)); plt.imshow(cost, cmap='viridis'); plt.colorbar(label='cost')
plt.savefig('pred_results/mountainLionCorridor.png')
