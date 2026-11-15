# Correct: ruggedness (raw 0-36) min-max rescaled to 1-10, the scale
# protected status already has, then the given weights / 4.5; landcover 255 masked.
import numpy as np, rasterio, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
with rasterio.open('dataset/ruggedness.tif') as s:
    rug = s.read(1).astype(float)
rug = 1 + 9 * (rug - rug.min()) / (rug.max() - rug.min())
L = {}
for n in ['distance', 'protected_status_reclassified', 'landcover_reclassified']:
    with rasterio.open(f'dataset/{n}.tif') as s:
        L[n] = s.read(1, masked=True)
dist = L['distance'].filled(0).astype(float)
pro = L['protected_status_reclassified'].filled(0).astype(float)
lc = L['landcover_reclassified'].filled(255).astype(float); lc[lc == 255] = np.nan
cost = (1 * dist + 1.25 * rug + 1 * pro + 1.25 * lc) / 4.5
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cost, cmap='RdYlGn_r'); plt.colorbar(im, label='Composite cost')
ax.set_title('Mountain Lion Corridor Composite Cost Surface'); ax.axis('off')
plt.savefig('pred_results/mountainLionCorridor.png', dpi=150, bbox_inches='tight'); plt.close()
