# Correct: defensible MCDA variant - min-max normalise every criterion to 0-1, then weighted overlay.
import numpy as np, rasterio, matplotlib.pyplot as plt
W = {'distance': 1.0, 'ruggedness': 1.25, 'protected_status_reclassified': 1.0, 'landcover_reclassified': 1.25}
def norm(a):
    return (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a))
def build():
    acc = 0
    for n, w in W.items():
        with rasterio.open(f'dataset/{n}.tif') as s:
            a = s.read(1).astype('float32')
        if n == 'landcover_reclassified':
            a[a == 255] = np.nan
        acc = acc + w * norm(a)
    return acc / sum(W.values())
surface = build()
fig, ax = plt.subplots(figsize=(10, 8)); im = ax.imshow(surface, cmap='magma'); fig.colorbar(im, label='normalised cost')
fig.savefig('pred_results/mountainLionCorridor.png')
