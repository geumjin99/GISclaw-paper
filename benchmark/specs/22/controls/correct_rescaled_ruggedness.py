# Correct: rescale every criterion to 0-10 first (ruggedness is 0-36), weighted overlay
# with weights normalised to 1, nodata 255 masked, plotted with matplotlib.
import numpy as np, rasterio, matplotlib.pyplot as plt
def rd(n):
    with rasterio.open(f'dataset/{n}.tif') as s:
        return s.read(1).astype('float32')
d = rd('distance'); r = rd('ruggedness'); p = rd('protected_status_reclassified'); l = rd('landcover_reclassified')
l[l == 255] = np.nan
r = (r - np.nanmin(r)) / (np.nanmax(r) - np.nanmin(r)) * 10
w = np.array([1, 1.25, 1, 1.25]); w = w / w.sum()
composite = w[0] * d + w[1] * r + w[2] * p + w[3] * l
plt.figure(figsize=(9, 7)); plt.imshow(composite, cmap='magma_r'); plt.colorbar(label='cost')
plt.title('Composite cost'); plt.savefig('pred_results/mountainLionCorridor.png')
