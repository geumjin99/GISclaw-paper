# Correct: DEM decimated by 2, TRI via scipy generic_filter as in the domain
# knowledge, alpha band used as mask; plotted with pcolormesh.
import numpy as np, rasterio, matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
def tri_cell(w):
    c = w[4]
    return np.sqrt(np.nanmean((np.delete(w, 4) - c) ** 2))
def main():
    with rasterio.open('dataset/Elevation.tif') as s:
        z = s.read(1).astype(float)[::2, ::2]
        alpha = s.read(4)[::2, ::2]
    z[alpha == 0] = np.nan
    tri = generic_filter(z, tri_cell, size=3, mode='nearest')
    fig, ax = plt.subplots(figsize=(11, 8))
    m = ax.pcolormesh(tri[::-1], cmap='viridis', vmax=np.nanpercentile(tri, 99))
    fig.colorbar(m, label='TRI')
    ax.set_aspect('equal')
    fig.savefig('pred_results/ruggedness.png', dpi=110)
main()
