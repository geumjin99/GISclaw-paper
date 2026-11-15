# Correct: xarray map algebra inside a function, given weights, no division.
import numpy as np, rioxarray, matplotlib.pyplot as plt
def main():
    r = {n: rioxarray.open_rasterio(f'dataset/{n}.tif').sel(band=1).astype('float32')
         for n in ['distance', 'ruggedness', 'protected_status_reclassified', 'landcover_reclassified']}
    lc = r['landcover_reclassified'].where(r['landcover_reclassified'] != 255)
    cost = r['distance'] + 1.25 * r['ruggedness'] + r['protected_status_reclassified'] + 1.25 * lc
    fig, ax = plt.subplots(figsize=(10, 8))
    cost.plot(ax=ax, cmap='viridis')
    fig.savefig('pred_results/mountainLionCorridor.png')
main()
