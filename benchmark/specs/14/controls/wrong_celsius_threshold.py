# Wrong: unit slip, the 280 K threshold is mis-converted to Celsius (12 C instead of 6.85 C).
import numpy as np, xarray as xr, matplotlib.pyplot as plt
da = xr.open_dataset('dataset/E1_north_america.nc').air_temperature
tc = da.values - 273.15
thr = 280 - 273.15 + 5          # unit slip: threshold mis-converted (12 C instead of 6.85 C)
hits = tc > thr
counts = np.lib.stride_tricks.sliding_window_view(hits, 5, axis=0).all(-1).sum(0)
plt.imshow(counts, origin='lower', cmap='RdYlBu_r'); plt.colorbar()
plt.savefig('pred_results/temperature_statistic_vis.png')
