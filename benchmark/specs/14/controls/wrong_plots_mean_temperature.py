# Wrong: the spell count is computed, but the saved map (imshow) shows the mean temperature.
import numpy as np, xarray as xr, matplotlib.pyplot as plt
da = xr.open_dataset('dataset/E1_north_america.nc').air_temperature
hits = da.values > 280
spells = np.lib.stride_tricks.sliding_window_view(hits, 5, axis=0).all(-1).sum(0)
plt.imshow(da.mean('time').values, origin='lower')    # maps mean temperature, not the count
plt.colorbar(); plt.savefig('pred_results/temperature_statistic_vis.png')
