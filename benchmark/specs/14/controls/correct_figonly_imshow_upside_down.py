# Correct analysis with a presentation error only: the spell count (overlapping windows) is right, but it is
# drawn with imshow's default origin='upper' while the latitudes ascend, so the saved map is upside down
# (Mexico at the top, Canada at the bottom).
import numpy as np, xarray as xr, matplotlib.pyplot as plt
air = xr.open_dataset('dataset/E1_north_america.nc')['air_temperature']
hot = air.values > 280
count = np.lib.stride_tricks.sliding_window_view(hot, 5, axis=0).all(-1).sum(0)
plt.figure(figsize=(9, 6))
plt.imshow(count, cmap='RdYlBu_r')
plt.colorbar(label='5-year spells > 280 K')
plt.title('Warm spells, 1860-2099')
plt.savefig('pred_results/temperature_statistic_vis.png')
