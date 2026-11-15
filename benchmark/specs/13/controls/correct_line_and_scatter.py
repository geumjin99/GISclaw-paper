# Correct: single column (340.5E, 4.8S), 0-1500 m, temperature in degC as a line and salinity as
# markers on a twin axis.
import numpy as np, xarray as xr, matplotlib.pyplot as plt
ds = xr.open_dataset('dataset/atlantic_profiles.nc')
c = ds.sel(lon=340.5, lat=-4.834, method='nearest').sel(depth=slice(0, 1500))
fig, ax = plt.subplots(figsize=(5, 7)); ax2 = ax.twiny()
ax.plot(c.theta.values - 273.15, c.depth.values, color='navy', label='Temperature')
ax2.scatter(c.salinity.values, c.depth.values, color='firebrick', s=14, label='Salinity')
ax.invert_yaxis(); ax.set_xlabel('Potential temperature (degC)'); ax2.set_xlabel('Salinity (PSU)'); ax.set_ylabel('Depth (m)')
plt.savefig('pred_results/ocean_profiles_vis.png')
