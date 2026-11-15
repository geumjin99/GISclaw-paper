# Correct (oceanographic convention): temperature and salinity profiles at 330.5E / 9.8S, 0-1000 m,
# shown as two depth profiles plus a T-S diagram (line) - three panels.
import numpy as np, xarray as xr, matplotlib.pyplot as plt
ds = xr.open_dataset('dataset/atlantic_profiles.nc')
c = ds.sel(lon=330.5).isel(lat=0).sel(depth=slice(0, 1000))
T, S, Z = c.theta.values, c.salinity.values, c.depth.values
fig, ax = plt.subplots(1, 3, figsize=(13, 6))
ax[0].plot(T, Z); ax[0].invert_yaxis(); ax[0].set_xlabel('theta (K)'); ax[0].set_ylabel('Depth (m)')
ax[1].plot(S, Z, 'r'); ax[1].invert_yaxis(); ax[1].set_xlabel('Salinity (PSU)')
ax[2].plot(S, T, 'k-'); ax[2].set_xlabel('Salinity'); ax[2].set_ylabel('theta (K)'); ax[2].set_title('T-S diagram')
plt.tight_layout(); plt.savefig('pred_results/ocean_profiles_vis.png')
