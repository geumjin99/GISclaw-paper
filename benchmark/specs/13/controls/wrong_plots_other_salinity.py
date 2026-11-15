# Wrong: profiles are extracted correctly at 330.5E / 9.8S, but the plotted salinity line comes from a
# different column (index error when re-selecting for the plot): the drawn T-S pairing is wrong.
import numpy as np, xarray as xr, matplotlib.pyplot as plt
ds = xr.open_dataset('dataset/atlantic_profiles.nc').sel(depth=slice(0, 1000))
col = ds.sel(lon=330.5).isel(lat=0)
t_prof = col.theta.values.astype(float)
s_prof = col.salinity.values.astype(float)
z = ds.depth.values
fig, ax = plt.subplots(figsize=(5, 7)); ax2 = ax.twiny()
ax.plot(t_prof, z, color='tab:blue')
ax2.plot(ds.salinity.isel(lat=-1, lon=5).values, z, color='tab:red')
ax.invert_yaxis(); ax.set_xlabel('Potential temperature (K)'); ax2.set_xlabel('Salinity (PSU)'); ax.set_ylabel('Depth (m)')
plt.savefig('pred_results/ocean_profiles_vis.png')
