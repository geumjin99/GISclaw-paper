# Wrong: regional mean profile over the whole file, with missing deep values filled with 0 before averaging.
import numpy as np, xarray as xr, matplotlib.pyplot as plt
ds = xr.open_dataset('dataset/atlantic_profiles.nc').fillna(0)
t = ds.theta.mean(dim=['lat', 'lon']).values
s = ds.salinity.mean(dim=['lat', 'lon']).values
z = ds.depth.values
fig, ax = plt.subplots(figsize=(5, 7)); ax2 = ax.twiny()
ax.plot(t, z); ax2.plot(s, z, 'r'); ax.invert_yaxis()
plt.savefig('pred_results/ocean_profiles_vis.png')
