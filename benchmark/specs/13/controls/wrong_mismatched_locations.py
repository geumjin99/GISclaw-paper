# Wrong: temperature and salinity profiles are taken from two different water columns.
import xarray as xr, matplotlib.pyplot as plt
ds = xr.open_dataset('dataset/atlantic_profiles.nc').sel(depth=slice(0, 1000))
t = ds.theta.isel(lat=0).sel(lon=330.5)          # temperature at 9.8S
s = ds.salinity.isel(lat=-1).sel(lon=345.5)      # salinity from another column (1.5S, 14.5W)
fig, ax = plt.subplots(); ax2 = ax.twiny()
ax.plot(t.values, ds.depth.values); ax2.plot(s.values, ds.depth.values, 'r'); ax.invert_yaxis()
plt.savefig('pred_results/ocean_profiles_vis.png')
