# Wrong: plots surface temperature and salinity along longitude, not a vertical profile.
import xarray as xr, matplotlib.pyplot as plt
ds = xr.open_dataset('dataset/atlantic_profiles.nc')
surf = ds.isel(depth=0, lat=0)                   # surface values along longitude, not a vertical profile
fig, ax = plt.subplots(); ax2 = ax.twinx()
ax.plot(surf.lon.values, surf.theta.values); ax2.plot(surf.lon.values, surf.salinity.values, 'r')
plt.savefig('pred_results/ocean_profiles_vis.png')
