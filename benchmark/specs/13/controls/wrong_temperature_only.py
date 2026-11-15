# Wrong: only the temperature profile is drawn; no salinity profile.
import xarray as xr, matplotlib.pyplot as plt
ds = xr.open_dataset('dataset/atlantic_profiles.nc')
col = ds.sel(lon=330.5).isel(lat=0).sel(depth=slice(0, 1000))
plt.plot(col.theta.values, col.depth.values); plt.gca().invert_yaxis()
plt.xlabel('Temperature (K)'); plt.ylabel('Depth (m)'); plt.title('Temperature-salinity profile')
plt.savefig('pred_results/ocean_profiles_vis.png')
