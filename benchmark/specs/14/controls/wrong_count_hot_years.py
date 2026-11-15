# Wrong: counts every year above 280 K and ignores the five-consecutive-year requirement.
import xarray as xr, matplotlib.pyplot as plt, cartopy.crs as ccrs
da = xr.open_dataset('dataset/E1_north_america.nc').air_temperature
n_hot = (da > 280).sum('time')                 # ignores the 5-consecutive-year requirement
ax = plt.axes(projection=ccrs.PlateCarree())
ax.pcolormesh(da.longitude, da.latitude, n_hot.values, transform=ccrs.PlateCarree()); ax.coastlines()
plt.savefig('pred_results/temperature_statistic_vis.png')
