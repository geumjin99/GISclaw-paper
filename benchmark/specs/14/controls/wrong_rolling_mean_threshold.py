# Wrong: uses the 5-year running MEAN > 280 K instead of all five years > 280 K.
import numpy as np, xarray as xr, matplotlib.pyplot as plt, cartopy.crs as ccrs
air = xr.open_dataset('dataset/E1_north_america.nc')['air_temperature']
count = (air.rolling(time=5).mean() > 280).sum('time')
ax = plt.axes(projection=ccrs.PlateCarree())
m = ax.pcolormesh(air.longitude, air.latitude, count.values, cmap='Reds', transform=ccrs.PlateCarree())
ax.coastlines(); plt.colorbar(m, label='5-year spells > 280 K')
plt.savefig('pred_results/temperature_statistic_vis.png')
