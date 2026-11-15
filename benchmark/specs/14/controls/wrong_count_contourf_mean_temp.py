# Wrong: spell count computed correctly, but the saved map (contourf) shows the 240-year mean temperature.
import numpy as np, xarray as xr, matplotlib.pyplot as plt, cartopy.crs as ccrs
air = xr.open_dataset('dataset/E1_north_america.nc')['air_temperature']
hot = (air > 280).astype(int)
spells = (hot.rolling(time=5).sum() == 5).sum('time').values
mean_t = air.mean('time')
ax = plt.axes(projection=ccrs.PlateCarree())
cs = ax.contourf(air.longitude, air.latitude, mean_t, levels=12, cmap='RdYlBu_r', transform=ccrs.PlateCarree())
ax.coastlines(); plt.colorbar(cs, label='Number of 5-year spells > 280 K')
plt.savefig('pred_results/temperature_statistic_vis.png')
