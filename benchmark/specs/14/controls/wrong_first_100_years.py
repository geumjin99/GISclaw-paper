# Wrong: only the first 100 years of the 240-year series are analysed.
import numpy as np, xarray as xr, matplotlib.pyplot as plt, cartopy.crs as ccrs
air = xr.open_dataset('dataset/E1_north_america.nc')['air_temperature'].isel(time=slice(0, 100))
hot = air.values > 280
spell_count = np.lib.stride_tricks.sliding_window_view(hot, 5, axis=0).all(-1).sum(0)
ax = plt.axes(projection=ccrs.PlateCarree())
m = ax.pcolormesh(air.longitude, air.latitude, spell_count, cmap='Reds', transform=ccrs.PlateCarree())
ax.coastlines(); plt.colorbar(m)
plt.savefig('pred_results/temperature_statistic_vis.png')
