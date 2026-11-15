# Correct: overlapping windows via an xarray rolling sum (== 5), drawn with DataArray.plot.contourf.
import xarray as xr, matplotlib.pyplot as plt, cartopy.crs as ccrs
da = xr.open_dataset('dataset/E1_north_america.nc').air_temperature
full = (da > 280).astype(int).rolling(time=5).sum() == 5
count = full.sum('time')
ax = plt.axes(projection=ccrs.PlateCarree())
count.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), cmap='RdYlBu_r')
ax.coastlines(); plt.savefig('pred_results/temperature_statistic_vis.png')
