# Wrong: rLon/rLat plotted as if they were geographic lon/lat, which draws the field ~45 degrees
# south of its true place.
import numpy.ma as ma, xarray as xr, matplotlib.pyplot as plt, cartopy.crs as ccrs
tec = xr.open_dataset('dataset/space_weather.nc')['TEC']
data = ma.masked_less(tec.values, 0)
lons, lats = tec['rLon'].values, tec['rLat'].values
ax = plt.axes(projection=ccrs.PlateCarree())
cs = ax.contourf(lons, lats, data, 100, transform=ccrs.PlateCarree())
ax.stock_img(); ax.coastlines(); plt.colorbar(cs)
plt.savefig('pred_results/TEC_vis.png')
