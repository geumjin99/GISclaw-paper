# Wrong: builds a RotatedPole CRS object, but passes transform=PlateCarree() to contourf, so rLon/rLat are
# read as geographic lon/lat: the field is drawn ~45 degrees south of its true place.
import numpy as np, xarray as xr, matplotlib.pyplot as plt, cartopy.crs as ccrs
ds = xr.open_dataset('dataset/space_weather.nc')
tec = np.ma.masked_less(ds.TEC.values, 0)
proj = ccrs.RotatedPole(pole_longitude=180, pole_latitude=45)
ax = plt.axes(projection=ccrs.PlateCarree())
cs = ax.contourf(ds.rLon.values, ds.rLat.values, tec, 50, transform=ccrs.PlateCarree(), cmap='viridis')
ax.coastlines(); ax.set_global()
plt.colorbar(cs, orientation='horizontal', label='TEC')
plt.savefig('pred_results/TEC_vis.png')
