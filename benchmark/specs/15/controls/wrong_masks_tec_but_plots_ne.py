# Wrong: TEC is masked, but the map draws the electron density (Ne) instead of the masked TEC.
import numpy as np, numpy.ma as ma, xarray as xr, matplotlib.pyplot as plt, cartopy.crs as ccrs
ds = xr.open_dataset('dataset/space_weather.nc')
rp = ccrs.RotatedPole(pole_longitude=180, pole_latitude=45)
tec = ma.masked_less(ds.TEC.values, 0)
ne = ds.Ne.isel(height=10).values
ax = plt.axes(projection=rp)
ax.contourf(ds.rLon, ds.rLat, ne, 50, transform=rp); ax.coastlines()      # draws Ne, not the masked TEC
plt.savefig('pred_results/TEC_vis.png')
