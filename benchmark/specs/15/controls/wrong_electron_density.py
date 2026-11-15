# Wrong: draws the electron density (Ne) at one height instead of TEC.
import numpy as np, xarray as xr, matplotlib.pyplot as plt, cartopy.crs as ccrs
ds = xr.open_dataset('dataset/space_weather.nc')
rp = ccrs.RotatedPole(pole_longitude=180, pole_latitude=45)
ne = ds.Ne.isel(height=10).values                     # electron density at one height, not TEC
ne = np.where(ne < 0, np.nan, ne)
ax = plt.axes(projection=rp)
ax.contourf(ds.rLon, ds.rLat, ne, 50, transform=rp); ax.coastlines()
plt.savefig('pred_results/TEC_vis.png')
