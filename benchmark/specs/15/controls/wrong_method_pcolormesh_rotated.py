# Wrong: the task prescribes a filled contour map; TEC is placed correctly with coastlines but
# drawn as a pcolormesh.
import numpy as np, xarray as xr, matplotlib.pyplot as plt, cartopy.crs as ccrs
ds = xr.open_dataset('dataset/space_weather.nc')
rp = ccrs.RotatedPole(pole_longitude=180, pole_latitude=45)
tec = np.where(ds.TEC.values < 0, np.nan, ds.TEC.values)
ax = plt.axes(projection=rp)
m = ax.pcolormesh(ds.rLon.values, ds.rLat.values, tec, transform=rp, shading='nearest')
ax.coastlines(); plt.colorbar(m, label='TEC')
plt.savefig('pred_results/TEC_vis.png')
