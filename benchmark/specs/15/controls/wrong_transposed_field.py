# Wrong: CRS and masking are right, but the (rLat, rLon) array is passed transposed (square 31x31 grid, so no error).
import numpy as np, xarray as xr, matplotlib.pyplot as plt, cartopy.crs as ccrs
ds = xr.open_dataset('dataset/space_weather.nc')
tec = np.ma.masked_less(ds.TEC.values, 0)
rp = ccrs.RotatedPole(pole_longitude=180, pole_latitude=45)
ax = plt.axes(projection=rp)
cs = ax.contourf(ds.rLon.values, ds.rLat.values, tec.T, 60, transform=rp, cmap='plasma')
ax.coastlines(); ax.stock_img()
plt.colorbar(cs, orientation='horizontal', label='TEC')
plt.savefig('pred_results/TEC_vis.png')
