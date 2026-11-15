# Correct: masking written only inside the plotting call (no separate masked variable).
import numpy as np, xarray as xr, matplotlib.pyplot as plt, cartopy.crs as ccrs
ds = xr.open_dataset('dataset/space_weather.nc')
rp = ccrs.RotatedPole(pole_longitude=180, pole_latitude=45)
ax = plt.axes(projection=ccrs.Robinson())
ax.contourf(ds.rLon.values, ds.rLat.values, np.where(ds.TEC.values < 0, np.nan, ds.TEC.values), 40, transform=rp)
ax.stock_img(); ax.coastlines()
plt.savefig('pred_results/TEC_vis.png')
