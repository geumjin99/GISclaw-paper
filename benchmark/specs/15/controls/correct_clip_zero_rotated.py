# Correct: negatives clipped to 0 instead of masked; the field is drawn at its true location
# (rotated-pole CRS).
import numpy as np, xarray as xr, matplotlib.pyplot as plt, cartopy.crs as ccrs
ds = xr.open_dataset('dataset/space_weather.nc')
rp = ccrs.RotatedPole(pole_longitude=180, pole_latitude=45)
tec = np.clip(ds.TEC.values, 0, None)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.contourf(ds.rLon.values, ds.rLat.values, tec, 30, transform=rp); ax.coastlines()
plt.savefig('pred_results/TEC_vis.png')
