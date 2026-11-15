"""Reference solution for task 15: Total Electron Content (TEC) filled-contour map with
geographic/coastline background.

The TEC grid is on a ROTATED-POLE grid (grid_north_pole lat 45, lon 180): rLat/rLon are
rotated coordinates centred on (45N, 0E), not geographic lat/lon. The 2-D latitude/longitude
auxiliary variables in the file are mostly fill values (9.97e36), so the field is georeferenced
through the rotated-pole CRS. Negative TEC values (192 cells) are physically invalid and masked.
"""
import os
import numpy as np
import numpy.ma as ma
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

os.makedirs("pred_results", exist_ok=True)
ds = xr.open_dataset("dataset/space_weather.nc")
gm = ds["rotated_pole"].attrs
rotated = ccrs.RotatedPole(pole_longitude=gm["grid_north_pole_longitude"],
                           pole_latitude=gm["grid_north_pole_latitude"])
rlon, rlat = ds["rLon"].values, ds["rLat"].values
tec = ma.masked_less(ds["TEC"].values, 0)                       # mask invalid (negative) TEC
tec_masked = tec.filled(np.nan)
print("masked cells:", int(tec.mask.sum()), " TEC range %.2f-%.2f (1e16 e/m^2)" % (tec.min(), tec.max()))

# geographic coordinates of the grid (for the record / checks)
RLON, RLAT = np.meshgrid(rlon, rlat)
geo = ccrs.PlateCarree().transform_points(rotated, RLON, RLAT)
geo_lon, geo_lat = geo[..., 0], geo[..., 1]
print("geographic extent: lon %.1f..%.1f, lat %.1f..%.1f" % (geo_lon.min(), geo_lon.max(), geo_lat.min(), geo_lat.max()))

fig = plt.figure(figsize=(9, 8))
ax = plt.axes(projection=rotated)
cs = ax.contourf(rlon, rlat, tec, 100, transform=rotated, cmap="viridis")
ax.stock_img()
ax.coastlines(resolution="110m")
ax.gridlines(draw_labels=True, linewidth=0.3)
fig.colorbar(cs, ax=ax, orientation="horizontal", pad=0.06, label="TEC (1e16 electrons / m$^2$)")
ax.set_title("Total Electron Content")
fig.savefig("pred_results/TEC_vis.png", dpi=150, bbox_inches="tight")
