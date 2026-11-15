# Correct: grid converted to true lon/lat with pyproj (ob_tran rotated pole), masked TEC drawn with
# contourf on a PlateCarree map with coastlines.
import numpy as np, xarray as xr, matplotlib.pyplot as plt, cartopy.crs as ccrs
from pyproj import Transformer, CRS
ds = xr.open_dataset('dataset/space_weather.nc')
tec = np.ma.masked_where(ds.TEC.values < 0, ds.TEC.values)
rot = CRS.from_proj4('+proj=ob_tran +o_proj=longlat +o_lon_p=0 +o_lat_p=45 +lon_0=0 +datum=WGS84 +no_defs')
t = Transformer.from_crs(rot, 'EPSG:4326', always_xy=True)
RLON, RLAT = np.meshgrid(ds.rLon.values, ds.rLat.values)
ll = np.array(t.transform(RLON, RLAT))
ax = plt.axes(projection=ccrs.PlateCarree())
cs = ax.contourf(ll[0], ll[1], tec, 40, transform=ccrs.PlateCarree(), cmap='viridis')
ax.coastlines(); ax.gridlines(draw_labels=True)
plt.colorbar(cs, orientation='horizontal', label='TEC (TECU)')
plt.savefig('pred_results/TEC_vis.png')
print(ll[1].min(), ll[1].max(), ll[1][15, 15], ll[0][15, 15])
