# Wrong: TEC masked correctly, but drawn against the file's 2-D latitude/longitude auxiliary variables,
# which are mostly fill values (9.97e36) - the field is not at its geographic location.
import numpy as np, xarray as xr, matplotlib.pyplot as plt, cartopy.crs as ccrs, cartopy.feature as cfeature
ds = xr.open_dataset('dataset/space_weather.nc')
tec_data = ds['TEC'].values.astype(float)
tec_data[tec_data < 0] = np.nan
latitudes = ds['latitude'].values
longitudes = ds['longitude'].values
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
try:
    cf = ax.contourf(longitudes, latitudes, tec_data, transform=ccrs.PlateCarree(), cmap='viridis')
    plt.colorbar(cf, label='TEC (TECU)')
except Exception:
    pass
ax.add_feature(cfeature.COASTLINE)
plt.savefig('pred_results/TEC_vis.png', dpi=150, bbox_inches='tight')
