# Correct: counts distinct runs of >= 5 warm years (each run once) and draws the count with pcolormesh.
import numpy as np, xarray as xr, matplotlib.pyplot as plt, cartopy.crs as ccrs
da = xr.open_dataset('dataset/E1_north_america.nc').air_temperature
hot = (da > 280).values
spells = np.zeros(hot.shape[1:], int)
for i in range(hot.shape[1]):
    for j in range(hot.shape[2]):
        run = 0
        for v in hot[:, i, j]:
            run = run + 1 if v else 0
            if run == 5:          # count each run once when it reaches 5 years
                spells[i, j] += 1
ax = plt.axes(projection=ccrs.PlateCarree())
m = ax.pcolormesh(da.longitude, da.latitude, spells, cmap='Reds', transform=ccrs.PlateCarree())
ax.coastlines(); plt.colorbar(m); plt.title('Warm spells >280K lasting >=5 years')
plt.savefig('pred_results/temperature_statistic_vis.png')
