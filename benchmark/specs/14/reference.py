"""Reference solution for task 14: number of 5-year warm spells (annual mean > 280 K for five consecutive years) per
grid cell over 1860-2099 (E1 scenario), North America.

Definition (as in the original Iris example): count every window of 5 consecutive years in
which all five annual means exceed 280 K (overlapping windows; a 7-year run counts 3).
"""
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

os.makedirs("pred_results", exist_ok=True)
ds = xr.open_dataset("dataset/E1_north_america.nc")
air = ds["air_temperature"]                        # (time=240, latitude=37, longitude=49), K
threshold, spell_years = 280.0, 5

hits = air.values > threshold
windows = np.lib.stride_tricks.sliding_window_view(hits, spell_years, axis=0)   # (236, 37, 49, 5)
warm_spells = windows.all(axis=-1).sum(axis=0).astype(int)                     # (37, 49)
print("cells with >=1 spell: %d / %d; max %d" % ((warm_spells > 0).sum(), warm_spells.size, warm_spells.max()))

lons, lats = air["longitude"].values, air["latitude"].values
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
levels = np.arange(0, warm_spells.max() + 21, 20)
cs = ax.contourf(lons, lats, warm_spells, levels=levels, cmap="RdYlBu_r", transform=ccrs.PlateCarree())
ax.coastlines()
fig.colorbar(cs, ax=ax, label="number of 5-year windows > 280 K")
ax.set_title("Number of 5-year warm spells (> 280 K) in 240 years, 1860-2099")
fig.savefig("pred_results/temperature_statistic_vis.png", dpi=150, bbox_inches="tight")
