"""Reference solution for task 13: vertical potential temperature and salinity profile, South Atlantic.

Selection as in the original Iris example (the instruction itself gives no numbers):
longitude 330.5 E (29.5 W), -10 < latitude < -9 (one grid row, -9.83), depth <= 1000 m.
Temperature and salinity are drawn against depth on twin x-axes, depth increasing downward.
"""
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

os.makedirs("pred_results", exist_ok=True)
ds = xr.open_dataset("dataset/atlantic_profiles.nc")
sel = ds.sel(lon=330.5).where((ds.lat > -10) & (ds.lat < -9), drop=True)
sel = sel.sel(depth=sel.depth[sel.depth <= 1000])
theta_profile = sel["theta"].squeeze("lat").values.astype(float)        # K
salinity_profile = sel["salinity"].squeeze("lat").values.astype(float)  # PSU
depth = sel["depth"].values.astype(float)
print("lat", sel.lat.values, "levels", len(depth), "theta %.2f-%.2f K, salinity %.2f-%.2f" % (
    np.nanmin(theta_profile), np.nanmax(theta_profile), np.nanmin(salinity_profile), np.nanmax(salinity_profile)))

fig, ax1 = plt.subplots(figsize=(5, 6))
tcol, scol = (0.3, 0.4, 0.5), (0.6, 0.1, 0.15)
ax1.plot(theta_profile, depth, lw=2, color=tcol, alpha=0.8)
ax1.set_xlabel("Potential temperature / K", color=tcol)
ax1.set_ylabel("Depth / m")
ax1.invert_yaxis()
ax1.tick_params(axis="x", colors=tcol)
ax2 = ax1.twiny()
ax2.plot(salinity_profile, depth, lw=2, color=scol, alpha=0.8)
ax2.set_xlabel("Salinity / PSU", color=scol)
ax2.tick_params(axis="x", colors=scol)
ax1.set_title("29.5W, 9.8S, 0-1000 m", fontsize=9, pad=40)
fig.tight_layout()
fig.savefig("pred_results/ocean_profiles_vis.png", dpi=150)
