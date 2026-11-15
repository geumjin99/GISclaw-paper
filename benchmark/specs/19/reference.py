"""Reference solution for task 19: interpolated dissolved-oxygen (DO) surface for the Chesapeake Bay.

26 455 DO samples (mg/L, 2014-2015) come from only 131 stations, each sampled at many depths
and dates. We (1) average all samples per station (the task gives no depth/date filter),
(2) interpolate the station means with Gaussian kernel regression (Nadaraya-Watson: the
kernel-weighted MEAN of nearby values; this is the value-interpolating form of the kernel
idea in the task's Domain Knowledge - a value-weighted KDE would estimate sampling density,
not DO), bandwidth chosen by leave-one-out cross-validation, (3) evaluate on a 500 m grid
clipped to the Bay polygon (EPSG:32618, metres) and map it.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely import contains_xy

os.makedirs("pred_results", exist_ok=True)
samples = gpd.read_file("dataset/DissolvedO2.geojson")
bay = gpd.read_file("dataset/Bay.geojson").to_crs(samples.crs)
samples = samples[samples["MeasureValue"].notna()]
samples["x"], samples["y"] = samples.geometry.x, samples.geometry.y
stations = samples.groupby("Station").agg(x=("x", "mean"), y=("y", "mean"),
                                          do_mean=("MeasureValue", "mean"), n=("MeasureValue", "size"))
print(len(samples), "samples at", len(stations), "stations; station DO %.2f-%.2f mg/L" % (
    stations.do_mean.min(), stations.do_mean.max()))

S = stations[["x", "y"]].to_numpy()
v = stations["do_mean"].to_numpy()


def nw(points, bw, S=S, v=v, exclude_self=False):
    d2 = ((points[:, None, :] - S[None, :, :]) ** 2).sum(-1)
    w = np.exp(-0.5 * d2 / bw ** 2)
    if exclude_self:
        np.fill_diagonal(w, 0.0)
    return (w * v).sum(1) / np.maximum(w.sum(1), 1e-300)


cv = {bw: float(np.sqrt(np.mean((nw(S, bw, exclude_self=True) - v) ** 2))) for bw in (2500, 5000, 7500, 10000, 15000, 20000)}
bandwidth = min(cv, key=cv.get)
print("LOO-CV RMSE by bandwidth:", {k: round(x, 3) for k, x in cv.items()}, "-> bandwidth", bandwidth)

res = 500.0
minx, miny, maxx, maxy = bay.total_bounds
gx = np.arange(minx + res / 2, maxx, res)
gy = np.arange(maxy - res / 2, miny, -res)            # row 0 = north
GX, GY = np.meshgrid(gx, gy)
inside = contains_xy(bay.geometry.union_all(), GX, GY)
do_surface = np.full(GX.shape, np.nan)
pts = np.c_[GX[inside], GY[inside]]
do_surface[inside] = np.concatenate([nw(pts[i:i + 5000], bandwidth) for i in range(0, len(pts), 5000)])
print("surface: %d cells, mean %.2f mg/L, range %.2f-%.2f" % (inside.sum(), np.nanmean(do_surface),
                                                              np.nanmin(do_surface), np.nanmax(do_surface)))

fig, ax = plt.subplots(figsize=(7, 10))
im = ax.imshow(do_surface, extent=(gx[0] - res / 2, gx[-1] + res / 2, gy[-1] - res / 2, gy[0] + res / 2),
               cmap="RdYlBu", origin="upper")
bay.boundary.plot(ax=ax, color="0.3", linewidth=0.6)
ax.scatter(S[:, 0], S[:, 1], s=4, c="k", label="stations (n=%d)" % len(S))
fig.colorbar(im, ax=ax, label="Dissolved oxygen, station mean (mg/L)", shrink=0.7)
ax.set_title("Interpolated dissolved oxygen, Chesapeake Bay 2014-2015\n(Gaussian kernel regression, bw = %.1f km)" % (bandwidth / 1000))
ax.set_xlabel("Easting (m, UTM 18N)"); ax.set_ylabel("Northing (m)"); ax.legend(loc="lower right")
fig.savefig("pred_results/interploated_water_quality.png", dpi=150, bbox_inches="tight")
