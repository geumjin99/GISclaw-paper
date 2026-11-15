"""Wrong: weekday 7-19 h instead of the 3-5 pm peak, Local Moran's I on the crash points (KNN
weights) of the first numeric column; no per-segment crash measure, no Gi*, and the supplied
network SWM is never used."""
import geopandas as gpd
import numpy as np
import libpysal
import esda
import matplotlib.pyplot as plt
import os

def model():
    crashes = gpd.read_file("dataset/crashes.shp")
    roads = gpd.read_file("dataset/roads.shp") if os.path.exists("dataset/roads.shp") else None

    # selection: weekdays + time window
    if "CRASH_DOW" in crashes.columns and "CRASH_HOD" in crashes.columns:
        weekday = crashes["CRASH_DOW"].isin(["Monday","Tuesday","Wednesday","Thursday","Friday"])
        peak = (crashes["CRASH_HOD"] >= 7) & (crashes["CRASH_HOD"] <= 19)
        crashes_sel = crashes[weekday & peak].copy()
    else:
        crashes_sel = crashes.copy()

    # Snap to Roads (simplified: spatial join only)
    crashes_proj = crashes_sel.to_crs(epsg=32617)
    if roads is not None:
        roads_proj = roads.to_crs(epsg=32617)
        crashes_proj = gpd.sjoin_nearest(crashes_proj, roads_proj, how="left", distance_col="dist")

    # hot-spot analysis -> Moran Local
    if len(crashes_proj) > 10:
        wq = libpysal.weights.KNN.from_dataframe(crashes_proj, k=min(8, len(crashes_proj)-1))
        wq.transform = "r"
        num_col = crashes_proj.select_dtypes(include=[np.number]).columns[0]
        y = crashes_proj[num_col].fillna(0).values
        lisa = esda.moran.Moran_Local(y, wq)
        crashes_proj["hotspot"] = lisa.q
        crashes_proj["sig"] = lisa.p_sim < 0.05

    os.makedirs("pred_results", exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    if "hotspot" in crashes_proj.columns:
        crashes_proj.plot(column="hotspot", categorical=True, ax=ax, legend=True,
                         cmap="RdBu_r", markersize=3)
    else:
        crashes_proj.plot(ax=ax, markersize=1)
    ax.set_title("Traffic Crash Hot Spots")
    plt.savefig("pred_results/crash_hotspots.png", dpi=150)

if True:
    model()
