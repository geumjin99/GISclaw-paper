# Wrong: the "hot spot z-score" is the standardised crash rate, computed without any spatial
# weights; no Gi* statistic on the network.
import os, geopandas as gpd, numpy as np, scipy.sparse as sp, matplotlib.pyplot as plt
os.makedirs("pred_results", exist_ok=True)
crashes = gpd.read_file("dataset/crashes.shp"); roads = gpd.read_file("dataset/roads.shp")
peak = crashes[crashes.CRASH_DOW.isin(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]) & crashes.CRASH_HOD.between(15, 17)].copy()
snap = gpd.sjoin_nearest(peak[["geometry"]], roads[["UniqID", "geometry"]], max_distance=1320)
snap = snap[~snap.index.duplicated()]
roads["crashes"] = roads.UniqID.map(snap.UniqID.value_counts()).fillna(0)
roads["CrashRate"] = roads.crashes / (6 * roads.Shape_Leng)
x = roads.CrashRate.to_numpy()
# "hot spot z-score" without any spatial weights: just the standardised crash rate
roads["GiZScore"] = (x - x.mean()) / x.std()
roads.plot(column="GiZScore", cmap="RdBu_r", vmin=-3, vmax=3, linewidth=0.4, figsize=(8, 10))
plt.savefig("pred_results/crash_hotspots.png")
