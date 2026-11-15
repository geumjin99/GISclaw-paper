"""Wrong: uses Getis-Ord Gi* instead of Local Moran's I, the statistic the task names."""
import os, numpy as np, pandas as pd, geopandas as gpd, libpysal, esda, matplotlib.pyplot as plt
os.makedirs("pred_results", exist_ok=True)
hoods = gpd.read_file("dataset/berlin-neighbourhoods.geojson")
lst = pd.read_csv("dataset/berlin-listings.csv", low_memory=False)
pts = gpd.GeoDataFrame(lst[["id", "price"]], geometry=gpd.points_from_xy(lst.longitude, lst.latitude), crs=4326)
joined = gpd.sjoin(pts, hoods, predicate="within")
med = joined.groupby("neighbourhood_group")["price"].median()
hoods["median_price"] = hoods["neighbourhood_group"].map(med)
hoods["median_price"] = hoods["median_price"].fillna(hoods["median_price"].median())
w = libpysal.weights.Queen.from_dataframe(hoods, use_index=False); w.transform = "r"
# Getis-Ord Gi* used instead of Local Moran's I
gi = esda.getisord.G_Local(hoods["median_price"].values, w, star=True, permutations=999, seed=1)
hoods["spot"] = np.where(gi.p_sim >= 0.05, "Not significant", np.where(gi.Zs > 0, "Hot spot", "Cold spot"))
ax = hoods.plot(column="spot", categorical=True, legend=True, cmap="coolwarm", figsize=(9, 9), edgecolor="w")
ax.set_axis_off(); ax.set_title("Airbnb median price: significant hot / cold spots")
plt.savefig("pred_results/moran_local.png")
