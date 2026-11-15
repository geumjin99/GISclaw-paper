"""Wrong: the esda quadrant codes are mixed up (1 -> 'cold', 3 -> 'hot'), so hot and cold spots carry
each other's label.
"""
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
lisa = esda.moran.Moran_Local(hoods["median_price"].values, w, permutations=999, seed=1)
# esda quadrant codes mis-remembered: 1 -> "cold", 3 -> "hot"
names = {1: "Cold spot", 2: "Low-High outlier", 3: "Hot spot", 4: "High-Low outlier"}
hoods["spot"] = [names[q] if p < 0.05 else "Not significant" for q, p in zip(lisa.q, lisa.p_sim)]
ax = hoods.plot(column="spot", categorical=True, legend=True, cmap="coolwarm", figsize=(9, 9), edgecolor="w")
ax.set_axis_off(); ax.set_title("Airbnb median price: significant hot / cold spots")
plt.savefig("pred_results/moran_local.png")
