"""Wrong: the number of posts per county is used as the heat exposure instead of the mean tweet value.
Otherwise the benchmark authors' method.

Adapted from the reference script of GeoAnalystBench (Zhang et al. 2025, Apache-2.0).
"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

os.makedirs("pred_results", exist_ok=True)

# 1. dictionary
content = open("dataset/data20000.txt").read()
dict_word = {}
for item in content.split(","):
    try:
        word = item.split(":")[0].split("'")[1]
        dict_word[word] = float(item.split(":")[1])
    except Exception:
        pass
print("dictionary terms:", len(dict_word))

# 2. weather-related tweets with a location
data = json.load(open("dataset/12500-tweets-2021-09-24_01-48-23.json"))
rows = []
for tw in data:
    text = tw.get("text") or ""
    loc = None
    if tw.get("geo"):
        lat, lon = tw["geo"]["coordinates"][:2]
        loc = Point(lon, lat)
    elif tw.get("place") and tw["place"].get("bounding_box"):
        poly = tw["place"]["bounding_box"]["coordinates"][0]
        loc = Point(sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly))
    val = sum(dict_word.get(w, 0.0) for w in re.findall(r"\w+", text.lower()))
    if loc is not None and val != 0:
        rows.append({"value": val, "geometry": loc})
heat_tweets = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
print("weather-related located tweets:", len(heat_tweets), "of", len(data))

# 3. assign to counties
US = gpd.read_file("dataset/cb_2020_us_county_20m.shp")
heat_tweets = heat_tweets.to_crs(US.crs)
joined = gpd.sjoin(heat_tweets, US[["GEOID", "geometry"]], how="inner", predicate="within")

# 4. per-county count, mean, min-max normalisation
county_stats = joined.groupby("GEOID")["value"].agg(number_of_twitter="count", heat_exposure="count").reset_index()
mn, mx = county_stats["heat_exposure"].min(), county_stats["heat_exposure"].max()
county_stats["normalized_heat_exposure"] = (county_stats["heat_exposure"] - mn) / (mx - mn)
US = US.merge(county_stats, on="GEOID", how="left")
US["number_of_twitter"] = US["number_of_twitter"].fillna(0).astype(int)
print("counties with posts:", int((US["number_of_twitter"] > 0).sum()))
county_stats.to_csv("pred_results/US_heat_exposure_by_county.csv", index=False)

# 5. map (contiguous US; natural breaks)
fig, ax = plt.subplots(figsize=(20, 10))
lower48 = US[~US["STATEFP"].isin(["02", "15", "72"])]
lower48.plot(ax=ax, color="#f0f0f0", edgecolor="#d0d0d0", linewidth=0.1)
lower48[lower48["normalized_heat_exposure"].notna()].plot(
    ax=ax, column="normalized_heat_exposure", scheme="natural_breaks", k=5, cmap="YlOrRd",
    legend=True, edgecolor="grey", linewidth=0.1, legend_kwds={"title": "normalised heat exposure"})
ax.set_title("Heat-exposure sentiment from tweets by county (2021-09-24), natural breaks")
ax.set_axis_off()
plt.savefig("pred_results/US_heat_exposure_map.png", bbox_inches="tight", dpi=150)
