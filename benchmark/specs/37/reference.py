"""Reference solution for task 37: county-level heat-exposure sentiment from geotagged tweets
(USA, 2021-09-24).

Follows the benchmark authors' method, with one correction:
  * heat vocabulary: data20000.txt ("'term': score, ...") -> dict term -> score;
  * a tweet's value = sum of scores of its lower-cased \\w+ tokens found in the dictionary;
    tweets with value 0 (no weather term) are discarded;
  * CORRECTION: the vocabulary is cased (e.g. 'Hot', 'Cold', 'Warm', 'Cool'
    exist only capitalised), so the authors' lower-cased tokens never matched these core terms.
    The lookup is made case-insensitive: keys are lower-cased (where case variants collide the
    entry that is already lower case wins, otherwise the first one), and lower-cased keys of
    <= 2 characters are dropped - they are unit fragments ('oF', 'oC', '0C', 'C'), and 'oF'
    would otherwise turn every "of" into a weather term (514 spurious tweets);
  * for the checks, the per-county results of two other defensible conventions are also kept
    (`county_variants`): the authors' literal lookup (lower-cased text, cased dictionary) and
    an exact-case lookup (original text, cased dictionary);
  * location: exact 'geo' point, else the centroid of the 'place' bounding box;
  * point-in-polygon assignment to 2020 counties (GEOID); per county: number of posts and
    mean value (= heat exposure); min-max normalisation over counties with posts;
  * choropleth with natural breaks (Jenks, k = 5).

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
heat_vocab = {}
for k, v in dict_word.items():
    kl = k.lower()
    if len(kl) <= 2:
        continue
    if kl not in heat_vocab or k == kl:
        heat_vocab[kl] = v
print("case-insensitive vocabulary terms:", len(heat_vocab))

# 2. weather-related tweets with a location
data = json.load(open("dataset/12500-tweets-2021-09-24_01-48-23.json"))


def tweet_points(lookup, lower_text):
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
        val = sum(lookup.get(w, 0.0) for w in re.findall(r"\w+", text.lower() if lower_text else text))
        if loc is not None and val != 0:
            rows.append({"value": val, "geometry": loc})
    return rows


rows = tweet_points(heat_vocab, True)
heat_tweets = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
print("weather-related located tweets:", len(heat_tweets), "of", len(data))

# 3. assign to counties
US = gpd.read_file("dataset/cb_2020_us_county_20m.shp")
heat_tweets = heat_tweets.to_crs(US.crs)
joined = gpd.sjoin(heat_tweets, US[["GEOID", "geometry"]], how="inner", predicate="within")

# 4. per-county count, mean, min-max normalisation
county_stats = joined.groupby("GEOID")["value"].agg(number_of_twitter="count", heat_exposure="mean").reset_index()
mn, mx = county_stats["heat_exposure"].min(), county_stats["heat_exposure"].max()
county_stats["normalized_heat_exposure"] = (county_stats["heat_exposure"] - mn) / (mx - mn)
US = US.merge(county_stats, on="GEOID", how="left")
US["number_of_twitter"] = US["number_of_twitter"].fillna(0).astype(int)
print("counties with posts:", int((US["number_of_twitter"] > 0).sum()))
county_stats.to_csv("pred_results/US_heat_exposure_by_county.csv", index=False)

# alternative lookup conventions (kept for the checks only)
variants = []
for vname, lookup, lower_text in (("case_insensitive", heat_vocab, True), ("authors_literal", dict_word, True),
                                  ("exact_case", dict_word, False)):
    pts = gpd.GeoDataFrame(tweet_points(lookup, lower_text), geometry="geometry", crs="EPSG:4326").to_crs(US.crs)
    j = gpd.sjoin(pts, US[["GEOID", "geometry"]], how="inner", predicate="within")
    st = j.groupby("GEOID")["value"].agg(count="count", mean="mean").reset_index()
    st["variant"] = vname
    variants.append(st)
county_variants = pd.concat(variants, ignore_index=True)
del st, j, pts, variants
print(county_variants.groupby("variant")["count"].agg(["size", "sum"]))

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
