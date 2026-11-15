"""Wrong: tweets are scored by the absolute value of the term scores (magnitude of heat words), which
discards the sign of the sentiment.
"""
import json, os, re
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from shapely.geometry import Point
os.makedirs("pred_results", exist_ok=True)
vocab = {}
for item in open("dataset/data20000.txt").read().split(","):
    if ":" in item:
        k, v = item.rsplit(":", 1)
        try:
            vocab[k.strip().strip("'\"{} ")] = float(v.strip(" }\n"))
        except ValueError:
            pass
tweets = json.load(open("dataset/12500-tweets-2021-09-24_01-48-23.json"))
recs = []
for tw in tweets:
    s = sum(vocab.get(w, 0.0) for w in re.findall(r"\w+", (tw.get("text") or "").lower()))
    if tw.get("geo"):
        la, lo = tw["geo"]["coordinates"][:2]; pt = Point(lo, la)
    elif tw.get("place") and tw["place"].get("bounding_box"):
        c = np.array(tw["place"]["bounding_box"]["coordinates"][0]); pt = Point(c[:, 0].mean(), c[:, 1].mean())
    else:
        continue
    if s != 0:
        recs.append({"value": abs(s), "geometry": pt})
counties = gpd.read_file("dataset/cb_2020_us_county_20m.shp")
pts = gpd.GeoDataFrame(recs, geometry="geometry", crs=4326).to_crs(counties.crs)
j = gpd.sjoin(pts, counties[["GEOID", "geometry"]], predicate="within")
agg = j.groupby("GEOID")["value"].mean().rename("heat").reset_index()  # magnitude of heat words
agg["heat_norm"] = (agg["heat"] - agg["heat"].min()) / (agg["heat"].max() - agg["heat"].min())
m = counties.merge(agg, on="GEOID", how="left")
fig, ax = plt.subplots(figsize=(16, 9))
m = m[~m.STATEFP.isin(["02", "15", "72"])]
m.plot(ax=ax, color="lightgrey")
m.dropna(subset=["heat_norm"]).plot(ax=ax, column="heat_norm", scheme="NaturalBreaks", k=5, cmap="OrRd", legend=True)
plt.savefig("pred_results/US_heat_exposure_map.png", dpi=110)
