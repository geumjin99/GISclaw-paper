"""Correct: one GeoDataFrame.plot with scheme=NaturalBreaks; counties without tweets are drawn grey
(missing_kwds).
"""
import json, re, os
import pandas as pd, geopandas as gpd
import mapclassify
import matplotlib.pyplot as plt

def load_lexicon(path):
    pairs = re.findall(r"'([^']+)'\s*:\s*(-?[0-9.eE+-]+)", open(path).read())
    return {k: float(v) for k, v in pairs}

def tweet_points(path, lex):
    out = []
    for t in json.load(open(path)):
        txt = (t.get("extended_tweet") or {}).get("full_text") or t.get("text") or ""
        score = sum(lex.get(tok, 0) for tok in re.findall(r"\w+", txt) if tok)
        if score == 0:
            continue
        if t.get("coordinates"):
            x, y = t["coordinates"]["coordinates"]
        elif t.get("place") and t["place"].get("bounding_box"):
            ring = t["place"]["bounding_box"]["coordinates"][0]
            x = sum(p[0] for p in ring) / len(ring); y = sum(p[1] for p in ring) / len(ring)
        else:
            continue
        out.append((score, x, y))
    df = pd.DataFrame(out, columns=["score", "x", "y"])
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs="EPSG:4326")

def run():
    lex = load_lexicon("dataset/data20000.txt")
    pts = tweet_points("dataset/12500-tweets-2021-09-24_01-48-23.json", lex)
    cty = gpd.read_file("dataset/cb_2020_us_county_20m.shp").to_crs(4326)
    hit = gpd.sjoin(cty, pts, predicate="contains")
    exposure = hit.groupby("GEOID").agg(n=("score", "size"), hx=("score", "mean"))
    exposure["hx01"] = (exposure.hx - exposure.hx.min()) / (exposure.hx.max() - exposure.hx.min())
    cty = cty.set_index("GEOID").join(exposure)
    ax = cty.plot(column="hx01", scheme="NaturalBreaks", k=5, cmap="OrRd", legend=True, figsize=(15, 8),
                  missing_kwds={"color": "lightgrey"})   # tweet-less counties grey, in the same plot call
    ax.set_xlim(-127, -65); ax.set_ylim(23, 50)
    plt.savefig("pred_results/US_heat_exposure_map.png")

run()
