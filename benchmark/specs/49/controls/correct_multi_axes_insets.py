# Correct: xlsx joined on the abbreviation, rate per 10 000 computed, the choropleth drawn in three
# axes (lower 48 + DC in EPSG:5070, Alaska and Hawaii insets) from the same column. The main axes
# has 95 patches for 49 features, matching neither the 51-row table nor its part count.
import geopandas as gpd, matplotlib.pyplot as plt
import zipfile, re, xml.etree.ElementTree as ET, pandas as pd
import geopandas as gpd, matplotlib.pyplot as plt
import zipfile, re, xml.etree.ElementTree as ET, pandas as pd
def read_sheet(path):
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    z = zipfile.ZipFile(path)
    shared = ["".join(t.text or "" for t in si.iter(ns + "t")) for si in ET.fromstring(z.read("xl/sharedStrings.xml"))]
    recs = []
    for r in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(ns + "row"):
        d = {}
        for c in r.findall(ns + "c"):
            v = c.find(ns + "v")
            if v is not None:
                d[re.sub(r"\d", "", c.get("r"))] = shared[int(v.text)] if c.get("t") == "s" else float(v.text)
        recs.append(d)
    head = recs[0]
    return pd.DataFrame([{head[k]: r.get(k) for k in head} for r in recs[1:]]).dropna(subset=["State"])
tab = read_sheet("dataset/homeless_data.xlsx")
tab["State"] = tab.State.astype(str).str.strip().str.upper()
states = gpd.read_file("dataset/USA_States.shp")
key = "L0USA_St_1"
base = states.drop(columns=[c for c in states.columns if c.startswith("Sheet1__") or c == "HomelessPe"])
gdf = base.merge(tab, left_on=key, right_on="State", how="left")
gdf["HomelessPer10k"] = gdf["Homeless13"] / gdf["Pop13"] * 10000

def make_map(col):
    fig = plt.figure(figsize=(13, 8.5))
    ax = fig.add_axes([0.02, 0.05, 0.96, 0.85])
    ak_ax = fig.add_axes([0.03, 0.07, 0.24, 0.28])
    hi_ax = fig.add_axes([0.27, 0.07, 0.15, 0.18])
    vmin, vmax = gdf[col].min(), gdf[col].max()
    for a, crs, mask in ((ax, "EPSG:5070", ~gdf[key].isin(["AK", "HI"])), (ak_ax, "EPSG:3338", gdf[key] == "AK"),
                         (hi_ax, "EPSG:26904", gdf[key] == "HI")):
        sub = gdf[mask].to_crs(crs)
        sub.plot(column=col, cmap="PuBu", vmin=vmin, vmax=vmax, edgecolor="grey", linewidth=0.5, ax=a)
        a.set_axis_off()
    return fig

fig = make_map("HomelessPer10k")
fig.savefig("pred_results/homeless_rate_map.png", dpi=80, bbox_inches="tight")
