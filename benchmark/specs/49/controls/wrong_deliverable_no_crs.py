"""Wrong: the reference solution of task 49 with every geographic deliverable written without a CRS.

The analysis: homelessness 2013 by state, North Dakota oil boom.

Join homeless_data.xlsx (Sheet1: State, Change, Pop13, Homeless13; 53 rows incl. PR and a
blank row) to USA_States.shp on the state abbreviation (L0USA_St_1 <-> State), add
HomelessPopPer10k = Homeless13 / Pop13 * 10 000, and save two maps: homeless per 10 000
residents and the 2012-13 change in homelessness (North Dakota +201 %).
Outputs: pred_results/homeless_map.png, pred_results/homeless_change_map.png,
pred_results/states_homeless_2013.shp
"""
import re
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


def read_xlsx_sheet1(path):
    """Minimal .xlsx reader (zip + XML), independent of any Excel library."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns):
                shared.append("".join(t.text or "" for t in si.iter("{%s}t" % ns["m"])))
        root = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in root.iter("{%s}row" % ns["m"]):
        vals = {}
        for c in row.findall("m:c", ns):
            col = re.match(r"[A-Z]+", c.get("r")).group(0)
            v = c.find("m:v", ns)
            if v is None:
                isv = c.find("m:is", ns)
                val = "".join(t.text or "" for t in isv.iter("{%s}t" % ns["m"])) if isv is not None else None
            elif c.get("t") == "s":
                val = shared[int(v.text)]
            elif c.get("t") in ("str", "inlineStr"):
                val = v.text
            else:
                val = float(v.text)
            vals[col] = val
        rows.append(vals)
    cols = sorted({k for r in rows for k in r}, key=lambda k: (len(k), k))
    df = pd.DataFrame([[r.get(k) for k in cols] for r in rows[1:]], columns=[rows[0].get(k) for k in cols])
    return df

states = gpd.read_file("dataset/USA_States.shp")
try:
    homeless = pd.read_excel("dataset/homeless_data.xlsx", sheet_name="Sheet1")
except ImportError:
    homeless = read_xlsx_sheet1("dataset/homeless_data.xlsx")
homeless = homeless.dropna(subset=["State"])
homeless = homeless[homeless["State"].astype(str).str.strip() != ""]
for c in ("Change", "Pop13", "Homeless13"):
    homeless[c] = pd.to_numeric(homeless[c])
# the shapefile already carries a join (Sheet1__*, HomelessPe); drop it and join afresh
states = states[["OBJECTID", "L0USA_Stat", "L0USA_St_1", "geometry"]].rename(
    columns={"L0USA_Stat": "STATE_NAME", "L0USA_St_1": "STATE_ABBR"})
joined = states.merge(homeless, left_on="STATE_ABBR", right_on="State", how="left", validate="1:1")
print("joined:", int(joined["Homeless13"].notna().sum()), "of", len(joined), "states; unmatched table rows:",
      sorted(set(homeless.State) - set(states.STATE_ABBR)))
joined["HomelessPopPer10k"] = joined["Homeless13"] / joined["Pop13"] * 10000
joined["Change_pct"] = joined["Change"] * 100
nd = joined[joined.STATE_ABBR == "ND"].iloc[0]
print("ND: %.1f homeless per 10k, change %+.0f %%; national median %.1f per 10k" % (
    nd.HomelessPopPer10k, nd.Change_pct, joined.HomelessPopPer10k.median()))
joined.to_file("pred_results/states_homeless_2013.shp")

plot_df = joined.to_crs("ESRI:102003")       # equal-area view of the contiguous states
lower48 = plot_df[~plot_df.STATE_ABBR.isin(["AK", "HI"])]
fig, ax = plt.subplots(figsize=(14, 8))
lower48.plot(column="HomelessPopPer10k", cmap="YlOrRd", scheme="quantiles", k=5, legend=True,
             edgecolor="black", linewidth=0.4, ax=ax, legend_kwds={"title": "Homeless per 10 000"})
lower48[lower48.STATE_ABBR == "ND"].boundary.plot(ax=ax, color="blue", linewidth=2)
ax.set_title("Homeless population per 10 000 residents, 2013 (North Dakota outlined)"); ax.set_axis_off()
plt.savefig("pred_results/homeless_map.png", dpi=110, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(14, 8))
lower48.plot(column="Change_pct", cmap="RdBu_r", vmin=-100, vmax=100, legend=True, edgecolor="black",
             linewidth=0.4, ax=ax, legend_kwds={"label": "Change in homelessness 2012-13 (%)", "shrink": 0.6})
lower48[lower48.STATE_ABBR == "ND"].boundary.plot(ax=ax, color="black", linewidth=2)
ax.set_title("Change in homeless population 2012-2013 (North Dakota +%.0f %%)" % nd.Change_pct); ax.set_axis_off()
plt.savefig("pred_results/homeless_change_map.png", dpi=110, bbox_inches="tight")


# ---------------------------------------------------------------------------------------------
# Negative control: the analysis above is the reference solution, but every geographic
# deliverable file is then rewritten without a CRS, which violates the deliverable contract.
def _write_without_crs(path):
    """Rewrite one file in place so that it carries no CRS. Returns a short description.
    .shp: drop the .prj sidecar. .tif/.tiff: rewrite without CRS (and drop .aux.xml, which can
    hold one). .gpkg: rewrite every spatial layer with crs=None. .geojson/.json: write projected
    coordinates (UTM if the data are lon/lat) without a 'crs' member - a lon/lat GeoJSON
    without 'crs' is WGS 84 by definition (RFC 7946), so dropping the member alone would not
    remove the CRS."""
    import os
    low = path.lower()
    stem, _ = os.path.splitext(path)
    if low.endswith(".shp"):
        d, b = os.path.split(stem)
        gone = []
        for n in os.listdir(d or "."):
            if n.lower() == (b + ".prj").lower():
                os.remove(os.path.join(d, n))
                gone.append(n)
        return f"removed {gone}"
    if low.endswith((".tif", ".tiff")):
        import rasterio
        with rasterio.open(path) as r:
            prof, data = r.profile.copy(), r.read()
            tags = r.tags()
        prof.pop("crs", None)
        tmp = stem + ".__nocrs__" + os.path.splitext(path)[1]
        with rasterio.open(tmp, "w", **prof) as w:
            w.write(data)
            if tags:
                w.update_tags(**tags)
        os.replace(tmp, path)
        if os.path.exists(path + ".aux.xml"):
            os.remove(path + ".aux.xml")
        return "raster rewritten without CRS"
    if low.endswith(".gpkg"):
        import geopandas as gpd
        import pyogrio
        tmp = stem + ".__nocrs__.gpkg"
        for lyr, gt in pyogrio.list_layers(path):
            g = gpd.read_file(path, layer=lyr)
            if gt is not None:
                g = g.set_crs(None, allow_override=True)
            g.to_file(tmp, layer=lyr, driver="GPKG")
        os.replace(tmp, path)
        return "gpkg rewritten without CRS"
    if low.endswith((".geojson", ".json")):
        import geopandas as gpd
        g = gpd.read_file(path)
        if g.crs is None or g.crs.is_geographic:
            if g.crs is None:
                g = g.set_crs(4326)
            g = g.to_crs(g.estimate_utm_crs())
        g = g.set_crs(None, allow_override=True)
        os.remove(path)
        g.to_file(path, driver="GeoJSON")
        return "GeoJSON rewritten in projected coordinates without 'crs'"
    return "not a geographic file"


def _strip_deliverable_crs(patterns):
    import fnmatch
    import os
    for n in sorted(os.listdir("pred_results")):
        low = n.lower()
        if low.endswith(('.shp', '.geojson', '.gpkg', '.json', '.tif', '.tiff')) and any(fnmatch.fnmatchcase(low, p.lower()) for p in patterns):
            if low.endswith(".json"):
                try:
                    import json
                    if json.load(open(os.path.join("pred_results", n))).get("type") is None:
                        continue
                except Exception:
                    continue
            print("no-CRS control:", n, _write_without_crs(os.path.join("pred_results", n)))


_strip_deliverable_crs(['*.geojson', '*.gpkg', '*.shp'])
