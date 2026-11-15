"""Wrong: the reference solution of task 47 with every geographic deliverable written without a CRS.

The analysis: impervious area per land parcel (Tabulate Area + join + copy).

impervious.tif: 0.5 ft cells (NAD83 / Kentucky North, US ft), classes from its attribute
table (impervious.tif.vat.dbf): Value 0 = Impervious, Value 1 = Pervious, 255 = NoData.
Tabulate Area (cell centres within each parcel) -> area per class in square feet
(cell count x 0.25 ft2), fields VALUE_0 / VALUE_1 as ArcGIS names them, joined to the
121 parcels on Parcel_ID and copied to pred_results/parcels_joint.shp.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats

parcels = gpd.read_file("dataset/parcels.shp")
with rasterio.open("dataset/impervious.tif") as src:
    cell_area = abs(src.res[0] * src.res[1])        # 0.25 ft2
    nodata = src.nodata
print("parcels:", len(parcels), "cell area ft2:", cell_area)

zs = zonal_stats(parcels, "dataset/impervious.tif", categorical=True, nodata=nodata, all_touched=False)
tab = pd.DataFrame([{"Parcel_ID": pid, "VALUE_0": z.get(0, 0) * cell_area, "VALUE_1": z.get(1, 0) * cell_area}
                    for pid, z in zip(parcels["Parcel_ID"], zs)])
impervious_area = tab                                 # Tabulate Area result (one row per parcel)

parcels_joint = parcels.merge(impervious_area, on="Parcel_ID", how="left")   # Add Join (KEEP_ALL)
parcels_joint["Imperv_ft2"] = parcels_joint["VALUE_0"]
parcels_joint["Imperv_pct"] = 100 * parcels_joint["VALUE_0"] / (parcels_joint["VALUE_0"] + parcels_joint["VALUE_1"])
parcels_joint.to_file("pred_results/parcels_joint.shp")             # Copy Features
print("total impervious ft2: %.0f, pervious ft2: %.0f, mean impervious share %.1f %%" % (
    parcels_joint.VALUE_0.sum(), parcels_joint.VALUE_1.sum(), parcels_joint.Imperv_pct.mean()))


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


_strip_deliverable_crs(['*.geojson', '*.gpkg', '*.json', '*.shp'])
