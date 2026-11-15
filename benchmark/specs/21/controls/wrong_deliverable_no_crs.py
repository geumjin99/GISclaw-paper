"""Wrong: the reference analysis (reclassification of NLCD land cover and protected status to
a common 1-10 cost scale), but every geographic deliverable file is then rewritten without a CRS."""
import os
import numpy as np
import rasterio

landCover_classification = {
    11: 10, 21: 8, 22: 7, 23: 8, 24: 9, 31: 6, 41: 2, 42: 1, 43: 2,
    52: 3, 71: 3, 72: 3, 81: 4, 82: 6, 90: 4, 95: 4, 255: 10,
}
protected_status_classification = {0: 1, 1: 3, 2: 6, 3: 9, 4: 10, 255: 10}


def reclassify(array, table, unmapped=0):
    """Lookup-table reclassification; every output depends only on the ORIGINAL value
    (no chained re-mapping). Values not in the table become `unmapped` (NoData)."""
    lut = np.full(256, unmapped, dtype=np.uint8)
    for k, v in table.items():
        lut[k] = v
    return lut[array]


def run(src_path, table, out_path):
    with rasterio.open(src_path) as src:
        a = src.read(1)
        prof = src.profile
    out = reclassify(a, table)
    unmapped = sorted(set(np.unique(a).tolist()) - set(table))
    print(os.path.basename(src_path), "unmapped input values ->", unmapped)
    prof.update(dtype="uint8", count=1, nodata=0, compress="lzw")
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(out, 1)
    return a, out


os.makedirs("pred_results", exist_ok=True)
landCover, landCover_reclassified = run("dataset/landCover.tif", landCover_classification,
                                        "pred_results/landCover_reclassified.tif")
protected_status, protected_status_reclassified = run("dataset/Protected_Status.tif",
                                                      protected_status_classification,
                                                      "pred_results/protected_status_reclassified.tif")
for name, arr in (("landCover", landCover_reclassified), ("protected", protected_status_reclassified)):
    u, c = np.unique(arr, return_counts=True)
    print(name, dict(zip(u.tolist(), c.tolist())))


# ---------------------------------------------------------------------------------------------
# Negative control: the analysis above is the reference solution, but every geographic
# deliverable file is then written without a CRS. Expected: wrong.
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


_strip_deliverable_crs(['landCover_reclassified.tif*', 'protected_status_reclassified.tif*'])
