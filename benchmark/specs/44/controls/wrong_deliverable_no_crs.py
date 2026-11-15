"""Wrong: the reference analysis of task 44 (fire-station coverage gaps in Etobicoke), with every geographic
deliverable then written without a CRS; the deliverables must be georeferenced.

Both layers are delivered in geographic coordinates (EPSG:4326), so a 2 500 m buffer must be
built in a metric CRS: NAD83 / UTM zone 17N (EPSG:26917). All 84 Toronto stations are buffered
(stations outside Etobicoke still serve its edges), the buffers are dissolved, and the
no-service area is Etobicoke minus the covered area. The 2 500 m distance is the one used by
the original solution (the instruction gives none).
Outputs: pred_results/fire_buffer.geojson, pred_results/no_service.geojson,
pred_results/fire_coverage_gaps.png

Adapted from the reference script of GeoAnalystBench (Zhang et al. 2025, Apache-2.0).
"""
import geopandas as gpd
import matplotlib.pyplot as plt

METRIC = "EPSG:26917"
DIST = 2500.0

etobicoke = gpd.read_file("dataset/etobicoke.geojson").to_crs(METRIC)
stations = gpd.read_file("dataset/fire_station.geojson").to_crs(METRIC)
print("stations:", len(stations), "in Etobicoke:", int(stations.within(etobicoke.union_all()).sum()))

fire_buffer = gpd.GeoDataFrame(stations[["NAME", "ADDRESS"]].copy(), geometry=stations.buffer(DIST, resolution=32),
                               crs=METRIC)
fire_buffer.to_file("pred_results/fire_buffer.geojson", driver="GeoJSON")

coverage = gpd.GeoDataFrame(geometry=[fire_buffer.union_all()], crs=METRIC)
no_service = gpd.overlay(etobicoke, coverage, how="difference")
no_service = no_service.explode(index_parts=False).reset_index(drop=True)
no_service["area_km2"] = no_service.area / 1e6
no_service = no_service[no_service.area_km2 > 1e-6].sort_values("area_km2", ascending=False)
no_service.to_file("pred_results/no_service.geojson", driver="GeoJSON")
total = etobicoke.area.sum() / 1e6
print(f"Etobicoke {total:.1f} km2, no service {no_service.area_km2.sum():.2f} km2 "
      f"({100 * no_service.area_km2.sum() / total:.1f} %) in {len(no_service)} pieces")

fig, ax = plt.subplots(figsize=(9, 9))
etobicoke.plot(ax=ax, color="whitesmoke", edgecolor="black")
fire_buffer.clip(etobicoke).plot(ax=ax, color="lightgreen", alpha=0.5)
no_service.plot(ax=ax, color="red")
stations.clip(etobicoke.buffer(DIST)).plot(ax=ax, color="darkred", markersize=15, marker="^")
ax.set_title(f"Etobicoke fire-station coverage ({DIST:.0f} m) and no-service areas (red)")
ax.set_axis_off()
plt.savefig("pred_results/fire_coverage_gaps.png", dpi=120, bbox_inches="tight")


# ---------------------------------------------------------------------------------------------
# NEGATIVE CONTROL: the analysis above is the reference solution, but every geographic
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


_strip_deliverable_crs(['*.geojson', '*.gpkg', '*.json', '*.shp'])
