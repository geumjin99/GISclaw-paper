"""Wrong: the reference analysis of task 23 (PAD-US open space: reclassify, vectorize, area), with every
geographic deliverable then written without a CRS; the deliverables must be georeferenced."""
import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape

SQM_TO_ACRES = 0.000247105
IMPERVIOUS_ACCURACY = 0.85  # NLCD impervious-surface accuracy used in the CRS lesson


def main():
    with rasterio.open("dataset/PADUS_CRS_clip_1.tif") as src:
        padus = src.read(1)
        profile = src.profile.copy()
        nodata = src.nodata                 # 2147483648 (uint32): outside any PAD-US unit
        transform, crs = src.transform, src.crs
    # The raster is thematic: each cell holds the OBJECTID of a PAD-US protected area
    # (eligible open space). Threshold: a valid PAD-US id (> 0 and != NoData) -> open space.
    valid = (padus != nodata) & (padus > 0)
    open_space = valid.astype("uint8")
    cell_area = abs(transform.a * transform.e)              # 900 m2
    print("open-space cells:", int(open_space.sum()), "of", open_space.size)

    profile.update(dtype="uint8", nodata=255, count=1, compress="lzw")
    with rasterio.open("pred_results/open_space.tif", "w", **profile) as dst:
        dst.write(open_space, 1)

    polys = [shape(geom) for geom, val in shapes(open_space, mask=open_space == 1,
                                                  transform=transform, connectivity=4) if val == 1]
    gdf = gpd.GeoDataFrame({"gridcode": np.ones(len(polys), dtype=int)}, geometry=polys, crs=crs)
    gdf["area_sqm"] = gdf.geometry.area
    gdf["acres"] = gdf["area_sqm"] * SQM_TO_ACRES
    gdf["acres_adj"] = gdf["acres"] * IMPERVIOUS_ACCURACY   # CRS-lesson credited acres
    gdf.to_file("pred_results/open_space.shp")

    total_sqm = float(gdf["area_sqm"].sum())
    assert abs(total_sqm - open_space.sum() * cell_area) < 1.0
    total_acres = total_sqm * SQM_TO_ACRES
    print(f"Open space polygons: {len(gdf)}")
    print(f"Total open space area: {total_sqm:,.0f} m2 = {total_sqm / 1e6:.2f} km2 = {total_acres:,.1f} acres"
          f" ({total_acres * IMPERVIOUS_ACCURACY:,.1f} acres after the 0.85 accuracy factor)")


main()


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


_strip_deliverable_crs(['open_space.geojson', 'open_space.gpkg', 'open_space.shp', 'open_space.tif', 'open_space.tiff'])
