"""Wrong: the reference solution of task 4 with every geographic deliverable written without a CRS.

The analysis: groundwater vulnerable areas that need protection, Morrow County OR.

Open-source port of the benchmark's ArcGIS ModelBuilder workflow:
  Project soils -> NAD83 / Oregon North (Lambert Conformal Conic, EPSG:32126 - the CRS of
  land_cover.tif, which rasterio only reads as a LOCAL_CS because of its ESRI header);
  PolygonToRaster (cell-centre) of drclassdcd and wdepannmin on the land_cover grid
  (cellSize + snapRaster = land_cover, extent = mc_boundary);
  Reclassify drainage (codes 1..5 = order in which ArcGIS meets the string values in the
  table: Well, Poorly, Somewhat excessively, Excessively, Somewhat poorly) with
  1->3, 2->1, 3->4, 4->5, 5->2, i.e. Excessively 5 > Somewhat excessively 4 > Well 3 >
  Somewhat poorly 2 > Poorly 1 (faster drainage = more vulnerable);
  Reclassify water depth 0-10->5, 10-33->4, 33-61->3, 61-80->2, 80-92->1 (ArcGIS ranges:
  upper bound inclusive);
  vulnerable = 5*drainage + 4*depth, rescaled linearly to [1, 10];
  Reclassify land cover (NLCD table); risk = 8*vulnerable + 10*land_cover;
  high risk = risk > 100; undeveloped = land-cover cells of the 8 undeveloped classes inside
  the high-risk zone (values = NLCD codes), written to pred_results/undeveloped_areas.tif.
"""
import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.windows import from_bounds, Window

LCC = CRS.from_epsg(32126)  # NAD83 / Oregon North (m)

DRAIN_CODES = ["Well drained", "Poorly drained", "Somewhat excessively drained",
               "Excessively drained", "Somewhat poorly drained"]  # ArcGIS value order (first seen)
DRAIN_RECLASS = {1: 3, 2: 1, 3: 4, 4: 5, 5: 2}
DEPTH_RECLASS = [(0, 10, 5), (10, 33, 4), (33, 61, 3), (61, 80, 2), (80, 92, 1)]
LC_RECLASS = {11: 1, 21: 6, 22: 7, 23: 8, 24: 10, 31: 4, 41: 3, 42: 1, 43: 3, 52: 3, 71: 2,
              81: 5, 82: 9, 90: 1, 95: 1}
UNDEVELOPED = {"Deciduous Forest": 41, "Emergent Herbaceous Wetlands": 95, "Hay/Pasture": 81,
               "Herbaceous": 71, "Mixed Forest": 43, "Shrub/Scrub": 52, "Woody Wetlands": 90,
               "Barren Land": 31}


def reclass_ranges(a, table):
    out = np.full(a.shape, np.nan)
    for i, (lo, hi, v) in enumerate(table):
        sel = (a >= lo) & (a <= hi) if i == 0 else (a > lo) & (a <= hi)
        out[sel & np.isnan(out)] = v
    return out


def main():
    soils = gpd.read_file("dataset/mc_soils.shp").to_crs(LCC)
    boundary = gpd.read_file("dataset/mc_boundary.shp").to_crs(4326).to_crs(LCC)

    with rasterio.open("dataset/land_cover.tif") as src:
        # extent = boundary, snapped to the land_cover grid
        win = from_bounds(*boundary.total_bounds, transform=src.transform)
        win = win.round_offsets(op="floor").round_lengths(op="ceil")
        win = win.intersection(Window(0, 0, src.width, src.height))
        land_cover = src.read(1, window=win)
        transform = src.window_transform(win)
        lc_nodata = src.nodata
    shape = land_cover.shape
    lc_valid = land_cover != lc_nodata

    # Polygon to Raster
    code = {s: i + 1 for i, s in enumerate(DRAIN_CODES)}
    soils["drain_code"] = soils["drclassdcd"].map(code)
    drainage_conditions = rasterize(
        ((g, v) for g, v in zip(soils.geometry, soils["drain_code"]) if v == v),
        out_shape=shape, transform=transform, fill=0, dtype="int16").astype(float)
    drainage_conditions[drainage_conditions == 0] = np.nan
    water_depth = rasterize(
        ((g, v) for g, v in zip(soils.geometry, soils["wdepannmin"]) if v == v),
        out_shape=shape, transform=transform, fill=-9999, dtype="float32").astype(float)
    water_depth[water_depth == -9999] = np.nan

    # Reclassify
    reclass_drainage = np.full(shape, np.nan)
    for k, v in DRAIN_RECLASS.items():
        reclass_drainage[drainage_conditions == k] = v
    reclass_depth = reclass_ranges(water_depth, DEPTH_RECLASS)

    vulnerable_areas = reclass_drainage * 5 + reclass_depth * 4
    vmin, vmax = np.nanmin(vulnerable_areas), np.nanmax(vulnerable_areas)
    reclass_vulnerable = 1 + 9 * (vulnerable_areas - vmin) / (vmax - vmin)

    reclass_land_cover = np.full(shape, np.nan)
    for k, v in LC_RECLASS.items():
        reclass_land_cover[land_cover == k] = v

    risk_zones = reclass_vulnerable * 8 + reclass_land_cover * 10
    high_risk_zones = np.where(risk_zones > 100, risk_zones, np.nan)
    undeveloped = np.isin(land_cover, list(UNDEVELOPED.values())) & np.isfinite(high_risk_zones) & lc_valid
    undeveloped_areas = np.where(undeveloped, land_cover, 0).astype("uint8")

    print(f"vulnerable range {vmin}-{vmax}; risk range {np.nanmin(risk_zones):.1f}-{np.nanmax(risk_zones):.1f}")
    print(f"high-risk cells {np.isfinite(high_risk_zones).sum()}, undeveloped high-risk cells {undeveloped.sum()} "
          f"({undeveloped.sum() * 900 / 1e6:.2f} km2)")
    vals, cnt = np.unique(undeveloped_areas[undeveloped], return_counts=True)
    print("classes:", dict(zip(vals.tolist(), cnt.tolist())))

    os.makedirs("pred_results", exist_ok=True)
    prof = dict(driver="GTiff", height=shape[0], width=shape[1], count=1, crs=LCC, transform=transform,
                compress="lzw")
    with rasterio.open("pred_results/undeveloped_areas.tif", "w", dtype="uint8", nodata=0, **prof) as dst:
        dst.write(undeveloped_areas, 1)
    with rasterio.open("pred_results/risk_zones.tif", "w", dtype="float32", nodata=-9999, **prof) as dst:
        dst.write(np.where(np.isfinite(risk_zones), risk_zones, -9999).astype("float32"), 1)
    return risk_zones, undeveloped_areas


risk_zones, undeveloped_areas = main()


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


_strip_deliverable_crs(['*undeveloped*.tif', '*undeveloped*.tiff'])
