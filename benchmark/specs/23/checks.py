"""Task 23 - PAD-US open space: reclassified raster + vectorized polygons with area.

The input raster is thematic: each cell holds the OBJECTID of a PAD-US protected area
(eligible open space); 2147483648 is NoData (outside any unit). Correct reclassification:
valid PAD-US id -> 1, NoData -> 0 (294 448 cells, 265.0 km2).

Correctness (required):
  1. open_space.tif exists; its open-space class (cells == 1 / > 0) has IoU >= 0.95 with the
     gold open-space cells (pred warped onto the gold grid if CRS/grid differ);
  2. open_space.shp (or .gpkg/.geojson) exists and is non-empty;
  3. polygon union has area IoU >= 0.95 with the gold polygons (if a class column holds both
     0 and 1, only class 1 is used);
  4. area statistics: a numeric attribute equal to the equal-area polygon area in a common
     unit (m2, km2, ha, acres, sq ft, sq mi; x0.85 allowed): per polygon (median rel. error
     <= 3 %, r >= 0.99) or, for < 3 features (e.g. dissolved), in total (+/-3 %). Square
     degrees and cell counts fail.
  6. total area (sum of the area attribute) equals the gold total in some common unit
     (m2, km2, ha, acres, sq ft, sq mi), with or without the 0.85 accuracy factor (+/-3 %)
     (required: the instruction asks to calculate the total area).
Reported, not required:
  5. polygon validity (>= 99 %; 8-connected vectorization legitimately gives invalid rings).
"""
import os

import numpy as np

from checklib import result, area_iou, valid_fraction


def _fix_proj():
    """Point PROJ at the data directory bundled with rasterio so that rasterio and pyproj agree
    (with a mismatched proj.db, GeoTIFF CRSs are read as LOCAL_CS)."""
    import importlib.util
    spec = importlib.util.find_spec("rasterio")
    if spec and spec.origin:
        d = os.path.join(os.path.dirname(spec.origin), "proj_data")
        if os.path.exists(os.path.join(d, "proj.db")):
            os.environ["PROJ_DATA"] = d


_fix_proj()

TIF = ["open_space.tif", "open_space.tiff", "Open_Space.tif"]
VEC = ["open_space.shp", "open_space.gpkg", "open_space.geojson", "Open_Space.shp"]
UNITS = {"m2": 1.0, "km2": 1e-6, "ha": 1e-4, "acres": 0.000247105, "sqft": 10.7639, "sqmi": 3.861e-7}


def _first(ctx, names):
    for n in names:
        if os.path.exists(ctx.pred(n)):
            return n
    return None


def _binary_on_gold_grid(pred_path, gold_path):
    import rasterio
    from rasterio.warp import reproject, Resampling
    with rasterio.open(gold_path) as g:
        gshape, gtr, gcrs = (g.height, g.width), g.transform, g.crs
        gold = g.read(1)
    with rasterio.open(pred_path) as p:
        a = p.read(1).astype("float64")
        if p.nodata is not None:
            a[a == p.nodata] = 0
        a[~np.isfinite(a)] = 0
        nvals = len(np.unique(a[a > 0]))
        b = (a > 0).astype("uint8")
        if p.crs is None:
            # no CRS guessing for the deliverable file: deliverable files must carry a CRS
            raise ValueError("pred raster has no CRS - not interpreted (uniform CRS rule)")
        if p.crs == gcrs and p.transform == gtr and b.shape == gshape:
            pb = b
        else:
            pb = np.zeros(gshape, "uint8")
            reproject(b, pb, src_transform=p.transform, src_crs=p.crs, dst_transform=gtr,
                      dst_crs=gcrs, resampling=Resampling.nearest)
    return pb == 1, gold == 1, nvals


def checks(ctx):
    out = []
    t = _first(ctx, TIF)
    if t is None:
        out.append(result("open_space.tif present", False, threshold=TIF))
        out.append(result("open-space cells agree with gold (IoU)", False, note="no raster"))
    else:
        out.append(result("open_space.tif present", True, value=t))
        try:
            p, g, nvals = _binary_on_gold_grid(ctx.pred(t), ctx.gold("open_space.tif"))
            iou = float((p & g).sum() / max(1, (p | g).sum()))
            # the raw PAD-US id raster copied unchanged also satisfies the IoU test, because every
            # valid id is > 0. The instruction asks for a reclassified 1/0 raster, so the
            # open-space class must be ONE value.
            binary = nvals <= 1
            out.append(result("open-space cells agree with gold (IoU)", iou >= 0.95 and binary, value=round(iou, 4),
                              threshold="IoU >= 0.95 and a single open-space value",
                              note=f"pred open cells {int(p.sum())}, gold {int(g.sum())}; distinct positive values {nvals}"))
        except Exception as e:
            out.append(result("open-space cells agree with gold (IoU)", False, note=repr(e)[:200]))

    v = _first(ctx, VEC)
    if v is None:
        out.append(result("open-space polygons present (non-empty)", False, threshold=VEC))
        out.append(result("polygons cover the gold open space (area IoU)", False))
        out.append(result("area statistics attribute in a real area unit", False))
        out.append(result("total area matches gold in some unit", False))
        return out
    import geopandas as gpd
    try:
        pv = gpd.read_file(ctx.pred(v))
        gv = gpd.read_file(ctx.gold("open_space.shp"))
    except Exception as e:
        out.append(result("open-space polygons present (non-empty)", False, note=repr(e)[:200]))
        return out
    if pv.crs is None:
        # no CRS guessing for the deliverable file: deliverable files must carry a CRS
        note = f"{v} has no CRS - not interpreted (uniform CRS rule)"
        out.append(result("open-space polygons present (non-empty)", False, value={"file": v, "n": len(pv)}, note=note))
        out.append(result("polygons cover the gold open space (area IoU)", False, note=note))
        out.append(result("area statistics attribute in a real area unit", False, note=note))
        out.append(result("total area matches gold in some unit", False, note=note))
        return out
    for c in pv.columns:
        if c != "geometry" and pv[c].dtype.kind in "fiu":
            vals = set(np.unique(pv[c].dropna()).tolist())
            if vals == {0, 1} and c.lower() in ("value", "gridcode", "dn", "raster_val", "class", "open_space", "val"):
                pv = pv[pv[c] == 1]
                break
    vf = valid_fraction(pv) if len(pv) else 0.0
    out.append(result("open-space polygons present (non-empty)", len(pv) > 0, value={"file": v, "n": len(pv)}))
    out.append(result("polygon geometries valid", vf >= 0.99, value=round(vf, 4), threshold=">=99 % valid",
                      required=False, note="8-connected vectorization yields self-touching rings; area is unaffected"))
    try:
        pvv = pv.copy()
        pvv["geometry"] = pvv.geometry.buffer(0)
        iou = area_iou(pvv, gv)
    except Exception as e:
        iou = float("nan")
    out.append(result("polygons cover the gold open space (area IoU)", iou >= 0.95,
                      value=round(iou, 4) if iou == iou else None, threshold=0.95))

    # Area attribute in a real area unit: correlation alone is scale-invariant and would
    # accept square degrees / cell counts. An attribute must equal
    # the equal-area (EPSG:5070) polygon area in a known unit (x0.85 allowed): per polygon
    # (median relative error <= 3 % and r >= 0.99 when >= 3 polygons) or, for a single
    # dissolved feature / few polygons, in total (+/-3 %).
    try:
        geom_area = (pv.to_crs(gv.crs) if pv.crs and pv.crs != gv.crs else pv).geometry.area.to_numpy()
    except Exception:
        geom_area = pv.geometry.area.to_numpy()
    best_c, best_info = None, None
    for c in pv.columns:
        if c == "geometry" or pv[c].dtype.kind not in "fiu" or not len(pv):
            continue
        x = pv[c].astype(float).to_numpy()
        if not np.isfinite(x).all() or x.sum() <= 0:
            continue
        for u, f in UNITS.items():
            for adj in (1.0, 0.85):
                exp = geom_area * f * adj
                if len(pv) >= 3:
                    rel = np.abs(x - exp) / np.maximum(exp, 1e-12)
                    ok = float(np.median(rel)) <= 0.03 and np.std(x) > 0 and np.corrcoef(x, exp)[0, 1] >= 0.99
                else:
                    ok = abs(x.sum() - exp.sum()) <= 0.03 * exp.sum()
                if ok and best_c is None:
                    best_c, best_info = c, f"{u}{' x0.85' if adj != 1 else ''}"
    out.append(result("area statistics attribute in a real area unit", best_c is not None,
                      value={"column": best_c, "unit": best_info},
                      threshold="attribute = equal-area polygon area in m2/km2/ha/acres/sqft/sqmi (x0.85 ok), +/-3 %"))
    gold_total = float(gv.geometry.area.sum())
    match = None
    if best_c is not None:
        tot = float(pv[best_c].sum())
        for u, f in UNITS.items():
            for adj in (1.0, 0.85):
                if abs(tot - gold_total * f * adj) <= 0.03 * gold_total * f * adj:
                    match = f"{u}{' x0.85' if adj != 1 else ''}"
    out.append(result("total area matches gold in some unit", match is not None, value=match,
                      threshold=f"gold {gold_total / 1e6:.2f} km2 +/-3 %"))
    return out
