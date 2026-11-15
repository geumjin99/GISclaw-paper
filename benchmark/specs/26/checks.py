"""Checks for task 26: groundwater vulnerable areas - drainage x 5 + water depth x 4 on
rasterized soils.

Task core: project land cover and soils to Lambert Conformal Conic (Oregon North StatePlane),
rasterize drainage class (text field, coded 1-5 by first appearance) and minimum water-table
depth from the soils on the land-cover grid within the county-boundary extent, reclassify both
with the given remaps and compute vulnerable_areas = drainage*5 + depth*4 (values 14-37).
The gold agrees 98.9 % cell-for-cell with the ArcGIS result shipped as the input of task 27.

Correctness (required):
  1. a vulnerability raster (GeoTIFF in pred_results) exists;
  2. sampled at the gold cell centres (reprojected with pyproj, so any CRS/grid is accepted),
     its valid area overlaps the gold valid area (IoU >= 0.90) - catches CRS/extent errors;
  3. its values equal the gold values on >= 90 % of cells valid in both - catches wrong
     drainage coding, skipped/incorrect reclassification, swapped weights.
  4. per class: decomposing values into drainage score a and depth score b (5a+4b is unique),
     every class with >= 50 sampled gold cells is reproduced on >= 80 % of its cells
     (catches coding errors confined to rare drainage classes).
  5. (required: the instruction says "project ... to Lambert Conformal
     Conic") the output CRS is a Lambert Conformal Conic projection (any LCC: State Plane,
     Oregon Lambert, custom; 1SP or 2SP; detected from the projection method of the CRS);
  5b. (required: the instruction says "land_cover as the cellSize") output cell size equals the
     land-cover cell (30 m = 98.42 US ft) within +/-10 %, in the units of the output CRS.
Reported, not required:
  6. the value set is a subset of the 25 possible 5a+4b combinations (a, b in 1..5).
"""
import glob
import os

import numpy as np

from checklib import result


def _fix_proj():
    """Point PROJ at the data directory bundled with rasterio (via the environment) so that
    the CRS of GeoTIFF files is identified consistently by rasterio and pyproj."""
    import importlib.util
    spec = importlib.util.find_spec("rasterio")
    if spec and spec.origin:
        d = os.path.join(os.path.dirname(spec.origin), "proj_data")
        if os.path.exists(os.path.join(d, "proj.db")):
            os.environ["PROJ_DATA"] = d


_fix_proj()

POSSIBLE = {5 * a + 4 * b for a in range(1, 6) for b in range(1, 6)}
STEP = 3


def sample_on_gold(pred_path, gold_path, step=STEP):
    """Pred values at gold cell centres (subsampled), via pyproj; invalid -> nan.
    Returns (pred_vals, gold_vals) as float arrays with nan for nodata/outside."""
    import rasterio
    from pyproj import CRS, Transformer
    with rasterio.open(gold_path) as g:
        G = g.read(1).astype(float)
        if g.nodata is not None:
            G[G == g.nodata] = np.nan
        gt, gcrs = g.transform, g.crs
    rows, cols = np.mgrid[0:G.shape[0]:step, 0:G.shape[1]:step]
    xs = gt.c + (cols + 0.5) * gt.a
    ys = gt.f + (rows + 0.5) * gt.e
    gv = G[rows, cols]
    with rasterio.open(pred_path) as p:
        P = p.read(1).astype(float)
        if p.nodata is not None:
            P[P == p.nodata] = np.nan
        pt, pcrs = p.transform, p.crs
    if pcrs is None:
        # no CRS guessing for deliverable files: a deliverable without a CRS is not interpreted
        raise ValueError("pred raster has no CRS - not interpreted (uniform CRS rule)")
    if pcrs is not None and gcrs is not None and CRS.from_wkt(pcrs.to_wkt()) != CRS.from_wkt(gcrs.to_wkt()):
        tr = Transformer.from_crs(CRS.from_wkt(gcrs.to_wkt()), CRS.from_wkt(pcrs.to_wkt()), always_xy=True)
        xs, ys = tr.transform(xs, ys)
    inv = ~pt
    c, r = inv * (np.asarray(xs), np.asarray(ys))
    c, r = np.floor(c).astype(int), np.floor(r).astype(int)
    ok = (r >= 0) & (r < P.shape[0]) & (c >= 0) & (c < P.shape[1])
    pv = np.full(gv.shape, np.nan)
    pv[ok] = P[r[ok], c[ok]]
    return pv, gv


def _alt_gold(ctx, gv_shape_step=STEP):
    """Gold values under the other reading of the shared RemapRange breaks ([lo, hi): a depth
    on a break - 10, 33, 61, 80 - goes to the upper class). The table
    [[0,10,5],[10,33,4],[33,61,3],[61,80,2],[80,92,1]] repeats every break, and 80 cm alone is
    13.7 % of the county, so this convention alone moves agreement to ~0.86. ArcGIS (and the
    gold) put a break value in the lower class; a Python analyst reading the table literally
    may not. Both are accepted. Needs the gold water_depth.tif (raw depth on the gold grid)."""
    import rasterio
    with rasterio.open(ctx.gold("water_depth.tif")) as w:
        wd = w.read(1).astype(float)[::gv_shape_step, ::gv_shape_step]
    return wd


def checks(ctx):
    out = []
    tifs = [f for f in glob.glob(os.path.join(ctx.pred_dir, "*.tif*"))]
    best = None
    try:
        wd = _alt_gold(ctx)
    except Exception:
        wd = None
    for f in tifs:
        try:
            pv, gv = sample_on_gold(f, ctx.gold("vulnerable_areas.tif"))
        except Exception as e:
            continue
        pv = np.where(pv > 0, pv, np.nan)
        vp, vg = np.isfinite(pv), np.isfinite(gv)
        iou = (vp & vg).sum() / max(1, (vp | vg).sum())
        both = vp & vg
        golds = [("breaks to lower class (ArcGIS)", gv)]
        if wd is not None and wd.shape == gv.shape:
            # same drainage score a, depth score b re-derived with [lo, hi) breaks
            a = np.floor((gv - 4 * np.select([wd <= 10, wd <= 33, wd <= 61, wd <= 80, wd <= 92],
                                             [5, 4, 3, 2, 1], 0)) / 5)
            b_alt = np.select([wd < 10, wd < 33, wd < 61, wd < 80, wd <= 92], [5, 4, 3, 2, 1], 0)
            golds.append(("breaks to upper class [lo, hi)", np.where(np.isfinite(gv), 5 * a + 4 * b_alt, np.nan)))
        for conv, gg in golds:
            agree = float((np.round(pv[both]) == gg[both]).mean()) if both.any() else 0.0
            score = agree + iou + (0.01 if "vulner" in os.path.basename(f).lower() else 0)
            if best is None or score > best[0]:
                best = (score, os.path.basename(f) + " / " + conv, iou, agree, pv[both], f, pv[both], gg[both])
    out.append(result("vulnerability raster written", best is not None, value=[os.path.basename(t) for t in tifs]))
    if best is None:
        out.append(result("valid area matches gold (IoU)", False, note="no readable GeoTIFF"))
        out.append(result("cell values equal gold (drainage*5 + depth*4)", False))
        out.append(result("drainage and depth scores correct for every class (>= 0.80)", False))
        out.append(result("output CRS is Lambert Conformal Conic", False))
        out.append(result("output cell size = land-cover cell size (30 m +/-10 %)", False))
        out.append(result("values are valid 5a+4b combinations", False, required=False))
        return out
    _, name, iou, agree, vals, path = best[:6]
    out.append(result("valid area matches gold (IoU)", iou >= 0.90, value=round(float(iou), 4), threshold=0.90,
                      note=name))
    out.append(result("cell values equal gold (drainage*5 + depth*4)", agree >= 0.90, value=round(agree, 4),
                      threshold=0.90, note=name))
    # per-component, per-class agreement: 5a+4b decomposes uniquely
    # into the drainage score a and the depth score b. "Well drained" dominates, so a coding
    # error confined to rare drainage classes can keep overall agreement >= 0.90; every score
    # class (>= 50 sampled gold cells) of each component must be reproduced on >= 80 % of cells.
    dec = {5 * a + 4 * b: (a, b) for a in range(1, 6) for b in range(1, 6)}
    pv_b, gv_b = best[6], best[7]
    P = np.array([dec.get(int(round(x)), (0, 0)) for x in pv_b]).reshape(-1, 2)
    G = np.array([dec.get(int(round(x)), (0, 0)) for x in gv_b]).reshape(-1, 2)
    per = {}
    for comp, nm in ((0, "drainage"), (1, "depth")):
        for c in range(1, 6):
            sel = G[:, comp] == c
            if sel.sum() >= 50:
                per[f"{nm}{c}"] = round(float((P[sel, comp] == c).mean()), 3)
    out.append(result("drainage and depth scores correct for every class (>= 0.80)",
                      bool(per) and min(per.values()) >= 0.80, value=per, threshold=0.80,
                      note="value decomposed into drainage a / depth b (5a+4b is unique)"))
    # the instructed projection (LCC) and cell size (land cover) are required
    lcc, method, cell_m = False, None, None
    try:
        import rasterio
        from pyproj import CRS
        with rasterio.open(path) as p:
            wkt = p.crs.to_wkt() if p.crs else ""
            res = abs(p.transform.a), abs(p.transform.e)
        lcc = "Lambert_Conformal_Conic" in wkt or "Lambert Conic Conformal" in wkt
        if wkt:
            c = CRS.from_wkt(wkt)
            op = c.coordinate_operation
            method = op.method_name if op is not None else None
            lcc = lcc or bool(method and "lambert conic conformal" in method.lower())
            if c.is_projected:
                f = c.axis_info[0].unit_conversion_factor or 1.0
                cell_m = (res[0] * f, res[1] * f)
    except Exception:
        pass
    out.append(result("output CRS is Lambert Conformal Conic", lcc, value=method,
                      note="instructed: 'project ... to Lambert Conformal Conic'"))
    ok_cell = cell_m is not None and all(27.0 <= v <= 33.0 for v in cell_m)
    out.append(result("output cell size = land-cover cell size (30 m +/-10 %)", ok_cell,
                      value=[round(v, 2) for v in cell_m] if cell_m else None, threshold="27-33 m",
                      note="instructed: 'land_cover as the cellSize'"))
    u = set(np.unique(np.round(vals)).astype(int).tolist()) if len(vals) else set()
    out.append(result("values are valid 5a+4b combinations", bool(u) and u <= POSSIBLE, value=sorted(u)[:20],
                      required=False))
    return out
