"""Task 4 - groundwater vulnerable areas that need protection (Morrow County, OR).

Task core: soils -> LCC (NAD83 / Oregon North), rasterise drainage class and water-table depth
on the land_cover grid, reclassify both, vulnerable = 5*drain + 4*depth, standardise to [1,10],
risk = 8*vulnerable + 10*reclassified land cover, keep the undeveloped land-cover classes inside
risk > 100 -> undeveloped_areas.tif.

The result sits on a knife edge: risk > 100 for undeveloped classes is reached only by a few
exact score combinations, so small changes of interpretation change the answer drastically
(gold 665 cells; alternatives 102 - 103 000 cells, mutual IoU 0.006 - 0.17):
  a. which drainage string gets raster code 1..5 - stated in the task text (1 = Well,
     2 = Poorly, 3 = Somewhat excessively, 4 = Excessively, 5 = Somewhat poorly; the ArcGIS
     order of appearance). Alphabetical coding is not accepted;
  b. range bounds of the depth reclass - the data take exactly the break values 10, 33, 61, 80
     (ArcGIS: upper bound inclusive; np.digitize style: lower bound inclusive) - both accepted;
  c. "standardisation to [1, 10]" - stated in the task text as continuous linear min-max
     rescaling, not rounded; 10 equal-interval classes or rounding are not accepted.
Same standard as task 26 (drainage coding) and task 27 (standardisation). Both depth-bound variants are
recomputed here from the inputs and accepted; anything else is not.

Correctness (required):
  1. undeveloped_areas.tif (or .tiff / a variant name containing 'undeveloped') exists;
  2. it is a georeferenced raster with at least one selected cell;
  3. its selected cells (valid, non-zero) match one accepted variant on the land_cover grid,
     with 1-cell tolerance for regridding: precision >= 0.85 and recall >= 0.85;
  4a. the prescribed projection step ("First, project 'mc_soils.shp' to Lambert
     Conformal Conic"): a captured soils layer in an LCC CRS, or the deliverable raster in LCC.
Reported, not required:
  4. selected cells carry undeveloped NLCD codes (or a 0/1 mask);
  5. which interpretation matched.
"""
import glob
import itertools
import os

import numpy as np

from checklib import result

DRAIN_APPEAR = ["Well drained", "Poorly drained", "Somewhat excessively drained",
                "Excessively drained", "Somewhat poorly drained"]
DRAIN_RECLASS = {1: 3, 2: 1, 3: 4, 4: 5, 5: 2}
DEPTH = [(0, 10, 5), (10, 33, 4), (33, 61, 3), (61, 80, 2), (80, 92, 1)]
LC_RECLASS = {11: 1, 21: 6, 22: 7, 23: 8, 24: 10, 31: 4, 41: 3, 42: 1, 43: 3, 52: 3, 71: 2,
              81: 5, 82: 9, 90: 1, 95: 1}
UNDEV = [41, 95, 81, 71, 43, 52, 90, 31]
LCC = "EPSG:32126"


def _variants(ctx):
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    with rasterio.open(f"{ctx.inputs_dir}/land_cover.tif") as s:
        lc, tr = s.read(1), s.transform
    sh = lc.shape
    soils = gpd.read_file(f"{ctx.inputs_dir}/mc_soils.shp").to_crs(LCC)
    wd = rasterize(((g, v) for g, v in zip(soils.geometry, soils.wdepannmin) if v == v),
                   out_shape=sh, transform=tr, fill=-1, dtype="float32").astype(float)
    wd[wd < 0] = np.nan
    rl = np.full(sh, np.nan)
    for k, v in LC_RECLASS.items():
        rl[lc == k] = v
    undev = np.isin(lc, UNDEV)
    depth = {}
    for bins in ("upper-inclusive", "lower-inclusive"):
        rw = np.full(sh, np.nan)
        for i, (lo, hi, v) in enumerate(DEPTH):
            if bins == "upper-inclusive":
                sel = ((wd >= lo) if i == 0 else (wd > lo)) & (wd <= hi)
            else:
                sel = (wd >= lo) & ((wd < hi) if i < len(DEPTH) - 1 else (wd <= hi))
            rw[sel & np.isnan(rw)] = v
        depth[bins] = rw
    out = {}
    for order in ("appearance",):   # the task states the coding (order of appearance)
        names = DRAIN_APPEAR
        code = {s: i + 1 for i, s in enumerate(names)}
        dc = rasterize(((g, code[v]) for g, v in zip(soils.geometry, soils.drclassdcd) if v in code),
                       out_shape=sh, transform=tr, fill=0, dtype="int16")
        rd = np.full(sh, np.nan)
        for k, v in DRAIN_RECLASS.items():
            rd[dc == k] = v
        for bins, std in itertools.product(depth, ("linear",)):   # the task states a linear rescale
            va = rd * 5 + depth[bins] * 4
            mn, mx = np.nanmin(va), np.nanmax(va)
            if std == "linear":
                rv = 1 + 9 * (va - mn) / (mx - mn)
            else:
                rv = np.clip(np.floor((va - mn) / (mx - mn) * 10) + 1, 1, 10)
            out[(order, bins, std)] = undev & (rv * 8 + rl * 10 > 100)
    return out, tr, sh


def _pred_cells(path, tr, sh):
    """Row/col indices on the land_cover grid of the pred raster's selected cells."""
    import rasterio
    from pyproj import CRS, Transformer
    with rasterio.open(path) as r:
        a = r.read(1).astype(float)
        nod = r.nodata
        ptr = r.transform
        try:
            wkt = r.crs.to_wkt() if r.crs else None
        except Exception:
            wkt = None
    if wkt is None:
        # no CRS guessing for the deliverable file: the uniform rule
        # "deliverable files carry a CRS" fails the cell; its cells are not placed on the grid
        return np.zeros(sh, bool), np.array([]), "no CRS - not interpreted (uniform CRS rule)", 0
    sel = np.isfinite(a) & (a != 0)
    if nod is not None:
        sel &= a != nod
    rows, cols = np.nonzero(sel)
    vals = a[rows, cols]
    xs = ptr.c + (cols + 0.5) * ptr.a + (rows + 0.5) * ptr.b
    ys = ptr.f + (cols + 0.5) * ptr.d + (rows + 0.5) * ptr.e
    note = "pred CRS"
    if wkt and "LOCAL_CS" not in wkt:
        c = CRS.from_wkt(wkt)
        if not c.equals(CRS.from_user_input(LCC)) and not ("Oregon North" in c.name or "Oregon_North" in c.name):
            xs, ys = Transformer.from_crs(c, LCC, always_xy=True).transform(xs, ys)
            note = f"reprojected from {c.name}"
    else:
        note = "LOCAL_CS (as the input land_cover.tif) - taken as the land_cover CRS"
    inv = ~tr
    cc, rr = inv * (np.asarray(xs), np.asarray(ys))
    rr, cc = np.floor(rr).astype(int), np.floor(cc).astype(int)
    ok = (rr >= 0) & (rr < sh[0]) & (cc >= 0) & (cc < sh[1])
    m = np.zeros(sh, bool)
    m[rr[ok], cc[ok]] = True
    return m, vals, note, int((~ok).sum())


def _dilate(m):
    from scipy.ndimage import binary_dilation
    return binary_dilation(m, structure=np.ones((3, 3), bool))


def _is_lcc(crs_text):
    t = str(crs_text or "").lower()
    return any(k in t for k in ("lambert_conformal", "lambert conformal", "proj=lcc", "oregon north",
                                "epsg:32126", "epsg\",\"32126", "epsg:2269", "epsg:2913"))


def _projected_to_lcc(ctx, name):
    """The text prescribes "First, project 'mc_soils.shp' to Lambert Conformal Conic".
    Evidence: a captured soils layer (drclassdcd / wdepannmin columns) in an LCC CRS, or the
    deliverable raster (built on the soils rasterised onto the land_cover grid) in an LCC CRS.
    Working in the soils' native Albers and warping land_cover to it fails."""
    for tag, meta, load in ctx.objects("pred", kind="gdf"):
        cols = set(meta.get("columns") or [])
        crs = meta.get("crs")
        if meta.get("source") == "file" or crs is None:
            try:
                g = load()
                cols, crs = set(g.columns), (g.crs.to_wkt() if g.crs is not None else None)
            except Exception:
                continue
        if cols & {"drclassdcd", "wdepannmin"} and _is_lcc(crs):
            return True, f"soils layer {meta.get('name')}@{tag} in LCC"
    try:
        import rasterio
        with rasterio.open(ctx.pred(name)) as r:
            w = r.crs.to_wkt() if r.crs else ""
        if _is_lcc(w):
            return True, "deliverable raster in LCC (" + w[:60] + ")"
        return False, "soils not captured in LCC; deliverable CRS " + (w[:80] or "none")
    except Exception as e:
        return False, "no LCC evidence: " + repr(e)[:100]


def checks(ctx):
    out = []
    cands = [f for f in ctx.pred_files() if "undeveloped" in f.lower() and f.lower().endswith((".tif", ".tiff"))]
    name = "undeveloped_areas.tif" if "undeveloped_areas.tif" in cands else (cands[0] if cands else None)
    out.append(result("required output present", name is not None, value=cands,
                      threshold=["undeveloped_areas.tif", "undeveloped_areas.tiff"]))
    if name is None:
        out.append(result("georeferenced raster with selected cells", False))
        out.append(result("selected cells match an accepted interpretation (1-cell tolerance)", False))
        return out
    try:
        variants, tr, sh = _variants(ctx)
        pm, vals, note, outside = _pred_cells(ctx.pred(name), tr, sh)
    except Exception as e:
        out.append(result("georeferenced raster with selected cells", False, note=repr(e)[:300]))
        out.append(result("selected cells match an accepted interpretation (1-cell tolerance)", False))
        return out
    n = int(pm.sum())
    out.append(result("georeferenced raster with selected cells", n > 0, value=n, threshold=">0",
                      note=f"{note}; {outside} selected cells outside the land_cover grid"))
    best = (-1, None, 0, 0)
    pd_ = _dilate(pm)
    for k, g in variants.items():
        if g.sum() == 0:
            continue
        prec = float((pm & _dilate(g)).sum() / max(1, n))
        rec = float((g & pd_).sum() / g.sum())
        if min(prec, rec) > best[0]:
            best = (min(prec, rec), k, prec, rec)
    s, k, prec, rec = best
    out.append(result("selected cells match an accepted interpretation (1-cell tolerance)",
                      prec >= 0.85 and rec >= 0.85,
                      value={"precision": round(prec, 3), "recall": round(rec, 3), "cells": n,
                             "variant_cells": int(variants[k].sum()) if k else None},
                      threshold="precision>=0.85 and recall>=0.85", note=" / ".join(k) if k else ""))
    lcc, lcc_note = _projected_to_lcc(ctx, name)
    out.append(result("soils projected to Lambert Conformal Conic (prescribed step)", lcc, note=lcc_note))
    u = np.unique(vals)
    codes_ok = bool(np.isin(u, UNDEV).all() or set(u.tolist()) <= {1.0})
    out.append(result("selected cells carry undeveloped NLCD codes or a 0/1 mask", codes_ok,
                      value=u[:20].tolist(), required=False))
    return out
