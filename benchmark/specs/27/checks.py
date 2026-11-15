"""T27 - Undeveloped land inside high-risk groundwater zones.

Task core: standardise vulnerable_areas (valid range 14-37) to 1-10, reclassify NLCD land cover
with the given table, risk_zones = vul*8 + lc*10, keep risk_zones > 100, and within those
cells keep the undeveloped NLCD classes (Barren, Deciduous/Mixed Forest, Shrub/Scrub,
Herbaceous, Hay/Pasture, Woody/Emergent Wetlands). The two inputs are on different grids and
CRS units (vulnerable: Oregon North ft, 98.4 ft cells; land cover: Oregon North m, 30 m,
larger extent), so they must be aligned geographically. Result: 665 cells (0.60 km2, mostly
Hay/Pasture) in the gold.

Correctness (required):
  1. undeveloped_areas.tif in pred_results (or output/undeveloped_areas.tif as the instruction
     literally says);
  2. its selected cells (valid and > 0), sampled at the gold cell centres via pyproj, overlap
     the gold selection with IoU >= 0.70 (the selection is tiny, so a half-cell grid offset
     costs a few %). Wrong min/max (NoData -32768 included), no grid alignment, skipping the
     undeveloped filter or the > 100 threshold all give IoU < 0.2.
  3. every land-cover class of the gold selection with >= 20 cells (Hay/Pasture, Barren) has
     recall >= 0.5 - catches a class missing from the undeveloped list.
Reported, not required:
  4. output values are the undeveloped land-cover codes (Con keeps the land-cover value);
  5. selected area within +/-25 % of gold.
"""
import os

import numpy as np

from checklib import result


def _fix_proj():
    """Point PROJ at the data directory bundled with rasterio so that rasterio and pyproj
    agree: with a proj.db layout too old for rasterio's GDAL, GeoTIFF CRSs are read as LOCAL_CS.
    The environment variable is set because checklib's set_proj_data_search_path alone does
    not fix the CRS identification of files."""
    import importlib.util
    spec = importlib.util.find_spec("rasterio")
    if spec and spec.origin:
        d = os.path.join(os.path.dirname(spec.origin), "proj_data")
        if os.path.exists(os.path.join(d, "proj.db")):
            os.environ["PROJ_DATA"] = d


_fix_proj()



def sample_on_gold(pred_path, gold_path, step=1):
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
        # no CRS guessing for deliverable files; the uniform CRS rule fails it
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


UNDEV = {31, 41, 43, 52, 71, 81, 90, 95}


def _find(ctx):
    for n in ("undeveloped_areas.tif", "undeveloped_areas.tiff", "Undeveloped_Areas.tif"):
        if os.path.exists(ctx.pred(n)):
            return ctx.pred(n)
    alt = os.path.join(os.path.dirname(os.path.abspath(ctx.pred_dir)), "_task_root", "output", "undeveloped_areas.tif")
    return alt if os.path.exists(alt) else None


def checks(ctx):
    out = []
    p = _find(ctx)
    out.append(result("undeveloped_areas.tif present", p is not None,
                      value=os.path.relpath(p, os.path.dirname(os.path.abspath(ctx.pred_dir))) if p else None))
    if p is None:
        out.append(result("selected cells match gold (IoU)", False, note="no output"))
        out.append(result("every undeveloped class of the gold selection is selected (recall >= 0.5)", False))
        out.append(result("values are undeveloped land-cover codes", False, required=False))
        out.append(result("selected area within 25 % of gold", False, required=False))
        return out
    try:
        pv, gv = sample_on_gold(p, ctx.gold("undeveloped_areas.tif"), step=1)
    except Exception as e:
        out.append(result("selected cells match gold (IoU)", False, note=repr(e)[:200]))
        out.append(result("every undeveloped class of the gold selection is selected (recall >= 0.5)", False))
        return out
    ps = np.isfinite(pv) & (pv > 0)
    gs = np.isfinite(gv) & (gv > 0)
    iou = float((ps & gs).sum() / max(1, (ps | gs).sum()))
    out.append(result("selected cells match gold (IoU)", iou >= 0.70, value=round(iou, 4), threshold=0.70,
                      note=f"pred {int(ps.sum())} cells, gold {int(gs.sum())} (on gold grid)"))
    # per-class recall: Hay/Pasture dominates the selection, so a
    # whole undeveloped class dropped from the filter (e.g. Barren, 62 of 665 cells) still
    # gives IoU ~0.9. Gold keeps the land-cover code, so for every gold class with >= 20
    # cells at least half of its cells must be selected (works for 0/1 masks too).
    rec = {}
    for c in np.unique(gv[gs]).astype(int):
        gc = gs & (gv == c)
        if gc.sum() >= 20:
            rec[int(c)] = round(float((ps & gc).sum() / gc.sum()), 3)
    out.append(result("every undeveloped class of the gold selection is selected (recall >= 0.5)",
                      bool(rec) and min(rec.values()) >= 0.5, value=rec, threshold=0.5,
                      note="per land-cover class with >= 20 gold cells"))
    vals = set(np.unique(pv[ps]).astype(int).tolist())
    out.append(result("values are undeveloped land-cover codes", bool(vals) and vals <= UNDEV,
                      value=sorted(vals)[:10], required=False, note="a 0/1 mask is acceptable but loses the class"))
    ratio = ps.sum() / max(1, gs.sum())
    out.append(result("selected area within 25 % of gold", 0.75 <= ratio <= 1.25, value=round(float(ratio), 3),
                      required=False))
    return out
