"""Task 45 - terrain ruggedness (3x3 focal statistics) and rescaled ruggedness cost.

Task core: on the 30 m DEM, compute ruggedness with a 3x3 focal statistic of terrain
variation (the benchmark's model uses RANGE; std / TRI / slope are legitimate alternatives,
Spearman >= 0.96 with the range here), then rescale it to a standardised cost scale and save
the cost layer.

Required:
  1. ruggedness computed: a raster file or captured 2-D array on the DEM grid whose values
     rank-correlate with the gold 3x3 range (|Spearman| >= 0.95). 3x3 range/std/TRI/slope give
     0.97-1.0, a 7x7 window 0.93-0.94, 9x9 0.91-0.92 (5x5 gives 0.96-0.97 and cannot be told
     from slope). A focal MEAN (the ArcGIS default statistic) or elevation itself give 0.53
     and fail;
  2. a cost raster is saved, georeferenced on the DEM footprint (same CRS, bounds IoU >= 0.95);
  3. the saved cost is a standardised monotone transform of ruggedness: |Spearman| with gold
     ruggedness >= 0.85 (gold MSLarge 10->1: 0.92 because half the cells tie at 10; linear
     rescaling: 1.0) and all finite values within [0, 100] or z-scores ("rescaling to
     standardize" read statistically; raw ruggedness 0-223 m fails), with values on >= 90 %
     of valid DEM cells;
  3b. >= 90 % of the DEM NoData cells (4 %) are NoData/NaN in the cost raster.
Arrays/rasters from 'valid'-mode focal code (1-2 cells shorter on every side) are placed by a
centred pad or by the file's geotransform instead of being stretched onto the DEM shape (Horn
slope by slicing: 0.937 stretched vs 0.979 placed).
The instruction says "saved in the geodatabase": raster tables inside a GeoPackage are read
as candidate rasters (checks 1-3b).
Reported, not required:
  4. direction as in the benchmark's model: rugged terrain = low cost (mountain lions prefer
     rugged terrain; the instruction does not state the direction);
  5. MSLarge function used as in the benchmark's model (code mentions MSLarge / mean + std form).
"""
import glob
import os

import numpy as np

try:  # point PROJ at the data directory bundled with rasterio so that rasterio and pyproj agree
    import rasterio as _rio
    _pd = os.path.join(os.path.dirname(_rio.__file__), "proj_data")
    if os.path.isdir(_pd):
        os.environ["PROJ_DATA"] = _pd
except Exception:
    pass

from checklib import result, find_object, resample_to

RAS = ("*.tif", "*.tiff", "*.TIF", "*.img", "*.asc")


def _gold_arrays(ctx):
    rug, _, _ = find_object(ctx, "gold", "array", lambda o, m: m["name"] == "ruggedness")
    return rug


def _read_raster(path, label, out):
    import rasterio
    with rasterio.open(path) as r:
        a = r.read(1).astype(float)
        if r.nodata is not None:
            a[a == r.nodata] = np.nan
        a[np.abs(a) > 1e30] = np.nan
        crs = r.crs.to_epsg() if r.crs else None
        out.append((label, a, crs, tuple(r.bounds), r.transform))


def _geopackages(root):
    """Files under root that are GeoPackages by content, whatever their extension."""
    import sqlite3
    out = []
    for p in sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True)):
        try:
            if not os.path.isfile(p):
                continue
            with open(p, "rb") as f:
                if f.read(16) != b"SQLite format 3\x00":
                    continue
            con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
            try:
                ok = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='gpkg_contents'").fetchone()
            finally:
                con.close()
            if ok:
                out.append(p)
        except Exception:
            continue
    return out


def _rasters(ctx):
    import rasterio
    out = []
    for pat in RAS:
        for p in sorted(glob.glob(os.path.join(ctx.pred_dir, "**", pat), recursive=True)):
            try:
                _read_raster(p, os.path.basename(p), out)
            except Exception:
                continue
    # The instruction says "saved in the geodatabase"; a GeoPackage raster is the open
    # counterpart. Every raster table of a GeoPackage is read (vector-only GeoPackages fail to
    # open and are skipped); NoData is read back from the table like any other raster.
    # GeoPackages are recognised by content (SQLite header + a gpkg_contents table), not by
    # the .gpkg extension (valid GeoPackages may be saved as .geopackage / .gdb).
    for p in _geopackages(ctx.pred_dir):
        try:
            with rasterio.open(p) as r:
                subs = list(r.subdatasets or [])
        except Exception:
            continue
        if not subs:
            try:
                _read_raster(p, os.path.basename(p), out)
            except Exception:
                pass
        for sd in subs:
            try:
                _read_raster(sd, os.path.basename(p) + ":" + sd.rsplit(":", 1)[-1], out)
            except Exception:
                continue
    return out


def _placements(a, shape, T=None, dem_T=None):
    """Candidate placements of a 2-D array on the DEM grid: the array itself
    when shapes agree; otherwise the shape-stretch (resample_to), a centred pad when the array
    is a 'valid'-mode focal result (k cells shorter on every side, k <= 2), and - for files on
    the DEM resolution - the placement given by the file's own geotransform."""
    if a.shape == tuple(shape):
        return [a]
    out = [resample_to(a, shape)]
    dh, dw = shape[0] - a.shape[0], shape[1] - a.shape[1]
    if dh == dw and dh in (2, 4):
        k = dh // 2
        o = np.full(shape, np.nan)
        o[k:shape[0] - k, k:shape[1] - k] = a
        out.append(o)
    if T is not None and dem_T is not None and abs(T.a - dem_T.a) < 1e-6 * abs(dem_T.a) \
            and abs(T.e - dem_T.e) < 1e-6 * abs(dem_T.e):
        r0 = int(round((T.f - dem_T.f) / dem_T.e))
        c0 = int(round((T.c - dem_T.c) / dem_T.a))
        o = np.full(shape, np.nan)
        rs, cs = max(r0, 0), max(c0, 0)
        re_, ce = min(r0 + a.shape[0], shape[0]), min(c0 + a.shape[1], shape[1])
        if re_ > rs and ce > cs:
            o[rs:re_, cs:ce] = a[rs - r0:re_ - r0, cs - c0:ce - c0]
            out.append(o)
    return out


def _sp(a, b, idx):
    from scipy.stats import spearmanr
    x, y = a.ravel()[idx], b.ravel()[idx]
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 100 or np.nanstd(x[m]) == 0:
        return float("nan")
    return float(spearmanr(x[m], y[m]).correlation)


def _biou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix
    return ix / ua if ua > 0 else 0.0


def checks(ctx):
    out = []
    rug = _gold_arrays(ctx)
    if rug is None:
        return [result("gold ruggedness available", False, note="gold snapshot missing")]
    shape = rug.shape
    valid = np.isfinite(rug)
    idx = np.flatnonzero(valid.ravel())[::37]
    import rasterio
    with rasterio.open(os.path.join(ctx.inputs_dir, "Elevation.tif")) as r:
        dem_bounds, dem_epsg, dem_T = tuple(r.bounds), r.crs.to_epsg(), r.transform

    rasters = _rasters(ctx)
    # 1 ruggedness (file or captured array)
    best = (-1.0, None)
    cands = [(n, a, T) for n, a, _, _, T in rasters]
    for tag, meta, load in ctx.objects("pred", "array"):
        shp = meta.get("shape") or []
        if len(shp) == 2 and min(shp) >= 100 and abs(shp[0] / shp[1] - shape[0] / shape[1]) < 0.05:
            try:
                cands.append((meta["name"] + "@" + tag, load().astype(float), None))
            except Exception:
                pass
    for n, a, T in cands:
        for g in _placements(a, shape, T, dem_T):
            try:
                rho = _sp(g, rug, idx)
            except Exception:
                continue
            if np.isfinite(rho) and abs(rho) > best[0]:
                best = (abs(rho), n)
    out.append(result("ruggedness from 3x3 focal variation matches gold (|Spearman| >= 0.95)", best[0] >= 0.95,
                      value=round(best[0], 3), threshold=0.95, note=best[1] or "no raster/2-D array"))

    # 2/3 cost raster
    bestc = None  # (score, name, rho, lo, hi, cover, georef, std_ok, nod_kept)
    nodata_dem = ~valid
    for n, a, epsg, bnd, T in rasters:
        pl = [(abs(r_) if np.isfinite(r_) else -1, r_, g_) for g_ in _placements(a, shape, T, dem_T)
              for r_ in [_sp(g_, rug, idx)]]
        _, rho, g = max(pl, key=lambda t: t[0])
        if not np.isfinite(rho):
            continue
        fin = a[np.isfinite(a)]
        lo, hi = float(fin.min()), float(fin.max())
        cover = float(np.isfinite(g[valid]).mean())
        georef = (epsg == dem_epsg) and _biou(bnd, dem_bounds) >= 0.95
        # standardised: a bounded cost scale (0-100) or z-scores (mean 0, sd 1; "rescaling to
        # standardize" read statistically). Raw metres fail both.
        mu, sd = float(fin.mean()), float(fin.std())
        std_ok = (lo >= -1e-6 and hi <= 100 and hi > lo) or (sd > 0 and abs(mu) <= 0.05 * sd and abs(sd - 1) <= 0.05)
        # DEM NoData must stay NoData (NoData read as 0 m creates cost values and cliff
        # artefacts there)
        nod_kept = float((~np.isfinite(g[nodata_dem])).mean()) if nodata_dem.any() else 1.0
        score = abs(rho) + (1 if std_ok else 0) + (1 if georef else 0) + (0.5 if nod_kept >= 0.9 else 0)
        if bestc is None or score > bestc[0]:
            bestc = (score, n, rho, lo, hi, cover, georef, std_ok, nod_kept)
    if bestc is None:
        out.append(result("cost raster saved and georeferenced on the DEM grid", False, note="no raster file"))
        out.append(result("cost = standardised monotone transform of ruggedness", False))
        out.append(result("direction: rugged = low cost (as original)", False, required=False))
    else:
        _, n, rho, lo, hi, cover, georef, std_ok, nod_kept = bestc
        out.append(result("cost raster saved and georeferenced on the DEM grid", georef, value=n,
                          threshold=f"EPSG:{dem_epsg}, bounds IoU>=0.95"))
        ok = abs(rho) >= 0.85 and std_ok and cover >= 0.9
        out.append(result("cost = standardised monotone transform of ruggedness", ok,
                          value=[round(rho, 3), round(lo, 3), round(hi, 3), round(cover, 3)],
                          threshold="|rho|>=0.85, values in [0,100] or z-scores, cover>=0.9", note=n))
        out.append(result("DEM NoData kept as NoData in the cost raster (>= 90 % of NoData cells)", nod_kept >= 0.9,
                          value=round(nod_kept, 3), threshold=0.9, note=n))
        out.append(result("direction: rugged = low cost (as original)", rho < 0, value=round(rho, 3), required=False))
    code = ctx.code_text()
    out.append(result("method fidelity: MSLarge rescale", "mslarge" in code.lower() or "MSLarge" in code,
                      required=False))
    return out
