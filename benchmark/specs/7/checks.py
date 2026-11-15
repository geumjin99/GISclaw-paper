"""T07 - Land subsidence and flooding: flood depth and building damage (2050).

The task text gives no water level; the original solution floods land below -200 cm NAP
(flood depth = -200 - elevation, cm). Cells equal to 0 are outside the study area.

The checks therefore test *internal consistency* for whatever water level W the agent chose,
and report agreement with the original -200 cm separately:
Correctness (required):
  1. flooding_analysis.png exists and is not blank;
  2. a flood-depth surface: a captured array on the elevation grid (878 x 1196) whose positive
     cells (either sign, cm or m) equal W - elevation (>= 95 % within 1 cm) for one W, and whose
     flooded set equals {study-area cells with elevation < W} (IoU >= 0.95); the unmodified
     elevation array (+-) does not count. Using raw elevation as depth, inconsistent units or
     counting the outside zeros fails;
  3. building damage: a captured building-level table carries a damage column that, aligned with
     the input buildings (geometry or row order), rank-correlates >= 0.8 with the damage
     recomputed here from the agent's W (mean depth of flooded cells touching each footprint,
     cost = (0.298 ln(0.01 d) + 1.4502) * 271 * Shape_Area) - compared per m2 of building, so that
     the depth-dependent part is tested, not just building size - over buildings damaged in both, and
     whose total over those buildings is within 0.75-1.33 x the recomputed total. Mis-aligned
     joins, forgetting the cm->m factor, or using perimeter instead of area fail.
  4. the damage function is the given one with the natural logarithm (the original solution's
     math.log): when a building table holds a depth column and a damage column, the damage must
     equal (0.298 ln(k d) + 1.4502) * 271 * area (k = 0.01 for cm, 1 for m) to 1 %; an exact
     log10 fit fails. Without a depth column the damage check (3) decides.
  Window-cropped depth arrays are located on the elevation grid (FFT search for the offset where
  depth + elevation is constant) and must cover >= 95 % of the buildings; IoU is then measured
  on the window.
Reported, not required:
  5. W equals the original -200 cm (+-1 cm).
"""
import numpy as np

from checklib import file_exists, png_not_blank, result

GOLD_W = -200.0


def _inputs(ctx):
    import geopandas as gpd
    import rasterio
    with rasterio.open(f"{ctx.inputs_dir}/Elevation_2050.tif") as s:
        elev, tr = s.read(1).astype(float), s.transform
    b = gpd.read_file(f"{ctx.inputs_dir}/StudyAreaBuildings.shp")
    if b.crs is None:
        b = b.set_crs(28992)
    return elev, tr, b


def _depth_fit(a, elev):
    """Best (score, W, scale, sign, iou, share_ok) for array a as flood depth."""
    valid = elev != 0
    # the input elevation itself (or its negative) is not a flood-depth result
    if np.allclose(np.abs(a[valid]), np.abs(elev[valid]), equal_nan=True):
        return None
    best = None
    for sign in (1, -1):
        for scale in (1.0, 100.0):
            d = sign * a * scale
            fl = np.isfinite(d) & (d > 0)
            if fl.sum() < 1000:
                continue
            w_all = d[fl] + elev[fl]
            W = float(np.median(w_all))
            share = float(np.mean(np.abs(w_all - W) <= 1.0))
            ref = valid & (elev < W)
            iou = float((fl & ref).sum() / max(1, (fl | ref).sum()))
            sc = min(share / 0.95, iou / 0.95)
            if best is None or sc > best[0]:
                best = (sc, W, scale, sign, iou, share)
    return best


def _place(a, elev):
    """Offset (r0, c0) of a smaller 2-D array `a` (a cropped window of a depth surface) on the
    elevation grid: the offset where sign*a*scale + elevation is most nearly constant over a's
    positive cells (FFT cross-correlation). Returns elev window or None."""
    from scipy.signal import fftconvolve
    H, W = elev.shape
    h, w = a.shape
    if h > H or w > W or h * w < 0.05 * H * W or (h, w) == (H, W):
        return None
    best = None
    E = np.where(elev != 0, elev, np.nan)
    E0 = np.nan_to_num(E)
    for sign in (1, -1):
        for scale in (1.0, 100.0):
            d = sign * a * scale
            m = (np.isfinite(d) & (d > 0)).astype(float)
            n = m.sum()
            if n < 1000:
                continue
            dm = np.where(m > 0, d, 0.0)
            k = lambda x: x[::-1, ::-1]
            s1 = fftconvolve(E0, k(m), mode="valid")          # sum m*E per offset
            s2 = fftconvolve(E0 ** 2, k(m), mode="valid")
            s3 = fftconvolve(E0, k(dm), mode="valid")         # sum d*m*E
            sd, sd2 = dm.sum(), (dm ** 2).sum()
            tot1 = (sd + s1) / n
            tot2 = (sd2 + 2 * s3 + s2) / n
            var = tot2 - tot1 ** 2
            i = np.unravel_index(np.argmin(var), var.shape)
            if best is None or var[i] < best[0]:
                best = (var[i], i)
    if best is None:
        return None
    r0, c0 = best[1]
    return elev[r0:r0 + h, c0:c0 + w], (r0, c0)


def _formula_fit(obj):
    """Is a damage column = (0.298*log(k*d) + 1.4502) * 271 * area for a depth column d of
    the same table? Returns ('ln'|'log10', depth col, damage col) for the best exact fit."""
    area = None
    for c in obj.columns:
        if c.lower() == "shape_area":
            area = obj[c].astype(float).to_numpy()
    if area is None and "geometry" in obj.columns and getattr(obj, "crs", None) is not None:
        try:
            area = obj.geometry.area.to_numpy()
        except Exception:
            pass
    if area is None:
        return None
    num = [c for c in obj.columns if c != "geometry" and obj[c].dtype.kind in "fi"
           and c.lower() not in ("shape_area", "shape_leng", "index", "index_right")]
    for dmg in num:
        y = obj[dmg].astype(float).to_numpy() / (271.0 * area)
        for dep in num:
            if dep == dmg:
                continue
            d = obj[dep].astype(float).to_numpy()
            m = np.isfinite(y) & np.isfinite(d) & (d > 0) & (y > 0)
            if m.sum() < 30:
                continue
            for name, f in (("ln", np.log), ("log10", np.log10)):
                for k in (0.01, 1.0):
                    with np.errstate(all="ignore"):
                        pred = 0.298 * f(k * d[m]) + 1.4502
                    ok = pred > 0
                    if ok.sum() < 30:
                        continue
                    err = np.median(np.abs(y[m][ok] - pred[ok]) / np.abs(pred[ok]))
                    if err < 0.01:
                        return name, dep, dmg
    return None


def _expected_damage(b, elev, tr, W):
    from rasterstats import zonal_stats
    fl = (elev != 0) & (elev < W)
    dep = np.where(fl, W - elev, -9999.0)
    zs = zonal_stats(b, dep, affine=tr, nodata=-9999.0, stats=["mean"], all_touched=True)
    md = np.array([z["mean"] if z["mean"] is not None else 0.0 for z in zs])
    with np.errstate(divide="ignore", invalid="ignore"):
        c = (0.298 * np.log(0.01 * md) + 1.4502) * 271 * b["Shape_Area"].to_numpy()
    return np.where(md > 0, np.clip(c, 0, None), 0.0)


def _align(obj, b):
    """Input-building index for every row of obj (or None)."""
    import geopandas as gpd
    if "geometry" in obj.columns and getattr(obj, "crs", None) is not None:
        try:
            g = gpd.GeoDataFrame(obj[["geometry"]].copy(), geometry="geometry", crs=obj.crs).to_crs(b.crs)
            g["geometry"] = g.geometry.representative_point()
            j = gpd.sjoin(g, b[["geometry"]], how="left", predicate="within")
            j = j[~j.index.duplicated()]
            idx = j["index_right"].reindex(g.index)
            if idx.notna().mean() > 0.8:
                return idx.to_numpy()
        except Exception:
            pass
    if len(obj) == len(b):
        return np.arange(len(b), dtype=float)
    return None


def checks(ctx):
    from scipy.stats import spearmanr
    out = []
    names = ["flooding_analysis.png"]
    r = file_exists(ctx, names)
    out.append(r)
    if r["passed"]:
        nb, std = png_not_blank(ctx.pred(names[0]))
        out.append(result("map not blank", nb, value=std, threshold=">=3 grey-level std"))
    else:
        out.append(result("map not blank", False, note="no map"))
    try:
        elev, tr, b = _inputs(ctx)
    except Exception as e:
        out.append(result("flood depth = W - elevation on the study area", False, note=repr(e)[:200]))
        out.append(result("building damage consistent with depth and damage function", False))
        return out

    best, where = None, None
    for tag, meta, load in ctx.objects("pred", "array"):
        sh = list(meta.get("shape", []))
        if len(sh) != 2:
            continue
        try:
            a = np.asarray(load(), float)
            if sh == list(elev.shape):
                f = _depth_fit(a, elev)
            else:   # cropped window of the grid: locate it, then test on the window
                pl = _place(a, elev)
                f = _depth_fit(a, pl[0]) if pl is not None else None
                if f is not None:
                    # a crop is legitimate if the window still covers the buildings (>= 95 % of
                    # building centroids inside it); the IoU is then measured on the window
                    (r0, c0), (h, w) = pl[1], a.shape
                    cx, cy = b.geometry.centroid.x.to_numpy(), b.geometry.centroid.y.to_numpy()
                    cc, rr = (~tr) * (cx, cy)
                    inside = float(np.mean((rr >= r0) & (rr < r0 + h) & (cc >= c0) & (cc < c0 + w)))
                    if inside < 0.95:
                        f = (min(f[0], 0.0),) + f[1:4] + (min(f[4], inside), f[5])
                    tag = tag + f" (window at {r0},{c0}; {inside:.0%} of buildings inside)"
        except Exception:
            f = None
        if f and (best is None or f[0] > best[0]):
            best, where = f, f"{meta['name']}@{tag}"
    ok = best is not None and best[4] >= 0.95 and best[5] >= 0.95
    out.append(result("flood depth = W - elevation on the study area", ok,
                      value=None if best is None else {"W_cm": round(best[1], 1), "iou": round(best[4], 3),
                                                       "share_consistent": round(best[5], 3),
                                                       "units": "m" if best[2] == 100 else "cm"},
                      threshold="share >= 0.95 and IoU >= 0.95", note=where or "no array on the elevation grid"))
    W = best[1] if ok else GOLD_W

    # damage
    try:
        exp = _expected_damage(b, elev, tr, W)
    except Exception as e:
        exp = None
        out.append(result("building damage consistent with depth and damage function", False, note=repr(e)[:200]))
    if exp is not None:
        area = b["Shape_Area"].to_numpy().astype(float)
        bestd = (-1.0, None, None)
        seen = set()
        for kind in ("gdf", "df"):
            for tag, meta, load in ctx.objects("pred", kind):
                n = meta.get("rows", 0)
                sig = (meta.get("name"), n, tuple(meta.get("columns", [])))
                if sig in seen or n < 50 or n > 2 * len(b):
                    continue
                seen.add(sig)
                try:
                    obj = load()
                    idx = _align(obj, b)
                    if idx is None:
                        continue
                    for c in obj.columns:
                        if c in ("geometry", "Shape_Area", "Shape_Leng", "index", "index_right") or obj[c].dtype.kind not in "fi":
                            continue
                        v = obj[c].astype(float).to_numpy()
                        m = np.isfinite(idx) & np.isfinite(v)
                        s = (np.bincount(idx[m].astype(int), weights=v[m], minlength=len(b)),
                             np.bincount(idx[m].astype(int), minlength=len(b)))
                        has = s[1] > 0
                        pv = np.where(has, s[0] / np.maximum(s[1], 1), np.nan)
                        both = has & (pv > 0) & (exp > 0)
                        if both.sum() < 50:
                            continue
                        # rank-compare damage per m2 (the depth-dependent factor): raw damage is
                        # dominated by building area and would hide wrong depths
                        rho = float(spearmanr(pv[both] / area[both], exp[both] / area[both]).correlation)
                        ratio = float(pv[both].sum() / exp[both].sum())
                        score = rho if 0.75 <= ratio <= 1.33 else min(rho, 0.0)
                        if score > bestd[0]:
                            bestd = (score, {"spearman": round(rho, 3), "total_ratio": round(ratio, 3),
                                             "buildings": int(both.sum())}, f"{c} in {meta['name']}@{tag}")
                except Exception:
                    continue
        out.append(result("building damage consistent with depth and damage function", bestd[0] >= 0.8,
                          value=bestd[1], threshold="spearman >= 0.8 and total ratio 0.75-1.33",
                          note=bestd[2] or "no building table with a damage column"))
    # damage function form: natural log as in the original solution (math.log); log10 is wrong
    fits = []
    seen = set()
    for kind in ("gdf", "df"):
        for tag, meta, load in ctx.objects("pred", kind):
            n = meta.get("rows", 0)
            sig = (meta.get("name"), n, tuple(meta.get("columns", [])))
            if sig in seen or n < 50 or n > 2 * len(b):
                continue
            seen.add(sig)
            try:
                f = _formula_fit(load())
            except Exception:
                f = None
            if f:
                fits.append((f, f"{meta['name']}@{tag}"))
    ln = [x for x in fits if x[0][0] == "ln"]
    lg = [x for x in fits if x[0][0] == "log10"]
    if ln:
        out.append(result("damage function uses the natural log (as in the original solution)", True,
                          value="ln", note=f"{ln[0][0][2]} = f({ln[0][0][1]}) in {ln[0][1]}"))
    elif lg:
        out.append(result("damage function uses the natural log (as in the original solution)", False,
                          value="log10", note=f"{lg[0][0][2]} = f_log10({lg[0][0][1]}) in {lg[0][1]}"))
    else:
        dmg_ok = bool(out[-1]["passed"]) if out[-1]["name"].startswith("building damage") else False
        out.append(result("damage function uses the natural log (as in the original solution)", dmg_ok,
                          note="no depth column next to the damage column; decided by the damage check"))
    out.append(result("water level equals the original -200 cm NAP", ok and abs(W - GOLD_W) <= 1, value=round(W, 1),
                      threshold=GOLD_W, required=False))
    return out
