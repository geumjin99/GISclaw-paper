"""Task 22 - mountain lion corridor: composite cost surface from four criterion rasters.

Task core: map algebra on four co-registered 2494 x 3062 rasters (30 m, UTM 11N):
cost = 1*distance + 1.25*ruggedness + 1*protected_status + 1.25*landcover (optionally divided
by the weight total), visualised as mountainLionCorridor.png.

Correctness (required):
  1. the PNG exists and is not blank;
  2. a captured 2-D array is the composite: Pearson r >= 0.85 with the gold surface on cells
     valid in gold (any linear rescaling, division by 4.5, colour map is fine);
  3. the composite is the *weighted* combination of all four layers: a least-squares
     decomposition of that array on the four inputs has R^2 >= 0.98 and every relative
     coefficient (layer : distance) within +/-10 % of the given weight ratio, either on the
     raw scale or after the layer was range-normalised (min-max to 0-10 / 0-1, so the raw
     coefficient is w_i * 10 / range_i): ruggedness 1.25 or 1.25*10/36 (or exactly 1.25*9/36,
     min-max to 1-10, +/-1 %), protected 1.0 or 1.0*10/9, land cover 1.25.
     Equal weights, ruggedness weight 1.0, a dropped layer or a plotted input layer fail here;
  4. the map shows that surface: a drawn image / mesh (any resolution, same aspect) passes the
     same decomposition (R^2 >= 0.98, weights as in 3) against the inputs resampled to it.
     The drawn array is accepted as drawn or flipped N-S (which way up the map is drawn is
     presentation; transposes / E-W mirrors fail).
Reported, not required:
  4a. (kind="figure") the map is drawn north-up: check 4 without the flip;
  4b. map readable: landcover-255 NoData summed as cost does not swamp the colour scale -
      kind="figure" (readability is presentation, not the main metric);
  5. ruggedness weight exactly 1.25 x distance as given (raw scale) vs. rescaled;
  6. landcover code 255 (ocean, NoData without a NoData flag) excluded from the surface.
"""
import os

import numpy as np

from checklib import file_exists, png_not_blank, result, resample_to

NAMES = ["mountainLionCorridor.png", "mountainlioncorridor.png", "mountainLionCorridor.PNG"]
LAYERS = ["distance", "ruggedness", "protected_status_reclassified", "landcover_reclassified"]
SHAPE = (2494, 3062)
STEP = 3  # subsample for speed


def _inputs(ctx):
    import rasterio
    out = []
    for n in LAYERS:
        with rasterio.open(os.path.join(ctx.inputs_dir, n + ".tif")) as s:
            out.append(s.read(1)[::STEP, ::STEP].astype(float))
    return out


RANGES = np.array([10.0, 36.0, 9.0, 10.0])   # value ranges of the four inputs (lc 255 masked)
W = np.array([1.0, 1.25, 1.0, 1.25])


def _weights_ok(ratios):
    """ratios = coefficient_i / coefficient_distance. Each layer may be on its raw scale or
    range-normalised like distance (factor 10/range_i)."""
    ok = []
    for i in range(1, 4):
        allowed = {W[i], W[i] * 10.0 / RANGES[i]}
        good = any(abs(ratios[i] - v) <= 0.10 * v for v in allowed)
        # Ruggedness min-max rescaled to 1-10 (the scale of protected/land cover) gives exactly
        # w*9/36 = 0.3125, which sits on the lower +/-10 % edge of 0.347, where float noise
        # decides. That exact target-span form is accepted (+/-1 %, room for drawn-array
        # decimation only). Sign is unchanged: inverted ruggedness (< 0) fails.
        if i == 1:
            v9 = W[i] * 9.0 / RANGES[i]
            good = good or abs(ratios[i] - v9) <= 0.01 * v9
        ok.append(good)
    return all(ok)


def _arrays(ctx, side, drawn=False):
    objs = ctx.objects(side, "array", include_files=not drawn)
    for tag, meta, load in objs:
        if drawn and (meta.get("source") != "figure" or meta.get("artist") not in ("image", "QuadMesh")):
            continue
        shp = meta.get("shape", [])
        if len(shp) != 2 or min(shp) < 100:
            continue
        # same aspect ratio as the study grid (allow other resolutions)
        if abs(shp[0] / shp[1] - SHAPE[0] / SHAPE[1]) > 0.02:
            continue
        try:
            a = np.asarray(load(), float)
        except Exception:
            continue
        if drawn and meta.get("origin") == "lower":
            a = a[::-1]
        if a.shape != SHAPE:
            a = resample_to(a, SHAPE)
        yield meta["name"] + "@" + tag, a[::STEP, ::STEP]


def _png_spread(path):
    """Spread (5th-95th percentile) of colormap positions of the coloured pixels of a PNG,
    after inverting the colours along the best-fitting registered colormap.
    Returns (spread, cmap, n_pixels) or (None, note, 0) when undecidable."""
    import matplotlib
    from PIL import Image
    a = np.asarray(Image.open(path).convert("RGB"), float) / 255.0
    st = max(1, int(np.sqrt(a.shape[0] * a.shape[1] / 3e5)))
    px = a[::st, ::st].reshape(-1, 3)
    px = px[(px.max(1) - px.min(1)) > 0.08]          # drop grey background/text/axes
    if len(px) < 1000:
        return None, "too few coloured pixels (grey colormap?)", len(px)
    uniq, inv, cnt = np.unique(np.round(px, 2), axis=0, return_inverse=True, return_counts=True)
    best = (None, None, 9.0, None)
    for name in matplotlib.colormaps():
        if name.endswith("_r"):
            continue
        lut = matplotlib.colormaps[name](np.linspace(0, 1, 256))[:, :3]
        dd = ((uniq[:, None, :] - lut[None]) ** 2).sum(-1)
        idx = dd.argmin(1)
        res = np.sqrt(dd[np.arange(len(uniq)), idx])
        r = float(np.average(res, weights=cnt))
        if r < best[2]:
            best = (idx / 255.0, name, r, res)
    pos, name, r, res = best
    if pos is None or r > 0.03:
        return None, "no registered colormap fits the map colours", len(px)
    inv = np.ravel(inv)
    keep = res[inv] < 0.05
    p = pos[inv][keep]
    if len(p) < 1000:
        return None, "too few pixels on the colormap", len(p)
    lo, hi = np.percentile(p, [5, 95])
    return float(hi - lo), name, int(len(p))


def _readable(ctx, best_img, best_d):
    """Landcover code 255 (ocean, NoData without a flag) summed as cost 255 gives the sea ~4x
    the land maximum; with the default colour stretch the whole land surface collapses into
    one colour and the map shows nothing.
    Readability of the map is presentation -> kind="figure" (never required; secondary
    analysis). Passes when either the drawn surface has no such outliers (255 masked, recoded, clipped),
    or the saved PNG still spreads the land over the colour scale (vmax set by the analyst)."""
    name = "map readable: NoData (landcover 255) does not swamp the colour scale"
    if best_img is None or not (best_d and best_d[0]):
        return result(name, False, note="no drawn composite surface", kind="figure")
    d, lc = best_img
    sea = lc == 255
    land = np.isfinite(d) & ~sea & (lc != 0)
    if not sea.any() or not land.any():
        return result(name, True, note="no landcover-255 cells on the drawn grid", kind="figure")
    sv = d[sea]
    lmax, lmin = float(np.nanmax(d[land])), float(np.nanmin(d[land]))
    if np.mean(~np.isfinite(sv)) > 0.9 or np.nanmax(sv) <= lmax + 0.01 * abs(lmax):
        return result(name, True, value={"sea_masked_or_in_range": True}, kind="figure")
    # sea drawn as extreme cost: decide from the picture whether the land is still readable
    pngs = [n for n in NAMES if os.path.exists(ctx.pred(n))]
    if not pngs:
        return result(name, False, note="no map", kind="figure")
    try:
        spread, cm, n = _png_spread(ctx.pred(pngs[0]))
    except Exception as e:
        spread, cm, n = None, repr(e)[:100], 0
    if spread is None:
        return result(name, True, note="sea drawn with extreme values; PNG colours undecidable: " + str(cm), kind="figure")
    return result(name, spread >= 0.15, value=round(spread, 3), threshold=">= 0.15 colormap span (5-95 %)",
                  note=f"sea drawn up to {np.nanmax(sv):.1f} vs land max {lmax:.1f}; cmap {cm}, {n} px", kind="figure")


def _gold(ctx):
    for name, a in _arrays(ctx, "gold"):
        if name.startswith("cost_surface@"):
            return a
    return None


def checks(ctx):
    out = []
    r = file_exists(ctx, NAMES)
    out.append(r)
    if r["passed"]:
        nb, std = png_not_blank(ctx.pred(r["value"][0]))
        out.append(result("map not blank", nb, value=std, threshold=">=3 grey-level std"))
    else:
        out.append(result("map not blank", False, note="no map"))

    try:
        gold = _gold(ctx)
        X = _inputs(ctx)
    except Exception as e:
        gold, X = None, None
        note = repr(e)[:200]
    if gold is None or X is None:
        out.append(result("composite surface agrees with gold (Pearson)", False, note="gold/inputs missing"))
        out.append(result("weighted combination of all four layers with the given weights", False))
        out.append(result("map shows the weighted composite surface", False))
        return out
    gv = np.isfinite(gold)

    best = (None, -2.0, None)
    seen = set()
    for name, a in _arrays(ctx, "pred"):
        key = (a.shape, float(np.nansum(a[::7, ::7])))
        if key in seen:
            continue
        seen.add(key)
        m = gv & np.isfinite(a)
        if m.sum() < 0.8 * gv.sum() or a[m].std() == 0:
            continue
        rr = float(np.corrcoef(a[m], gold[m])[0, 1])
        if rr > best[1]:
            best = (name, rr, a)
    name, rr, a = best
    out.append(result("composite surface agrees with gold (Pearson)", rr >= 0.85,
                      value=round(rr, 4) if name else None, threshold=0.85,
                      note=name or "no 2-D array on the study grid covering >=80 % of valid cells"))
    if name is None:
        out.append(result("weighted combination of all four layers with the given weights", False, note="no surface"))
        out.append(result("map shows the weighted composite surface", False, note="no surface"))
        out.append(result("map drawn north-up (composite as drawn)", False, note="no surface", kind="figure"))
        out.append(result("map readable: NoData (landcover 255) does not swamp the colour scale", False, note="no surface", kind="figure"))
        out.append(result("ruggedness weight exactly 1.25 (no rescaling)", False, required=False))
        out.append(result("landcover 255 (NoData) excluded", False, required=False))
        return out

    m = gv & np.isfinite(a)
    A = np.column_stack([np.ones(m.sum())] + [x[m] for x in X])
    coef, *_ = np.linalg.lstsq(A, a[m], rcond=None)
    pred = A @ coef
    r2 = 1 - np.sum((a[m] - pred) ** 2) / np.sum((a[m] - a[m].mean()) ** 2)
    b = coef[1:]
    if abs(b[0]) > 1e-6 * (np.abs(b).max() + 1e-12):
        ratios = b / b[0]
    else:
        ratios = np.array([np.nan] * 4)
    ok = bool(r2 >= 0.98 and np.all(b > 0) and np.all(np.isfinite(ratios)) and _weights_ok(ratios))
    out.append(result("weighted combination of all four layers with the given weights", bool(ok),
                      value={"r2": round(float(r2), 4), "w_dist": 1.0, "w_rug": round(float(ratios[1]), 3),
                             "w_prot": round(float(ratios[2]), 3), "w_lc": round(float(ratios[3]), 3)},
                      threshold="R2>=0.98; relative to distance, +/-10 %: rug 1.25 or 0.347 (or 0.3125 +/-1 %, 1-10 rescale), prot 1.0 or 1.111, lc 1.25"))
    # the map must show the weighted surface: decompose each drawn image / mesh (at its own
    # resolution, inputs resampled to it the same way) and require the same weight structure
    best_d, best_img, best_up = None, None, None
    try:
        import rasterio
        full = []
        for n in LAYERS:
            with rasterio.open(os.path.join(ctx.inputs_dir, n + ".tif")) as s_:
                full.append(s_.read(1).astype(float))
        for tag, meta, load in ctx.objects("pred", "array", include_files=False):
            if meta.get("source") != "figure" or meta.get("artist") not in ("image", "QuadMesh"):
                continue
            shp = meta.get("shape", [])
            if len(shp) != 2 or min(shp) < 50 or abs(shp[0] / shp[1] - SHAPE[0] / SHAPE[1]) > 0.02:
                continue
            d0 = np.asarray(load(), float)
            if meta.get("origin") == "lower":
                d0 = d0[::-1]
            # Which way up the map is drawn is presentation. The required
            # check accepts the drawn array as drawn or flipped N-S (never transposed / mirrored
            # E-W); the orientation as drawn is the kind="figure" row below.
            for flipped, d in ((False, d0), (True, d0[::-1])):
                st = max(1, int(np.sqrt(d.size / 4e5)))
                dd = d[::st, ::st]
                # inputs brought to the drawn grid: integer-stride decimation (a[::k, ::k]) when
                # the shape fits, else nearest-neighbour resampling
                variants = [[resample_to(x, d.shape) for x in full]]
                k = int(round(SHAPE[0] / d.shape[0]))
                if k > 1 and -(-SHAPE[0] // k) == d.shape[0] and -(-SHAPE[1] // k) == d.shape[1]:
                    variants.insert(0, [x[::k, ::k] for x in full])
                for V in variants:
                    Xd = [x[::st, ::st] for x in V]
                    md = np.isfinite(dd) & (Xd[3] != 255)
                    if md.sum() < 1000 or dd[md].std() == 0:
                        continue
                    Ad = np.column_stack([np.ones(md.sum())] + [x[md] for x in Xd])
                    cd, *_ = np.linalg.lstsq(Ad, dd[md], rcond=None)
                    r2d = 1 - np.sum((dd[md] - Ad @ cd) ** 2) / np.sum((dd[md] - dd[md].mean()) ** 2)
                    bd = cd[1:]
                    rd = bd / bd[0] if abs(bd[0]) > 1e-9 else np.full(4, np.nan)
                    okd = bool(r2d >= 0.98 and np.all(bd > 0) and np.all(np.isfinite(rd)) and _weights_ok(rd))
                    lab = meta["name"] + "@" + tag + (" (flipped N-S)" if flipped else "")
                    if best_d is None or (okd, r2d) > (best_d[0], best_d[1]):
                        best_d = (okd, float(r2d), np.round(rd, 3).tolist(), lab)
                        best_img = (d, V[3])
                    if not flipped and (best_up is None or (okd, r2d) > (best_up[0], best_up[1])):
                        best_up = (okd, float(r2d), np.round(rd, 3).tolist(), lab)
    except Exception as e:
        best_d = best_d or (False, float("nan"), None, repr(e)[:120])
    out.append(result("map shows the weighted composite surface", bool(best_d and best_d[0]),
                      value=best_d[1:3] if best_d else None,
                      threshold="drawn array (as drawn or flipped N-S): R2>=0.98, weights as check 3",
                      note=best_d[3] if best_d else "no drawn image/mesh on the study grid"))
    out.append(result("map drawn north-up (composite as drawn)", bool(best_up and best_up[0]),
                      value=best_up[1:3] if best_up else None, kind="figure",
                      note=best_up[3] if best_up else "no drawn image/mesh on the study grid"))
    out.append(_readable(ctx, best_img, best_d))
    out.append(result("ruggedness weight exactly 1.25 (no rescaling)", abs(ratios[1] - 1.25) <= 0.125,
                      value=round(float(ratios[1]), 3), threshold="1.25+/-0.125", required=False,
                      note="~0.35 means ruggedness was min-max rescaled to 0-10 first"))
    lc = X[3]
    sea = lc == 255
    vals = a[sea]
    excluded = bool(sea.any() and (np.mean(~np.isfinite(vals)) > 0.9 or np.nanmax(np.abs(vals)) <= np.nanmax(np.abs(a[gv])) * 1.01))
    out.append(result("landcover 255 (NoData) excluded", excluded, required=False,
                      note="255 cells are ocean in the SW corner; adding 1.25*255 gives spurious extreme cost"))
    return out
