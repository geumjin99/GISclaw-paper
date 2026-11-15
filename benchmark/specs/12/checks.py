"""Task 12 - Terrain ruggedness (TRI) from Elevation.tif, visualised as ruggedness.png.

Required:
  1. ruggedness.png exists and is not blank;
  2. a TRI-family ruggedness surface was computed: some captured 2-D array with the DEM's
     aspect ratio rank-correlates with the TRI at Spearman >= 0.95. The yardstick is the gold
     TRI (compared on a 1/4-resolution grid) or, for arrays at a coarser resolution (the DEM
     was decimated/aggregated), the TRI recomputed from the gold elevation decimated to
     that grid - so a correct TRI on a downsampled DEM is not penalised (2x decimated
     generic_filter TRI: 1.00 resolution-matched, only 0.898 against full-res gold).
     Neighbour-difference measures pass (Riley sum form 1.00, mean |diff| 0.99, local std
     0.98, local range 0.965, block-mean x2 TRI 0.97); plain slope (0.933 full res, 0.908 x2)
     is a different variable and fails, as do the raw elevation (0.26) and a uint8-overflow TRI
     (0.66).
  3. the figure shows the ruggedness surface (>= 0.95 by the same yardstick), judged on the
     data actually drawn, in EITHER vertical orientation
     (which way up the map is drawn is presentation; transposes/left-right mirrors still fail).
     The orientation as drawn is the kind="figure" check 5, determined as follows:
       * imshow arrays as stored (flipped when origin='lower');
       * pcolormesh in index space (axes limits ~ 0..ncols / 0..nrows): row 0 is at the
         bottom unless the y-axis is inverted, so a north-up map must have been drawn flipped;
       * pcolormesh in coordinate space: the y direction of the mesh is not captured, so both
         vertical orientations are accepted (never transposes/rotations);
       * contour/contourf: the drawn Z array is identified from the plotting call in the
         code and must be a captured TRI-like array whose range the contour levels span.
     A figure with none of these, or whose drawn variable cannot be identified, is
     UNVERIFIABLE (not passed).
  Integer-valued surfaces (5..64 levels; TRI truncated by a uint8 output array) are compared
  with the yardstick quantised into matching quantile bins (the exact TRI stored as uint8
  scores only 0.942 against the unquantised yardstick).
Reported, not required:
  4. TRI as defined (RMS neighbour difference): Spearman >= 0.99 (resolution-matched);
  5. (kind="figure", secondary analysis) the surface is drawn north-up: check 3 in the
     orientation drawn.
"""
import re

import numpy as np

from checklib import drawn_arrays, file_exists, png_not_blank, resample_to, result, unverifiable

NAME = "ruggedness.png"
FULL = (2494, 3062)
COARSE = (624, 766)  # 1/4 of the DEM grid
THR = 0.95


def _dem_shaped(sh):
    sh = list(sh or [])
    if len(sh) == 3 and 1 <= sh[0] <= 4:
        sh = sh[1:]
    if len(sh) != 2 or min(sh) < 100:
        return False
    return abs(sh[1] / sh[0] - FULL[1] / FULL[0]) <= 0.05 * FULL[1] / FULL[0]


def _as2d(a):
    a = np.asarray(a, dtype=float)
    return a[0] if a.ndim == 3 else a


def _tri(z):
    """NaN-aware RMS neighbour-difference TRI (as the reference)."""
    H, W = z.shape
    p = np.pad(z, 1, constant_values=np.nan)
    s, n = np.zeros((H, W)), np.zeros((H, W))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                d2 = (p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W] - z) ** 2
                ok = np.isfinite(d2)
                s[ok] += d2[ok]
                n += ok
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.sqrt(s / n)
    t[~np.isfinite(z) | (n == 0)] = np.nan
    return t


def _rho(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 1000 or np.nanstd(x[m]) == 0:
        return -1.0
    from scipy.stats import spearmanr
    return float(spearmanr(x[m], y[m]).correlation)


def _rho_q(a, ref):
    """Rank agreement allowing for integer quantisation of the candidate.

    An integer-valued candidate (5..64 distinct levels, e.g. a TRI written into the uint8 output
    of generic_filter on the 8-bit DEM, which truncates values whose median is ~1.3) is compared
    with the yardstick quantised into the same number of quantile bins (bin sizes = the
    candidate's level frequencies). A monotone quantisation of the TRI then scores ~1.0 instead
    of 0.94; the threshold is unchanged, so quantised slope (0.91-0.93) still fails."""
    v = a[np.isfinite(a)]
    if v.size < 1000 or np.any(v != np.round(v)):
        return -1.0
    m = np.isfinite(a) & np.isfinite(ref)
    lv, cnt = np.unique(a[m], return_counts=True)
    if not 5 <= len(lv) <= 64:
        return -1.0
    th = np.quantile(ref[m], np.cumsum(cnt)[:-1] / cnt.sum())
    q = np.full(ref.shape, np.nan)
    q[m] = np.searchsorted(th, ref[m], side="right")
    return _rho(a, q)


def _cmp(a, ref):
    return max(_rho(a, ref), _rho_q(a, ref))


class _Yard:
    """TRI yardstick at any resolution: full-res gold, or TRI of the decimated gold DEM."""

    def __init__(self, gold_tri, gold_elev):
        self.gold_c = resample_to(gold_tri, COARSE)
        self.elev = gold_elev
        self.cache = {}

    def score(self, a):
        a = _as2d(a)
        best = _cmp(resample_to(a, COARSE), self.gold_c)
        if self.elev is not None and a.shape != FULL and a.shape[0] < FULL[0]:
            if a.shape not in self.cache:
                ref = _tri(resample_to(self.elev, a.shape))
                cmp_shape = COARSE if a.shape[0] > COARSE[0] else a.shape
                self.cache[a.shape] = (resample_to(ref, cmp_shape), cmp_shape)
            ref, cs = self.cache[a.shape]
            best = max(best, _cmp(resample_to(a, cs), ref))
        return best


def _limits(ctx, tag, ax):
    for t, m, a in drawn_arrays(ctx, "pred", "axes_limits"):
        if t == tag and m["name"] == f"fig_ax{ax}_limits":
            return np.asarray(a, float)
    return None


def _split_args(s):
    depth, cur, out = 0, "", []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            if depth == 0:
                break
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _contour_z_names(code):
    """Identifiers used in the Z argument of contour/contourf calls in the code."""
    names = []
    for m in re.finditer(r"\bcontourf?\s*\(", code):
        args = [a for a in _split_args(code[m.end():]) if not re.match(r"^\w+\s*=", a)]
        if not args:
            continue
        z = args[0] if len(args) <= 2 else args[2]
        ids = [i for i in re.findall(r"[A-Za-z_]\w*", z)
               if i not in ("np", "numpy", "ma", "flipud", "fliplr", "T", "values", "data", "filled", "nan_to_num")]
        names.extend(ids)
    return names


def _figure_check(ctx, yard, name, name_or):
    """Score what the figure actually shows. Returns [main result, orientation result].

    Which way up the raster is drawn is presentation, not analysis. The
    required check accepts the drawn surface in either vertical orientation (never transposes
    or mirror images left-right); the orientation as drawn is a separate kind="figure" check."""
    best, where, seen = -1.0, None, False
    best_or, where_or = -1.0, None
    for tag, meta, a in drawn_arrays(ctx, "pred"):
        art = meta.get("artist")
        a = np.asarray(a, float)
        if art == "image":
            if a.ndim != 2 or not _dem_shaped(a.shape):
                continue
            seen = True
            cands = [a[::-1] if meta.get("origin") == "lower" else a]
            how = f"imshow origin={meta.get('origin')}"
        elif "QuadMesh" in str(art) and meta["name"].endswith("_values"):
            if a.ndim != 2 or not _dem_shaped(a.shape):
                continue
            seen = True
            ax = int(re.match(r"fig_ax(\d+)_", meta["name"]).group(1))
            lim = _limits(ctx, tag, ax)
            nr, nc = a.shape
            if lim is not None and max(abs(lim[0]), abs(lim[1])) <= nc + 2 and max(abs(lim[2]), abs(lim[3])) <= nr + 2:
                cands = [a[::-1] if lim[2] < lim[3] else a]          # index space
                how = "pcolormesh index space"
            else:
                cands = [a, a[::-1]]                                 # coordinate space
                how = "pcolormesh coordinate space (both vertical orientations)"
        else:
            continue
        for c in cands:
            r = yard.score(c)
            if r > best_or:
                best_or, where_or = r, f"{meta['name']}@{tag} ({how})"
        for c in cands + [c[::-1] for c in cands]:
            r = yard.score(c)
            if r > best:
                best, where = r, f"{meta['name']}@{tag} ({how}; either vertical orientation)"
    if seen:
        return [result(name, best >= THR, value=round(best, 3), threshold=THR, note=where or ""),
                result(name_or, best_or >= THR, value=round(best_or, 3), threshold=THR,
                       note=where_or or "", kind="figure")]
    # contour / contourf: identify the drawn Z array from the code
    levels = [np.asarray(a, float).ravel() for t, m, a in drawn_arrays(ctx, "pred")
              if "Contour" in str(m.get("artist")) and m["name"].endswith("_values")]
    if not levels:
        u = unverifiable(name, "no raster/mesh/contour artist with DEM-shaped data captured in the figure")
        return [u, result(name_or, False, note="no drawn surface", kind="figure")]
    zn = set(_contour_z_names(ctx.code_text()))
    lv = np.concatenate(levels)
    lv = lv[np.isfinite(lv) & (np.abs(lv) < 1e100)]   # extend='min'/'max' layers are +-huge
    best, where = -1.0, None
    for tag, meta, load in ctx.objects("pred", "array", include_files=False):
        if meta.get("source") == "figure" or meta["name"] not in zn or not _dem_shaped(meta.get("shape")):
            continue
        try:
            a = _as2d(load())
        except Exception:
            continue
        lo, hi = np.nanmin(a), np.nanmax(a)
        span = hi - lo if hi > lo else 1.0
        p1, p99 = np.nanpercentile(a, [1, 99])
        # contour levels must lie within the drawn variable's value range and span most of
        # its bulk (1st-99th percentile; levels are often capped at a high percentile)
        consistent = len(lv) >= 2 and lv.min() >= lo - 0.1 * span and lv.max() <= hi + 0.1 * span \
            and (lv.max() - lv.min()) >= 0.5 * (p99 - p1)
        if not consistent:
            continue
        r = max(yard.score(a), yard.score(a[::-1]))
        if r > best:
            best, where = r, f"contour of {meta['name']}@{tag}"
    if where is None:
        u = unverifiable(name, f"contour plot; drawn Z ({sorted(zn) or 'unparsed'}) not identified as a captured "
                               "array consistent with the contour levels")
        return [u, result(name_or, False, note="drawn surface not identified", kind="figure")]
    return [result(name, best >= THR, value=round(best, 3), threshold=THR, note=where),
            result(name_or, best >= THR, value=round(best, 3), threshold=THR,
                   note=where + " (contour: vertical direction not captured, both accepted)", kind="figure")]


def checks(ctx):
    out = []
    r = file_exists(ctx, [NAME, "Ruggedness.png"])
    out.append(r)
    if r["passed"]:
        nb, std = png_not_blank(ctx.pred(r["value"][0]))
        out.append(result("map not blank", nb, value=std, threshold=">=3 grey-level std"))
    else:
        out.append(result("map not blank", False, note="no map"))

    N_SURF = "TRI-family ruggedness surface computed (Spearman >= 0.95, resolution-matched)"
    N_FIG = "figure shows the ruggedness surface (Spearman >= 0.95, either vertical orientation)"
    N_OR = "figure: ruggedness surface drawn north-up (Spearman >= 0.95 as drawn)"
    N_DEF = "TRI as defined (Spearman >= 0.99)"
    gold = elev = None
    for tag, meta, load in ctx.objects("gold", "array", include_files=False):
        if meta["name"] == "ruggedness":
            gold = load()
        elif meta["name"] == "elevation":
            elev = load()
    if gold is None:
        out.append(result(N_SURF, False, note="gold array missing"))
        out.append(result(N_FIG, False, note="gold array missing"))
        out.append(result(N_OR, False, note="gold array missing", kind="figure"))
        out.append(result(N_DEF, False, required=False))
        return out
    yard = _Yard(gold, elev)
    best, where = -1.0, None
    for tag, meta, load in ctx.objects("pred", "array"):
        if meta.get("source") == "figure" or not _dem_shaped(meta.get("shape")):
            continue
        try:
            rho = yard.score(load())
        except Exception:
            continue
        if rho > best:
            best, where = rho, f"{meta['name']}@{tag}"
    out.append(result(N_SURF, best >= THR, value=round(best, 3), threshold=THR,
                      note=where or "no DEM-shaped 2-D array captured"))
    try:
        out.extend(_figure_check(ctx, yard, N_FIG, N_OR))
    except Exception as e:  # never crash
        out.append(result(N_FIG, False, note=repr(e)[:200]))
        out.append(result(N_OR, False, note=repr(e)[:200], kind="figure"))
    out.append(result(N_DEF, best >= 0.99, value=round(best, 3), threshold=0.99, required=False))
    return out
