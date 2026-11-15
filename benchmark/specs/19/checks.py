"""Checks for task 19: interpolated dissolved-oxygen surface (Chesapeake Bay),
interploated_water_quality.png.

The samples are depth profiles at 131 stations; the task does not fix depth or date, so
four station aggregations are admissible: mean of all samples (reference), surface layer
('S') only, bottom layer ('B') only, and near-surface water by depth
(Depth <= 2 m). checks.py builds an IDW surface of each on a 1 km grid
inside the Bay (11 789 cells) as yardsticks.

Required:
  1. the map exists (instructed misspelt name, or the corrected 'interpolated_...');
  2. map not blank;
  3. the field DRAWN in the figure is an interpolated DO surface. It is scored in any of
     the 8 orientations (flips / transposes) within its own
     drawn bounding box: which way up / where the map is drawn is presentation. The field is
     first placed by its own georeference: imshow arrays by their extent and origin (row 0 at the top for 'upper';
     arrays drawn in pixel space are placed on the Bay / sample bounding box as displayed,
     never flipped; an extent in km - x1000 matches the Bay / sample box,
     as given overlaps neither - is rescaled to metres), pcolormesh cells by their captured path bboxes, scatter points by their
     offsets and values (or, for GeoDataFrame.plot colours, the captured point layer whose
     column the colours encode), contour/contourf/tricontourf by evaluating the X/Y/Z
     arguments of the call on the captured arrays (lon/lat coordinates are projected).
     Sampled at the 11 789 Bay cells (nearest drawn sample within 1.5 spacings, >= 70 %
     coverage), it rank-correlates with one yardstick at Spearman >= 0.65 (IDW 0.99,
     kriging 0.86, kernel regression 0.87, nearest-station 0.74; value-weighted KDE 0.18,
     smoothed noise 0.16). A different layer drawn than the one computed fails. A correctly
     computed grid drawn rotated/flipped passes here and fails the kind="figure" check 7.
     Nothing placeable -> UNVERIFIABLE;
  4. the drawn values are DO in mg/L: mean over the Bay within 1.0 mg/L of the matching
     yardstick and 1st-99th percentile inside [0, 20]. A density surface (KDE, ~1e-10) or
     another variable (e.g. TotalDepth) fails. If the drawn image is only a display
     rescaling (e.g. rasterio.plot.show normalises to 0..1), an output GeoTIFF that the drawn
     values are a linear function of (r >= 0.99 at the Bay cells, placed by its own
     transform) supplies the values.
Reported, not required:
  5. a DO surface was computed (any captured grid, all 8 orientations);
  6. surface clipped to the Bay (<= 25 % of finite cells outside the polygon);
  7. (kind="figure", secondary analysis) the drawn field placed by its own georeference, as
     drawn (no re-orientation), reaches Spearman >= 0.65.
"""
import os

import numpy as np

from checklib import drawn_arrays, file_exists, png_not_blank, result, unverifiable

NAMES = ["interploated_water_quality.png", "interpolated_water_quality.png"]
RES = 1000.0


def _setup(inputs_dir):
    import geopandas as gpd
    import pandas as pd
    from shapely import contains_xy
    pts = gpd.read_file(os.path.join(inputs_dir, "DissolvedO2.geojson"))
    bay = gpd.read_file(os.path.join(inputs_dir, "Bay.geojson")).to_crs(pts.crs)
    pts = pts[pts["MeasureValue"].notna()].copy()
    pts["x"], pts["y"] = pts.geometry.x, pts.geometry.y
    pts["L"] = pts["Layer"].astype(str).str.strip()
    xy = pts.groupby("Station")[["x", "y"]].mean()
    cands = {"all samples": pts.groupby("Station")["MeasureValue"].mean(),
             "surface layer": pts[pts.L == "S"].groupby("Station")["MeasureValue"].mean(),
             "bottom layer": pts[pts.L == "B"].groupby("Station")["MeasureValue"].mean(),
             # near-surface water defined by depth (Depth <= 2 m, all
             # 131 stations) is an admissible surface definition besides the Layer == 'S' flag
             "surface (depth <= 2 m)": pts[pd.to_numeric(pts["Depth"], errors="coerce") <= 2]
             .groupby("Station")["MeasureValue"].mean()}
    minx, miny, maxx, maxy = bay.total_bounds
    gx = np.arange(minx + RES / 2, maxx, RES)
    gy = np.arange(maxy - RES / 2, miny, -RES)
    GX, GY = np.meshgrid(gx, gy)
    inside = contains_xy(bay.geometry.union_all(), GX, GY)
    P = np.c_[GX[inside], GY[inside]]
    yard = {}
    for k, s in cands.items():
        S = xy.loc[s.index].to_numpy()
        d2 = ((P[:, None, :] - S[None]) ** 2).sum(-1) + 1.0
        yard[k] = (s.to_numpy() / d2).sum(1) / (1 / d2).sum(1)
    exts = [tuple(bay.total_bounds), (pts.x.min(), pts.y.min(), pts.x.max(), pts.y.max())]
    outside_pts = np.c_[GX[~inside], GY[~inside]]
    return P, yard, exts, outside_pts, pts.crs


def _is_coord_grid(a):
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = np.nanstd(a, axis=1)
            c = np.nanstd(a, axis=0)
            tot = np.nanstd(a)
    return tot == 0 or np.nanmedian(r) <= 1e-9 * max(tot, 1) or np.nanmedian(c) <= 1e-9 * max(tot, 1)


def _sample(arr, e, Q):
    x0, y0, x1, y1 = e
    ny, nx = arr.shape
    c = np.floor((Q[:, 0] - x0) / (x1 - x0) * nx).astype(int)
    r = np.floor((y1 - Q[:, 1]) / (y1 - y0) * ny).astype(int)
    ok = (c >= 0) & (c < nx) & (r >= 0) & (r < ny)
    s = np.full(len(Q), np.nan)
    s[ok] = arr[r[ok], c[ok]]
    return s


def _score_grid(A, P, yard, exts, outside):
    from scipy.stats import spearmanr
    best = None
    variants = [A, A[::-1], A[:, ::-1], A[::-1, ::-1], A.T, A.T[::-1], A.T[:, ::-1], A.T[::-1, ::-1]]
    for e in exts:
        for arr in variants:
            s = _sample(arr, e, P)
            m = np.isfinite(s)
            if m.mean() < 0.7:
                continue
            for k, g in yard.items():
                rho = float(spearmanr(s[m], g[m]).correlation)
                if best is None or rho > best[0]:
                    so = _sample(arr, e, outside)
                    best = (rho, k, s[m], g[m], float(np.isfinite(so).sum() / max(1, np.isfinite(so).sum() + m.sum())))
    return best


def _score_points(df, P, yard):
    from scipy.spatial import cKDTree
    from scipy.stats import spearmanr
    xy = np.c_[df.geometry.x, df.geometry.y]
    fin = np.isfinite(xy).all(axis=1)   # missing / empty points would crash the tree
    if fin.sum() < 10:
        return None
    df, xy = df[fin], xy[fin]
    tree = cKDTree(xy)
    dist, idx = tree.query(P)
    best = None
    for c in df.columns:
        if c == "geometry" or df[c].dtype.kind not in "fi":
            continue
        s = df[c].to_numpy(float)[idx]
        s[dist > 3 * RES] = np.nan
        m = np.isfinite(s)
        if m.mean() < 0.7 or np.nanstd(s) == 0:
            continue
        for k, g in yard.items():
            rho = float(spearmanr(s[m], g[m]).correlation)
            if best is None or rho > best[0]:
                best = (rho, k, s[m], g[m], 0.0)
    return best


# ----------------------------------------------------------------------------- what is drawn
def _to_crs_xy(x, y, crs):
    """Drawn coordinates that look like lon/lat are converted to the data CRS."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    fx, fy = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(fx) and np.abs(fx).max() <= 180 and np.abs(fy).max() <= 90:
        from pyproj import Transformer
        t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        x, y = t.transform(x, y)
    return np.asarray(x, float), np.asarray(y, float)


def _km_extent(ext, exts):
    """A map drawn in km (imshow extent = metre coordinates / 1000, axes 'Easting (km)')
    is rescaled to metres when the extent, x1000, covers the Bay or sample bounding box
    (box IoU >= 0.9) and the extent as given overlaps neither. Otherwise the extent unchanged."""
    l, r, b, t = [float(v) for v in ext]
    x0, x1, y0, y1 = min(l, r), max(l, r), min(b, t), max(b, t)

    def box_iou(p, q):
        ix = max(0.0, min(p[2], q[2]) - max(p[0], q[0]))
        iy = max(0.0, min(p[3], q[3]) - max(p[1], q[1]))
        inter = ix * iy
        un = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - inter
        return inter / un if un > 0 else 0.0
    if any(box_iou((x0, y0, x1, y1), e) > 0 for e in exts):
        return ext
    if any(box_iou((1000 * x0, 1000 * y0, 1000 * x1, 1000 * y1), e) >= 0.9 for e in exts):
        return [1000.0 * l, 1000.0 * r, 1000.0 * b, 1000.0 * t]
    return ext


def _image_points(a, ext, origin, exts):
    """Cell centres (x, y) of an imshow array placed by its extent/origin (row 0 at the top
    for origin='upper'). Arrays drawn in pixel space are placed on the Bay / sample bounding
    box as DISPLAYED (top row = north), never flipped."""
    ny, nx = a.shape
    e = list(ext)
    pixel = abs(min(e[0], e[1]) + 0.5) < 1e-6 and abs(max(e[0], e[1]) - (nx - 0.5)) < 1e-6
    cand = []
    if pixel:
        disp = a if origin != "lower" else a[::-1]
        for x0, y0, x1, y1 in exts:
            cand.append((disp, (x0, x1, y0, y1), "upper", "pixel extent -> bbox " + ("bay" if not cand else "samples")))
    else:
        cand.append((a, tuple(e), origin, "image extent"))
    out = []
    for arr, (l, r, b, t), org, how in cand:
        yr0, yr1 = (t, b) if org != "lower" else (b, t)
        cx = l + (np.arange(nx) + 0.5) * (r - l) / nx
        cy = yr0 + (np.arange(ny) + 0.5) * (yr1 - yr0) / ny
        X, Y = np.meshgrid(cx, cy)
        out.append((X.ravel(), Y.ravel(), np.asarray(arr, float).ravel(), how))
    return out


def _split_args(s):
    depth, cur, out, quote = 0, "", [], None
    for ch in s:
        if quote:
            cur += ch
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
        elif ch in "([{":
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


class _Arr(np.ndarray):
    @property
    def values(self):
        return self

    @property
    def data(self):
        return self

    def filled(self, *a, **k):
        return self

    def to_numpy(self):
        return self


def _contour_points(ctx, tag):
    """Evaluate X, Y, Z of contour/contourf/tricontourf/pcolor calls in the code on the
    captured arrays of snapshot `tag` (no builtins). Returns list of (x, y, v, how)."""
    import re
    code = ctx.code_text()
    ns = {"np": np, "numpy": np}
    for t, meta, load in ctx.objects("pred", "array", include_files=False):
        if t == tag and meta.get("source") != "figure" and "." not in meta["name"]:
            try:
                ns[meta["name"]] = np.asarray(load(), float).view(_Arr)
            except Exception:
                pass
    out = []
    for m in re.finditer(r"\b(tricontourf|tricontour|contourf|contour|pcolormesh|pcolor|tripcolor)\s*\(", code):
        args = _split_args(code[m.end():])
        pos = [a for a in args if not re.match(r"^\w+\s*=[^=]", a)]
        if len(pos) < 3:
            continue
        try:
            vals = []
            for e in pos[:3]:
                if "__" in e or re.search(r"\b(import|open|exec|eval|lambda|getattr)\b", e):
                    raise ValueError
                vals.append(np.asarray(eval(e, {"__builtins__": {}}, ns), float))
            X, Y, Z = vals
        except Exception:
            continue
        if Z.ndim == 2 and X.ndim == 1 and Y.ndim == 1 and Z.shape == (len(Y), len(X)):
            X, Y = np.meshgrid(X, Y)
        if X.shape != Z.shape or Y.shape != Z.shape:
            continue
        out.append((X.ravel(), Y.ravel(), Z.ravel(), f"{m.group(1)} call evaluated"))
    return out


def _drawn_fields(ctx, exts, crs):
    """Every field drawn in the figure as (x, y, value, label), in the data CRS."""
    from checklib import drawn_color_column_matches
    fields = []
    arrs = {}
    for tag, meta, a in drawn_arrays(ctx, "pred"):
        arrs[(tag, meta["name"])] = (meta, a)
    tags = sorted({t for t, _ in arrs})
    for (tag, nm), (meta, a) in arrs.items():
        a = np.asarray(a, float)
        base = nm.rsplit("_", 1)[0]
        try:
            if meta.get("artist") == "image" and a.ndim == 2 and min(a.shape) >= 10:
                ext = arrs.get((tag, nm + "_extent"), (None, None))[1]
                if ext is None:
                    continue
                ext_m = _km_extent(ext, exts)
                for x, y, v, how in _image_points(a, ext_m, meta.get("origin"), exts):
                    if ext_m is not ext:
                        how += ", km axes x1000"
                    x, y = _to_crs_xy(x, y, crs)
                    fields.append((x, y, v, f"{nm}@{tag} ({how})"))
            elif nm.endswith("_values") and "QuadMesh" in str(meta.get("artist")) and a.size >= 100:
                bb = arrs.get((tag, base + "_paths_bbox"), (None, None))[1]
                if bb is None or len(bb) != a.size:
                    continue
                bb = np.asarray(bb, float)
                x, y = _to_crs_xy((bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2, crs)
                fields.append((x, y, a.ravel(), f"{nm}@{tag} (mesh cells)"))
            elif nm.endswith("_offsets") and meta.get("artist") == "PathCollection" and len(a) >= 100:
                v = arrs.get((tag, base + "_values"), (None, None))[1]
                if v is not None and np.asarray(v).size == len(a):
                    x, y = _to_crs_xy(a[:, 0], a[:, 1], crs)
                    fields.append((x, y, np.asarray(v, float).ravel(), f"{nm}@{tag} (scatter values)"))
        except Exception:
            continue
    # scatter/point layers drawn with colours only (GeoDataFrame.plot): the matching captured column
    try:
        for tname, col, rho, cmap, tg in drawn_color_column_matches(ctx, "pred", min_rho=0.95):
            for t, meta, load in ctx.objects("pred", "gdf", include_files=False):
                if t != tg or meta["name"] != tname:
                    continue
                g = load()
                if not (g.geom_type == "Point").all() or len(g) < 100:
                    continue
                if g.crs is not None and crs is not None and g.crs != crs:
                    g = g.to_crs(crs)
                fields.append((g.geometry.x.to_numpy(), g.geometry.y.to_numpy(), g[col].to_numpy(float),
                               f"{tname}.{col}@{tg} (points drawn by colour)"))
    except Exception:
        pass
    for tag in tags:
        for x, y, v, how in _contour_points(ctx, tag):
            x, y = _to_crs_xy(x, y, crs)
            fields.append((x, y, v, f"{how}@{tag}"))
    return fields


_ORIENT = ["as drawn", "mirrored E-W", "flipped N-S", "rotated 180", "transposed",
           "transposed + mirrored E-W", "transposed + flipped N-S", "transposed + rotated 180"]


def _orientations(x, y):
    """The 8 orientations (flips / transposes) of a drawn field within its
    own bounding box: which way the map is drawn is presentation, not analysis. Yields
    (label, x', y'); the first is the field as drawn."""
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any():
        return
    x0, x1, y0, y1 = x[ok].min(), x[ok].max(), y[ok].min(), y[ok].max()
    if x1 <= x0 or y1 <= y0:
        yield _ORIENT[0], x, y
        return
    u, v = (x - x0) / (x1 - x0), (y - y0) / (y1 - y0)
    for lab, (a, b) in zip(_ORIENT, [(u, v), (1 - u, v), (u, 1 - v), (1 - u, 1 - v),
                                     (v, u), (1 - v, u), (v, 1 - u), (1 - v, 1 - u)]):
        yield lab, x0 + a * (x1 - x0), y0 + b * (y1 - y0)


def _score_field(x, y, v, P, yard):
    """Sample a drawn field at the Bay cells (nearest drawn sample within ~1.5 spacings)."""
    from scipy.spatial import cKDTree
    from scipy.stats import spearmanr
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, v = x[ok], y[ok], v[ok]
    if len(x) < 10:
        return None
    xy = np.c_[x, y]
    tree = cKDTree(xy)
    sub = xy[np.linspace(0, len(xy) - 1, min(len(xy), 3000)).astype(int)]
    sp = float(np.median(tree.query(sub, k=2)[0][:, 1]))
    dist, idx = tree.query(P)
    s = v[idx].astype(float)
    s[dist > max(1.5 * sp, 0.75 * RES)] = np.nan
    m = np.isfinite(s)
    if m.mean() < 0.7 or np.nanstd(s[m]) == 0:
        return None
    best = None
    for k, g in yard.items():
        rho = float(spearmanr(s[m], g[m]).correlation)
        if best is None or rho > best[0]:
            best = (rho, k, s[m], g[m])
    return best


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
        P, yard, exts, outside, crs = _setup(ctx.inputs_dir)
    except Exception as e:
        out.append(result("the DRAWN map is an interpolated DO surface (Spearman >= 0.65 vs a yardstick, any orientation)", False, note=repr(e)[:200]))
        out.append(result("drawn values are DO in mg/L", False))
        out.append(result("figure: drawn surface placed correctly on the Bay (Spearman >= 0.65 as georeferenced)", False,
                          kind="figure"))
        return out
    N_S = "the DRAWN map is an interpolated DO surface (Spearman >= 0.65 vs a yardstick, any orientation)"
    N_V = "drawn values are DO in mg/L"
    N_G = "figure: drawn surface placed correctly on the Bay (Spearman >= 0.65 as georeferenced)"
    # (a) what the figure shows; required: any orientation within its own bounding box
    # (orientation / georegistration on the map is presentation);
    # kind="figure": as georeferenced (orientation as drawn)
    best, where, best_field = None, None, None
    geo_best, geo_where = None, None
    for x, y, v, label in _drawn_fields(ctx, exts, crs):
        for olab, xo, yo in _orientations(x, y):
            try:
                b = _score_field(xo, yo, v, P, yard)
            except Exception:
                continue
            if b is None:
                continue
            if olab == _ORIENT[0] and (geo_best is None or b[0] > geo_best[0]):
                geo_best, geo_where = b, label
            if best is None or b[0] > best[0] + 1e-9:
                lab2 = label if olab == _ORIENT[0] else f"{label} [{olab}]"
                best, where, best_field = b, lab2, (xo, yo, v, lab2)
    # (b) reported only: any captured grid, orientation-free
    any_best = None
    for tag, meta, load in ctx.objects("pred"):
        try:
            if meta.get("kind") == "array" and meta.get("source") != "figure":
                sh = meta.get("shape") or []
                if len(sh) == 3 and sh[0] == 1:
                    sh = sh[1:]
                if len(sh) != 2 or min(sh) < 30:
                    continue
                A = np.asarray(load(), float).reshape(sh)
                if _is_coord_grid(A):
                    continue
                b = _score_grid(A, P, yard, exts, outside)
            elif meta.get("kind") == "gdf" and meta.get("rows", 0) >= 500:
                g = load()
                if not (g.geom_type == "Point").all():
                    continue
                if g.crs is not None and crs is not None and g.crs != crs:
                    g = g.to_crs(crs)
                b = _score_points(g, P, yard)
            else:
                continue
        except Exception:
            continue
        if b is not None and (any_best is None or b[0] > any_best[0]):
            any_best = b
    # output GeoTIFFs carry their own georeference: used as unit evidence when the drawn image
    # is a display rescaling (e.g. rasterio.plot.show normalises to 0..1)
    tif_fields = []
    for f in ctx.pred_files("*"):
        if f.lower().endswith((".tif", ".tiff")):
            try:
                import rasterio
                with rasterio.open(ctx.pred(f)) as r:
                    a = r.read(1).astype(float)
                    if r.nodata is not None:
                        a[a == r.nodata] = np.nan
                    rows, cols = np.mgrid[0:a.shape[0], 0:a.shape[1]]
                    xs, ys = rasterio.transform.xy(r.transform, rows.ravel(), cols.ravel())
                    rc = r.crs
                x, y = np.asarray(xs, float), np.asarray(ys, float)
                if rc is not None and crs is not None and str(rc) != str(crs):
                    from pyproj import Transformer
                    x, y = Transformer.from_crs(rc, crs, always_xy=True).transform(x, y)
                tif_fields.append((x, y, a.ravel(), f))
            except Exception:
                continue
    if best is None:
        out.append(unverifiable(N_S, "no drawn field (image with extent, mesh, scatter, contour call) could be "
                                     "placed on the Bay"))
        out.append(unverifiable(N_V, "no drawn field"))
        out.append(result(N_G, False, note="no drawn field", kind="figure"))
    else:
        rho, k, s_, g = best
        out.append(result(N_S, rho >= 0.65, value=round(rho, 3), threshold=0.65, note=f"{where} ~ IDW of {k}"))
        lo, hi = np.nanpercentile(s_, [1, 99])
        ok = abs(np.nanmean(s_) - np.nanmean(g)) <= 1.0 and lo >= 0 and hi <= 20
        if not ok and rho >= 0.65:
            # drawn image may be a linear display rescaling of a georeferenced mg/L raster
            dx, dy, dv, _ = best_field
            for x, y, v, f in tif_fields:
                try:
                    from scipy.spatial import cKDTree
                    okm = np.isfinite(v) & np.isfinite(x) & np.isfinite(y)
                    _, i = cKDTree(np.c_[x[okm], y[okm]]).query(P)
                    tv = v[okm][i]
                    okd = np.isfinite(dx) & np.isfinite(dy)
                    _, j = cKDTree(np.c_[dx[okd], dy[okd]]).query(P)
                    dvv = dv[okd][j]
                    m = np.isfinite(tv) & np.isfinite(dvv)
                    r_lin = np.corrcoef(tv[m], dvv[m])[0, 1]
                    t_lo, t_hi = np.nanpercentile(tv[m], [1, 99])
                    if r_lin >= 0.99 and abs(np.nanmean(tv[m]) - np.nanmean(g)) <= 1.0 and t_lo >= 0 and t_hi <= 20:
                        ok, where = True, f"{where}; drawn image is a display rescaling of {f} (r={r_lin:.3f}), values from the GeoTIFF"
                        s_ = tv[m]
                        lo, hi = t_lo, t_hi
                        break
                except Exception:
                    continue
        out.append(result(N_V, ok, value=[round(float(np.nanmean(s_)), 3), round(float(lo), 3), round(float(hi), 3)],
                          threshold=f"mean {np.nanmean(g):.2f}+/-1.0, p1-p99 in [0,20]", note=where))
        out.append(result(N_G, geo_best is not None and geo_best[0] >= 0.65,
                          value=None if geo_best is None else round(geo_best[0], 3), threshold=0.65,
                          note=geo_where or "", kind="figure"))
    out.append(result("a DO surface was computed (any captured grid, orientation-free)",
                      any_best is not None and any_best[0] >= 0.65,
                      value=None if any_best is None else round(any_best[0], 3), required=False))
    frac_out = any_best[4] if any_best is not None else 1.0
    out.append(result("surface clipped to the Bay", frac_out <= 0.25, value=round(frac_out, 3), threshold=0.25,
                      required=False))
    return out
