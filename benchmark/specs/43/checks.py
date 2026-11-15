"""Checks for task 43: elk home range - convex hull, kernel density, density-based clustering.

Task core: 150 191 GPS fixes of elk (SW Alberta, 2009) -> (1) minimum bounding geometry as a
convex hull (home range), (2) kernel density surface, (3) density-based clustering
(DBSCAN/HDBSCAN); the spatial outputs are to be stored (the original model writes all three).

Correctness (required):
  1. home range stored: a polygon vector file in pred_results whose union matches the gold
     convex hull of all fixes (area IoU >= 0.9) - or, as the legitimate per-animal reading of
     "home ranges", the union of per-individual convex hulls (IoU >= 0.9 with that union).
     An envelope (bounding box) gives IoU 0.68 and fails. A file with several
     labelled geometry types per animal (a string column with 2-6 values) is also scored on the
     subset whose label names the convex hull ('hull' / 'convex');
  2. kernel density stored as a georeferenced raster that agrees with the gold density:
     the pred raster is sampled at the gold 100 m cell centres (re-projected as needed),
     both are averaged to 3 km blocks and the Pearson r must be >= 0.6. Bandwidths from the
     original 469 m up to ~5 km Gaussian give r >= 0.69; a north-south flip (0.32) or a 5 km
     shift (0.42) fail;
  3. density-based clusters: per-fix cluster labels (vector file, captured table or array in
     input order) with a noise class (-1 / NaN), >= 2 clusters, covering >= 90 % of fixes, and
     clustered fixes lie in clearly denser terrain than noise fixes (median gold density ratio
     >= 1.5; gold 2.55, random labels ~1). k-means (no noise) or labels unrelated to density fail.
  4. a DBSCAN/HDBSCAN/OPTICS estimator is called in the code (the task names DBSCAN; a call,
     not an import - density-threshold regions are not density-based clustering).
Check 1 also accepts the union of per-collar (tag_ident) hulls.
A log-density raster (all values <= 0, e.g. sklearn score_samples written as is)
is exponentiated before the comparison (block r 0.13 -> 0.92 on the same surface).
The KDE raster must be smoother than the plain fix-count grid
on its own grid (lag-1 autocorrelation inside the hull >= counts + 0.1, or >= 0.95; raw count
grid 0.66 vs 0.66, gold 0.98 vs 0.58), and cluster labels must cover >= 90 % of the 150 191
fixes.
"""
import os
import glob
import re as _re

import numpy as np

try:  # point PROJ at the data directory bundled with rasterio so that rasterio and pyproj agree
    import rasterio as _rio
    _pd = os.path.join(os.path.dirname(_rio.__file__), "proj_data")
    if os.path.isdir(_pd):
        os.environ["PROJ_DATA"] = _pd
except Exception:
    pass

from checklib import result, area_iou, find_object

GOLD_CRS = "EPSG:3400"
N_FIXES = 150191
INPUT_COLS = {"OBJECTID", "timestamp", "long", "lat", "comments", "external_t", "dop", "fix_type_r",
              "satellite_", "height", "crc_status", "outlier_ma", "sensor_typ", "individual", "tag_ident",
              "ind_ident", "study_name", "date", "time", "timestamp_Converted", "summer_indicator", "geometry",
              "index", "level_0", "x", "y", "X", "Y", "lon", "longitude", "latitude", "index_right"}
VEC_EXT = ("*.geojson", "*.json", "*.shp", "*.gpkg")
RAS_EXT = ("*.tif", "*.tiff", "*.TIF", "*.img", "*.asc")


def _vector_files(ctx):
    import geopandas as gpd
    out = []
    for pat in VEC_EXT:
        for p in glob.glob(os.path.join(ctx.pred_dir, "**", pat), recursive=True):
            try:
                if p.endswith(".gpkg"):
                    import pyogrio
                    for lyr in pyogrio.list_layers(p)[:, 0]:
                        out.append((os.path.basename(p) + ":" + lyr, gpd.read_file(p, layer=lyr)))
                else:
                    out.append((os.path.basename(p), gpd.read_file(p)))
            except Exception:
                continue
    return out


def _gold_points(ctx):
    obj, _, _ = find_object(ctx, "gold", "gdf", lambda o, m: len(o) > 100000 and "cluster" in o.columns)
    return obj


def _sample(path, xs, ys, src_crs):
    """Values of raster `path` at coordinates (xs, ys) given in src_crs; NaN outside."""
    import rasterio
    from pyproj import Transformer
    with rasterio.open(path) as r:
        a = r.read(1).astype(float)
        if r.nodata is not None:
            a[a == r.nodata] = np.nan
        crs = None
        try:
            crs = r.crs.to_wkt() if r.crs else None
        except Exception:
            crs = None
        T = r.transform
        b = r.bounds
    if crs is None:
        # no CRS guessing for deliverable files: deliverable files must carry a CRS, so a
        # raster without one is not interpreted
        return np.full(len(xs), np.nan)
    px, py = Transformer.from_crs(src_crs, crs, always_xy=True).transform(xs, ys)
    inv = ~T
    c, rr = inv * (np.asarray(px), np.asarray(py))
    c, rr = np.floor(c).astype(int), np.floor(rr).astype(int)
    ok = (rr >= 0) & (rr < a.shape[0]) & (c >= 0) & (c < a.shape[1])
    v = np.full(len(xs), np.nan)
    v[ok] = a[rr[ok], c[ok]]
    return v


def _lag1(a):
    def ac(x, y):
        k = np.isfinite(x) & np.isfinite(y)
        if k.sum() < 100 or np.std(x[k]) == 0 or np.std(y[k]) == 0:
            return float("nan")
        return float(np.corrcoef(x[k], y[k])[0, 1])
    return np.nanmean([ac(a[:, :-1].ravel(), a[:, 1:].ravel()), ac(a[:-1].ravel(), a[1:].ravel())])


def _smoothness(path, pts, hull, expo=False):
    """Lag-1 autocorrelation of the raster inside the hull, and of the plain fix-count grid on
    the same raster grid (a kernel density must be clearly smoother than the counts)."""
    import rasterio
    from rasterio.features import geometry_mask
    with rasterio.open(path) as r:
        a = r.read(1).astype(float)
        if r.nodata is not None:
            a[a == r.nodata] = np.nan
        crs, T = r.crs, r.transform
    if expo:  # log-density raster: judge smoothness on the density scale
        a = np.exp(a - np.nanmax(np.where(np.isfinite(a), a, np.nan)))
    if crs is None:
        return float("nan"), float("nan")
    h = hull.to_crs(crs)
    ins = ~geometry_mask(h.geometry, a.shape, T)
    p = pts.to_crs(crs)
    c, rr = ~T * (p.geometry.x.to_numpy(), p.geometry.y.to_numpy())
    c, rr = np.floor(c).astype(int), np.floor(rr).astype(int)
    ok = (rr >= 0) & (rr < a.shape[0]) & (c >= 0) & (c < a.shape[1])
    cnt = np.bincount(rr[ok] * a.shape[1] + c[ok], minlength=a.size).reshape(a.shape).astype(float)
    return _lag1(np.where(ins, a, np.nan)), _lag1(np.where(ins, cnt, np.nan))


def _blocks(a, b=30):
    h, w = a.shape[0] // b * b, a.shape[1] // b * b
    return a[:h, :w].reshape(h // b, b, w // b, b).mean((1, 3))


def _label_candidates(ctx, gold_pts):
    """(name, x, y, labels) in GOLD_CRS for per-fix clustering results."""
    import geopandas as gpd
    out = []
    srcs = [(n, g) for n, g in _vector_files(ctx) if len(g) >= 10000]
    for tag, meta, load in ctx.objects("pred", "gdf"):
        if meta.get("rows", 0) >= 10000:
            try:
                srcs.append((meta["name"] + "@" + tag, load()))
            except Exception:
                pass
    for name, g in srcs:
        try:
            if g.crs is None or not (g.geom_type == "Point").mean() > 0.95:
                continue
            gg = g.to_crs(GOLD_CRS)
            x, y = gg.geometry.x.to_numpy(), gg.geometry.y.to_numpy()
        except Exception:
            continue
        for c in g.columns:
            if c in INPUT_COLS or g[c].dtype.kind not in "fi":
                continue
            v = g[c].to_numpy(float)
            fin = v[np.isfinite(v)]
            if len(fin) == 0 or not np.allclose(fin, np.round(fin)):
                continue
            out.append((f"{c} in {name}", x, y, np.where(np.isfinite(v), v, -1)))
    if gold_pts is not None:
        gx, gy = gold_pts.geometry.x.to_numpy(), gold_pts.geometry.y.to_numpy()
        for tag, meta, load in ctx.objects("pred", "array"):
            if meta.get("shape") != [len(gold_pts)]:
                continue
            try:
                a = load()
            except Exception:
                continue
            if a.dtype.kind in "iu" or (a.dtype.kind == "f" and np.allclose(a[np.isfinite(a)], np.round(a[np.isfinite(a)]))):
                out.append((meta["name"] + "@" + tag, gx, gy, np.where(np.isfinite(a.astype(float)), a, -1)))
    return out


def checks(ctx):
    import geopandas as gpd
    out = []
    gold_hull_p = ctx.gold("elk_home_range.geojson")
    gold_kd_p = ctx.gold("elk_kernel_density.tif")
    gold_pts = _gold_points(ctx)

    # ---------------------------------------------------------------- 1 home range
    best = (-1.0, None)
    hull = None
    try:
        hull = gpd.read_file(gold_hull_p)
        alt = alt_tag = None
        if gold_pts is not None:
            ih = gold_pts.groupby("ind_ident").geometry.apply(lambda s: s.union_all().convex_hull)
            alt = gpd.GeoDataFrame(geometry=list(ih.values), crs=gold_pts.crs)
            # per-collar hulls (tag_ident, 61 collars) are an equally
            # legitimate "per animal" reading (union IoU 0.868 with the ind_ident union)
            # (the gold snapshot does not keep tag_ident, so the collars come from the input fixes)
            try:
                src = gpd.read_file(os.path.join(ctx.inputs_dir, "Elk_in_Southwestern_Alberta_2009.geojson"),
                                    columns=["tag_ident"]).to_crs(GOLD_CRS)
                it = src.groupby("tag_ident").geometry.apply(lambda s: s.union_all().convex_hull)
                alt_tag = gpd.GeoDataFrame(geometry=list(it.values), crs=GOLD_CRS)
            except Exception:
                alt_tag = None
        for name, g in _vector_files(ctx):
            if g.crs is None or len(g) == 0 or len(g) > 1000:
                continue
            if not g.geom_type.isin(["Polygon", "MultiPolygon"]).all():
                continue
            g = g[g.geometry.notna()]
            # A file may hold several labelled geometry types per animal (e.g. mbg_type =
            # Convex Hull / Minimum Bounding Envelope / Minimum Bounding Circle, 68 rows each); its
            # union as a whole gives IoU 0.71. The subset labelled as the hull is scored too: a string
            # column with 2-6 values, the subset whose value names a convex hull ('hull' / 'convex').
            subsets = [(name, g)]
            for c in g.columns:
                if c == "geometry" or g[c].dtype != object:
                    continue
                vc = g[c].astype(str).value_counts()
                if 2 <= len(vc) <= 6:
                    subsets += [(f"{name}[{c} == {k!r}]", g[g[c].astype(str) == k]) for k in vc.index
                                if _re.search(r"hull|convex", k, _re.I)]
            for sname, gs in subsets:
                for ref, lab in ((hull, "hull of all fixes"), (alt, "per-animal hulls"),
                                 (alt_tag, "per-collar (tag_ident) hulls")):
                    if ref is None:
                        continue
                    iou = area_iou(gs.to_crs(GOLD_CRS), ref.to_crs(GOLD_CRS))
                    if iou > best[0]:
                        best = (iou, f"{sname} vs {lab}")
    except Exception as e:
        best = (best[0], f"error {e!r}"[:200])
    out.append(result("home range (convex hull) stored and matches gold (IoU >= 0.9)", best[0] >= 0.9,
                      value=round(best[0], 3), threshold=0.9, note=best[1] or "no polygon vector file"))

    # ---------------------------------------------------------------- 2 kernel density
    best = (-1.0, None)
    gdens = None
    try:
        import rasterio
        with rasterio.open(gold_kd_p) as r:
            gdens = r.read(1).astype(float)
            T = r.transform
        rows, cols = np.mgrid[0:gdens.shape[0], 0:gdens.shape[1]]
        xs, ys = T * (cols.ravel() + 0.5, rows.ravel() + 0.5)
        G = _blocks(gdens)
        for pat in RAS_EXT:
            for p in glob.glob(os.path.join(ctx.pred_dir, "**", pat), recursive=True):
                try:
                    v = _sample(p, xs, ys, GOLD_CRS).reshape(gdens.shape)
                except Exception:
                    continue
                cover = float(np.isfinite(v[gdens > 0]).mean()) if (gdens > 0).any() else 0
                # log-density surface (sklearn KernelDensity.score_samples written as is):
                # all values <= 0 over a wide range -> compare on the density scale
                fin = v[np.isfinite(v)]
                logd = bool(len(fin) and fin.max() <= 0 and fin.min() < -5)
                if logd:
                    v = np.exp(v - fin.max())
                v = np.where(np.isfinite(v), v, 0.0)
                P = _blocks(v)
                if P.std() == 0:
                    continue
                rr = float(np.corrcoef(P.ravel(), G.ravel())[0, 1])
                if cover < 0.8:
                    rr = min(rr, 0.0)
                if rr >= 0.6:   # a kernel density, not a raw count grid
                    sm, sc = _smoothness(p, gold_pts, hull, logd) if gold_pts is not None and hull is not None else (float("nan"),) * 2
                    if not (sm >= 0.95 or sm >= sc + 0.1):
                        rr = min(rr, 0.59)
                        p = p + f" [not smoothed: lag-1 r {sm:.2f} vs fix counts on same grid {sc:.2f}]"
                if rr > best[0]:
                    best = (rr, os.path.basename(p) + f" (coverage {cover:.2f})" + (" [log density, exponentiated]" if logd else ""))
    except Exception as e:
        best = (best[0], f"error {e!r}"[:200])
    out.append(result("kernel density raster stored and agrees with gold (3 km block Pearson >= 0.6)",
                      best[0] >= 0.6, value=round(best[0], 3), threshold=0.6,
                      note=best[1] or "no readable georeferenced raster in pred_results"))

    # ---------------------------------------------------------------- 3 clusters
    best = (-1.0, None, None)
    try:
        cands = _label_candidates(ctx, gold_pts)
        import rasterio
        for name, x, y, lab in cands:
            noise = lab < 0
            ncl = len(np.unique(lab[~noise]))
            if not noise.any() or ncl < 2 or noise.all() or len(lab) < 0.9 * N_FIXES:
                continue
            d = _sample(gold_kd_p, x, y, GOLD_CRS)
            dn, dc = np.nanmedian(d[noise]), np.nanmedian(d[~noise])
            ratio = float(dc / max(dn, 1e-9))
            if ratio > best[0]:
                best = (ratio, name, [ncl, round(float(noise.mean()), 3), len(lab)])
    except Exception as e:
        best = (best[0], f"error {e!r}"[:200], None)
    out.append(result("density-based clusters with noise; clustered fixes denser than noise (ratio >= 1.5)",
                      best[0] >= 1.5, value=round(best[0], 2), threshold=1.5,
                      note=f"{best[1]} [n_clusters, noise share, n]={best[2]}" if best[1] else
                      "no per-fix cluster labels with a noise class"))

    # 4 the task names DBSCAN: a density-based clustering estimator must actually be called
    # (KDE-threshold regions are not density-based clustering, and a bare import is not a call)
    import re
    code = "\n".join(l.split("#")[0] for l in ctx.code_text().splitlines())
    called = re.search(r"\b(H?DBSCAN|OPTICS|dbscan|hdbscan|cluster_optics_dbscan)\s*\(", code)
    out.append(result("density-based clustering estimator called (DBSCAN/HDBSCAN/OPTICS)", bool(called),
                      value=called.group(0) if called else None,
                      note="call required, not just an import"))
    return out
