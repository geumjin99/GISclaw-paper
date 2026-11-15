"""Task 6 - elk home range (convex hull), kernel density and DBSCAN clusters, SW Alberta 2009.

Correctness (required):
  1. Elk_Analysis.png exists and is not blank;
  2. home range by minimum bounding geometry: a captured polygon layer matches the convex hull
     of all fixes, or the union of per-animal convex hulls - grouped by ind_ident (68 animals) or
     by tag_ident (61 collars; as in task 43) - (area IoU >= 0.90 after
     reprojection); envelopes/rectangles, hulls of a subset or of the 7 'comments' groups fail;
  3. kernel density surface: a captured 2-D array (any resolution/CRS; assumed to span the fix
     extent, optionally padded, either row order) or a per-fix density vector agrees with the gold
     KDE (Pearson >= 0.6 on sqrt-density over a 40 x 40 comparison grid; Spearman >= 0.7 for
     per-fix values). Bandwidths of ~0.2-5 km pass (5 km Gaussian: grid r ~0.75, per-fix rho
     ~0.73; see control correct_gaussian_5km_padded); a bandwidth applied in the wrong units
     (468 m on degrees -> flat surface) fails. A 2-D array must be kernel-smoothed (corr. with
     its own 3x3 mean >= 0.9): a raw 2-D histogram of the fixes (0.80-0.85) is not a kernel
     density. When the run holds no 2-D array at all (seaborn.kdeplot, contourf of an inline
     KDE), a kernel-density call drawn as contours/image in the code is accepted;
  4. density-based clustering: a captured label vector of the fixes (gdf column, aligned array,
     or plotted scatter values) contains noise (-1) and >= 2 clusters, and clusters are spatially
     coherent (>= 90 % of clustered fixes share the label of their nearest clustered neighbour).
     k-means (no noise label) and random labels fail.
  4b. the prescribed clustering method (as in task 43): a DBSCAN / HDBSCAN / OPTICS call
     in the code (an import alone does not count; see _dbscan_called);
  5. the saved Elk_Analysis.png overlays the three results: the figure captured at the last
     savefig to that name (save snapshots mapped to the savefig calls of the code in order;
     any save snapshot when the mapping is ambiguous) holds a hull outline (line / polygon
     collection), a density layer (image, mesh, contours, or fixes coloured by a continuous
     value) and a cluster layer (a point collection coloured by >= 2 integer labels, or >= 3
     point collections). A map that only shows the clusters or only the hull fails.
Reported, not required:
  6. distances computed in a projected CRS (code mentions to_crs / a projected EPSG).
  7. (kind="figure") a density image drawn in the saved map sits north-up at
     its true location (as drawn, it agrees with the gold KDE at least as well as its mirror).

Georeferencing of captured objects: polygons without a CRS (shapely hulls, groupby results) are
tried in EPSG:4326 (when their coordinates are geographic) and in every EPSG code named in
the code; a 2-D density array is also georeferenced by captured coordinate grids of the same
shape (np.mgrid / np.meshgrid, either axis order - the scipy gaussian_kde idiom gives a
transposed array) and by the extent of a drawn image.
"""
import numpy as np

from checklib import file_exists, png_not_blank, result

CRS_M = "EPSG:32611"
G = 40


def _elk(ctx):
    import geopandas as gpd
    import pyogrio
    df = pyogrio.read_dataframe(f"{ctx.inputs_dir}/Elk_in_Southwestern_Alberta_2009.geojson",
                                columns=["ind_ident", "tag_ident"])
    return df.to_crs(CRS_M)


def _iou(u, ref):
    i, o = u.intersection(ref).area, u.union(ref).area
    return i / o if o > 0 else 0.0


def _sample(a, extent, X, Y, flip):
    """Value of 2-D array a (spanning extent x0,x1,y0,y1; row 0 = north unless flip) at X, Y."""
    x0, x1, y0, y1 = extent
    ny, nx = a.shape
    c = np.floor((X - x0) / (x1 - x0) * nx).astype(int)
    r = np.floor((y1 - Y) / (y1 - y0) * ny).astype(int)
    if flip:
        r = ny - 1 - r
    ok = (r >= 0) & (r < ny) & (c >= 0) & (c < nx)
    out = np.full(X.shape, np.nan)
    out[ok] = a[r[ok], c[ok]]
    return out


def _code_crs(code):
    import re
    out = []
    for m in re.finditer(r"(?:epsg\s*=\s*|EPSG:|epsg:)\s*['\"]?(\d{4,5})", code or ""):
        c = "EPSG:" + m.group(1)
        if c not in out:
            out.append(c)
    return out


def _is_geo(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    return bool(np.nanmax(np.abs(xs)) <= 180 and np.nanmax(np.abs(ys)) <= 90)


def _grid_axes(cx, cy):
    """(axis, x vector, y vector) when cx/cy are coordinate grids of one array (mgrid / meshgrid)."""
    out = []
    for ax in (0, 1):
        xv = cx[:, 0] if ax == 0 else cx[0, :]
        yv = cy[0, :] if ax == 0 else cy[:, 0]
        tx = np.ptp(cx, axis=1 - ax).max() if cx.shape[1 - ax] > 1 else 0
        ty = np.ptp(cy, axis=ax).max() if cy.shape[ax] > 1 else 0
        if tx <= 1e-9 * max(1.0, np.ptp(cx)) and ty <= 1e-9 * max(1.0, np.ptp(cy)) \
                and np.ptp(xv) > 0 and np.ptp(yv) > 0:
            out.append((ax, xv, yv))
    return out


def _saved_map_tags(ctx, target):
    """Save-snapshot tags whose figure is the one written to `target` (its last save).
    Save snapshots (saveNN, one per executed savefig) are mapped onto the savefig calls of the
    code: when every call is a literal file name at top level (no loop / function), one to one;
    otherwise from the END - valid when the target's last call and every call after it are
    literal top-level calls (each executes exactly once, so they own the last snapshots).
    Returns (set of tags, how) or (None, how) = ambiguous, any saved figure."""
    import checklib as _cl
    if _cl.saved_figures(ctx):  # exact record of which savefig wrote which file
        rec = _cl.figure_record_for_file(ctx, target)
        if rec is None:
            return set(), f"no recorded savefig wrote {target}"
        if rec.get("tag"):
            return {rec["tag"]}, f"{rec['tag']} wrote {target} last (savefig record)"
        # beyond the 20 data snapshots: fall back to inferring from the code
    import ast
    import os
    import re
    root = ctx.pred_snap
    if not root or not os.path.isdir(root):
        return None, "no snapshots"
    saves = sorted((t for t in os.listdir(root) if re.fullmatch(r"save\d+", t)), key=lambda t: int(t[4:]))
    try:
        tree = ast.parse(ctx.code_text())
    except Exception:
        return None, "code not parseable; any saved figure"
    calls = []

    def walk(node, simple):
        for ch in ast.iter_child_nodes(node):
            s = simple and not isinstance(ch, (ast.For, ast.AsyncFor, ast.While, ast.FunctionDef,
                                               ast.AsyncFunctionDef, ast.Lambda, ast.ListComp,
                                               ast.GeneratorExp, ast.DictComp, ast.SetComp, ast.With))
            if isinstance(ch, ast.Call) and isinstance(ch.func, ast.Attribute) and ch.func.attr == "savefig":
                a = ch.args[0] if ch.args else next((k.value for k in ch.keywords if k.arg == "fname"), None)
                name = a.value if isinstance(a, ast.Constant) and isinstance(a.value, str) else None
                calls.append(((ch.lineno, ch.col_offset), name, s and name is not None))
            walk(ch, s)
    walk(tree, True)
    calls.sort()
    if not saves or len(saves) >= 20 or not calls:
        return None, "save snapshots not mappable; any saved figure"
    hits = [i for i, c in enumerate(calls) if c[1] and os.path.basename(c[1]).lower() == target.lower()]
    if not hits:
        return None, "no literal savefig of the target; any saved figure"
    i = hits[-1]
    tail = calls[i:]
    if all(c[2] for c in tail) and len(saves) >= len(tail):
        t = saves[len(saves) - len(tail)]
        return {t}, f"{t} = savefig call #{i + 1} of {len(calls)} (mapped from the end)"
    return None, "save snapshots not mappable (loops/functions/non-literal names); any saved figure"


def _figure_layers(ctx, tag):
    """Which of hull outline / density / cluster layers the figure of snapshot `tag` holds."""
    arts = {}
    for t, meta, load in ctx.objects("pred", "array", include_files=False):
        if t == tag and meta.get("source") == "figure":
            arts[meta["name"]] = (meta, load)
    have = {"outline": None, "density": None, "clusters": None}
    point_colls = 0
    for nm, (meta, load) in arts.items():
        art, sh = meta.get("artist"), meta.get("shape", [])
        try:
            if art == "line" and sh and sh[0] >= 4:
                have["outline"] = have["outline"] or nm
            elif art in ("LineCollection", "PatchCollection", "PolyCollection") and nm.endswith("_paths_bbox"):
                have["outline"] = have["outline"] or nm
            elif art == "image" and len(sh) >= 2 and min(sh[:2]) >= 10:
                have["density"] = have["density"] or nm
            elif art in ("QuadMesh", "PolyQuadMesh") and nm.endswith("_values") and int(np.prod(sh)) >= 100 \
                    and (len(sh) < 2 or min(sh) > 1):
                have["density"] = have["density"] or nm
            elif "Contour" in str(art):
                have["density"] = have["density"] or nm
            elif art == "PathCollection" and nm.endswith("_offsets") and sh and sh[0] >= 500:
                point_colls += 1
                base = nm[: -len("_offsets")]
                if base + "_values" in arts:
                    v = np.asarray(arts[base + "_values"][1](), float).ravel()
                    v = v[np.isfinite(v)]
                    if len(v) and np.allclose(v, np.round(v)) and len(np.unique(v)) >= 2:
                        have["clusters"] = have["clusters"] or base
                    elif len(np.unique(v)) > 50:
                        have["density"] = have["density"] or base + " (fixes coloured by density)"
                elif base + "_facecolors" in arts:
                    fc = np.asarray(arts[base + "_facecolors"][1](), float)
                    if fc.ndim == 2 and len(np.unique(np.round(fc, 3), axis=0)) >= 3:
                        have["clusters"] = have["clusters"] or base + " (per-point colours)"
        except Exception:
            continue
    if have["clusters"] is None and point_colls >= 3:
        have["clusters"] = f"{point_colls} point collections"
    return have


def _smooth(a, k):
    from scipy.ndimage import uniform_filter
    a = np.where(np.isfinite(a), a, 0.0)
    return uniform_filter(a, size=max(1, k))


def _smoothness(a):
    """Correlation of the raw surface with its own 3x3 mean (on cells above 0.1 % of the max):
    a kernel density on a grid finer than ~2x its bandwidth is > 0.9; a raw 2-D histogram of the
    fixes (no kernel) is 0.80-0.85 at 0.2-1 km cells."""
    a = np.where(np.isfinite(a), a, 0.0)
    s = _smooth(a, 3)
    m = a > 0.001 * np.nanmax(np.abs(a))
    if m.sum() < 50 or a[m].std() == 0:
        return 0.0
    return float(np.corrcoef(a[m], s[m])[0, 1])


def _drawn_density_registration(ctx, tags, gold, gs, X, Y, code_crs):
    """Figure check (kind="figure"): a density image drawn in the
    saved Elk_Analysis.png, placed as matplotlib draws it (extent l, r, b, t; row 0 at t for
    origin 'upper', at b for 'lower'), agrees with the gold KDE at least as well as its
    north-south mirror image. Contours / meshes / per-fix colours carry their own coordinates
    and are not assessed; images in pixel coordinates are not assessable."""
    name = "density image in Elk_Analysis.png drawn north-up at its true location (figure)"
    if gold is None or not tags:
        return result(name, True, note="not assessable (no gold KDE or no saved figure)", kind="figure")
    try:
        from pyproj import Transformer
        pts = {CRS_M: (X, Y)}
        for c in ["EPSG:4326"] + code_crs:
            try:
                pts[c] = Transformer.from_crs(CRS_M, c, always_xy=True).transform(X, Y)
            except Exception:
                pass
        arrs = {}
        for t, meta, load in ctx.objects("pred", "array", include_files=False):
            if t in tags and meta.get("source") == "figure":
                arrs[(t, meta.get("name"))] = (meta, load)
        best = None
        for (t, nm), (meta, load) in arrs.items():
            sh = meta.get("shape", [])
            if meta.get("artist") != "image" or len(sh) < 2 or min(sh[:2]) < 10 or (t, nm + "_extent") not in arrs:
                continue
            a = np.asarray(load(), float)
            if a.ndim == 3:   # RGB(A) image: no values to compare
                continue
            a = _smooth(np.where(np.isfinite(a), a, 0.0), max(1, a.shape[1] // (2 * G)))
            e = np.asarray(arrs[(t, nm + "_extent")][1](), float).ravel()
            upper = (meta.get("origin") or "upper") == "upper"
            ny, nx = a.shape
            for c, (px, py) in pts.items():
                px, py = np.asarray(px, float), np.asarray(py, float)
                if e[1] == e[0] or e[3] == e[2]:
                    continue
                cc = np.floor((px - e[0]) / (e[1] - e[0]) * nx).astype(int)
                y0, y1 = (e[3], e[2]) if upper else (e[2], e[3])      # row 0 edge, last-row edge
                rr = np.floor((py - y0) / (y1 - y0) * ny).astype(int)
                ok = (rr >= 0) & (rr < ny) & (cc >= 0) & (cc < nx)
                if ok.mean() < 0.5:
                    continue
                sc = {}
                for lab, r_ in (("drawn", rr), ("mirrored", ny - 1 - rr)):
                    v = np.full(px.shape, np.nan)
                    v[ok] = a[r_[ok], cc[ok]]
                    m = np.isfinite(v) & np.isfinite(gs)
                    if m.sum() < 100 or v[m].std() == 0:
                        sc[lab] = -1.0
                        continue
                    sc[lab] = float(np.corrcoef(np.sqrt(np.clip(v[m], 0, None)),
                                                np.sqrt(np.clip(gs[m], 0, None)))[0, 1])
                key = max(sc.values())
                if key >= 0.5 and (best is None or key > best[0]):
                    best = (key, sc, f"{nm}@{t} ({c}, origin {'upper' if upper else 'lower'})")
        if best is None:
            return result(name, True, note="not assessable (no density image in map coordinates; "
                                           "contours / mesh / scatter carry their own coordinates)", kind="figure")
        _, sc, where = best
        return result(name, sc["drawn"] >= sc["mirrored"],
                      value={k: round(v, 3) for k, v in sc.items()},
                      threshold="r(as drawn) >= r(north-south mirror)", note=where, kind="figure")
    except Exception as ex:
        return result(name, True, note=f"not assessable ({repr(ex)[:120]})", kind="figure")


def _dbscan_called(code):
    """The instruction prescribes "Density-based Clustering(DBSCAN)" (same rule as task 43).
    A DBSCAN-family call (DBSCAN / HDBSCAN / OPTICS, sklearn or hdbscan) outside comments is
    required; an import alone does not count. The checked code holds only the successful blocks,
    so a call in a block that raised later (its labels survive) is recognised by DBSCAN being
    named together with its parameters (eps / min_samples / min_cluster_size) in the code."""
    import re
    c = "\n".join(ln.split(" #")[0] for ln in (code or "").splitlines() if not ln.lstrip().startswith("#"))
    m = re.search(r"\b(H?DBSCAN|OPTICS|dbscan|hdbscan|cluster_optics_dbscan)\s*\(", c)
    if m:
        return True, "call: " + m.group(0)
    if re.search(r"dbscan", c, re.I) and re.search(r"\b(eps|min_samples|min_cluster_size)\b", c):
        return True, "DBSCAN named with eps/min_samples in the code (call in a block not in the checked code)"
    return False, "no DBSCAN/HDBSCAN/OPTICS call in the code"


def checks(ctx):
    out = []
    names = ["Elk_Analysis.png"]
    r = file_exists(ctx, names)
    out.append(r)
    if r["passed"]:
        nb, std = png_not_blank(ctx.pred(names[0]))
        out.append(result("map not blank", nb, value=std, threshold=">=3 grey-level std"))
    else:
        out.append(result("map not blank", False, note="no map"))
    try:
        elk = _elk(ctx)
    except Exception as e:
        for n in ("home range polygon matches convex hull (IoU >= 0.90)", "kernel density matches gold (grid r >= 0.6 or per-fix rho >= 0.7)",
                  "DBSCAN-style clusters with noise, spatially coherent"):
            out.append(result(n, False, note=repr(e)[:200]))
        return out
    xy = np.c_[elk.geometry.x, elk.geometry.y]
    n = len(elk)

    code_all = ctx.code_text()
    code_crs = _code_crs(code_all)

    # --- 2. convex hull ------------------------------------------------------------------------
    pop = elk.geometry.union_all().convex_hull
    per = elk.dissolve(by="ind_ident").convex_hull.union_all()
    per_tag = elk.dissolve(by="tag_ident").convex_hull.union_all()   # per collar (as in task 43)
    best, where = 0.0, None
    seen = set()
    for tag, meta, load in ctx.objects("pred", "gdf"):
        sig = (meta.get("name"), meta.get("rows"), tuple(meta.get("columns", [])))
        if sig in seen or not meta.get("rows") or meta.get("rows") > 5000:
            continue
        seen.add(sig)
        try:
            g = load()
            g = g[g.geom_type.isin(["Polygon", "MultiPolygon"])]
            if not len(g):
                continue
            if g.crs is None and meta.get("source") == "file":
                continue  # no CRS guessing for files written by the run
            if g.crs is not None:
                trials = [(g.crs, "")]
            else:   # shapely hull / groupby result without CRS: try plausible CRSs
                b = g.total_bounds
                trials = ([("EPSG:4326", " (no CRS; assumed EPSG:4326)")] if _is_geo(b[[0, 2]], b[[1, 3]]) else []) + \
                         [(c, f" (no CRS; assumed {c} from code)") for c in code_crs]
            for crs_, how in trials:
                try:
                    u = g.set_crs(crs_, allow_override=True).to_crs(CRS_M).geometry.union_all()
                except Exception:
                    continue
                for k, ref in (("population hull", pop), ("per-animal hulls", per), ("per-collar hulls", per_tag)):
                    v = _iou(u, ref)
                    if v > best:
                        best, where = v, f"{meta['name']}@{tag} ~ {k}{how}"
        except Exception:
            continue
    out.append(result("home range polygon matches convex hull (IoU >= 0.90)", best >= 0.9, value=round(best, 3),
                      threshold=0.9, note=where or "no polygon layer"))

    # --- 3. kernel density ------------------------------------------------------------------------
    gold, gext = None, None
    for tag, meta, load in ctx.objects("gold", "array"):
        if meta.get("name") == "kde":
            gold = load()
    xmin, ymin, xmax, ymax = elk.total_bounds
    if gold is not None:
        bw5 = 5 * 468.805393200003
        gext = (xmin - bw5, xmin - bw5 + 200.0 * gold.shape[1], ymin - bw5, ymin - bw5 + 200.0 * gold.shape[0])
    best_k, where_k = -1.0, None
    best_pts, where_p = -1.0, None
    n2d, rough_best = 0, 0.0
    if gold is not None:
        gx = np.linspace(xmin, xmax, G + 1)
        gy = np.linspace(ymin, ymax, G + 1)
        X, Y = np.meshgrid((gx[:-1] + gx[1:]) / 2, (gy[:-1] + gy[1:]) / 2)
        gs = _sample(_smooth(gold, max(1, gold.shape[1] // (2 * G))), gext, X, Y, False)
        gpts = _sample(gold, gext, xy[:, 0], xy[:, 1], False)
        dx, dy = xmax - xmin, ymax - ymin
        exts = {"fix extent": (xmin, xmax, ymin, ymax)}
        for p in (0.05, 0.1):
            exts[f"fix extent +{int(p * 100)}%"] = (xmin - p * dx, xmax + p * dx, ymin - p * dy, ymax + p * dy)
        exts["gold extent"] = gext
        for tag, meta, load in ctx.objects("pred", "array"):
            sh = meta.get("shape", [])
            try:
                if len(sh) == 2 and min(sh) >= 20 and 1 not in sh:
                    a = np.asarray(load(), float)
                    if not np.isfinite(a).any() or np.nanstd(a) == 0:
                        continue
                    n2d += 1
                    sm = _smoothness(a)
                    if sm < 0.9:      # no kernel smoothing (raw counts): not a kernel density
                        rough_best = max(rough_best, sm)
                        continue
                    a = np.where(np.isfinite(a), a, 0.0)
                    a = _smooth(a, max(1, a.shape[1] // (2 * G)))
                    for en, e in exts.items():
                        for flip in (False, True):
                            p = _sample(a, e, X, Y, flip)
                            m = np.isfinite(p) & np.isfinite(gs)
                            if m.sum() < 100 or p[m].std() == 0:
                                continue
                            rr = float(np.corrcoef(np.sqrt(np.clip(p[m], 0, None)), np.sqrt(np.clip(gs[m], 0, None)))[0, 1])
                            if rr > best_k:
                                best_k, where_k = rr, f"{meta['name']}@{tag} ({en}{', flipped' if flip else ''})"
                elif (len(sh) == 1 and sh[0] == n) or (len(sh) == 2 and n in sh and 1 in sh):
                    a = np.asarray(load(), float).ravel()
                    from scipy.stats import spearmanr
                    m = np.isfinite(a) & np.isfinite(gpts)
                    if m.sum() > 0.8 * n and a[m].std() > 0:
                        rho = float(spearmanr(a[m], gpts[m]).correlation)
                        if rho > best_pts:
                            best_pts, where_p = rho, f"{meta['name']}@{tag} (per-fix density)"
            except Exception:
                continue
        # georeferenced candidates: coordinate grids of the same shape (mgrid / meshgrid, either
        # axis order) and drawn images with their extent; geographic grids are compared on the
        # comparison points transformed to EPSG:4326, projected ones in UTM 11N or a code CRS.
        try:
            from pyproj import Transformer
            from scipy.interpolate import RegularGridInterpolator
            pts = {CRS_M: (X, Y)}
            for c in ["EPSG:4326"] + code_crs:
                try:
                    pts[c] = Transformer.from_crs(CRS_M, c, always_xy=True).transform(X, Y)
                except Exception:
                    pass
            snaps = {}
            for tag, meta, load in ctx.objects("pred", "array", include_files=False):
                snaps.setdefault(tag, {})[meta.get("name")] = (meta, load)

            def score(vals, label):
                nonlocal best_k, where_k
                m = np.isfinite(vals) & np.isfinite(gs)
                if m.sum() < 100 or vals[m].std() == 0:
                    return
                rr = float(np.corrcoef(np.sqrt(np.clip(vals[m], 0, None)), np.sqrt(np.clip(gs[m], 0, None)))[0, 1])
                if rr > best_k:
                    best_k, where_k = rr, label

            for tag, arrs in snaps.items():
                two = {nm: (m_, l_) for nm, (m_, l_) in arrs.items()
                       if len(m_.get("shape", [])) == 2 and min(m_["shape"]) >= 20}
                if len(two) > 40:
                    continue
                loaded = {}

                def get(nm):
                    if nm not in loaded:
                        loaded[nm] = np.asarray(two[nm][1](), float)
                    return loaded[nm]
                for nm, (m_, l_) in two.items():
                    sh = tuple(m_["shape"])
                    peers = [p for p in two if p != nm and tuple(two[p][0]["shape"]) == sh]
                    img_ext = arrs.get(nm + "_extent") if m_.get("artist") == "image" else None
                    if len(peers) < 2 and img_ext is None:
                        continue
                    a = get(nm)
                    if not np.isfinite(a).any() or np.nanstd(a) == 0 or _smoothness(a) < 0.9:
                        continue
                    for cxn in peers:
                        for cyn in peers:
                            if cxn == cyn:
                                continue
                            for axis, xv, yv in _grid_axes(get(cxn), get(cyn)):
                                A = a if axis == 0 else a.T          # A[ix, iy]
                                if xv[0] > xv[-1]:
                                    xv, A = xv[::-1], A[::-1, :]
                                if yv[0] > yv[-1]:
                                    yv, A = yv[::-1], A[:, ::-1]
                                if not (np.all(np.diff(xv) > 0) and np.all(np.diff(yv) > 0)):
                                    continue
                                A = _smooth(A, max(1, A.shape[0] // (2 * G)))
                                f = RegularGridInterpolator((xv, yv), A, bounds_error=False, fill_value=np.nan)
                                for c in (["EPSG:4326"] if _is_geo(xv, yv) else [CRS_M] + code_crs):
                                    if c in pts:
                                        px, py = pts[c]
                                        score(f(np.c_[np.ravel(px), np.ravel(py)]).reshape(X.shape),
                                              f"{nm}@{tag} on grid ({cxn}, {cyn}){' transposed' if axis == 0 else ''}, {c}")
                    if img_ext is not None:
                        e = np.asarray(img_ext[1](), float).ravel()
                        ext = (min(e[0], e[1]), max(e[0], e[1]), min(e[2], e[3]), max(e[2], e[3]))
                        As = _smooth(a, max(1, a.shape[1] // (2 * G)))
                        for c in (["EPSG:4326"] if _is_geo(ext[:2], ext[2:]) else [CRS_M] + code_crs):
                            if c in pts:
                                px, py = pts[c]
                                for flip in (False, True):
                                    score(_sample(As, ext, np.asarray(px), np.asarray(py), flip),
                                          f"{nm}@{tag} (drawn image extent, {c}{', flipped' if flip else ''})")
        except Exception:
            pass
    kde_ok = best_k >= 0.6 or best_pts >= 0.7
    note_k = (where_k or where_p) or ("gold kde missing" if gold is None else
                                      ("only rough (unsmoothed, histogram-like) 2-D arrays, smoothness <= %.2f" % rough_best
                                       if rough_best else "no density array"))
    if not kde_ok and n2d == 0:
        # KDE drawn directly (seaborn.kdeplot, contourf of an inline KDE): no surface is captured.
        # Accept a kernel-density call whose result is drawn as contours/image, decided on the code.
        import re
        code = ctx.code_text()
        if re.search(r"kdeplot\s*\(", code) or (
                re.search(r"(gaussian_kde|KernelDensity|kde\.KDE|gaussian_filter)\s*\(", code)
                and re.search(r"(contourf?|imshow|pcolormesh)\s*\(", code)):
            kde_ok = True
            note_k = "KDE drawn without a captured surface (code: kdeplot / kernel density + contour)"
    out.append(result("kernel density matches gold (grid r >= 0.6 or per-fix rho >= 0.7)",
                      kde_ok,
                      value={"grid_r": round(best_k, 3), "per_fix_rho": round(best_pts, 3)},
                      threshold="grid r >= 0.6 or per-fix rho >= 0.7",
                      note=note_k))

    # --- 4. DBSCAN ----------------------------------------------------------------------------------
    from scipy.spatial import cKDTree

    def coherent(coords, lab):
        lab = np.asarray(lab)
        if lab.dtype.kind == "f":
            if not np.all(np.isfinite(lab)) or not np.allclose(lab, np.round(lab)):
                return None
            lab = lab.astype(int)
        if lab.dtype.kind not in "iu" or -1 not in lab:
            return None
        ks = set(np.unique(lab)) - {-1}
        if len(ks) < 2 or len(ks) > 0.2 * len(lab):
            return None
        m = (lab >= 0) & np.isfinite(coords).all(axis=1)   # no NaN / empty points in the tree
        c = coords[m]
        if len(c) < 20:
            return None
        nn = cKDTree(c).query(c, k=2)[1][:, 1]
        return float(np.mean(lab[m][nn] == lab[m])), len(ks), float(np.mean(~m))

    best_c, where_c = (0.0, 0, 0.0), None
    seen = set()
    cands = []
    for tag, meta, load in ctx.objects("pred", "gdf"):
        sig = (meta.get("name"), meta.get("rows"), tuple(meta.get("columns", [])))
        if sig in seen or meta.get("rows", 0) < 1000:
            continue
        seen.add(sig)
        try:
            g = load()
            if not (g.geom_type == "Point").all():
                continue
            coords = np.c_[g.geometry.x, g.geometry.y]
            for c in g.columns:
                if c != "geometry" and g[c].dtype.kind in "if" and c not in ("OBJECTID", "tag_ident", "satellite_",
                                                                              "summer_indicator", "outlier_ma"):
                    cands.append((f"{c} in {meta['name']}@{tag}", coords, g[c].to_numpy()))
        except Exception:
            continue
    figs = {}
    for tag, meta, load in ctx.objects("pred", "array"):
        sh = meta.get("shape", [])
        nm = meta.get("name", "")
        try:
            if meta.get("source") == "figure" and nm.endswith(("_values", "_offsets")):
                figs.setdefault((tag, nm.rsplit("_", 1)[0]), {})[nm.rsplit("_", 1)[1]] = load
            elif len(sh) == 1 and sh[0] == n:
                cands.append((f"{nm}@{tag}", xy, load()))
        except Exception:
            continue
    for (tag, base), d in figs.items():
        if "values" in d and "offsets" in d:
            try:
                v, o = np.asarray(d["values"]()).ravel(), np.asarray(d["offsets"]())
                if len(v) == len(o) and len(v) >= 1000:
                    cands.append((f"{base}@{tag} (plotted)", o, v))
            except Exception:
                pass
    for nm, coords, lab in cands:
        try:
            res = coherent(coords, lab)
        except Exception:
            res = None
        if res and res[0] > best_c[0]:
            best_c, where_c = res, nm
    coh, ncl, noise = best_c
    out.append(result("DBSCAN-style clusters with noise, spatially coherent", coh >= 0.9,
                      value={"nn_same_label": round(coh, 3), "clusters": ncl, "noise_share": round(noise, 3)},
                      threshold="noise label present, >=2 clusters, coherence >= 0.90",
                      note=where_c or "no label vector with a noise class"))
    dcall, dnote = _dbscan_called(ctx.code_text())
    out.append(result("density-based clustering estimator called (DBSCAN/HDBSCAN/OPTICS)", dcall, note=dnote))
    # --- 5. the saved map overlays the three results ------------------------------------------
    tags, how = _saved_map_tags(ctx, names[0])
    if tags is None:   # ambiguous: any saved figure
        import os as _os
        import re as _re
        _r = ctx.pred_snap
        tags = sorted(x for x in _os.listdir(_r) if _re.fullmatch(r"save\d+", x)) \
            if _r and _os.path.isdir(_r) else []
    best_l, best_tag = None, None
    for t in tags:
        h = _figure_layers(ctx, t)
        if best_l is None or sum(v is not None for v in h.values()) > sum(v is not None for v in best_l.values()):
            best_l, best_tag = h, t
    ok_l = bool(best_l) and all(v is not None for v in best_l.values())
    out.append(result("Elk_Analysis.png overlays hull, density and clusters", ok_l,
                      value={k: (v is not None) for k, v in (best_l or {}).items()} or None,
                      threshold="hull outline, density layer and cluster layer in the saved figure",
                      note=(f"{how}; " + "; ".join(f"{k}: {v}" for k, v in best_l.items() if v)) if best_l
                      else how))
    # --- 7. (figure) the density image is drawn where the density is ------------------------
    out.append(_drawn_density_registration(ctx, tags, gold, gs if gold is not None else None,
                                           X if gold is not None else None, Y if gold is not None else None,
                                           code_crs))
    code = ctx.code_text()
    proj = any(k in code for k in ("to_crs", "EPSG:326", "epsg=326", "32611", "32612", "3400", "estimate_utm_crs"))
    out.append(result("distances computed in a projected CRS", proj, required=False))
    return out
