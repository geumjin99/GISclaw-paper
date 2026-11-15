"""T32 - Mountain-lion corridors: weighted cost surface + optimal region connections.

Gold: cost = 1.25 ruggedness cost (focal range, MSLarge 10->1) + distance-to-roads cost
(Euclidean distance from all roads, MSLarge 10->1: near roads expensive) + land-cover cost
(authors' table) + protected-status cost (1/3/6/9, no status = 9). Least-cost MST connections
between the 4 habitat regions (3 paths, ~35 km each). The benchmark distributes no Roads.geojson
for this task; the file is taken from the data of task 21.

Correctness (required):
  1. optimal_path.png exists and is not blank;
  2. the cost surface is a weighted sum of all four cost layers with the right direction:
     a captured 2-D array (input grid, or a coarser grid of the same aspect ratio, resampled)
     or written GeoTIFF is regressed (standardised, on a 4x-subsampled grid) on the four gold
     component layers; required R^2 >= 0.75, every standardised weight >= 0.08 and every
     partial correlation >= 0.35 (a dropped layer gives ~0 / <= 0.25, an inverted one a
     negative value; the weights themselves are free, the task says only "different weights").
     The road layer of the regression is the best-fitting (max R^2) of three definitions (all
     roads, FUNC_CLASS <= 4, FUNC_CLASS <= 3 - the data description calls them "major
     roads"), so a major-roads-only distance is accepted, but a wrong road model cannot be
     picked just to absorb a missing layer. The task text leaves the rescaling of the road
     distance open, so a fourth road layer is offered - the decile rank of the gold
     distance-to-all-roads (10-class quantile reverse reclassification, as produced by
     quantile / rank-decile / manual-bin rescaling). Linear 10 -> 1 rescalings of the gold
     ruggedness and distance to all roads are offered as well, and the check passes when ANY
     combination of the offered ruggedness x road variants meets every threshold, provided
     every partial r of the max-R^2 combination also reaches 0.35 (layer presence is read from the best-fitting model, so a
     mismatched variant cannot absorb a missing layer);
  3. corridors follow the gold least-cost paths: path vertices (vector lines, (N,2) coordinate
     or row/col arrays, a sparse path raster, or lines drawn in the figure) more than 1 km from
     the habitats are compared with the gold paths: >= 80 % of gold path points lie within
     1.5 km of the prediction (recall) and >= 80 % of predicted points within 1.5 km of gold
     (precision); or, for a network that connects all habitats but is a superset of the MST
     (e.g. least-cost links between every pair; the task says only "connect"), recall >= 0.90
     and precision >= 0.35;
  4. the map draws the corridors: either some figure-drawn candidate - lines drawn in the figure,
     or a drawn image whose sparse cells are a path raster (ax.imshow of a path mask) - passes
     the check-3 criterion on its own (every figure candidate is tested, not only the best
     candidate of check 3, which is usually a saved file), or a drawn collection (<= 200 paths, e.g. GeoDataFrame.plot of
     the paths; per-path bounding boxes captured, in map or pixel coordinates) whose boxes
     (+1.5 km) cover >= 80 % of the gold corridor points and >= 80 % of whose boxes contain
     gold corridor points.
"""
import numpy as np

from checklib import drawn_arrays, file_exists, png_not_blank, resample_to, result

SHAPE = (2494, 3062)
SUB = 4                      # cost regression on every 4th cell
R2_MIN, BETA_MIN, PCORR_MIN = 0.75, 0.08, 0.35
RADIUS = 1500.0


def _grid(ctx):
    import rasterio
    with rasterio.open(f"{ctx.inputs_dir}/Elevation.tif") as s:
        return s.transform, s.bounds


def _gold(ctx):
    cost, paths, comps = None, None, {}
    names = ("ruggedness_cost", "road_cost", "landcover_cost", "protected_cost")
    for tag, meta, load in ctx.objects("gold", include_files=False):
        if tag != "final":
            continue
        if meta["name"] == "cost_surface" and meta.get("kind") == "array":
            cost = np.asarray(load(), float)
        if meta["name"] in names and meta.get("kind") == "array":
            comps[meta["name"]] = np.asarray(load(), float)[::SUB, ::SUB]
        if meta["name"] in ("distance_to_roads", "ruggedness") and meta.get("kind") == "array":
            comps[meta["name"]] = np.asarray(load(), float)[::SUB, ::SUB]
        if meta["name"] == "optimal_paths" and meta.get("kind") == "gdf":
            paths = load()
    return cost, paths, comps


_ROAD_CACHE = {}


def _mslarge(x):
    m, s = float(x.mean()), float(x.std())
    return 10.0 - 9.0 * np.where(x > m, 1.0 - s / (x - m + s), 0.0)


def _decile_cost(d):
    """10-class reverse reclassification of distance by rank (deciles): nearest decile = 10."""
    from scipy.stats import rankdata
    f = np.isfinite(d)
    out = np.full(d.shape, np.nan)
    r = rankdata(d[f]) / f.sum()
    out[f] = 11.0 - np.clip(np.ceil(r * 10.0), 1, 10)
    return out


def _linear_cost(x):
    """Linear rescale 10 -> 1 (smallest value costs 10, largest 1), the same direction as MSLarge
    10 -> 1 (the task text leaves the rescaling function open, so linear is accepted next to MSLarge)."""
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    return 10.0 - 9.0 * (x - lo) / (hi - lo) if hi > lo else np.full(x.shape, np.nan)


def _road_variants(ctx, transform, shape):
    """MSLarge distance-to-roads cost on the subsampled grid for road subsets
    (FUNC_CLASS <= 4 and <= 3; the full set comes from the gold run)."""
    key = (ctx.inputs_dir, shape)
    if key in _ROAD_CACHE:
        return _ROAD_CACHE[key]
    import geopandas as gpd
    from rasterio.features import rasterize
    from scipy.ndimage import distance_transform_edt
    out = {}
    try:
        roads = gpd.read_file(f"{ctx.inputs_dir}/Roads.geojson").to_crs(26911)
        fc = roads["FUNC_CLASS"].astype(int)
        t = transform * transform.scale(SUB)
        for k in (4, 3):
            sub = roads[fc <= k]
            r = rasterize(((g, 1) for g in sub.geometry if g is not None), out_shape=shape, transform=t,
                          fill=0, dtype="uint8", all_touched=True)
            out[f"roads FUNC_CLASS<={k}"] = _mslarge(distance_transform_edt(r == 0) * abs(t.a))
    except Exception:
        pass
    _ROAD_CACHE[key] = out
    return out


def _fit(y, X):
    """Standardised multiple regression of y on columns of X: (R^2, betas, partial correlations)."""
    m = np.isfinite(y) & np.isfinite(X).all(1)
    if m.sum() < 1000 or y[m].std() == 0:
        return -1.0, None, None
    Xm, ym = X[m], y[m]
    Z = (Xm - Xm.mean(0)) / Xm.std(0)
    yz = (ym - ym.mean()) / ym.std()
    A = np.c_[np.ones(len(Z)), Z]
    b, *_ = np.linalg.lstsq(A, yz, rcond=None)
    pc = []
    for j in range(Z.shape[1]):
        o = np.delete(A, j + 1, 1)
        ry = yz - o @ np.linalg.lstsq(o, yz, rcond=None)[0]
        rx = Z[:, j] - o @ np.linalg.lstsq(o, Z[:, j], rcond=None)[0]
        pc.append(float(np.corrcoef(ry, rx)[0, 1]) if ry.std() > 0 and rx.std() > 0 else 0.0)
    return float(1 - ((yz - A @ b) ** 2).mean()), b[1:], np.array(pc)


def _same_aspect(shp):
    return len(shp) == 2 and shp[0] >= 300 and abs(shp[0] / shp[1] - SHAPE[0] / SHAPE[1]) < 0.015 \
        and shp[0] <= SHAPE[0] + 2


def _densify(lines, step=30.0):
    out = []
    for g in lines:
        if g is None or g.is_empty:
            continue
        parts = getattr(g, "geoms", [g])
        for p in parts:
            if p.length == 0:
                continue
            n = max(2, int(p.length / step))
            out += [p.interpolate(t, normalized=True).coords[0] for t in np.linspace(0, 1, n)]
    return np.array(out) if out else np.zeros((0, 2))


def _xy_from_rc(rc, transform):
    rows, cols = rc[:, 0], rc[:, 1]
    x = transform.c + (cols + 0.5) * transform.a
    y = transform.f + (rows + 0.5) * transform.e
    return np.c_[x, y]


def _candidate_paths(ctx, transform, bounds):
    """(label, points[N,2] in UTM 11N metres)."""
    import geopandas as gpd
    out = []
    in_map = lambda p: (p[:, 0].min() >= bounds.left - 1e3 and p[:, 0].max() <= bounds.right + 1e3 and
                        p[:, 1].min() >= bounds.bottom - 1e3 and p[:, 1].max() <= bounds.top + 1e3)
    in_pix = lambda p: (p.min() >= -1 and p[:, 0].max() <= SHAPE[0] + 1 and p[:, 1].max() <= SHAPE[1] + 1)
    for tag, meta, load in ctx.objects("pred"):
        k = meta.get("kind")
        try:
            if k == "gdf":
                g = load()
                g = g[g.geometry.notna()]
                g = g[g.geom_type.isin(["LineString", "MultiLineString"])]
                if len(g) == 0:
                    continue
                if g.crs is None and meta.get("source") == "file":
                    continue  # no CRS guessing for files written by the run
                if g.crs is not None and g.crs.to_epsg() != 26911:
                    g = g.to_crs(26911)
                out.append((f"{meta['name']}@{tag}", _densify(g.geometry)))
            elif k == "array":
                shp = meta.get("shape") or []
                if len(shp) == 2 and _same_aspect(shp):
                    a = np.asarray(load())
                    nz = np.argwhere(np.nan_to_num(a.astype(float)) > 0)
                    if 0 < len(nz) <= 0.02 * a.size:  # sparse raster = path cells (any grid factor)
                        f = SHAPE[0] / shp[0]
                        out.append((f"{meta['name']}@{tag} (path raster)",
                                    _xy_from_rc((nz + 0.5) * f - 0.5, transform)))
                elif len(shp) == 2 and shp[1] == 2 and shp[0] >= 20:
                    p = np.asarray(load(), float)
                    p = p[np.isfinite(p).all(1)]
                    if len(p) < 20:
                        continue
                    if in_map(p):
                        out.append((f"{meta['name']}@{tag} (xy)", p))
                    elif in_pix(p):  # (row, col) or (col, row) pixel coordinates
                        out.append((f"{meta['name']}@{tag} (row,col)", _xy_from_rc(p, transform)))
                        out.append((f"{meta['name']}@{tag} (col,row)", _xy_from_rc(p[:, ::-1], transform)))
                    elif in_pix(p[:, ::-1]):
                        out.append((f"{meta['name']}@{tag} (col,row)", _xy_from_rc(p[:, ::-1], transform)))
        except Exception:
            continue
    # all lines drawn in one figure snapshot together (one line per corridor is typical)
    by_tag = {}
    for lab, p in out:
        if "fig_ax" in lab and "_line" in lab:
            by_tag.setdefault((lab.split("@")[1].split(" ")[0], lab.split(" ")[-1]), []).append(p)
    for (tag, kind), ps in by_tag.items():
        if len(ps) > 1:
            out.append((f"all figure lines@{tag} {kind}", np.vstack(ps)))
    return out


def checks(ctx):
    out = []
    names = ["optimal_path.png", "Optimal_path.png", "optimal_path.PNG"]
    r = file_exists(ctx, names)
    out.append(r)
    if r["passed"]:
        nb, std = png_not_blank(ctx.pred(r["value"][0]))
        out.append(result("map not blank", nb, value=std, threshold=">=3 grey-level std"))
    else:
        out.append(result("map not blank", False, note="no optimal_path.png"))
    nm = ["cost surface = weighted sum of ruggedness, road-distance, land-cover and protection costs", "corridors follow gold least-cost paths (recall & precision >= 0.80 within 1.5 km)"]
    try:
        transform, bounds = _grid(ctx)
        gcost, gpaths, comps = _gold(ctx)
        assert gcost is not None and gpaths is not None and all(k in comps for k in ("ruggedness_cost", "road_cost", "landcover_cost", "protected_cost"))
    except Exception as e:
        for n in nm:
            out.append(result(n, False, note=f"gold unavailable {e!r}"[:200]))
        return out

    # 2. cost surface: regression on the four gold component layers
    sub_shape = comps["road_cost"].shape
    other = np.stack([comps[k].ravel() for k in ("landcover_cost", "protected_cost")], 1)
    ruggs = {"MSLarge": comps["ruggedness_cost"]}
    roads = {"all roads": comps["road_cost"]}
    if "distance_to_roads" in comps:   # quantile / equal-count 10-class rescaling
        roads["all roads, decile rank"] = _decile_cost(comps["distance_to_roads"])
        roads["all roads, linear"] = _linear_cost(comps["distance_to_roads"])   # linear rescaling
    if "ruggedness" in comps:
        ruggs["linear"] = _linear_cost(comps["ruggedness"])                     # linear rescaling
    roads.update(_road_variants(ctx, transform, sub_shape))
    best = (None, -9.0, None, None)
    seen = 0
    for tag, meta, load in ctx.objects("pred", "array"):
        shp = tuple(meta.get("shape") or [])
        if not _same_aspect(shp):
            continue
        try:
            a = np.asarray(load(), float)
            a = a[::SUB, ::SUB] if a.shape == SHAPE else resample_to(a, SHAPE)[::SUB, ::SUB]
        except Exception:
            continue
        if a.shape != sub_shape or np.isfinite(a).mean() < 0.3:
            continue
        seen += 1
        y = a.ravel()
        fits = []
        for gn, gc in ruggs.items():
            for rn, rc in roads.items():
                r2, b, pc = _fit(y, np.c_[gc.ravel(), other, rc.ravel()])
                if b is not None:
                    sc = min(r2 - R2_MIN, float(b.min()) - BETA_MIN, float(pc.min()) - PCORR_MIN)   # >= 0 means pass
                    fits.append((sc, r2, f"{rn}; ruggedness {gn}", b, pc))
        if not fits:
            continue
        # the thresholds may be met by ANY offered variant combination. Layer presence is read from the
        # max-R^2 fit: every partial r of that fit must reach PCORR_MIN, so a mismatched variant
        # cannot be picked to let another layer absorb a missing one (control
        # wrong_no_protected_status_with_roads: with the decile road layer the absent protection
        # layer reaches partial r 0.41, in the max-R^2 fit 0.02).
        top = max(fits, key=lambda f: f[1])
        score, r2, rn, b, pc = max(fits, key=lambda f: (f[0], f[1]))
        score = min(score, float(top[4].min()) - PCORR_MIN)
        if top[2] != rn:
            rn = f"{rn}; presence from max-R2 fit ({top[2]}): partial r {np.round(top[4], 2).tolist()}"
        if score > best[1]:
            best = (f"{meta['name']}@{tag}", score, (r2, b, pc), rn)
    lab, score, fit, rn = best
    if lab:
        r2, b, pc = fit
        val = (f"R2={r2:.3f}; weights rug={b[0]:.2f} landcover={b[1]:.2f} protected={b[2]:.2f} "
               f"roads={b[3]:.2f} ({rn}); partial r {np.round(pc, 2).tolist()}")
    else:
        val = None
    out.append(result(nm[0], score >= 0, value=val, threshold=f"R2>={R2_MIN}, every weight>={BETA_MIN}, every partial r>={PCORR_MIN}",
                      note=lab or f"no 2-D surface on the input grid ({seen} candidates)"))

    # 3. paths
    import geopandas as gpd
    from scipy.spatial import cKDTree
    hab = gpd.read_file(f"{ctx.inputs_dir}/habitats.geojson").to_crs(26911)
    near = hab.buffer(1000).union_all()
    import shapely
    def away(p):
        p = np.asarray(p, float).reshape(-1, 2)
        p = p[np.isfinite(p).all(axis=1)]   # NaN vertices would crash cKDTree
        if len(p) == 0:
            return p
        return p[~shapely.contains_xy(near, p[:, 0], p[:, 1])]
    G = away(_densify(gpaths.to_crs(26911).geometry))
    bestp = (None, -1.0, None, None)
    fig_ok = []   # figure-drawn candidates (lines, images) that pass the check-3 criterion
    for lab, P in _candidate_paths(ctx, transform, bounds):
        P = away(P)
        if len(P) < 10:
            continue
        rec = float((cKDTree(P).query(G)[0] <= RADIUS).mean())
        prec = float((cKDTree(G).query(P)[0] <= RADIUS).mean())
        ok = (rec >= 0.80 and prec >= 0.80) or (rec >= 0.90 and prec >= 0.35)
        key = (ok, min(rec, prec))
        if bestp[0] is None or key > bestp[1]:
            bestp = (lab, key, rec, prec)
        if ok and (lab.startswith("fig_ax") or lab.startswith("all figure lines")):
            fig_ok.append(f"{lab} (recall={rec:.2f}, precision={prec:.2f})")
    lab, key, rec, prec = bestp
    path_ok = bool(lab) and key[0]
    out.append(result(nm[1], path_ok, value=f"recall={rec:.2f}, precision={prec:.2f}" if lab else None,
                      threshold="recall>=0.80 and precision>=0.80, or recall>=0.90 and precision>=0.35 (superset network)",
                      note=lab or "no path geometry / coordinates / path raster captured"))

    # 4. corridors drawn in the map
    # every figure candidate is tested, not only the best candidate of check 3 (a saved geojson /
    # path raster usually scores above the same lines drawn with ax.plot); a drawn image whose
    # sparse cells are the path raster (ax.imshow of a path mask) is a figure candidate as well
    # ('fig_ax<i>_image<j> (path raster)')
    drawn_ok = bool(fig_ok)
    note = fig_ok[0] if drawn_ok else ""
    if not drawn_ok:
        bestc = (None, -1.0)
        for tag, meta, arr in drawn_arrays(ctx):
            if not meta["name"].endswith("_paths_bbox"):
                continue
            bb = np.asarray(arr, float).reshape(-1, 4)
            bb = bb[np.isfinite(bb).all(1)]
            if not (1 <= len(bb) <= 200):
                continue
            variants = []
            if bb[:, 0].min() >= bounds.left - 5e3 and bb[:, 2].max() <= bounds.right + 5e3 and \
                    bb[:, 1].min() >= bounds.bottom - 5e3 and bb[:, 3].max() <= bounds.top + 5e3:
                variants.append(bb)
            elif bb.min() >= -2 and bb[:, 2].max() <= SHAPE[1] + 2 and bb[:, 3].max() <= SHAPE[0] + 2:
                x0 = transform.c + bb[:, 0] * transform.a
                x1 = transform.c + bb[:, 2] * transform.a
                y0 = transform.f + bb[:, 3] * transform.e
                y1 = transform.f + bb[:, 1] * transform.e
                variants.append(np.c_[x0, y0, x1, y1])
            for v in variants:
                e = v + np.array([-RADIUS, -RADIUS, RADIUS, RADIUS])
                inside = ((G[:, None, 0] >= e[None, :, 0]) & (G[:, None, 0] <= e[None, :, 2]) &
                          (G[:, None, 1] >= e[None, :, 1]) & (G[:, None, 1] <= e[None, :, 3]))
                cov = float(inside.any(1).mean())
                prec_b = float(inside.any(0).mean())
                if min(cov, prec_b) > bestc[1]:
                    bestc = (f"{meta['name']}@{tag} (cover={cov:.2f}, boxes on corridors={prec_b:.2f})", min(cov, prec_b))
        drawn_ok = bestc[1] >= 0.80
        note = bestc[0] or "no drawn lines / collections found"
    out.append(result("map draws the corridors", drawn_ok, threshold="figure lines pass check 3, or drawn path boxes cover/precision >= 0.80",
                      note=note))
    return out
