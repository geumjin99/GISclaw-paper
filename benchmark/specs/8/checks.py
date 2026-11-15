"""Task 8 - fire station coverage gaps in Etobicoke (buffer + difference overlay).

The buffer distance is not given in the task (the benchmark's script uses 2.5 km). Checks infer
the agent's distance d from its gap layer and verify the geometry for that d:
Correctness (required):
  1. Fire_Service_Analysis.png exists and is not blank;
  2. a captured polygon layer equals Etobicoke minus the union of d-metre buffers around ALL
     stations, for some d in 0.4-5 km (area IoU >= 0.90 after reprojection to UTM 17N; d searched
     in 50 m steps; layers that are just the district outline - IoU >= 0.85 with it and not
     matched better (by > 0.02) by a gap reference - are ignored).
     This rejects: buffering in degrees (ellipses / world-sized buffers), using only
     the stations inside Etobicoke (border areas served from neighbouring districts appear as
     gaps), reversed difference (buffers minus Etobicoke), or mapping the buffers instead of gaps.
     The inferred true radius d must equal a buffer distance stated in the code (constant
     .buffer() arguments, resolved through simple assignments; +-10 %), which rejects
     unit/CRS errors that scale every buffer uniformly (2500 m in Web Mercator = 1.8 km true).
     Captured geometries without CRS (shapely objects) are tried in the CRSs used by the run.
  Also for (2): the gap layer must not be better explained by buffers drawn in DEGREES
     (stations buffered in EPSG:4326: ellipses ~0.72 x 1 of the N-S radius at 43.7 N); a
     degree reference Etobicoke - U buffer(r deg) is searched like the metric one, and a layer
     whose best degree IoU exceeds its best metric IoU by > 0.02 fails (with 1 km buffers the gap
     is ~60 % of the district, so IoU alone cannot see the ellipses).
  3. the saved map DRAWS the gap: the figure captured at the last savefig of
     Fire_Service_Analysis.png (save snapshots mapped onto the literal savefig calls in order;
     any save snapshot if ambiguous) holds a polygon/line collection whose path bounding boxes
     reproduce the parts of a passing gap layer (>= 90 % of its area; layer coordinates tried
     natively and in the CRSs of the run). A map of only the buffers fails.
     The complement drawing is accepted too - the district drawn as a filled base and
     the covered part (district minus the gap = buffers clipped to the district) drawn over it in
     later polygon collections of the same axes (union rebuilt from the captured vertices, area
     IoU >= 0.90 with the covered part, and district minus that union IoU >= 0.90 with the gap
     layer; redraws of the district itself are not counted).
     Unclipped buffers over the district fail. Every binding of a name is a candidate
     stated distance, and layers whose radius matches a stated distance rank first.
Reported, not required:
  4. d within 1.5-3 km (typical urban fire-response distance; gold 2.5 km).
"""
import numpy as np

from checklib import file_exists, png_not_blank, result

CRS_M = "EPSG:32617"
SUBSET_SHARE = 0.005


def stated_buffer_distances(code):
    """Numeric buffer distances written in the code: constant arguments of .buffer(...) calls,
    resolving simple name bindings (DIST = 2500) and constant arithmetic (2.5 * 1000).
    Unresolvable expressions (calls, computed values) are skipped."""
    import ast
    try:
        tree = ast.parse(code)
    except Exception:
        return []
    # every binding of a name is a candidate (R = 2500 ... R = 1500, the map redone with the
    # second value: the true radius is 1500 m)
    env = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            env.setdefault(n.targets[0].id, []).append(n.value)

    def ev(e, depth=0):
        """Set of values the expression can take over the candidate bindings."""
        if depth > 5:
            raise ValueError
        if isinstance(e, ast.Constant) and isinstance(e.value, (int, float)) and not isinstance(e.value, bool):
            return {float(e.value)}
        if isinstance(e, ast.Name) and e.id in env:
            vals = set()
            for b in env[e.id]:
                try:
                    vals |= ev(b, depth + 1)
                except Exception:
                    pass
            if not vals:
                raise ValueError
            return vals
        if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.USub):
            return {-x for x in ev(e.operand, depth + 1)}
        if isinstance(e, ast.BinOp) and isinstance(e.op, (ast.Mult, ast.Div, ast.Add, ast.Sub)):
            A, B = ev(e.left, depth + 1), ev(e.right, depth + 1)
            f = {ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b if b else float("nan"),
                 ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b}[type(e.op)]
            return {f(a, b) for a in A for b in B}
        raise ValueError
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "buffer":
            arg = n.args[0] if n.args else next((k.value for k in n.keywords if k.arg == "distance"), None)
            if arg is None:
                continue
            try:
                vs = ev(arg)
            except Exception:
                continue
            out.extend(v for v in sorted(vs) if v == v and v > 0)
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


def _drawn_share(ctx, tag, layer, crss):
    """Area share of the layer's polygon parts whose bounding box equals the bounding box of a
    path drawn in the figure of snapshot `tag` (polygon/line collections), trying the layer's
    own coordinates and its reprojection to every CRS of the run."""
    from scipy.spatial import cKDTree
    boxes = []
    for t, meta, load in ctx.objects("pred", "array", include_files=False):
        if t == tag and meta.get("source") == "figure" and meta["name"].endswith("_paths_bbox") \
                and meta.get("artist") in ("PatchCollection", "LineCollection", "PolyCollection", "PathCollection"):
            try:
                b = np.asarray(load(), float).reshape(-1, 4)
                boxes.append(b[np.isfinite(b).all(axis=1)])
            except Exception:
                pass
    if not boxes:
        return 0.0
    B = np.vstack(boxes)
    best = 0.0
    variants = [layer]
    if layer.crs is not None:
        for c in crss:
            try:
                variants.append(layer.to_crs(c))
            except Exception:
                pass
    for g in variants:
        parts = g.geometry.explode(index_parts=False)
        parts = parts[parts.geom_type == "Polygon"]
        if not len(parts):
            continue
        pb = np.asarray(parts.bounds, float)
        ar = parts.area.to_numpy()
        fin = np.isfinite(pb).all(axis=1)   # empty parts have NaN bounds
        if not fin.any() or not len(B):
            continue
        pb, ar = pb[fin], ar[fin]
        ext = max(np.ptp(pb[:, [0, 2]]), np.ptp(pb[:, [1, 3]]), 1e-9)
        d, _ = cKDTree(B).query(pb, k=1)
        hit = d <= 1e-6 * ext + 1e-9
        best = max(best, float(ar[hit].sum() / max(ar.sum(), 1e-12)))
    return best


def _coll_geometry(vv):
    """Polygon area of a captured collection from its vertices (cols: path index, x, y; codes are not
    captured). Each path's rings are split where a vertex returns to the ring's first vertex and are
    combined even-odd (holes cancel), then the paths are unioned. None if nothing can be rebuilt."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    vv = np.asarray(vv, float).reshape(-1, 3)
    vv = vv[np.isfinite(vv).all(axis=1)]
    geoms = []
    for q in np.unique(vv[:, 0]):
        pts = vv[vv[:, 0] == q, 1:]
        g, k = None, 0
        while k < len(pts):
            s = pts[k]
            same = np.flatnonzero((pts[k + 1:] == s).all(axis=1))
            e = k + 1 + same[0] if len(same) else len(pts) - 1
            ring = pts[k:e + 1]
            k = e + 1
            if len(ring) >= 4:
                try:
                    p = Polygon(ring).buffer(0)
                except Exception:
                    continue
                if not p.is_empty:
                    g = p if g is None else g.symmetric_difference(p)
        if g is not None and not g.is_empty:
            geoms.append(g)
    return unary_union(geoms) if geoms else None


def _drawn_complement_share(ctx, tag, layer, district, crss):
    """Complement drawing: the district is drawn as a filled base and the COVERED part
    (district minus this gap layer = district intersected with the buffers) is drawn over it, so the
    gap is the base fill left showing. Both must be polygon collections on the same axes, the
    district parts first (>= 90 % of their area bbox-matched in one collection, facecolour not fully
    transparent where captured), then the coverage in later collections. The union of those later
    collections (rebuilt from the captured vertices) must have area IoU >= 0.90 with the covered
    part; returns the area IoU of the implied gap (district minus that union) with the gap layer.
    Later collections that are a redraw of the district itself (rebuilt geometry closer to the
    district than to the covered part, e.g. an outline with facecolor='none') are left out: at
    2.5-3 km the covered part is one polygon with the district's bbox, so a bbox hit alone would admit
    any district redraw. Unclipped buffers, outlines only, or coverage at
    another radius do not match."""
    import re
    from scipy.spatial import cKDTree
    colls, verts, fcs = {}, {}, {}
    for t, meta, load in ctx.objects("pred", "array", include_files=False):
        if t != tag or meta.get("source") != "figure" or meta.get("artist") not in ("PatchCollection", "PolyCollection"):
            continue
        m = re.fullmatch(r"fig_ax(\d+)_coll(\d+)_(paths_bbox|vertices|facecolors)", meta.get("name", ""))
        if not m:
            continue
        key = (int(m.group(1)), int(m.group(2)))
        try:
            a = np.asarray(load(), float)
        except Exception:
            continue
        if m.group(3) == "paths_bbox":
            b = a.reshape(-1, 4)
            colls[key] = b[np.isfinite(b).all(axis=1)]
        elif m.group(3) == "vertices":
            verts[key] = a
        else:
            fcs[key] = a
    if len(colls) < 2 or layer.crs is None:
        return 0.0
    variants = [layer.crs] + list(crss)

    def hit_share(gs, B):
        parts = gs.explode(index_parts=False)
        parts = parts[parts.geom_type == "Polygon"]
        if not len(parts) or not len(B):
            return 0.0
        pb = np.asarray(parts.bounds, float)
        ar = parts.area.to_numpy()
        fin = np.isfinite(pb).all(axis=1)
        if not fin.any():
            return 0.0
        pb, ar = pb[fin], ar[fin]
        ext = max(np.ptp(pb[:, [0, 2]]), np.ptp(pb[:, [1, 3]]), 1e-9)
        d, _ = cKDTree(B).query(pb, k=1)
        return float(ar[d <= 1e-6 * ext + 1e-9].sum() / max(ar.sum(), 1e-12))

    def iou(a, b):
        u = a.union(b).area
        return a.intersection(b).area / u if u > 0 else 0.0

    rebuilt = {}
    for k, vv in verts.items():
        try:
            rebuilt[k] = _coll_geometry(vv)
        except Exception:
            rebuilt[k] = None
    best = 0.0
    for c in variants:
        try:
            g = layer.to_crs(c).geometry.buffer(0).union_all()
            dist = district.to_crs(c).geometry.buffer(0).union_all()
            dparts = district.to_crs(c).geometry
            cov = dist.difference(g)
        except Exception:
            continue
        if cov.is_empty:
            continue
        import geopandas as gpd
        dist_gs = gpd.GeoSeries([dist], crs=c)
        for (i, jd), Bd in colls.items():
            if max(hit_share(dist_gs, Bd), hit_share(dparts, Bd)) < 0.9:
                continue
            fc = fcs.get((i, jd))
            if fc is not None and fc.ndim == 2 and fc.shape[1] == 4 and not (fc[:, 3] > 0).any():
                continue   # outline-only base (facecolours captured only for >= 2 patches)
            later = []
            for (ii, j), gj in rebuilt.items():
                if ii != i or j <= jd or gj is None or gj.is_empty:
                    continue
                if iou(gj, dist) >= iou(gj, cov):
                    continue   # a redraw of the district itself, not the covered part
                later.append(gj)
            if later:
                from shapely.ops import unary_union
                U = unary_union(later)
                # Score what the complement SHOWS, the implied gap
                # (district minus the drawn coverage), against the gap layer; at 2.5 km the gap is
                # ~2 % of the district, so IoU with the covered part alone would admit coverage at 2-2.25 km.
                # The coverage must still match the covered part (rejects unclipped buffers).
                if iou(U, cov) >= 0.9:
                    best = max(best, iou(dist.difference(U), g))
    return best


def checks(ctx):
    import geopandas as gpd
    out = []
    names = ["Fire_Service_Analysis.png"]
    r = file_exists(ctx, names)
    out.append(r)
    if r["passed"]:
        nb, std = png_not_blank(ctx.pred(names[0]))
        out.append(result("map not blank", nb, value=std, threshold=">=3 grey-level std"))
    else:
        out.append(result("map not blank", False, note="no map"))
    try:
        st = gpd.read_file(f"{ctx.inputs_dir}/fire_station.geojson").to_crs(CRS_M)
        area = gpd.read_file(f"{ctx.inputs_dir}/etobicoke.geojson").to_crs(CRS_M).geometry.union_all()
    except Exception as e:
        out.append(result("coverage-gap polygons match Etobicoke minus station buffers (IoU >= 0.90)", False,
                          note=repr(e)[:200]))
        return out
    ds = np.arange(400, 5001, 50)   # lower bound 400 m (same as task 44)
    refs = {}

    def gap(d):
        if d not in refs:
            refs[d] = area.difference(st.buffer(d, 32).union_all())
        return refs[d]

    st_deg = gpd.read_file(f"{ctx.inputs_dir}/fire_station.geojson").to_crs("EPSG:4326")
    area_deg = gpd.read_file(f"{ctx.inputs_dir}/etobicoke.geojson").to_crs("EPSG:4326").geometry.union_all()
    rs = np.round(np.arange(0.0035, 0.0451, 0.0005), 4)   # from ~0.3-0.4 km
    refs_deg = {}

    def gap_deg(r):
        if r not in refs_deg:
            gd = area_deg.difference(st_deg.buffer(r, 32).union_all())
            refs_deg[r] = gpd.GeoSeries([gd], crs="EPSG:4326").to_crs(CRS_M).iloc[0]
        return refs_deg[r]

    # the inferred true radius must be the distance the code states (a 2500 m buffer drawn in Web
    # Mercator is really ~1.8 km; feet/metre mix-ups likewise shrink/grow every buffer)
    stated = [x for x in stated_buffer_distances(ctx.code_text()) if x >= 50]
    stated += [x * 1000 for x in stated_buffer_distances(ctx.code_text()) if 0.4 <= x <= 10]  # km literals

    def dist_match(d):
        return d is not None and (not stated or any(abs(d - x) <= max(0.1 * x, 100) for x in stated))

    best = (0.0, None, None)
    best_key = None
    layers = []      # (metric iou, d, label, native layer, degree iou, r)
    seen = set()
    crss = sorted({m.get("crs") for _, m, _ in ctx.objects("pred", "gdf", include_files=False)
                   if m.get("crs") and m.get("crs") != "None"})
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
            if g.crs is None:   # e.g. a captured shapely geometry: try the CRSs used in the run
                for c in crss:
                    try:
                        gg = g.set_crs(c, allow_override=True).to_crs(CRS_M)
                    except Exception:
                        continue
                    if gg.geometry.union_all().intersects(area):
                        g = gg
                        break
                if g.crs is None:
                    continue
            u = g.to_crs(CRS_M).geometry.buffer(0).union_all()
            if u.is_empty or u.area > 2 * area.area:
                continue
            # the district outline itself is not a gap layer. With short radii
            # (0.4-0.6 km) the true gap covers ~90 % of the district, so a layer close to the
            # outline is kept when the gap reference explains it better (checked after the search)
            oi = u.intersection(area).area / u.union(area).area
            # coarse search then refine
            def iou(d):
                ref = gap(d)
                un = u.union(ref).area
                return u.intersection(ref).area / un if un > 0 else 0.0
            # pre-screen by area: gap area is monotone in d
            ua = u.area
            areas = np.array([gap(d).area for d in ds[::4]])
            k = int(np.argmin(np.abs(areas - ua)))
            cand = ds[max(0, 4 * k - 8): 4 * k + 9]
            lb = (0.0, None)
            for d in cand:
                v = iou(int(d))
                if v > lb[0]:
                    lb = (v, int(d))
            # same search with buffers drawn in degrees (EPSG:4326)
            def iou_deg(r):
                ref = gap_deg(r)
                un = u.union(ref).area
                return u.intersection(ref).area / un if un > 0 else 0.0
            dareas = np.array([gap_deg(r).area for r in rs[::4]])
            kd = int(np.argmin(np.abs(dareas - ua)))
            db = (0.0, None)
            for r in rs[max(0, 4 * kd - 8): 4 * kd + 9]:
                v = iou_deg(float(r))
                if v > db[0]:
                    db = (v, float(r))
            if oi >= 0.85 and not lb[0] > oi + 0.02:
                continue   # explained as well by the district outline: not a gap layer
            # subset constraint (same as task 44): a gap built from a subset of the
            # stations keeps areas within reach of the omitted ones; <= 0.5 % of the layer may lie
            # within 0.9 d of any station
            share = 1.0
            if lb[1] is not None:
                ui = u.intersection(area)
                share = float(ui.intersection(st.buffer(0.9 * lb[1], 32).union_all()).area / ui.area) \
                    if ui.area > 0 else 1.0
            layers.append((lb[0], lb[1], f"{meta['name']}@{tag}", g, db[0], db[1], share))
            # rank layers by metric IoU, a degree-explained layer only after every metric one
            okm = not db[0] > lb[0] + 0.02
            oks = share <= SUBSET_SHARE
            # a layer whose radius matches the stated distance ranks first (otherwise an
            # intermediate sensitivity layer, e.g. 1 km inside a radius loop, outranks the stated
            # 2 km gap and then fails the distance test)
            key = (okm and oks and lb[0] >= 0.9 and dist_match(lb[1]), lb[0])
            if best_key is None or key > best_key:
                best, best_key = (lb[0], lb[1], f"{meta['name']}@{tag}", okm, oks, share), key
        except Exception:
            continue
    v, d, w = best[:3]
    not_deg = best[3] if len(best) > 3 else True
    not_subset = best[4] if len(best) > 4 else True
    dist_ok = True
    if d is not None and stated:
        dist_ok = any(abs(d - x) <= max(0.1 * x, 100) for x in stated)
    note = w or "no polygon layer"
    if stated:
        note += f"; stated buffer distance(s) {sorted(set(round(x) for x in stated))} m"
    if len(best) > 5:
        note += f"; {100 * best[5]:.2f} % of the gap within 0.9 d of a station"
    if not not_subset:
        note += " -> areas served by some stations left in the gap (subset of stations buffered)"
    if not not_deg:
        note += " -> better explained by buffers drawn in degrees (EPSG:4326): CRS error"
    out.append(result("coverage-gap polygons match Etobicoke minus station buffers (IoU >= 0.90)",
                      v >= 0.9 and dist_ok and not_deg and not_subset,
                      value={"iou": round(v, 3), "buffer_m": d}, threshold="IoU 0.9; true radius = stated distance +-10 %; <= 0.5 % of the gap within 0.9 d of a station",
                      note=note if dist_ok else note + " -> true radius differs from the stated distance (CRS/unit error)"))
    deg_layers = [l for l in layers if l[4] > l[0] + 0.02]
    if not not_deg and deg_layers:
        l = max(deg_layers, key=lambda x: x[4])
        out[-1]["note"] += f" ({l[2]}: degree-buffer IoU {l[4]:.3f} at r = {l[5]} deg vs metric {l[0]:.3f})"

    # --- 3. the saved map draws the gap --------------------------------------------------------------
    passing = [l for l in layers if l[0] >= 0.9 and not l[4] > l[0] + 0.02 and l[1] is not None
               and l[6] <= SUBSET_SHARE
               and (not stated or any(abs(l[1] - x) <= max(0.1 * x, 100) for x in stated))]
    tags, how = _saved_map_tags(ctx, names[0])
    if tags is None:   # ambiguous: any saved figure
        import os as _os
        import re as _re
        _r = ctx.pred_snap
        tags = sorted(x for x in _os.listdir(_r) if _re.fullmatch(r"save\d+", x)) \
            if _r and _os.path.isdir(_r) else []
    crss_all = sorted(set(crss) | {CRS_M, "EPSG:4326"})
    share, where_d = 0.0, None
    for t in tags:
        for l in passing:
            try:
                sh_ = _drawn_share(ctx, t, l[3], crss_all)
            except Exception:
                sh_ = 0.0
            if sh_ > share:
                share, where_d = sh_, f"{l[2]} drawn in {t}"
    if share < 0.9:   # complement drawing: district base + covered part over it
        try:
            dist_gdf = gpd.read_file(f"{ctx.inputs_dir}/etobicoke.geojson")
        except Exception:
            dist_gdf = None
        for t in (tags if dist_gdf is not None else []):
            for l in passing:
                try:
                    sh_ = _drawn_complement_share(ctx, t, l[3], dist_gdf, crss_all)
                except Exception:
                    sh_ = 0.0
                if sh_ > share:
                    share, where_d = sh_, f"{l[2]} drawn as its complement (district base + covered part) in {t}"
    out.append(result("the saved map draws the coverage-gap polygons", share >= 0.9, value=round(share, 3),
                      threshold=">= 90 % of the gap area drawn",
                      note=f"{how}; " + (where_d or ("no passing gap layer" if not passing
                                                      else "gap layer not found among the drawn polygons"))))
    out.append(result("buffer distance 1.5-3 km", d is not None and 1500 <= d <= 3000, value=d,
                      threshold="1500-3000 m", required=False))
    return out
