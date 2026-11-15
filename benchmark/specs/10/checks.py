"""T10 - Deforestation risk from the planned road BR 421, Rondonia.

Correctness (required):
  1. predictedRiskyArea.png exists and is not blank;
  2. the risk analysis: a captured polygon layer matches the 5.5 km buffer of the planned road
     minus the already deforested area (gold "risky area", 1 350 km2) or that area intersected
     with the protected forests (791 km2) or that area minus the protected forests (559 km2)
     - area IoU >= 0.85 after reprojection. The bare buffer
     (IoU 0.58), a wrong distance, a buffer in degrees or the wrong overlay direction
     (deforested part instead of the remaining forest) fail;
     In both cases <= 3 % of the candidate may lie on already-deforested land (gold buffer minus
     gold risky area): "protected forest inside the buffer" without the deforestation overlay
     (IoU 0.887 with the protected product, 11 % deforested) fails;
     A captured layer without CRS (shapely geometry) is tried in the CRSs used in
     the run (the first that puts it on the planned-road buffer); files without CRS are not used;
  3. the context layers named in the task are DRAWN: the code plots layers derived directly from
     roads.geojson, protectedForest.geojson and deforestedArea.geojson (read / to_crs / explode /
     make_valid / filter / clip - not a product that merely used them). Single-colour layers
     leave no data in the figure capture, so this is decided on the code (AST).
     The code reading follows conditional
     expressions (`R if c else R.cx[...]`) and user functions that return their first argument
     (`deforested = fix(raw).to_crs(...)`); and a layer also counts as drawn when the capture
     recorded a GeoDataFrame/GeoSeries .plot of it (>= 90 % of the drawn features have an input
     feature's bounding box, >= 50 % of the input's features) - for layers built in a loop into a
     dict and unpacked, which the code reading cannot follow.
     The code reading also follows the
     layer dissolved / split in function form (shapely.union_all / unary_union / get_parts /
     list of the layer, not of a literal list of layers); and a layer counts as drawn when a
     plain line artist (ax.plot of a coordinate array, LineCollection) has >= 90 % of its
     vertices on input vertices and reaches >= 50 % of the input's features and >= 25 % of
     its vertices.
Reported, not required:
  4. a Rondonia boundary layer (state-sized polygon, ~240 000 km2) is drawn - no boundary file is
     supplied, it must come from elsewhere (e.g. Natural Earth via cartopy).
"""
import numpy as np

from checklib import file_exists, png_not_blank, result

CRS_M = "ESRI:102033"


# ----------------------------------------------------------------------------- what is plotted
# Collections drawn with a single colour carry no data in the figure capture, so whether a context
# layer is drawn is read from the code (AST): a `.plot()` whose receiver is a layer derived
# *directly* from the input file - read, to_crs, explode, make_valid, filtering, clip(layer, ...) -
# and not a product that merely used it (e.g. buffer.difference(deforested)).
_READERS = {"read_file", "read_dataframe", "clip", "GeoDataFrame", "GeoSeries", "explode", "dissolve"}
# the layer dissolved / split into parts in
# function form - shapely.union_all(L.geometry), unary_union(L.geometry), shapely.get_parts(...),
# list(...) - is still the layer. Only when the FIRST argument itself is the layer: a literal
# list / tuple (union_all([buffer, deforested])) is a combination, not the layer.
_SAME_LAYER = {"union_all", "unary_union", "get_parts", "list"}


def _direct(e, key, names, helpers=frozenset()):
    import ast
    while True:
        if isinstance(e, ast.Constant) and isinstance(e.value, str):
            return key in e.value
        if isinstance(e, ast.Name):
            return e.id in names
        if isinstance(e, (ast.List, ast.Tuple)):
            return any(_direct(x, key, names, helpers) for x in e.elts)
        if isinstance(e, ast.IfExp):   # `R if c else R.cx[...]`
            return _direct(e.body, key, names, helpers) or _direct(e.orelse, key, names, helpers)
        if isinstance(e, ast.Call):
            fn = e.func.attr if isinstance(e.func, ast.Attribute) else getattr(e.func, "id", "")
            if fn in _READERS or (isinstance(e.func, ast.Name) and fn in helpers):
                args = list(e.args[:1]) + [k.value for k in e.keywords if k.arg in ("geometry", "data", "filename")]
                if any(_direct(a, key, names, helpers) for a in args):
                    return True
            elif fn in _SAME_LAYER and e.args and not isinstance(e.args[0], (ast.List, ast.Tuple, ast.Set)):
                if _direct(e.args[0], key, names, helpers):
                    return True
            e = e.func
            continue
        if isinstance(e, ast.Attribute):
            if e.attr == "crs":
                return False
            e = e.value
            continue
        if isinstance(e, ast.Subscript):
            e = e.value
            continue
        return False


def _passthrough_helpers(tree):
    """Names of user-defined functions that return (a layer derived directly from) their first
    argument, e.g. `def fix(g): g = g.copy(); g["geometry"] = ...; return g`.
    A function returning something else (buf.difference(defo) with defo as first
    argument) is not a pass-through."""
    import ast
    out = set()
    for f in ast.walk(tree):
        if not isinstance(f, ast.FunctionDef) or not f.args.args:
            continue
        names = {f.args.args[0].arg}
        body = sorted([n for n in ast.walk(f) if isinstance(n, ast.Assign)], key=lambda n: (n.lineno, n.col_offset))
        for n in body:
            d = _direct(n.value, "\0", names)
            for t in n.targets:
                if isinstance(t, ast.Name):
                    (names.add if d else names.discard)(t.id)
        rets = [n for n in ast.walk(f) if isinstance(n, ast.Return) and n.value is not None]
        if rets and all(_direct(r.value, "\0", names) for r in rets):
            out.add(f.name)
    return frozenset(out)


def plotted_layers(code, keys):
    """{key: source line of a .plot() of a layer derived directly from that input file}."""
    import ast
    try:
        tree = ast.parse(code)
    except Exception:
        return {}
    helpers = _passthrough_helpers(tree)
    lines = code.splitlines()
    assigns = sorted([n for n in ast.walk(tree) if isinstance(n, ast.Assign)], key=lambda n: (n.lineno, n.col_offset))
    out = {}
    for key in keys:
        names = set()
        for n in assigns:
            tg = [t for t in n.targets if isinstance(t, ast.Name)]
            if not tg:
                continue
            d = _direct(n.value, key, names, helpers)
            for t in tg:
                (names.add if d else names.discard)(t.id)
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "plot":
                if _direct(n.func.value, key, names, helpers):
                    out[key] = lines[n.lineno - 1].strip()[:100] if n.lineno <= len(lines) else ""
                    break
    return out


def drawn_input_layers(ctx, files):
    """{file key: note} for input layers whose drawing was recorded at savefig (capture.py
    records rows / CRS / per-feature bounds of every GeoDataFrame.plot / GeoSeries.plot call).
    A drawn layer is the input layer when >= 90 % of its features
    have the bounding box of an input feature (input reprojected to the drawn CRS, tolerance
    1e-5 of the input extent) and it has >= 50 % of the input's features: the layer itself,
    reprojected / repaired / filtered lightly / clipped to the state - not a product that merely
    used it (a buffer difference, an overlay) and not a single feature of it. Complements the
    code reading for layers built in loops / dicts that the AST cannot follow."""
    import geopandas as gpd
    from scipy.spatial import cKDTree
    from checklib import drawn_geo_layers
    recs = drawn_geo_layers(ctx, file="predictedRiskyArea.png") or drawn_geo_layers(ctx)
    recs = [(t, r, b) for t, r, b in recs if b is not None and r.get("crs") and len(b)]
    out = {}
    if not recs:
        return out
    for key, fname in files.items():
        try:
            inp = gpd.read_file(f"{ctx.inputs_dir}/{fname}")
            inp = inp[inp.geometry.notna() & ~inp.geometry.is_empty]
        except Exception:
            continue
        cache = {}
        for tag, r, b in recs:
            try:
                if r["rows"] < 0.5 * len(inp):
                    continue
                if r["crs"] not in cache:
                    ib = inp.to_crs(r["crs"]).geometry.bounds.to_numpy()
                    ib = ib[np.isfinite(ib).all(1)]
                    ext = max(np.nanmax(ib[:, 2]) - np.nanmin(ib[:, 0]), np.nanmax(ib[:, 3]) - np.nanmin(ib[:, 1]))
                    cache[r["crs"]] = (cKDTree(ib), 1e-5 * ext)
                tree, tol = cache[r["crs"]]
                bb = np.asarray(b, float)
                bb = bb[np.isfinite(bb).all(1)]
                if not len(bb):
                    continue
                dist, _ = tree.query(bb, k=1, p=np.inf, distance_upper_bound=tol)
                share = float(np.isfinite(dist).mean())
                if share >= 0.9:
                    out[key] = f"drawn layer #{r.get('i')}@{tag}: {r['rows']} features, {share:.0%} = {fname} features"
                    break
            except Exception:
                continue
    return out


def drawn_input_lines(ctx, files):
    """{file key: note} for input layers drawn as plain line artists (ax.plot of a coordinate
    array, or a LineCollection), e.g. roads turned by a helper into one NaN-separated array and
    drawn with ax.plot. A line artist shows the input layer
    when >= 90 % of its finite vertices are vertices of the input layer (input reprojected to a
    CRS of a layer drawn in the same run, or kept as is; tolerance 1e-6 of the input extent -
    simplification keeps original vertices) AND it reaches >= 50 % of the input's features AND
    >= 25 % of the input's vertices. A product that merely used the layer (its outline crossing
    a few input vertices) or one feature of it fails the coverage terms."""
    import geopandas as gpd
    import shapely
    from scipy.spatial import cKDTree
    from checklib import drawn_arrays, drawn_geo_layers
    arrs, boxes = [], []
    coll = ("LineCollection", "PolyCollection", "PatchCollection")
    got = list(drawn_arrays(ctx, file="predictedRiskyArea.png") or drawn_arrays(ctx))
    with_verts = {(t, m["name"][:-len("_vertices")]) for t, m, a in got if m["name"].endswith("_vertices")}
    for t, m, a in got:
        a = np.asarray(a, float)
        if m.get("artist") == "line" and a.ndim == 2 and a.shape[1] == 2:
            v = a
        elif m.get("artist") in coll and m["name"].endswith("_vertices") and a.ndim == 2 and a.shape[1] == 3:
            # polygon collections built from the layer's rings too
            v = a[:, 1:]
        elif m.get("artist") in coll and m["name"].endswith("_paths_bbox") and a.ndim == 2 and a.shape[1] == 4 \
                and (t, m["name"][:-len("_paths_bbox")]) not in with_verts:
            # vertices not kept (> 200 000): per-path bounding boxes instead
            b = a[np.isfinite(a).all(1)]
            if len(b) >= 10:
                boxes.append((f"{m['name']}@{t}", b))
            continue
        else:
            continue
        v = v[np.isfinite(v).all(1)]
        if len(v) >= 10:
            arrs.append((f"{m['name']}@{t}", v))
    out = {}
    if not arrs and not boxes:
        return out
    crss = {r.get("crs") for _, r, _ in drawn_geo_layers(ctx) if r.get("crs")}
    for key, fname in files.items():
        try:
            inp = gpd.read_file(f"{ctx.inputs_dir}/{fname}")
            inp = inp[inp.geometry.notna() & ~inp.geometry.is_empty]
        except Exception:
            continue
        for c in [None] + sorted(crss):
            try:
                g = inp.geometry if c is None else inp.geometry.to_crs(c)
                xy, ix = shapely.get_coordinates(g.values, return_index=True)
                ok = np.isfinite(xy).all(1)
                xy, ix = xy[ok], ix[ok]
                if not len(xy):
                    continue
                ext = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))
                tol = 1e-6 * ext
                tin = cKDTree(xy)
                for lab, v in arrs:
                    d, _ = tin.query(v, k=1, p=np.inf, distance_upper_bound=tol)
                    prec = float(np.isfinite(d).mean())
                    if prec < 0.9:
                        continue
                    dd, _ = cKDTree(v).query(xy, k=1, p=np.inf, distance_upper_bound=tol)
                    hit = np.isfinite(dd)
                    vshare = float(hit.mean())
                    fshare = len(np.unique(ix[hit])) / len(g)
                    if fshare >= 0.5 and vshare >= 0.25:
                        out[key] = (f"line artist {lab}: {prec:.0%} of its vertices are {fname} vertices, "
                                    f"{fshare:.0%} of its features / {vshare:.0%} of its vertices drawn")
                        break
                if key in out:
                    break
                # collections without vertices: >= 90 % of the drawn paths have the bounding box of
                # a part of the input layer (tolerance as above), covering >= 50 % of its features
                # and >= 25 % of its parts
                if boxes:
                    parts = g.explode(index_parts=False)
                    pb = parts.bounds.to_numpy(float)
                    pix = parts.index.to_numpy()
                    okp = np.isfinite(pb).all(1)
                    pb, pix = pb[okp], pix[okp]
                    if len(pb):
                        tb = cKDTree(pb)
                        for lab, b in boxes:
                            d, _ = tb.query(b, k=1, p=np.inf, distance_upper_bound=tol)
                            inside = np.isfinite(d)
                            # a path not equal to a part may be a piece of one (make_valid / union
                            # split it): its box lies inside that part's box
                            miss = np.flatnonzero(~inside)[:100000]
                            for k0 in range(0, len(miss), 256):
                                blk = miss[k0:k0 + 256]
                                bb = b[blk]
                                c = ((pb[None, :, 0] <= bb[:, None, 0] + tol) & (pb[None, :, 1] <= bb[:, None, 1] + tol)
                                     & (pb[None, :, 2] >= bb[:, None, 2] - tol) & (pb[None, :, 3] >= bb[:, None, 3] - tol))
                                inside[blk] = c.any(1)
                            prec = float(inside.mean())
                            if prec < 0.9:
                                continue
                            dd, _ = cKDTree(b).query(pb, k=1, p=np.inf, distance_upper_bound=tol)
                            hit = np.isfinite(dd)
                            pshare = float(hit.mean())
                            fshare = len(np.unique(pix[hit])) / len(g)
                            if fshare >= 0.5 and pshare >= 0.25:
                                out[key] = (f"collection {lab}: {prec:.0%} of its paths have the bounds of a "
                                            f"{fname} part, {fshare:.0%} of its features / {pshare:.0%} of its parts drawn")
                                break
                    if key in out:
                        break
            except Exception:
                continue
    return out


def checks(ctx):
    import geopandas as gpd
    out = []
    names = ["predictedRiskyArea.png"]
    r = file_exists(ctx, names)
    out.append(r)
    if r["passed"]:
        nb, std = png_not_blank(ctx.pred(names[0]))
        out.append(result("map not blank", nb, value=std, threshold=">=3 grey-level std"))
    else:
        out.append(result("map not blank", False, note="no map"))

    refs = {}
    for tag, meta, load in ctx.objects("gold", "gdf"):
        if meta.get("name") in ("risky_area", "prot_at_risk", "buffer"):
            try:
                refs[meta["name"]] = load().to_crs(CRS_M).geometry.union_all()
            except Exception:
                pass
    defo_in_buf = None
    if "buffer" in refs and "risky_area" in refs:
        defo_in_buf = refs["buffer"].difference(refs["risky_area"])
    # third reading of "the change" - forest newly exposed
    # outside the protected areas (buffer - deforested - protected forest, ~559 km2), computed
    # from the gold risky area and the protected-forest input
    if "risky_area" in refs:
        try:
            import shapely
            prot = gpd.read_file(f"{ctx.inputs_dir}/protectedForest.geojson").to_crs(CRS_M)
            prot_u = shapely.make_valid(prot.geometry.union_all())
            refs["risky_unprotected"] = refs["risky_area"].difference(prot_u)
        except Exception:
            pass
    best, where = 0.0, None
    run_crss = sorted({m.get("crs") for _, m, _ in ctx.objects("pred", "gdf", include_files=False)
                       if m.get("crs") and m.get("crs") != "None"})
    have = {"roads": False, "protected": False, "deforested": False}
    boundary = False
    seen = set()
    for tag, meta, load in ctx.objects("pred", "gdf"):
        rows = meta.get("rows", 0)
        sig = (meta.get("name"), rows, tuple(meta.get("columns", [])))
        if sig in seen or not rows:
            continue
        seen.add(sig)
        try:
            if rows >= 20000:
                g = load()
                if g.geom_type.isin(["LineString", "MultiLineString"]).mean() > 0.9:
                    have["roads"] = True
                continue
            g = load()
            if g.crs is None and meta.get("source") == "file":
                continue  # no CRS guessing for files written by the run
            if g.crs is None:
                # a captured shapely geometry has no CRS:
                # try the CRSs used in the run, the first one that puts it on the gold
                # buffer of the planned road
                hit = None
                for c in run_crss:
                    try:
                        gg = g.set_crs(c, allow_override=True).to_crs(CRS_M)
                        if "buffer" in refs and gg.geometry.buffer(0).union_all().intersects(refs["buffer"]):
                            hit = gg
                            break
                    except Exception:
                        continue
                if hit is None:
                    continue
                g = hit
            g = g.to_crs(CRS_M)
            area = float(g.area.sum()) / 1e6  # before type filtering: make_valid may give collections
            if rows == 52 or ("Category" in g.columns and rows >= 40):
                have["protected"] = True
            if 50000 <= area <= 120000 and rows <= 10:
                have["deforested"] = True
            if 200000 <= area <= 280000 and rows <= 5:
                boundary = True
            g = g[g.geom_type.isin(["Polygon", "MultiPolygon"])]
            if not len(g):
                continue
            if area > 20000:
                continue
            u = g.geometry.buffer(0).union_all()
            # share of the candidate lying on already-deforested land (buffer minus gold risky)
            defo_share = u.intersection(defo_in_buf).area / u.area if u.area > 0 and defo_in_buf is not None else 1.0
            for k in ("risky_area", "prot_at_risk", "risky_unprotected"):
                if k not in refs:
                    continue
                ref = refs[k]
                inter, uni = u.intersection(ref).area, u.union(ref).area
                v = inter / uni if uni else 0.0
                if defo_share > 0.03:
                    v = min(v, 0.0) if v >= 0.85 else v
                    if inter / uni >= 0.85 and (where is None or best < 0.85):
                        where = f"{meta['name']}@{tag} ~ gold {k} (IoU {inter / uni:.3f}) but {defo_share:.0%} of it is already deforested"
                if v > best:
                    best, where = v, f"{meta['name']}@{tag} ~ gold {k} (deforested share {defo_share:.1%})"
        except Exception:
            continue
    out.append(result("risky area = planned-road buffer minus deforested (or within protected forest)",
                      best >= 0.85, value=round(best, 3), threshold=0.85,
                      note=where or ("gold layers missing" if not refs else "no polygon layer")))
    drawn = plotted_layers(ctx.code_text(), ("roads.geojson", "protectedForest", "deforestedArea"))
    try:  # recorded drawings (capture)
        for k, v in drawn_input_layers(ctx, {"roads.geojson": "roads.geojson",
                                             "protectedForest": "protectedForest.geojson",
                                             "deforestedArea": "deforestedArea.geojson"}).items():
            drawn.setdefault(k, v)
    except Exception:
        pass
    try:  # plain line artists showing an input layer
        miss = {k: f for k, f in {"roads.geojson": "roads.geojson", "protectedForest": "protectedForest.geojson",
                                  "deforestedArea": "deforestedArea.geojson"}.items() if k not in drawn}
        if miss:
            for k, v in drawn_input_lines(ctx, miss).items():
                drawn.setdefault(k, v)
    except Exception:
        pass
    dv = {"roads": "roads.geojson" in drawn, "protected": "protectedForest" in drawn,
          "deforested": "deforestedArea" in drawn}
    out.append(result("context layers drawn (existing roads, protected forest, deforested area)",
                      all(dv.values()), value=dv, note="; ".join(drawn.values())[:300]
                      or "no plot of the input context layers found in the code"))
    out.append(result("Rondonia boundary layer drawn", boundary, required=False))
    return out
