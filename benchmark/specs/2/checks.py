"""Task 2 - public transit access, Hamilton County TN.

Task core: map the bus-stop service area together with three block-group demographic factors
(population density, households below poverty `households_ACSHHBPOV`, households without a
vehicle `AtRisk_ACSOVEH0`), each with its own transparent colour scheme (an overlay).

Correctness (required):
  1. the map exists and is not blank;
  2. the three factors are DRAWN: three different polygon collections of the figure whose colour
     values, mapped back onto the plotted block-group table (or an overlay of it), rank-correlate
     with poverty (rho >= 0.9), no-vehicle (rho >= 0.9) and population density (pop/area,
     rho >= 0.8; drawing the literal `populationToPolygonSizeRating` column is also accepted
     because that column is constant in the benchmark data - a benchmark defect). Any
     monotone scaling or classification (quantiles, normalisation) passes. Collections drawn
     with explicit colours (plot(color=[...]), no drawn values) are read from their face
     colours inverted to colormap positions: a column of the plotted table counts as drawn
     when the colours are value classes of it (class IoU >= 0.9) or rank it (|rho| >= 0.95);
  3. the bus service area is DRAWN and correctly located: the solution code plots (`.plot`,
     `.boundary.plot`) a layer derived from BusServiceArea through geometry-preserving steps
     (read, to_crs, dissolve, overlay/clip), not replaced by centroids/points or CRS-overridden;
     when that layer is a captured GeoDataFrame it must match the service area (or its
     intersection with the block groups) with area IoU >= 0.90. (Figure captures hold no
     outline geometry, so the plotting call is identified in the code.)
  The drawn layers of (2) are taken from the figure saved as transit_access.png (save
  snapshots mapped onto the literal savefig calls of the code in order; any saved figure when
  the mapping is ambiguous) - factors drawn only in other PNGs do not count.
Reported, not required:
  4. density is meaningful: a varying density (pop/area) was derived and drawn;
  5. a geometric overlay (intersection/difference/clip) of block groups and service area exists.
"""
import ast

import numpy as np
import pandas as pd

from checklib import file_exists, png_not_blank, result, key_values, drawn_arrays, _cmap_positions, rank_fit_to_positions

SA_FILE = "BusServiceArea"


def _layers(ctx):
    import geopandas as gpd
    svc = gpd.read_file(f"{ctx.inputs_dir}/BusServiceArea.geojson").to_crs("EPSG:32616")
    bg = gpd.read_file(f"{ctx.inputs_dir}/HamiltonDemographics.geojson").to_crs("EPSG:32616")
    s = svc.geometry.union_all()
    b = bg.geometry.union_all()
    return bg, {"service area": s, "service area within block groups": s.intersection(b)}


def _iou(u, ref):
    inter, uni = u.intersection(ref).area, u.union(ref).area
    return inter / uni if uni > 0 else 0.0


def _gdf_union(obj):
    g = obj
    if g.crs is None:
        return None
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    g = g[g.geom_type.isin(["Polygon", "MultiPolygon"])]
    if not len(g):
        return None
    g = g.to_crs("EPSG:32616")
    return g.geometry.buffer(0).union_all()


def _nparts(geom):
    try:
        return len(geom.geoms) if hasattr(geom, "geoms") else 1
    except Exception:
        return 1


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


def _patch_groups(ctx, side, tags=None):
    """(tag, group) for the hand-drawn patch groups (capture.drawn_patch_groups) of the saved
    figures, limited to `tags` when given."""
    import os
    from capture import drawn_patch_groups
    root = ctx.pred_snap if side == "pred" else ctx.gold_snap
    if not root or not os.path.isdir(root):
        return []
    return [(t, g) for t in sorted(os.listdir(root)) if t.startswith("save") and (tags is None or t in tags)
            for g in drawn_patch_groups(os.path.join(root, t))]


def _patch_union(g):
    """Union of a patch group's polygons (shapely), or None."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    if not g.get("polygons"):
        return None
    polys = []
    for v in g["polygons"]:
        v = v[np.isfinite(v).all(1)]
        if len(v) >= 3:
            p = Polygon(v).buffer(0)
            if not p.is_empty:
                polys.append(p)
    return unary_union(polys) if polys else None


def drawn_per_feature(ctx, side, inputs, key, max_rows=5000, tags=None):
    """Per-feature series behind each drawn polygon collection (GeoDataFrame.plot(column=...)),
    mapped back onto a captured table with geometry whose polygon parts reproduce the drawn
    sequence: (a) one of its numeric columns (non-null rows) equals the drawn values, or
    (b) an array was passed as column (drawn values constant within each row's parts).
    Returns list of (drawn_id, label, column or None, Series indexed by key)."""
    arrs = [(t, m, a) for t, m, a in drawn_arrays(ctx, side) if tags is None or t in tags
            if m.get("artist") == "PatchCollection"]
    drawn = [(t + "/" + m["name"], np.asarray(a, float).ravel()) for t, m, a in arrs
             if m["name"].endswith("_values")]
    # collections drawn with explicit colours (no `_values`, e.g. plot(color=[...])): face
    # colours inverted to colormap positions
    with_values = {t + "/" + m["name"][:-len("_values")] for t, m, a in arrs if m["name"].endswith("_values")}
    colored = []
    for t, m, a in arrs:
        if not m["name"].endswith("_facecolors") or t + "/" + m["name"][:-len("_facecolors")] in with_values:
            continue
        fc = np.asarray(a, float)
        if fc.ndim != 2 or fc.shape[1] < 3 or len(np.unique(np.round(fc[:, :3], 4), axis=0)) < 3:
            continue
        try:
            pos, cname, res = _cmap_positions(fc)
        except Exception:
            continue
        if pos is not None and res <= 0.02:
            colored.append((t + "/" + m["name"], np.asarray(pos, float), cname))
    # polygons drawn one by one with ax.add_patch (no collection): each patch group's face
    # colours, inverted the same way
    for t, g in _patch_groups(ctx, side, tags):
        fc = np.asarray(g["facecolors"], float)
        if not g.get("fill") or fc.ndim != 2 or len(fc) < 3 or len(np.unique(np.round(fc[:, :3], 4), axis=0)) < 3:
            continue
        try:
            pos, cname, res = _cmap_positions(fc)
        except Exception:
            continue
        if pos is not None and res <= 0.02:
            colored.append((f"{t}/patches_g{g['group']}", np.asarray(pos, float), cname))
    out = []
    if not drawn and not colored:
        return out
    tables = []
    for tag, meta, load in ctx.objects(side, "gdf", include_files=False):
        if not 0 < meta.get("rows", 0) <= max_rows:
            continue
        try:
            obj = load()
            keys = key_values(obj, inputs, key)
        except Exception:
            continue
        if keys is None:
            continue
        parts = np.array([_nparts(g) for g in obj.geometry])
        tables.append((meta["name"] + "@" + tag, obj, keys, parts))
    for did, d in drawn:
        for name, obj, keys, parts in tables:
            hit = False
            for c in obj.columns:
                if c == "geometry" or obj[c].dtype.kind not in "fiub":
                    continue
                v = obj[c].astype(float)
                ok = v.notna().to_numpy() & ~obj.geometry.is_empty.to_numpy()
                exp = np.repeat(v.to_numpy()[ok], parts[ok])
                if len(exp) == len(d) and np.allclose(exp, d, rtol=1e-6, atol=1e-9, equal_nan=True):
                    out.append((did, f"{name}.{c}", c, pd.Series(v.to_numpy()[ok], index=keys.to_numpy()[ok])))
                    hit = True
                    break
            if not hit and parts.sum() == len(d):
                starts = np.r_[0, np.cumsum(parts)[:-1]]
                first = d[starts]
                if np.allclose(np.repeat(first, parts), d, equal_nan=True):
                    out.append((did, f"array drawn on {name}", None, pd.Series(first, index=keys.to_numpy())))
    # colour-only collections: a column of the plotted table is the one drawn when the drawn
    # colours are value classes of it (3-12 classes, class IoU >= 0.9) or rank it continuously
    # (|rho| >= 0.95); the column's own values then stand for the drawn series (factor
    # identity is decided by _rho against the inputs, as for drawn values)
    for did, pos, cname in colored:
        ncls = len(np.unique(np.round(pos, 3)))
        for name, obj, keys, parts in tables:
            for c in obj.columns:
                if c == "geometry" or obj[c].dtype.kind not in "fiub":
                    continue
                v = obj[c].astype(float).to_numpy()
                ok = np.isfinite(v) & ~obj.geometry.is_empty.to_numpy()
                if ok.sum() == len(pos):
                    pv = pos
                elif parts[ok].sum() == len(pos):
                    pv = pos[np.r_[0, np.cumsum(parts[ok])[:-1]]]
                else:
                    continue
                rho, ceiling, ciou = rank_fit_to_positions(v[ok], pv)
                if rho != rho:
                    continue
                if abs(rho) >= 0.95 or (3 <= ncls <= 12 and ciou >= 0.9):
                    out.append((did, f"{name}.{c} (face colours, {cname}, class IoU {ciou:.2f})", c,
                                pd.Series(v[ok], index=keys.to_numpy()[ok])))
    return out


def _rho(s, ref, min_n=20):
    s = s.groupby(level=0).mean()
    common = ref.index.intersection(s.index)
    x, y = s.loc[common].astype(float), ref.loc[common].astype(float)
    m = x.notna() & y.notna()
    if m.sum() < min_n or x[m].std() == 0 or y[m].std() == 0:
        return -1.0
    return float(x[m].rank().corr(y[m].rank()))


# ----------------------------------------------------------------------------- code analysis
_TAINT_ATTRS = {"centroid", "representative_point", "points_from_xy", "sample_points"}


def _src_has_sa(node, sa):
    # `x.crs` only borrows the CRS of the service area, it does not derive geometry from it
    crs_names = {id(n.value) for n in ast.walk(node) if isinstance(n, ast.Attribute) and n.attr == "crs"}
    for n in ast.walk(node):
        if id(n) in crs_names:
            continue
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and SA_FILE in n.value:
            return True
        if isinstance(n, ast.Name) and n.id in sa:
            return True
    return False


def _taints(node, tainted):
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and n.attr in _TAINT_ATTRS:
            return True
        if isinstance(n, ast.Name) and n.id in tainted:
            return True
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "set_crs" \
                and any(k.arg == "allow_override" for k in n.keywords):
            return True
    return False


def service_area_plots(code):
    """Names / inline expressions derived from BusServiceArea that are plotted.
    Returns list of (base_name or None for inline, source line)."""
    try:
        tree = ast.parse(code)
    except Exception:
        return []
    sa, tainted = set(), set()
    nodes = sorted([n for n in ast.walk(tree) if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign))],
                   key=lambda n: (n.lineno, n.col_offset))
    for n in nodes:
        val = n.value
        if val is None:
            continue
        targets = n.targets if isinstance(n, ast.Assign) else [n.target]
        refs, bad = _src_has_sa(val, sa), _taints(val, tainted)
        for t in targets:
            names = [e for e in (t.elts if isinstance(t, ast.Tuple) else [t])]
            for e in names:
                if isinstance(e, ast.Name):
                    if refs and not bad:
                        sa.add(e.id); tainted.discard(e.id)
                    elif refs and bad:
                        tainted.add(e.id); sa.discard(e.id)
                    elif e.id in sa and not refs:
                        sa.discard(e.id)  # rebound to something else
                elif isinstance(e, (ast.Subscript, ast.Attribute)):
                    base = e.value
                    while isinstance(base, (ast.Subscript, ast.Attribute)):
                        base = base.value
                    if isinstance(base, ast.Name) and base.id in sa and bad:
                        tainted.add(base.id); sa.discard(base.id)
    lines = code.splitlines()
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "plot":
            recv = n.func.value
            if _src_has_sa(recv, sa) and not _taints(recv, tainted):
                base = recv
                while isinstance(base, (ast.Subscript, ast.Attribute, ast.Call)):
                    base = base.func if isinstance(base, ast.Call) else base.value
                name = base.id if isinstance(base, ast.Name) and base.id in sa else None
                out.append((name, lines[n.lineno - 1].strip()[:120] if n.lineno <= len(lines) else ""))
    return out


def checks(ctx):
    out = []
    names = ["transit_access.png"]
    r = file_exists(ctx, names)
    out.append(r)
    if r["passed"]:
        nb, std = png_not_blank(ctx.pred(names[0]))
        out.append(result("map not blank", nb, value=std, threshold=">=3 grey-level std"))
    else:
        out.append(result("map not blank", False, note="no map"))

    import geopandas as gpd
    try:
        bg, refs = _layers(ctx)
    except Exception as e:
        return out + [result("three factors drawn (poverty, no-vehicle, density)", False, note=repr(e)[:200]),
                      result("bus service area drawn and correctly located", False, note=repr(e)[:200])]
    raw = gpd.read_file(f"{ctx.inputs_dir}/HamiltonDemographics.geojson")
    ids = bg["OBJECTID"].to_numpy()
    pov = pd.Series(bg["households_ACSHHBPOV"].astype(float).to_numpy(), index=ids)
    veh = pd.Series(bg["AtRisk_ACSOVEH0"].astype(float).to_numpy(), index=ids)
    dens = pd.Series((bg["populationtotals_TOTPOP_CY"] / (bg.geometry.area / 1e6)).to_numpy(), index=ids)

    # --- 2. the three factors as drawn ---------------------------------------------------------
    map_tags, map_how = _saved_map_tags(ctx, names[0])
    drawn = drawn_per_feature(ctx, "pred", raw, "OBJECTID", tags=map_tags)
    found = {}
    for fac, ref, thr in (("poverty", pov, 0.9), ("no-vehicle", veh, 0.9), ("density", dens, 0.8)):
        best = None
        for did, lab, col, s in drawn:
            rho = _rho(s, ref)
            literal = fac == "density" and col == "populationToPolygonSizeRating"
            if rho >= thr or literal:
                score = 1.0 if literal else rho
                if best is None or score > best[1]:
                    best = (did, score, lab + (" (literal constant rating column)" if literal else ""), rho)
        found[fac] = best
    used = [b[0] for b in found.values() if b]
    ok = all(found.values()) and len(set(used)) == 3
    out.append(result("three factors drawn (poverty, no-vehicle, density)", ok,
                      value={k: (round(v[3], 3) if v else None) for k, v in found.items()},
                      threshold="poverty/no-vehicle rho>=0.9, density rho>=0.8 (or literal rating column); distinct layers",
                      note="; ".join(f"{k}: {v[2]}" for k, v in found.items() if v)
                      or f"{len(drawn)} drawn polygon layers mapped to block groups ({map_how})"))

    # --- 3. service area drawn and located --------------------------------------------------------
    ious = {}
    overlay_best = 0.0
    seen = set()
    for tag, meta, load in ctx.objects("pred", "gdf"):
        sig = (meta.get("name"), meta.get("rows"), tuple(meta.get("columns", [])))
        if sig in seen or meta.get("rows", 0) == 0 or meta.get("rows", 0) > 5000:
            continue
        seen.add(sig)
        try:
            u = _gdf_union(load())
            if u is None:
                continue
            for k, ref in refs.items():
                v = _iou(u, ref)
                ious[meta["name"]] = max(ious.get(meta["name"], 0.0), v)
                if k != "service area" and meta.get("rows", 0) > 1:
                    overlay_best = max(overlay_best, v)
        except Exception:
            continue
    code = ctx.code_text()
    plots = service_area_plots(code) if code else []
    ok_sa, note = False, "no plot of a layer derived from BusServiceArea found in the code" if code else "no code available"
    val = None
    for name, line in plots:
        if name is None:                     # inline read -> plot, geometry-preserving chain
            ok_sa, note, val = True, "inline: " + line, None
            break
        if name in ious:
            val = round(ious[name], 3)
            if ious[name] >= 0.90:
                ok_sa, note = True, f"{name} (IoU {val}): {line}"
                break
            note = f"{name} drawn but misplaced/wrong (IoU {val}): {line}"
        else:
            ok_sa, note = True, f"{name} not captured; drawn via: {line}"
    if not ok_sa:
        # drawn by hand as patches (ax.add_patch, no .plot call): a patch group of the saved map
        # whose polygons cover the service area with IoU >= 0.90, in the CRS of the input or of
        # any captured layer (the figure itself has no CRS)
        crss = {str(gpd.read_file(f"{ctx.inputs_dir}/BusServiceArea.geojson").crs)}
        crss |= {m.get("crs") for _, m, _ in ctx.objects("pred", "gdf") if m.get("crs") not in (None, "None")}
        # an outline drawn from the exterior rings shows the area with its holes filled: both
        # the service area and its hole-free outline are accepted as the reference
        from shapely.geometry import MultiPolygon, Polygon
        sa = refs["service area"]
        filled = MultiPolygon([Polygon(p.exterior) for p in getattr(sa, "geoms", [sa]) if p.geom_type == "Polygon"])
        best = (0.0, None)
        for t, g in _patch_groups(ctx, "pred", map_tags):
            u = _patch_union(g)
            if u is None or u.is_empty:
                continue
            for c in crss:
                try:
                    refs_c = gpd.GeoSeries([sa, filled.buffer(0)], crs="EPSG:32616").to_crs(c)
                except Exception:
                    continue
                v = max(_iou(u, refs_c.iloc[0]), _iou(u, refs_c.iloc[1]))
                if v > best[0]:
                    best = (v, f"{t} patch group {g['group']} ({g['n']} {g['type']} patches, {c})")
        if best[0] >= 0.90:
            ok_sa, val, note = True, round(best[0], 3), f"hand-drawn patches: {best[1]} (IoU {best[0]:.3f})"
        elif best[1]:
            note += f"; best hand-drawn patch group IoU {best[0]:.3f} ({best[1]})"
    out.append(result("bus service area drawn and correctly located", ok_sa, value=val, threshold="IoU >= 0.90",
                      note=note))

    # --- optional ---------------------------------------------------------------------------------
    d = found.get("density")
    out.append(result("population density derived (pop/area; rating column is constant)",
                      bool(d) and d[3] >= 0.8, value=round(d[3], 3) if d else None, threshold=0.8, required=False))
    out.append(result("geometric overlay of block groups and service area", overlay_best >= 0.90,
                      value=round(overlay_best, 3), threshold=0.90, required=False))
    return out
