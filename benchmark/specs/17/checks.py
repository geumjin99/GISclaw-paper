"""Checks for task 17: map of 2016 fatal car crashes in NYC over the borough polygons (geoplot).

The file holds exactly the 72 fatal collisions of 2016, so the analysis is: show all of
them, at their true positions, on the borough map. Checked on the data actually drawn
(scatter offsets / marker lines captured at savefig), which are in the plotting CRS.
Required:
  1. collisions_map_vis.png exists and is not blank;
  2. all 72 collisions are drawn at the right places: pooled point artists contain 71-81
     distinct points (two crashes share a location), and after min-max normalising both sets (makes the test CRS/projection-free)
     >= 71 input collisions have a drawn point within 1 % of the extent. Dropping or
     filtering crashes, swapped lat/lon, or plotting something else fails;
     Only points visible in the saved map count: on plain axes those inside the captured axes
     limits; on cartopy/geoplot axes (limits in projected units) those inside the map extent
     set in the code (set_extent / extent=, literal or <captured layer>.total_bounds), or all points when no extent is set (the
     autoscaled frame shows all data). An extent that cannot be read -> not counted.
  3. the borough polygons are DRAWN and share the points' coordinate space: a drawing call
     in the code (geoplot polyplot/choropleth, add_geometries, ShapelyFeature,
     <layer>.plot / .boundary.plot) is resolved to a captured polygon layer (or an inline
     read of nyc_boroughs.geojson, incl. .to_crs(EPSG)) covering >= 95 % of the borough area,
     and >= 90 % of drawn points lie inside its (5 %-expanded) bounds in the CRS it is drawn
     in. A borough layer that is only loaded, never drawn, fails; no resolvable drawing call
     -> UNVERIFIABLE (not passed);
  Points (2) are taken only from the savefig that wrote collisions_map_vis.png last, when
  that can be identified from the code (see _final_save_tags).
Required (a method the task text prescribes is required):
  4. the crashes are plotted with geoplot ("plot ... on a map by geoplot"): a geoplot point-layer
     call (pointplot / kdeplot / quadtree / voronoi / webmap) in the executed code, comments
     ignored; a map drawn only with geopandas / cartopy / matplotlib fails.
Reported, not required:
  5. fatalities encoded (marker size or colour by 'NUMBER OF PERSONS KILLED').
Presentation (kind="figure"; secondary analysis, never required):
  6. the crash markers are visible: in the figure that wrote the map, >= 90 % of the points of
     the scatter collections whose colours were captured have a marker (face, or edge when
     captured per point) clearly darker than a white background (alpha-composited luminance
     <= 0.88; e.g. the lightest 'Reds' colour, 0.97, is near-white).
     The background is assumed white (axes facecolor and fills under the points
     are not used). Uniform marker colours are not captured -> UNVERIFIABLE (not passed).
"""
import os
import re

import numpy as np

from checklib import drawn_arrays, file_exists, png_not_blank, result, unverifiable

NAME = "collisions_map_vis.png"


def _norm(a):
    lo, hi = a.min(0), a.max(0)
    return (a - lo) / np.where(hi - lo > 0, hi - lo, 1)


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


def _receiver(code, i):
    """Expression immediately before position i (e.g. 'b.to_crs(2263)' before '.plot(')."""
    j, depth = i, 0
    while j > 0:
        ch = code[j - 1]
        if ch in ")]":
            depth += 1
        elif ch in "([":
            if depth == 0:
                break
            depth -= 1
        elif depth == 0 and not (ch.isalnum() or ch in "_."):
            break
        j -= 1
    return code[j:i]


def _polygon_draw_exprs(code):
    """Expressions of layers drawn as polygons: geoplot polyplot/choropleth, add_geometries,
    ShapelyFeature, <layer>.plot / <layer>.boundary.plot."""
    exprs = []
    # geoplot module aliases / imported names resolved as in the point-layer check
    # (e.g. `import geoplot as gp; gp.polyplot(...)`)
    mods, names = _geoplot_aliases(code, ("polyplot", "choropleth"))
    alt = [re.escape(x) + r"\.(?:polyplot|choropleth)" for x in sorted(mods | {"geoplot.geoplot"})] + [re.escape(n) for n in sorted(names)]
    rx = r"(?<![\w.])(?:" + "|".join(alt) + r")\s*\(|add_geometries\s*\(|ShapelyFeature\s*\("
    for m in re.finditer(rx, code):
        args = _split_args(code[m.end():])
        pos = [a for a in args if not re.match(r"^\w+\s*=[^=]", a)]
        kw = {a.split("=", 1)[0].strip(): a.split("=", 1)[1] for a in args if re.match(r"^\w+\s*=[^=]", a)}
        e = pos[0] if pos else kw.get("df") or kw.get("geoms")
        if e:
            exprs.append(e)
    for m in re.finditer(r"\.plot\s*\(", code):
        e = _receiver(code, m.start())
        if e and not re.search(r"\b(plt|ax|axes|ax\d*)$", e):
            exprs.append(e)
    return exprs


def _boroughs_drawn(ctx, inp_b):
    """Borough layers that the code draws, as GeoDataFrames in the CRS they are drawn in."""
    code = ctx.code_text()
    gdfs = {}
    for tag, meta, load in ctx.objects("pred", "gdf", include_files=False):
        gdfs.setdefault(meta["name"], load)
    layers, names = [], []
    for e in _polygon_draw_exprs(code):
        lay = None
        if re.search(r"\[|\.query\(|\.loc\b|\.iloc\b|\.head\(|\.sample\(", e) and "nyc_boroughs" not in e:
            continue   # a subset of a layer (e.g. one borough) is not the borough layer
        ids = re.findall(r"[A-Za-z_]\w*", e)
        for i in ids:
            if i in gdfs:
                try:
                    g = gdfs[i]()
                    if len(g) and g.geom_type.isin(["Polygon", "MultiPolygon"]).all():
                        lay = g
                        break
                except Exception:
                    pass
        if lay is None and "nyc_boroughs" in e:
            lay = inp_b
        if lay is None:
            continue
        m = re.search(r"to_crs\(([^)]*)\)", e)
        if m:
            d = re.search(r"\d{4,6}", m.group(1))
            if not d:
                continue
            try:
                lay = lay.to_crs(int(d.group()))
            except Exception:
                continue
        try:   # must be the borough layer (or most of it), not some other polygon layer
            ref = inp_b.to_crs(lay.crs) if lay.crs and inp_b.crs and lay.crs != inp_b.crs else inp_b
            cov = lay.geometry.union_all().intersection(ref.geometry.union_all()).area / ref.geometry.union_all().area
        except Exception:
            continue
        if cov >= 0.95:
            layers.append(lay)
            names.append(e[:60])
    return layers, names


def _layer_bounds(ctx, name):
    """total_bounds of the captured (in-memory) layer `name`, or None."""
    if ctx is None:
        return None
    for tag, meta, load in ctx.objects("pred", "gdf", include_files=False):
        if meta.get("name") != name:
            continue
        try:
            g = load()
            if len(g):
                return [float(x) for x in g.total_bounds]
        except Exception:
            pass
    return None


def _extent_from_code(code, ctx=None):
    """(xmin, xmax, ymin, ymax) in lon/lat if the code sets a geographic map extent;
    'none' if it sets none; None if it sets one that cannot be read.
    The argument may also be `<layer>.total_bounds` (directly or through a
    name assigned to it), where <layer> is a captured layer: its bounds are used, read with
    the same positional meaning as a literal (e.g. extent=b.total_bounds)."""
    found = False
    for m in re.finditer(r"set_extent\s*\(|\bextent\s*=", code):
        found = True
        txt = code[m.end():]
        if txt.lstrip().startswith(("[", "(")):
            arg = _split_args(txt.lstrip()[1:])
        else:
            first = (_split_args(txt)[0] if _split_args(txt) else "").strip()
            tb = re.fullmatch(r"([A-Za-z_]\w*)\.total_bounds", first)
            if not tb and re.fullmatch(r"[A-Za-z_]\w*", first):
                al = re.findall(r"^\s*" + re.escape(first) + r"\s*=\s*([A-Za-z_]\w*)\.total_bounds\s*$", code, re.M)
                tb = re.fullmatch(r"([A-Za-z_]\w*)", al[-1]) if al else None
            if tb:
                arg = _layer_bounds(ctx, tb.group(1)) or []
            else:
                a = re.findall(r"^\s*" + re.escape(first) + r"\s*=\s*[\[\(]([^\]\)]*)", code, re.M) \
                    if re.fullmatch(r"[A-Za-z_]\w*", first) else []
                arg = a[-1].split(",") if a else []
        try:
            v = [float(x) for x in arg[:4]]
        except ValueError:
            return None
        if len(v) != 4 or not all(abs(x) <= 180 for x in v):
            return None
        if "set_extent" in m.group(0):
            return v[0], v[1], v[2], v[3]
        return v[0], v[2], v[1], v[3]          # geoplot extent=(minx, miny, maxx, maxy)
    if re.search(r"set_xlim|set_ylim|plt\.xlim|plt\.ylim|\baxis\s*\(\s*\[", code) and found is False:
        return "limits"
    return "none"


def _final_save_tags(ctx, name):
    """Snapshot tag(s) of the savefig that wrote the required file LAST.

    A figure saved earlier (or to another file, e.g. a debug plot) does not count when the
    required PNG is later overwritten with something else (an empty axes, or boroughs without
    the points). When the capture holds an exact record of which savefig wrote which file, it
    is used. Otherwise savefig calls are read from the code in
    order; when their number equals the number of save snapshots, the k-th call maps to
    saveNN = k and only the last snapshot that wrote `name` counts. If the calls cannot be
    matched to the snapshots, all snapshots count."""
    import checklib as _cl
    if _cl.saved_figures(ctx):  # exact record of which savefig wrote which file
        rec = _cl.figure_record_for_file(ctx, name)
        if rec is None:
            return set()
        if rec.get("tag"):
            return {rec["tag"]}
        # beyond the 20 data snapshots: fall back to inferring from the code
    try:
        code = ctx.code_text()
        root = ctx.pred_snap
        tags = sorted(d for d in os.listdir(root) if re.fullmatch(r"save\d\d", d)) if root and os.path.isdir(root) else []
    except Exception:
        return None
    calls = re.findall(r"savefig\(\s*(?:f?['\"]([^'\"]+)['\"]|([A-Za-z_][\w\.]*))", code)
    targets = [os.path.basename(a) if a else None for a, b in calls]
    if not tags or len(targets) != len(tags):
        return None
    hits = [t for t, f in zip(tags, targets) if f and f.lower() == name.lower()]
    return {hits[-1]} if hits else None


N_VIS = "crash markers visible against a white background"


def _lum(c):
    """Luminance of RGBA colours composited over white."""
    c = np.asarray(c, float)
    rgb = c[:, :3] * c[:, 3:4] + (1 - c[:, 3:4])
    return rgb @ np.array([0.2126, 0.7152, 0.0722])


def _markers_visible(ctx, tag, code):
    """kind="figure": share of drawn crash markers darker than white."""
    if tag is None:
        return result(N_VIS, False, kind="figure", note="no drawn points")
    arrs = {}
    for t, meta, load in ctx.objects("pred", "array", include_files=False):
        if t != tag or meta.get("source") != "figure" or meta.get("artist") != "PathCollection":
            continue
        arrs[meta["name"]] = load
    n_vis = n_all = 0
    unknown = 0
    for nm, load in arrs.items():
        if not nm.endswith("_offsets"):
            continue
        pre = nm[:-len("_offsets")]
        try:
            off = np.asarray(load(), float)
            fc = np.asarray(arrs[pre + "_facecolors"](), float) if pre + "_facecolors" in arrs else None
            ec = np.asarray(arrs[pre + "_edgecolors"](), float) if pre + "_edgecolors" in arrs else None
        except Exception:
            continue
        n = len(off)
        if fc is None or fc.ndim != 2 or len(fc) != n:
            unknown += n
            continue
        dark = 1 - _lum(fc)
        if ec is not None and ec.ndim == 2 and len(ec) == n:
            dark = np.maximum(dark, 1 - _lum(ec))
        elif re.search(r"edgecolors?\s*=|\bec\s*=", code) and (dark < 0.12).any():
            unknown += n            # uniform edge colour set in the code but not captured
            continue
        n_all += n
        n_vis += int((dark >= 0.12).sum())
    if not n_all:
        r = result(N_VIS, False, kind="figure",
                   note="UNVERIFIABLE: marker colours not captured (uniform colour or line markers)")
        r["unverifiable"] = True
        return r
    share = n_vis / n_all
    return result(N_VIS, share >= 0.9, value=round(share, 3), threshold=">=0.90 of markers with luminance <=0.88",
                  kind="figure", note=f"{n_vis}/{n_all} markers visible" + (f"; {unknown} not assessable" if unknown else ""))



# the method named by the task text is required
GP_POINT_FUNCS = ("pointplot", "kdeplot", "quadtree", "voronoi", "webmap")


def _geoplot_calls(code, funcs):
    """Names of geoplot plotting functions in `funcs` that are called in the code (comment lines and
    trailing comments removed). Module aliases come from `import geoplot [as x]` / `import
    geoplot.geoplot as x`; `gplt` and `geoplot` are always accepted (some harness settings pre-import
    geoplot as gplt); `from geoplot import pointplot [as f]` is followed as well."""
    code = "\n".join("" if re.match(r"\s*#", ln) else ln.split("#")[0] for ln in code.splitlines())
    mods, names = _geoplot_aliases(code, funcs)
    fn = "|".join(funcs)
    hits = set()
    for mod in mods:
        hits.update(re.findall(r"(?<![\w.])" + re.escape(mod) + r"\.(" + fn + r")\s*\(", code))
    for n in names:
        if re.search(r"(?<![\w.])" + re.escape(n) + r"\s*\(", code):
            hits.add(n)
    return sorted(hits)


def _geoplot_aliases(code, funcs):
    """(module aliases of geoplot, local names of geoplot functions in `funcs` imported directly)."""
    mods, names = {"gplt", "geoplot"}, set()
    for m in re.finditer(r"^\s*import\s+(.+)$", code, re.M):
        for part in m.group(1).split(","):
            mm = re.match(r"\s*(geoplot(?:\.geoplot)?)\s*(?:as\s+(\w+))?\s*$", part)
            if mm:
                mods.add(mm.group(2) or mm.group(1))
    for m in re.finditer(r"^\s*from\s+geoplot(?:\.geoplot)?\s+import\s+\(?([^)\n]+)", code, re.M):
        for part in m.group(1).split(","):
            mm = re.match(r"\s*(\w+)\s*(?:as\s+(\w+))?", part)
            if mm and mm.group(1) in funcs:
                names.add(mm.group(2) or mm.group(1))
    return mods, names

def checks(ctx):
    import geopandas as gpd
    from scipy.spatial import cKDTree
    out = []
    r = file_exists(ctx, [NAME])
    out.append(r)
    if r["passed"]:
        nb, std = png_not_blank(ctx.pred(NAME))
        out.append(result("map not blank", nb, value=std, threshold=">=3 grey-level std"))
    else:
        out.append(result("map not blank", False, note="no map"))
    try:
        inp = gpd.read_file(os.path.join(ctx.inputs_dir, "fatal_collisions.geojson"))
        ref = np.c_[inp.geometry.x, inp.geometry.y]
    except Exception as e:
        out.append(result("all 72 collisions drawn at their positions and visible on the map", False, note=repr(e)[:200]))
        out.append(result("points and boroughs in the same coordinate space", False))
        return out
    # drawn points per snapshot, keeping only points visible inside the map frame
    lims = {}
    for tag, meta, a in drawn_arrays(ctx, "pred", "axes_limits"):
        lims[(tag, meta["name"].split("_limits")[0])] = np.asarray(a, float)
    ext = _extent_from_code(ctx.code_text(), ctx)
    final_tags = _final_save_tags(ctx, NAME)
    snaps, vis_notes = {}, []
    for tag, meta, load in ctx.objects("pred", "array", include_files=False):
        if meta.get("source") != "figure":
            continue
        if final_tags is not None and tag not in final_tags:
            continue   # figure saved earlier / to another file, later overwritten
        nm = meta["name"]
        try:
            if nm.endswith("_offsets") and meta.get("artist") == "PathCollection":
                a = np.asarray(load(), float)
            elif "_line" in nm and meta.get("artist") == "line":
                a = np.asarray(load(), float)
                if len(a) > 200:          # long polylines are outlines, not markers
                    continue
            else:
                continue
        except Exception:
            continue
        a = a[np.isfinite(a).all(1)] if a.ndim == 2 and a.shape[1] == 2 else None
        if a is None or not len(a):
            continue
        lim = lims.get((tag, nm.split("_")[0] + "_" + nm.split("_")[1]))
        if lim is not None:
            x0, x1 = sorted(lim[:2]); y0, y1 = sorted(lim[2:])
            inside = (a[:, 0] >= x0) & (a[:, 0] <= x1) & (a[:, 1] >= y0) & (a[:, 1] <= y1)
            # same data coordinates as the limits (plain axes) only if the points also fill a
            # sizeable part of the frame; lon/lat offsets inside a metre-based projected frame
            # centred on NYC would otherwise look "inside"
            spread = (np.ptp(a[:, 0]) / max(x1 - x0, 1e-12), np.ptp(a[:, 1]) / max(y1 - y0, 1e-12))
            plain = inside.mean() >= 0.5 and min(spread) >= 0.05
            if plain or ext == "none":
                if plain:
                    a = a[inside]            # plain axes: offsets and limits share data coordinates
                    vis_notes.append("visibility from axes limits")
                else:
                    vis_notes.append("projected axes, no extent set: autoscaled frame shows all points")
            elif ext is None or ext == "limits":
                vis_notes.append("map extent set in code but not readable in lon/lat")
                continue
            else:
                a = a[(a[:, 0] >= ext[0]) & (a[:, 0] <= ext[1]) & (a[:, 1] >= ext[2]) & (a[:, 1] <= ext[3])]
                vis_notes.append("visibility from map extent in code")
        if len(a):
            snaps.setdefault(tag, []).append(a)
    best = (0, 0, None, None)
    for tag, arrs in snaps.items():
        pts = np.unique(np.vstack(arrs), axis=0)
        if len(pts) < 5:
            continue
        d, _ = cKDTree(_norm(pts)).query(_norm(ref))
        matched = int((d <= 0.01).sum())
        if (matched, -abs(len(pts) - 72)) > (best[0], -abs(best[1] - 72)):
            best = (matched, len(pts), tag, pts)
    matched, n, tag, pts = best
    nu = len(np.unique(ref, axis=0))   # 71: two crashes share a location
    out.append(result("all 72 collisions drawn at their positions and visible on the map", nu <= n <= nu + 10 and matched >= 71,
                      value={"drawn_points": n, "inputs_matched": matched}, threshold=f"{nu}-{nu + 10} distinct drawn, >=71 matched",
                      note=(tag or "no visible point artist captured") + "; " + "; ".join(sorted(set(vis_notes)))))
    N_B = "borough polygons drawn, in the points' coordinate space"
    try:
        inp_b = gpd.read_file(os.path.join(ctx.inputs_dir, "nyc_boroughs.geojson"))
        layers, lnames = _boroughs_drawn(ctx, inp_b)
    except Exception as e:
        layers, lnames = [], [repr(e)[:100]]
    if not layers:
        out.append(unverifiable(N_B, "no drawing call of the borough layer identified in the code "
                                     "(polyplot/choropleth/add_geometries/<layer>.plot)"))
    elif pts is None:
        out.append(result(N_B, False, note="no drawn points"))
    else:
        inside_best = 0.0
        for g in layers:
            x0, y0, x1, y1 = g.total_bounds
            dx, dy = 0.05 * (x1 - x0), 0.05 * (y1 - y0)
            ins = ((pts[:, 0] >= x0 - dx) & (pts[:, 0] <= x1 + dx) & (pts[:, 1] >= y0 - dy) & (pts[:, 1] <= y1 + dy)).mean()
            inside_best = max(inside_best, float(ins))
        out.append(result(N_B, inside_best >= 0.9, value=round(inside_best, 3), threshold=0.9,
                          note="drawn: " + "; ".join(lnames)))
    code = ctx.code_text() if hasattr(ctx, "code_text") else ""
    out.append(_markers_visible(ctx, tag, code))
    # the text prescribes the tool ("plot ... on a map by geoplot"), so the
    # crashes must be drawn by a geoplot point-layer call (an import alone, or a comment, is not enough)
    gcalls = _geoplot_calls(code, GP_POINT_FUNCS)
    out.append(result("crashes plotted with geoplot (task: 'on a map by geoplot')", bool(gcalls),
                      value=gcalls or None,
                      note="geoplot pointplot/kdeplot/quadtree/voronoi/webmap call in the executed code" if gcalls
                      else "no geoplot point-layer call (pointplot/kdeplot/quadtree/voronoi/webmap) in the code"))
    out.append(result("fatalities encoded (size/colour by persons killed)",
                      "NUMBER OF PERSONS KILLED" in code and bool(re.search(r"scale\s*=|markersize\s*=|\bs\s*=|hue\s*=|column\s*=|c\s*=", code)),
                      required=False))
    return out
