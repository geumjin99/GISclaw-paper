"""T30 - Bike routes for commuting professionals, Capitol Hill, Seattle (Esri case study).

Task core (Esri model + domain knowledge): study area Capitol Hill (L_HOOD), arterial
network (SEGMENT_TY = 1 and ARTERIAL_C = 1), three proposed corridors - East: (E) Madison St;
West: 10th Ave E; Central: 23rd Ave / 23rd Ave E / 24th Ave E / Turner Way E -,
half-mile (2640 ft; CRS in US feet) buffers, union, clip to Capitol Hill, coverage share
(gold 96.8 % of 3.77 sq mi). Gold corridor lengths East / West / Central: 3.67 (incl. Madison
St downtown) / 1.23 / 1.64 mi. The Esri tutorial reading (Broadway in West, no Madison
downtown) gives 97.7 %, 2.48 / 1.99 / 1.64 mi and is accepted as well.

Correctness (required):
  1. (optional: the instruction asks for no map) a non-blank map PNG;
  2. proposed routes: a line layer (captured or written) whose length near Capitol Hill is
     0.5-2 x the gold route length (rules out 'all streets/arterials'; legitimate variants
     are 1.0-1.45 x) and that covers
     >= 50 % of EACH gold corridor's length within 50 ft;
  3. half-mile accessibility of Capitol Hill: a polygon (captured or written) with area IoU
     >= 0.85 with the gold clipped coverage (and not simply the Capitol Hill polygon: IoU with
     it < 0.99), or a coverage share from 95 % up to gold (96.8 %) + 1.5 points (the domain
     knowledge read literally gives 95.9 %) (fraction or %; scalars,
     dict entries, long or wide tables; numeric text in a long table counts).
     An unclipped buffer alone does not answer the
     question (it needs the share). The band stays narrow because a wider one lets a
     saturated 100 % (2640 m buffer) pass. Coverage is nearly saturated,
     so this check mainly catches unit/CRS and clipping errors; design errors are caught by 2.
     Wrong buffer units (0.5 or 804.7 'metres' in a feet CRS) or an unclipped buffer fail.
  3b. (optional, as 1) the map shows the proposed routes: >= 50 % of the route segments' bounding boxes match
     a drawn collection path / line (State Plane ft, WGS 84, Web Mercator or UTM 10N).
  5. (required: the instruction says "Use Network Analyst to propose new
     routes") network analysis in the executed code, comments removed: a graph library
     (networkx / osmnx / igraph / pandana / scipy.sparse.csgraph) or a shortest-path routine.
Reported, not required:
  4. precision: >= 50 % of the proposed length near Capitol Hill lies on gold corridors;
  6. existing trails measured (5227 ft = 0.99 mi, SEGMENT_TY = 8 within Capitol Hill).
"""
import glob
import os

import numpy as np

from checklib import result, png_not_blank

GOLD_SHARE = None
NEAR = 50.0
LEN_LO, LEN_HI = 0.5, 2.0   # window of the route length relative to gold


def _numeric_str(v):
    """'96.77' -> True; 'True', '', 'nan', '1e3x' -> False."""
    try:
        return np.isfinite(float(v.strip()))
    except (ValueError, AttributeError):
        return False


def _ft(g, crs, from_file=False):
    """To the input CRS. Captured in-memory layers whose coordinates are State Plane feet but
    that are unlabelled / labelled geographic get the input CRS assigned. Files written by the
    run are not guessed: such a file has no usable CRS, the uniform rule
    "deliverable files carry a CRS" fails the cell, and the layer is not interpreted."""
    b = g.total_bounds
    if g.crs is None or (g.crs.is_geographic and (abs(b[0]) > 360 or abs(b[1]) > 360)):
        if from_file:
            raise ValueError("file layer without a usable CRS - not interpreted")
        return g.set_crs(crs, allow_override=True)
    return g if g.crs == crs else g.to_crs(crs)


def _layers(ctx, side="pred"):
    import geopandas as gpd
    for tag, meta, load in ctx.objects(side, "gdf"):
        if meta.get("rows", 0) > 30000:
            continue
        try:
            yield meta["name"] + "@" + tag, load()
        except Exception:
            continue
    if side == "pred":
        for f in glob.glob(os.path.join(ctx.pred_dir, "*")):
            if f.lower().endswith((".shp", ".geojson", ".gpkg", ".json")):
                try:
                    yield os.path.basename(f), gpd.read_file(f)
                except Exception:
                    continue


def _map_shows_routes(ctx, layer, crs):
    name = "map shows the proposed routes"
    from checklib import drawn_arrays, unverifiable
    if layer is None:
        return result(name, False, note="no proposed route layer")
    drawn = [(t, m, np.asarray(a, float)) for t, m, a in drawn_arrays(ctx, "pred")
             if m["name"].endswith("_paths_bbox") or m.get("artist") == "line"]
    if not drawn:
        return unverifiable(name, "no drawn line data captured")
    seg = layer.explode(index_parts=False)
    seg = seg[seg.geometry.length > 0]
    best = 0.0
    for c in (crs, "EPSG:4326", "EPSG:3857", "EPSG:32610"):
        try:
            sc = seg.to_crs(c)
        except Exception:
            continue
        bb = sc.geometry.bounds.to_numpy()
        ext = sc.total_bounds
        tol = 0.002 * max(ext[2] - ext[0], ext[3] - ext[1])
        for t, m, a in drawn:
            if m["name"].endswith("_paths_bbox"):
                d = a[np.isfinite(a).all(1)] if a.ndim == 2 and a.shape[1] == 4 else np.empty((0, 4))
            elif a.ndim == 2 and a.shape[1] == 2 and len(a) >= 2:
                d = np.r_[a.min(0), a.max(0)][None, :]
            else:
                continue
            if not len(d):
                continue
            hit = np.zeros(len(bb), bool)
            for row in d:
                hit |= np.all(np.abs(bb - row) <= tol, axis=1)
            best = max(best, float(hit.mean()))
    return result(name, best >= 0.5, value=round(best, 3), threshold=0.5,
                  note="share of proposed route segments whose bbox matches a drawn path")


def checks(ctx):
    import geopandas as gpd
    import pandas as pd
    out = []
    pngs = ctx.pred_files("*.png")
    # the instruction asks for no map, so both map checks are reported, not required
    # (required/optional follows the task text)
    out.append(result("map written (non-blank PNG)", any(png_not_blank(ctx.pred(p))[0] for p in pngs), value=pngs,
                      required=False))

    nb = gpd.read_file(os.path.join(ctx.inputs_dir, "Neighborhoods.shp"))
    crs = nb.crs
    ch = nb[nb["L_HOOD"] == "CAPITOL HILL"].union_all()
    ch_near = ch.buffer(100)
    gold_routes = gpd.read_file(ctx.gold("proposed_bike_routes.gpkg")).to_crs(crs)
    gold_cov = gpd.read_file(ctx.gold("half_mile_coverage.gpkg")).to_crs(crs)
    gold_share = float(gold_cov["share"].iloc[0])
    gold_len_near = gold_routes.intersection(ch_near).length.sum()
    corridors = {k: d.union_all() for k, d in gold_routes.groupby("corridor")}
    gold_union = gold_routes.union_all()

    # ---- proposed routes
    best, best_layer = None, None
    for name, g in _layers(ctx):
        try:
            g = g[g.geometry.notna()]
            if not len(g) or not g.geom_type.isin(["LineString", "MultiLineString"]).all():
                continue
            g = _ft(g, crs, from_file="@" not in name or name.endswith("@file"))
            u = g.union_all()
            near_len = u.intersection(ch_near).length
            if not LEN_LO * gold_len_near <= near_len <= LEN_HI * gold_len_near:
                continue
            ub = u.buffer(NEAR)
            rec = {k: c.intersection(ub).length / c.length for k, c in corridors.items()}
            prec = u.intersection(ch_near).intersection(gold_union.buffer(NEAR)).length / near_len
            score = min(rec.values())
            if best is None or score > best[0]:
                best = (score, name, rec, prec, near_len)
                best_layer = g
        except Exception:
            continue
    if best is None:
        out.append(result("proposed routes cover the three designated corridors", False,
                          note=f"no line layer of plausible length ({LEN_LO}-{LEN_HI}x gold) near Capitol Hill"))
        out.append(result("precision: proposed length lies on designated corridors", False, required=False))
    else:
        score, name, rec, prec, near_len = best
        out.append(result("proposed routes cover the three designated corridors", score >= 0.5,
                          value={k: round(v, 2) for k, v in rec.items()}, threshold=">=0.5 each",
                          note=f"{name}; {near_len / 5280:.2f} mi near Capitol Hill (gold {gold_len_near / 5280:.2f})"))
        out.append(result("precision: proposed length lies on designated corridors", prec >= 0.5,
                          value=round(prec, 3), threshold=0.5, required=False))

    # ---- the map shows the proposed routes: a drawn collection whose
    # per-path bounding boxes coincide with >= 50 % of the proposed route segments (any of the
    # usual CRSs: State Plane ft, WGS 84, Web Mercator, UTM 10N), or line artists doing so.
    r_map = _map_shows_routes(ctx, best_layer, crs)
    r_map["required"] = False   # see "map written"
    out.append(r_map)

    # ---- half-mile coverage
    cov_ok, cov_note = False, "no coverage polygon or share found"
    gc = gold_cov.union_all()
    for name, g in _layers(ctx):
        try:
            g = g[g.geometry.notna()]
            if not len(g) or not g.geom_type.isin(["Polygon", "MultiPolygon"]).all() or len(g) > 50:
                continue
            g = _ft(g, crs, from_file="@" not in name or name.endswith("@file"))
            u = g.union_all()
            iou = u.intersection(gc).area / u.union(gc).area
            iou_ch = u.intersection(ch).area / u.union(ch).area
            if iou >= 0.85 and iou_ch < 0.99:          # the neighbourhood polygon itself is not a result
                cov_ok, cov_note = True, f"{name}: IoU {iou:.3f}"
                break
        except Exception:
            continue
    if not cov_ok:
        vals = []
        for tag, meta, load in ctx.objects("pred", "scalar"):
            n = meta["name"].lower()
            if any(k in n for k in ("cover", "share", "pct", "percent", "access", "within", "frac", "ratio")):
                vals.append((meta["name"], meta["value"]))
        # small captured tables / dict entries (series) / output tables: keyword in a row's text
        # (long format) or in a column name (wide format)
        kw = ("cover", "share", "pct", "percent", "access", "within", "frac", "ratio", "served")
        for tag, meta, load in ctx.objects("pred"):
            if meta.get("kind") not in ("df", "series") or not 0 < meta.get("rows", 0) <= 50:
                continue
            try:
                df = load()
            except Exception:
                continue
            for _, r in df.iterrows():
                txt = " ".join(str(v) for v in r.values).lower()
                if any(k in txt for k in kw):
                    vals += [(meta["name"], float(v)) for v in r.values
                             if isinstance(v, (int, float, np.number)) and not isinstance(v, bool)]
                    # a metric/value table whose value column also holds True/False is
                    # read back from CSV as text; numeric strings count
                    vals += [(meta["name"], float(v)) for v in r.values
                             if isinstance(v, str) and _numeric_str(v)]
            for c in df.columns:
                if any(k in str(c).lower() for k in kw) and df[c].dtype.kind in "fi":
                    vals += [(f"{meta['name']}.{c}", float(v)) for v in df[c].dropna()]
        # upper bound gold + 1.5 points: coverage saturates, so 100 % (buffer far too
        # large, e.g. 2640 m instead of ft, or all arterials) must not pass on the share alone;
        # lower bound 0.95: the domain knowledge read literally (West corridor = 10th Ave E
        # only, without Broadway) gives 95.9 %
        for n, v in vals:
            if 0.95 <= v <= gold_share + 0.015 or 95.0 <= v <= 100 * gold_share + 1.5:
                cov_ok, cov_note = True, f"{n} = {v:.3f}"
                break
        if not cov_ok and vals:
            cov_note = f"values seen: {[(n, round(v, 3)) for n, v in vals[:5]]}"
    out.append(result("half-mile coverage of Capitol Hill matches gold", cov_ok, note=cov_note,
                      threshold=f"IoU>=0.85 (clipped to Capitol Hill) or share in [0.95, {gold_share + 0.015:.3f}]"))

    # "Use Network Analyst to propose new routes" is an instructed method ->
    # required; any graph/network library or shortest-path routine counts (comments removed).
    code = "\n".join(l.split("#", 1)[0] for l in ctx.code_text().splitlines())
    keys = ("networkx", "osmnx", "igraph", "pandana", "csgraph", "shortest_path", "dijkstra", "nx.")
    net = [k for k in keys if k in code]
    out.append(result("network analysis performed (graph shortest paths)", bool(net), value=net,
                      note="instructed: 'Use Network Analyst to propose new routes'"))
    trails = False
    for tag, meta, load in ctx.objects("pred", "scalar"):
        v = meta["value"]
        if abs(v - 5227.06) <= 30 or abs(v - 0.99) <= 0.02 or abs(v - 1593.2) <= 10:
            trails = True
    out.append(result("existing trail length in Capitol Hill measured", trails, required=False,
                      note="5227 ft / 0.99 mi / 1593 m"))
    return out
