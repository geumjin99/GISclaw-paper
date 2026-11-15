"""Checks for task 3: burn scars from Landsat 8 dNBR (2014 pre-fire vs 2015 post-fire).

Task core: NBR = (B5 - B7)/(B5 + B7) for both years, dNBR = NBR_2014 - NBR_2015, threshold to a
burned mask and convert it to vector polygons, then map them.

The gold dNBR is recomputed here from the inputs (deterministic band arithmetic).

Correctness (required):
  1. the map exists and is not blank;
  2. a dNBR surface was computed: some captured 2-D array agrees cell-by-cell with the gold
     dNBR, |Pearson r| >= 0.98 (either sign convention, pre-post or post-pre; a coarser array
     is compared with the gold dNBR area-averaged onto its grid; the gold is
     either the DN dNBR or the dNBR of TOA reflectance calibrated with the bands'
     ReflectanceGain/ReflectanceBias tags). Wrong bands
     (B5/B6: r 0.95, NDVI: 0.61), a single year (0.42) or unsigned-integer overflow (-0.48) fail;
  3. burn-scar polygons in the right place: the polygon layer that is MAPPED (a captured layer
     whose variable the code plots; best captured layer only when the plotted object is
     constructed inline / not captured; a variable reassigned in a loop as a row subset of a
     captured layer, e.g. one severity class per iteration, stands for the union of the drawn
     subsets of its parent layer) (reprojected to the
     image CRS; a captured in-memory layer without CRS is assumed to be in it, a file without
     CRS is not used) covers >= 70 % of the clearly burned
     core (gold dNBR > 0.44, USGS moderate-high) and >= 60 % of its area has gold dNBR > 0.10
     (USGS low-severity break). This accepts any threshold from ~0.1 to ~0.44 and raster or
     contour vectorisation, and rejects the image footprint, regrowth areas, pixel-coordinate
     polygons and mis-placed layers.
Reported, not required:
  4. burned area is in a plausible range (15-350 km2; gold 62 km2 at dNBR > 0.27).
"""
import numpy as np

from checklib import file_exists, png_not_blank, result, resample_to


def _gold(ctx):
    import rasterio
    out = []
    for y in ("2014", "2015"):
        with rasterio.open(f"{ctx.inputs_dir}/G_{y}.tif") as s:
            b5, b7 = s.read(5).astype(float), s.read(7).astype(float)
            nod, tr, crs = s.nodata, s.transform, s.crs
        bad = (b5 >= nod) | (b7 >= nod)
        with np.errstate(all="ignore"):
            v = (b5 - b7) / (b5 + b7)
        v[bad] = np.nan
        out.append(v)
    return out[0] - out[1], tr, crs


def _gold_refl(ctx):
    """The same dNBR from top-of-atmosphere reflectance:
    each band calibrated with the ReflectanceGain / ReflectanceBias tags stored in the GeoTIFF
    (DN * gain + bias; the sun-elevation factor cancels in the ratio). Physically the standard
    NBR; r = 0.86 with the DN dNBR because of the -0.1 bias. None if the tags are missing."""
    import rasterio
    out = []
    for y in ("2014", "2015"):
        with rasterio.open(f"{ctx.inputs_dir}/G_{y}.tif") as s:
            bands = []
            for b in (5, 7):
                t = s.tags(b)
                if "ReflectanceGain" not in t or "ReflectanceBias" not in t:
                    return None
                dn = s.read(b).astype(float)
                bad = dn >= s.nodata
                r = dn * float(t["ReflectanceGain"]) + float(t["ReflectanceBias"])
                r[bad] = np.nan
                bands.append(r)
        with np.errstate(all="ignore"):
            out.append((bands[0] - bands[1]) / (bands[0] + bands[1]))
    return out[0] - out[1]


def _block_mean(g, shape):
    """Area-average the gold grid onto a coarser grid of `shape` (same extent)."""
    H, W = g.shape
    h, w = shape
    ri = (np.arange(H) * h // H)[:, None]
    ci = (np.arange(W) * w // W)[None, :]
    idx = (ri * w + ci).ravel()
    v = g.ravel()
    ok = np.isfinite(v)
    s = np.bincount(idx[ok], weights=v[ok], minlength=h * w)
    n = np.bincount(idx[ok], minlength=h * w)
    with np.errstate(all="ignore"):
        out = s / n
    out[n < 0.5 * (H * W) / (h * w)] = np.nan
    return out.reshape(h, w)


def plotted_names(code):
    """Base variable names of every `.plot(...)` receiver in the code (e.g. `scars` for
    scars.to_crs(c).boundary.plot()), and whether some plot receiver is not a plain name
    (inline object, ax.plot ...)."""
    import ast
    names, other = set(), False
    try:
        tree = ast.parse(code)
    except Exception:
        return names, True
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "plot":
            b = n.func.value
            while isinstance(b, (ast.Subscript, ast.Attribute, ast.Call)):
                b = b.func if isinstance(b, ast.Call) else b.value
            if isinstance(b, ast.Name) and b.id not in ("plt", "ax", "axes", "axs", "ax1", "ax2", "fig"):
                names.add(b.id)
            else:
                other = True
    return names, other


def loop_subset_parents(code):
    """Plotted variables that are reassigned inside a `for` loop as a row subset of another
    variable that depends on the loop variable (e.g. `for k in cols: sub = gdf[gdf['class'] == k];
    sub.plot()`): {plotted name: (parent, column, values)}. The map shows the union over the
    iterations, i.e. the parent layer; when the subset is `parent[parent[col] == loopvar]` and
    the iterable is a literal (list/tuple/set/dict keys/range, directly or via a name bound to
    one), only rows with col in those values (column, values; else None, None)."""
    import ast
    try:
        tree = ast.parse(code)
    except Exception:
        return {}
    plotted, _ = plotted_names(code)
    literals = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            literals[n.targets[0].id] = n.value

    def values_of(it):
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute) and it.func.attr in ("keys", "items"):
            it = it.func.value
        if isinstance(it, ast.Name) and it.id in literals:
            it = literals[it.id]
        try:
            if isinstance(it, ast.Dict):
                return [ast.literal_eval(k) for k in it.keys]
            if isinstance(it, (ast.List, ast.Tuple, ast.Set)):
                return [ast.literal_eval(e) for e in it.elts]
            if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range":
                return list(range(*[ast.literal_eval(a) for a in it.args]))
        except Exception:
            return None
        return None

    out = {}
    for f in ast.walk(tree):
        if not isinstance(f, ast.For) or not isinstance(f.target, (ast.Name, ast.Tuple)):
            continue
        lv = {t.id for t in ast.walk(f.target) if isinstance(t, ast.Name)}
        for n in (x for b in f.body for x in ast.walk(b)):
            if not (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                    and n.targets[0].id in plotted and isinstance(n.value, ast.Subscript)):
                continue
            base = n.value.value
            if isinstance(base, ast.Attribute) and base.attr in ("loc",):
                base = base.value
            if not isinstance(base, ast.Name):
                continue
            if not any(isinstance(x, ast.Name) and x.id in lv for x in ast.walk(n.value.slice)):
                continue   # subset does not change with the iteration
            col = vals = None
            sl = n.value.slice
            if (isinstance(sl, ast.Compare) and len(sl.ops) == 1 and isinstance(sl.ops[0], ast.Eq)
                    and isinstance(sl.comparators[0], ast.Name) and sl.comparators[0].id in lv
                    and isinstance(sl.left, ast.Subscript) and isinstance(sl.left.value, ast.Name)
                    and sl.left.value.id == base.id):
                try:
                    col = ast.literal_eval(sl.left.slice)
                    vals = values_of(f.iter)
                except Exception:
                    col = vals = None
                if vals is None:
                    col = None
            out[n.targets[0].id] = (base.id, col, vals)
    return out


def checks(ctx):
    out = []
    names = ["burn_scar_analysis.png"]
    r = file_exists(ctx, names)
    out.append(r)
    if r["passed"]:
        nb, std = png_not_blank(ctx.pred(names[0]))
        out.append(result("map not blank", nb, value=std, threshold=">=3 grey-level std"))
    else:
        out.append(result("map not blank", False, note="no map"))
    try:
        gold, tr, crs = _gold(ctx)
    except Exception as e:
        out.append(result("dNBR surface matches gold (|r| >= 0.98)", False, note=repr(e)[:200]))
        out.append(result("burn-scar polygons cover the burned core and little else", False))
        return out

    # --- dNBR array --------------------------------------------------------------------------
    # gold variants: dNBR from raw DN, and from gain/bias-calibrated reflectance
    best, where = 0.0, None
    gv = np.isfinite(gold)
    ratio = gold.shape[0] / gold.shape[1]
    golds = [("DN", gold)]
    try:
        gr = _gold_refl(ctx)
        if gr is not None:
            golds.append(("reflectance", gr))
    except Exception:
        pass
    for tag, meta, load in ctx.objects("pred", "array"):
        sh = meta.get("shape", [])
        if len(sh) != 2 or min(sh) < 50 or abs(sh[0] / sh[1] - ratio) > 0.05:
            continue
        try:
            a = np.asarray(load(), float)
            for gname, gold_v in golds:
                if a.shape[0] < gold_v.shape[0] * 0.9:      # coarser grid: aggregate gold onto it
                    g2 = _block_mean(gold_v, a.shape)
                    p, gg = a, g2
                else:
                    p, gg = resample_to(a, gold_v.shape), gold_v
                gvv = np.isfinite(gg)
                m = gvv & np.isfinite(p)
                if m.sum() < 0.5 * gvv.sum() or p[m].std() == 0:
                    continue
                rr = float(np.corrcoef(p[m], gg[m])[0, 1])
                if abs(rr) > abs(best):
                    best, where = rr, f"{meta['name']}@{tag}" + (f" (vs {gname} gold)" if gname != "DN" else "")
        except Exception:
            continue
    out.append(result("dNBR surface matches gold (|r| >= 0.98)", abs(best) >= 0.98, value=round(best, 4),
                      threshold=0.98, note=where or "no raster-shaped array"))

    # --- burn polygons ----------------------------------------------------------------------
    from rasterio import features
    core = gv & (gold > 0.44)
    burnt = gv & (gold > 0.10)
    bestp = (0.0, 0.0, None, 0.0)
    seen = set()
    drawn_names, other_plot = plotted_names(ctx.code_text())
    loop_par = loop_subset_parents(ctx.code_text())
    parents = {}            # parent layer name -> (column, values) drawn class by class in a loop
    for child, (par, col, vals) in loop_par.items():
        parents[par] = (col, vals)
    plotted_direct = set(drawn_names)
    drawn_names = drawn_names | set(parents)
    layers = []
    for tag, meta, load in ctx.objects("pred", "gdf"):
        sig = (meta.get("name"), meta.get("rows"), tuple(meta.get("columns", [])))
        if sig in seen or not meta.get("rows"):
            continue
        seen.add(sig)
        try:
            g = load()
            g = g[g.geometry.notna() & g.geom_type.isin(["Polygon", "MultiPolygon"])]
            if meta.get("name") in parents and meta.get("name") not in plotted_direct:
                col, vals = parents[meta["name"]]
                if col is not None and col in g.columns:
                    g = g[g[col].isin(vals)]
            if not len(g):
                continue
            if g.crs is None:
                if meta.get("source") == "file":
                    continue  # no CRS guessing for files written by the run
                g = g.set_crs(crs)  # captured in-memory layer: assumed to be in the image CRS
            elif g.crs != crs:
                g = g.to_crs(crs)
            m = features.rasterize(((geom, 1) for geom in g.geometry), out_shape=gold.shape, transform=tr,
                                   fill=0, dtype="uint8").astype(bool)
            if m.sum() == 0:
                continue
            cov = float((m & core).sum() / max(1, core.sum()))
            prec = float((m & burnt).sum() / max(1, (m & gv).sum()))
            area = float(m.sum() * abs(tr.a * tr.e) / 1e6)
            layers.append((meta["name"], (cov, prec, f"{meta['name']}@{tag}", area)))
        except Exception:
            continue
    # the layer that is mapped decides: polygon layers that the code plots; only when the plotted
    # object is not a captured layer (inline construction, ax.plot of coordinates) fall back to
    # the best captured polygon layer.
    drawn = [l for l in layers if l[0] in drawn_names]
    pool = drawn if drawn else (layers if other_plot or (drawn_names - {l[0] for l in layers}) else [])
    for _, cand in pool:
        if bestp[2] is None or min(cand[0] / 0.7, cand[1] / 0.6) > min(bestp[0] / 0.7, bestp[1] / 0.6):
            bestp = cand
    cov, prec, w, area = bestp
    if w and not drawn:
        w += " (plotted object not captured; best captured polygon layer)"
    if not pool:
        w = "no plotted polygon layer (code plots: %s)" % sorted(drawn_names)
    out.append(result("burn-scar polygons cover the burned core and little else", cov >= 0.7 and prec >= 0.6,
                      value={"core_coverage": round(cov, 3), "precision_dNBR>0.1": round(prec, 3)},
                      threshold="coverage>=0.70, precision>=0.60", note=w or "no polygon layer"))
    out.append(result("burned area plausible (15-350 km2)", 15 <= area <= 350, value=round(area, 1),
                      threshold="15-350", required=False))
    return out
