"""Checks for task 1: urban heat islands and at-risk elderly population, Madison WI.

Task core: interpolate 139 point temperatures (Kriging) over the city, average the surface per
census block group (269 polygons), map it as a choropleth, and highlight block groups that are
both hot and dense in people aged 65+.

Correctness (required):
  1. the requested map exists and is not blank;
  2. a per-block-group mean temperature was computed: a captured table (one row per block
     group, +/- 5 %), a per-block series, or a 1-D array of exactly 269 values (input order)
     whose ranks agree with the gold block means (Spearman rho >= 0.90 aligned on OBJECTID /
     geometry / position). Kriging, GP and IDW give 0.93-0.98; nearest-station values 0.87,
     raw points / unrelated columns < 0.5;
  3. the values are temperatures of this data set (mean within 2 F of the gold mean);
  4. the values come from a surface, not copied station readings (< 50 % of block values equal
     a station temperature) - rejects "each block = nearest station";
  5. the choropleth DRAWN in the figure shows that temperature: the colour values of a drawn
     polygon collection, mapped back onto the plotted block table (multi-part polygons expand
     to several patches), rank-correlate with the gold block means (rho >= 0.90);
  6. a hot & elderly-dense highlight set was derived: a captured subset of block groups
     (table rows, or a boolean flag column/array) of 1-60 % of the block groups whose members
     are on average hotter than the median block group AND denser in people aged 65+ than the
     median (gold temperatures and input densities).
  The drawn choropleth of (5) is taken from the figure saved as interpolated_urban_heat.png
  (save snapshots mapped onto the literal savefig calls of the code in order; any saved figure
  when the mapping is ambiguous) - a choropleth drawn only in another PNG does not count.
  7. the prescribed method was used (the instruction says "using Kriging
     interpolation", "apply the Kriging method"): pykrige OK/UK, gstools kriging, a Gaussian-
     process regressor, or a hand-written kriging system (variogram + linear solve) in the code.
"""
import numpy as np
import pandas as pd

from checklib import file_exists, png_not_blank, result, key_values, drawn_arrays

N_BLOCKS = 269
GOLD_COL = "temperature"
RHO_MIN = 0.90
EXCLUDE = {"Block_Groups_TOTPOP10", "Block_Groups_PopOver65", "Block_Groups_Over65Density",
           "Shape_Length", "Shape_Area", "index_right", "index", "OBJECTID_right", "SID",
           "LATITUDE", "LONGITUDE", "OBJECTID", "OBJECTID_left"}


def _kriging_used(code):
    """The instruction prescribes Kriging. Accept pykrige (OK/UK), gstools kriging,
    a Gaussian-process regressor (fitted kernel = covariance model) or a hand-written kriging
    system (variogram/covariance model + linear solve). Comment lines are ignored."""
    import re
    lines = [ln.split(" #")[0] for ln in code.splitlines() if not ln.lstrip().startswith("#")]
    c = "\n".join(lines)
    pats = [("pykrige", r"\bpykrige\b|\b(Ordinary|Universal)Kriging\w*\s*\("),
            ("gstools", r"\bgstools\b|\bkrige\.\w+\s*\(|\bgs\.krige"),
            ("GP regressor", r"GaussianProcessRegressor\s*\("),
            ("skgstat", r"\bskgstat\b")]
    for lab, p in pats:
        if re.search(p, c):
            return True, lab
    if re.search(r"variogram|semivar|covariance|\bgamma\w*\s*=", c, re.I) and \
            re.search(r"linalg\.(solve|inv|lstsq|pinv)|scipy\.linalg|\bsolve\s*\(", c):
        return True, "hand-written kriging system (variogram + linear solve)"
    return False, "no Kriging/GP interpolator in the code (e.g. IDW, nearest, RBF, griddata)"


def _rho(s, g):
    common = g.index.intersection(s.index)
    x, y = s.loc[common].astype(float), g.loc[common].astype(float)
    m = x.notna() & y.notna()
    if m.sum() < 0.8 * N_BLOCKS or x[m].std() == 0:
        return -1.0
    return float(x[m].rank().corr(y[m].rank()))


def _per_block_candidates(ctx, side, blocks):
    """(label, Series indexed by OBJECTID) for every per-block numeric result captured."""
    ids = blocks["OBJECTID"].to_numpy()
    for tag, meta, load in ctx.objects(side):
        k = meta.get("kind")
        try:
            if k in ("gdf", "df", "series"):
                if abs(meta.get("rows", 0) - N_BLOCKS) > 0.05 * N_BLOCKS:
                    continue
                obj = load()
                keys = key_values(obj, blocks, "OBJECTID")
                if keys is None and k == "series" and len(obj) == N_BLOCKS:
                    keys = pd.Series(ids, index=obj.index)
                if keys is None:
                    continue
                for c in obj.columns:
                    if c in EXCLUDE or c == "geometry" or obj[c].dtype.kind not in "fi":
                        continue
                    yield f"{meta['name']}.{c}@{tag}", obj[c].astype(float).groupby(keys.values).mean()
            elif k == "array" and meta.get("source") != "figure":
                a = np.asarray(load(), float).squeeze()
                if a.ndim == 1 and len(a) == N_BLOCKS:
                    yield f"{meta['name']}@{tag}", pd.Series(a, index=ids)
        except Exception:
            continue


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


def drawn_per_feature(ctx, side, inputs, key, n_expected, tol=0.05, tags=None):
    """Per-feature series behind each drawn polygon collection (GeoDataFrame.plot(column=...)).
    The drawn colour values are mapped back onto a captured table with geometry whose polygon
    parts (multi-part polygons give several patches) reproduce the drawn sequence: either one
    of its numeric columns (non-null rows) equals the drawn values, or - when an array was
    passed as column - the drawn values are constant within each row's parts.
    Returns list of (label, Series indexed by key)."""
    drawn = [(t, m, np.asarray(a, float).ravel()) for t, m, a in drawn_arrays(ctx, side)
             if tags is None or t in tags
             if m["name"].endswith("_values") and m.get("artist") == "PatchCollection"]
    out = []
    if not drawn:
        return out
    tables = []
    for tag, meta, load in ctx.objects(side, "gdf", include_files=False):
        if abs(meta.get("rows", 0) - n_expected) > tol * n_expected:
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
    for dt, dm, d in drawn:
        for name, obj, keys, parts in tables:
            # (a) a column of the table was drawn
            for c in obj.columns:
                if c == "geometry" or obj[c].dtype.kind not in "fiub":
                    continue
                v = obj[c].astype(float)
                ok = v.notna().to_numpy() & ~obj.geometry.is_empty.to_numpy()
                exp = np.repeat(v.to_numpy()[ok], parts[ok])
                if len(exp) == len(d) and np.allclose(exp, d, rtol=1e-6, atol=1e-9, equal_nan=True):
                    out.append((f"{name}.{c} drawn ({dm['name']})",
                                pd.Series(v.to_numpy()[ok], index=keys.to_numpy()[ok])))
                    break
            else:
                # (b) an array passed as column: drawn values constant within each row's parts
                if parts.sum() == len(d):
                    starts = np.r_[0, np.cumsum(parts)[:-1]]
                    first = d[starts]
                    if np.allclose(np.repeat(first, parts), d, equal_nan=True):
                        out.append((f"array drawn on {name} ({dm['name']})",
                                    pd.Series(first, index=keys.to_numpy())))
    return out


def _highlight_candidates(ctx, side, blocks):
    """(label, set of OBJECTIDs) for captured subsets of block groups."""
    ids = blocks["OBJECTID"].to_numpy()
    hi = int(0.6 * N_BLOCKS)
    for tag, meta, load in ctx.objects(side):
        k = meta.get("kind")
        try:
            if k in ("gdf", "df"):
                rows = meta.get("rows", 0)
                obj = None
                if 1 <= rows <= hi:
                    obj = load()
                    keys = key_values(obj, blocks, "OBJECTID")
                    if keys is not None and keys.notna().all() and keys.nunique() == len(obj):
                        yield f"{meta['name']}@{tag}", set(keys.tolist())
                if abs(rows - N_BLOCKS) <= 0.05 * N_BLOCKS:   # boolean flag column
                    obj = load() if obj is None else obj
                    keys = key_values(obj, blocks, "OBJECTID")
                    if keys is None:
                        continue
                    for c in obj.columns:
                        if c != "geometry" and obj[c].dtype.kind == "b":
                            yield f"{meta['name']}.{c}@{tag}", set(keys[obj[c].to_numpy()].tolist())
            elif k == "array" and meta.get("source") != "figure":
                a = np.asarray(load()).squeeze()
                if a.ndim == 1 and len(a) == N_BLOCKS and set(np.unique(a[np.isfinite(a)])) <= {0.0, 1.0}:
                    yield f"{meta['name']}@{tag}", set(ids[a == 1].tolist())
        except Exception:
            continue
    # boolean masks kept as separate Series (hot = t > x; eld = d > y; b[hot & eld].plot(...)):
    # each mask, and the AND of two masks combined in the code
    yield from _mask_candidates(ctx, side, blocks)


def _mask_candidates(ctx, side, blocks):
    import re
    ids = blocks["OBJECTID"].to_numpy()
    frames = {}   # gdf/df with OBJECTID and N_BLOCKS rows, by tag: their index -> OBJECTID
    masks = {}    # name -> (tag, set of OBJECTIDs); last capture of a name wins
    for tag, meta, load in ctx.objects(side):
        try:
            if meta.get("kind") in ("gdf", "df") and abs(meta.get("rows", 0) - N_BLOCKS) <= 0.05 * N_BLOCKS:
                obj = load()
                if "OBJECTID" in obj.columns and obj["OBJECTID"].is_unique:
                    frames.setdefault(tag, []).append(obj["OBJECTID"])
        except Exception:
            continue
    for tag, meta, load in ctx.objects(side):
        try:
            if meta.get("kind") != "series" or meta.get("rows", N_BLOCKS) != N_BLOCKS:
                continue
            df = load()   # a captured Series is stored as reset_index(): [index column, values]
            if not isinstance(df, pd.DataFrame) or df.shape[1] != 2:
                continue
            s = pd.Series(df.iloc[:, 1].to_numpy(), index=pd.Index(df.iloc[:, 0].to_numpy()))
            if s.dtype.kind != "b" or len(s) != N_BLOCKS:
                continue
            oid = None
            for o in frames.get(tag, []):   # a frame of the same snapshot with the same index
                if o.index.equals(s.index):
                    oid = o.to_numpy()
                    break
            if oid is None and s.index.equals(pd.RangeIndex(N_BLOCKS)):
                oid = ids                  # positional, as in _per_block_candidates
            if oid is None:
                continue
            masks[meta["name"]] = (tag, set(oid[s.to_numpy()].tolist()))
        except Exception:
            continue
    for n, (tag, sel) in masks.items():
        yield f"mask {n}@{tag}", sel
    code = ctx.code_text() if hasattr(ctx, "code_text") else ""
    names = sorted(masks)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if re.search(rf"(?<![~\w.]){re.escape(a)}\s*&\s*{re.escape(b)}\b|(?<![~\w.]){re.escape(b)}\s*&\s*{re.escape(a)}\b", code):
                yield f"mask {a} & {b} (combined in the code)", masks[a][1] & masks[b][1]


def checks(ctx):
    out = []
    names = ["interpolated_urban_heat.png"]
    r = file_exists(ctx, names)
    out.append(r)
    if r["passed"]:
        nb, std = png_not_blank(ctx.pred(names[0]))
        out.append(result("map not blank", nb, value=std, threshold=">=3 grey-level std"))
    else:
        out.append(result("map not blank", False, note="no map"))

    import geopandas as gpd
    blocks = gpd.read_file(f"{ctx.inputs_dir}/block.geojson")
    gold = None
    for tag, meta, load in ctx.objects("gold", "gdf", include_files=False):
        if meta["name"] == "choropleth_map":
            gold = load()
    names_req = ["per-block mean temperature matches gold pattern (Spearman >= 0.90)",
                 "values are temperatures of this data (mean within 2 F)",
                 "values come from an interpolated surface, not copied station readings",
                 "drawn choropleth shows the per-block temperature",
                 "hot & elderly-dense highlight set derived"]
    if gold is None or GOLD_COL not in gold.columns:
        return out + [result(n, False, note="gold table missing") for n in names_req]
    g = gold.drop_duplicates("OBJECTID").set_index("OBJECTID")[GOLD_COL].astype(float)

    best = (None, -1.0, None)
    for label, s in _per_block_candidates(ctx, "pred", blocks):
        rho = _rho(s, g)
        if rho > best[1]:
            best = (label, rho, s)
    label, rho, series = best
    out.append(result(names_req[0], rho >= RHO_MIN, value=round(rho, 3) if label else None, threshold=RHO_MIN,
                      note=label or "no per-block numeric result with >=80% coverage"))
    if label:
        pm = float(series.mean())
        out.append(result(names_req[1], abs(pm - float(g.mean())) <= 2.0, value=round(pm, 2),
                          threshold=f"{g.mean():.2f}+/-2"))
        st = gpd.read_file(f"{ctx.inputs_dir}/Temperature.geojson")["TemperatureF"].astype(float).to_numpy()
        v = series.dropna().to_numpy()
        copied = float(np.mean(np.min(np.abs(v[:, None] - st[None, :]), axis=1) < 1e-6)) if len(v) else 1.0
        out.append(result(names_req[2], copied < 0.5, value=round(copied, 3), threshold="< 0.5 of blocks equal a station value"))
    else:
        out.append(result(names_req[1], False, note="no per-block result"))
        out.append(result(names_req[2], False, note="no per-block result"))

    map_tags, map_how = _saved_map_tags(ctx, names[0])
    drawn = drawn_per_feature(ctx, "pred", blocks, "OBJECTID", N_BLOCKS, tags=map_tags)
    dbest = (None, -1.0)
    for lab, s in drawn:
        s = s.groupby(level=0).mean()
        rr = _rho(s, g)
        if rr > dbest[1]:
            dbest = (lab, rr)
    out.append(result(names_req[3], dbest[1] >= RHO_MIN, value=round(dbest[1], 3) if dbest[0] else None,
                      threshold=RHO_MIN, note=dbest[0] or f"no drawn block-group choropleth found ({len(drawn)} candidates; {map_how})"))

    # highlight: hot & elderly-dense subset (properties judged on gold temperatures / input density)
    dens = blocks.set_index("OBJECTID")["Block_Groups_Over65Density"].astype(float)
    tq, dq = g.rank(pct=True), dens.rank(pct=True)
    hbest = None
    for lab, ids in _highlight_candidates(ctx, "pred", blocks):
        ids = [i for i in ids if i in tq.index]
        if not (1 <= len(ids) <= 0.6 * N_BLOCKS):
            continue
        tm, dm = float(tq.loc[ids].mean()), float(dq.loc[ids].mean())
        if tm > 0.6 and dm > 0.6:
            hbest = (lab, len(ids), round(tm, 2), round(dm, 2))
            break
    code = ctx.code_text() if hasattr(ctx, "code_text") else ""
    if hbest is None:
        # inline selections (b[(b.t > x) & (b.Over65Density > y)].boundary.plot(...)) are not
        # captured as objects and outline collections carry no data; accept a selection line
        # in the code that combines a comparison on the elderly density with another condition.
        import re
        for ln in code.splitlines():
            if re.search(r"Over65|PopOver65", ln) and re.search(r"[<>]|quantile|percentile|nlargest", ln) \
                    and re.search(r"&|\band\b", ln) and not ln.lstrip().startswith("#"):
                hbest = ("code: " + ln.strip()[:100], "code-only", None, None)
                break
    out.append(result(names_req[4], hbest is not None, value=hbest and list(hbest[1:]),
                      threshold="1-60 % of blocks; mean temp & density percentile > 0.6",
                      note=hbest[0] if hbest else "no hot & elderly-dense subset of block groups captured"))

    krig, krig_how = _kriging_used(code)
    out.append(result("method: Kriging / GP interpolation (prescribed by the task)", krig,
                      note=krig_how if code else "no code available"))
    return out
