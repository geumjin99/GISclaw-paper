"""Task 49 - homelessness 2013 by state (North Dakota oil boom): join + maps.

The instruction asks to join the state-level homelessness table to the state polygons on the
state abbreviation and to save the maps; it names no calculated field (the benchmark's script
adds homeless per 10 000 residents).

Required:
  1. at least one map (PNG) saved and not blank;
  J. the join: homeless_data.xlsx read (code) and a table of the states layer (geometry or the
     layer's own attribute columns) carries an xlsx value - Homeless13, Pop13, Change (x1/x100)
     or a rate derived from them - aligned per state for >= 95 % of states; positional joins and
     the pre-joined Sheet1__* / HomelessPe fields of the input shapefile fail;
  2b. the map shows the joined homelessness data: a figure's colour values (choropleth, raw or
     binned) match the rate, the 2012-13 change or the 2013 homeless count of a same-length
     table (population alone fails); no captured colour values at all -> UNVERIFIABLE.
     Drawn values / inverted face colours are compared per feature (polygon parts collapsed),
     with |rho| >= 0.9 (reversed colormaps) or, for classified maps (3-12 classes), drawn
     classes = value intervals of the column (class IoU >= 0.9): equal interval k=5 reaches
     only rho 0.66 on these skewed rates. A multi-axes map (lower 48 + AK / HI insets) is
     matched per axes: the drawn values of one axes equal the column on an in-order subset of
     >= 40 features (exact values).
Reported, not required:
  2. calculated field per state: a captured/saved table with one row per state (51 +/- 1) has
     a column equal to Homeless13 / Pop13 in some rate unit (per 10 000, per 100 000, per
     capita, percent) within 5 % (or 0.6 per 10 000 for the rounded pre-joined field) for
     >= 95 % of states, aligned on state abbreviation / name / geometry, or the 2012-13
     relative change joined from the xlsx (x1 or x100; code must read the xlsx). A column
     equal to the rounded HomelessPe field already present in the input shapefile (renamed
     copy) is not credited unless the code reads homeless_data.xlsx;
  3. the 2012-13 change (the 'trend' the instruction mentions, ND +201 %) is carried per state;
  4. the tabular file is read and joined (code reads homeless_data.xlsx) - the input
     shapefile already contains a join (Sheet1__*, HomelessPe), so a solution may map those.
"""
import glob
import os

import numpy as np

from checklib import result, png_not_blank, key_values, drawn_arrays, unverifiable

# columns of a join already stored in USA_States.shp - not the agent's own result
PREJOINED = {"HomelessPe", "Sheet1__Ch", "Sheet1__Po", "Sheet1__Ho", "OBJECTID", "Shape_Leng", "Shape_Area"}
# attribute columns of the states layer itself (a joined attribute table exported without geometry)
STATE_COLS = {"L0USA_Stat", "L0USA_St_1", "STATE_NAME", "STATE_ABBR", "Shape_Leng", "Shape_Area", "OBJECTID"}
SCALES = {"per 10k": 1.0, "per 100k": 10.0, "per capita": 1e-4, "percent": 1e-2, "per 1k": 0.1}


def _rank_fit(v, pos, tol=1e-3):
    """(rho, class_iou) of column v against drawn values/colour positions pos.
    class_iou: mean over drawn classes of the IoU between the rows drawn in that class and the
    rows a value-sorted assignment of the same class sizes puts there - 1.0 when the drawn
    classes are value intervals of v (any scheme: quantiles, natural breaks, equal interval),
    whatever rank correlation the class sizes allow (equal interval k=5 on these rates: 0.66)."""
    from scipy.stats import spearmanr
    v, pos = np.asarray(v, float), np.asarray(pos, float)
    if len(v) != len(pos) or len(v) < 3 or np.std(v) == 0 or np.std(pos) == 0:
        return float("nan"), float("nan")
    rho = float(spearmanr(v, pos).correlation)
    ps = np.sort(pos) if rho >= 0 else np.sort(pos)[::-1]
    ideal = np.empty_like(pos)
    ideal[np.argsort(v, kind="stable")] = ps
    q, qi = np.round(pos / tol).astype(np.int64), np.round(ideal / tol).astype(np.int64)
    ious = [((q == c) & (qi == c)).sum() / max(1, ((q == c) | (qi == c)).sum()) for c in np.unique(q)]
    return rho, float(np.mean(ious))


def _drawn_match(v, d, parts=None, min_rho=0.9, min_iou=0.9):
    """Does column v (one value per feature) explain drawn per-patch values d? Patches are one
    per feature, or one per polygon PART (geopandas); part-expanded draws are collapsed to the
    first patch of each feature so multi-part states (Alaska: 32 parts) do not dominate.
    Accept |rho| >= min_rho (abs: reversed colormaps invert colour positions) or, for a
    classified map (3-12 drawn classes), class IoU >= min_iou."""
    v, d = np.asarray(v, float), np.asarray(d, float)
    if parts is not None and len(d) == int(np.sum(parts)) and len(d) != len(v):
        d = d[np.r_[0, np.cumsum(parts)[:-1]].astype(int)]
    if len(d) != len(v):
        return None
    m = np.isfinite(d) & np.isfinite(v)
    if m.sum() < 40:
        return None
    rho, ciou = _rank_fit(v[m], d[m])
    if rho != rho:
        return None
    ncls = len(np.unique(np.round(d[m], 3)))
    if abs(rho) >= min_rho or (3 <= ncls <= 12 and ciou >= min_iou):
        return rho, ciou, ncls
    return None


def _subset_match(v, d, parts=None, min_features=40):
    """Multi-axes maps (lower 48 + Alaska / Hawaii insets, one axes per region): one
    axes draws only a subset of the rows, so its patch count matches neither the table nor its
    part count. Accept when the drawn per-patch VALUES are exactly the column values of an
    in-order subset of the rows (each row's patches = its polygon parts, all with the row's
    value; rows not drawn skipped), covering >= min_features features. Raw drawn values only
    (no colour inversion): an exact equality, so another column or a re-ordered join fails.
    Returns (rho, class_iou, ncls) of the matched subset via _drawn_match, or None."""
    v, d = np.asarray(v, float), np.asarray(d, float)
    if parts is None:
        parts = np.ones(len(v), int)
    parts = np.asarray(parts, int)
    if len(parts) != len(v) or len(d) < min_features:
        return None
    i = j = 0
    rows, starts = [], []
    while i < len(v) and j < len(d):
        p = max(int(parts[i]), 1)
        seg = d[j:j + p]
        if len(seg) == p and np.isfinite(v[i]) and np.allclose(seg, v[i], rtol=1e-9, atol=1e-12):
            rows.append(i)
            starts.append(j)
            j += p
        i += 1
    if j != len(d) or len(rows) < min_features:
        return None
    return _drawn_match(v[rows], d[starts])


def _gold(ctx):
    import geopandas as gpd
    g = gpd.read_file(ctx.gold("states_homeless_2013.shp"))
    return g


def _tables(ctx):
    import geopandas as gpd
    import pandas as pd
    out = []
    for tag, meta, load in ctx.objects("pred"):
        if meta.get("kind") in ("gdf", "df") and 49 <= meta.get("rows", 0) <= 53:
            try:
                out.append((meta["name"] + "@" + tag, load()))
            except Exception:
                pass
    for p in glob.glob(os.path.join(ctx.pred_dir, "**", "*.*"), recursive=True):
        if p.endswith((".shp", ".geojson", ".gpkg", ".csv")):
            try:
                t = pd.read_csv(p) if p.endswith(".csv") else gpd.read_file(p)
                if 49 <= len(t) <= 53:
                    out.append((os.path.basename(p), t))
            except Exception:
                pass
    return out


def _keys(t, states):
    """state abbreviation for every row of t, or None."""
    abbr = set(states.L0USA_St_1)
    names = dict(zip(states.L0USA_Stat.str.lower(), states.L0USA_St_1))
    for c in t.columns:
        if t[c].dtype == object:
            v = t[c].astype(str).str.strip()
            if v.isin(abbr).mean() > 0.9:
                return v.to_numpy()
            low = v.str.lower()
            if low.isin(names).mean() > 0.9:
                return low.map(names).to_numpy()
    if "geometry" in t.columns:
        k = key_values(t, states, "L0USA_St_1")
        if k is not None:
            return np.asarray(k)
    return None


def checks(ctx):
    import geopandas as gpd
    out = []
    pngs = ctx.pred_files("*.png")
    ok_png = [p for p in pngs if png_not_blank(ctx.pred(p))[0]]
    out.append(result("map saved and not blank", bool(ok_png), value=pngs))

    states = gpd.read_file(f"{ctx.inputs_dir}/USA_States.shp")
    try:
        gold = _gold(ctx).set_index("STATE_ABBR")
    except Exception as e:
        out.append(result("join + homeless rate per state correct", False, note=f"gold missing {e!r}"[:200]))
        return out
    grate = gold["HomelessPo"].astype(float) if "HomelessPo" in gold.columns else \
        gold[[c for c in gold.columns if c.startswith("HomelessP")][0]].astype(float)
    gchg = gold["Change"].astype(float)

    code = _strip_comments(ctx.code_text())   # a file named only in a comment is not read
    read_xlsx = "homeless_data" in code
    hpe = states.set_index("L0USA_St_1")["HomelessPe"].astype(float)
    best, bestc = (0.0, None), (0.0, None)
    copies = []
    acc_rate, acc_chg = set(), set()
    for name, t in _tables(ctx):
        k = _keys(t, states)
        if k is None:
            continue
        for c in t.columns:
            if c == "geometry" or c in PREJOINED or t[c].dtype.kind not in "fiu":
                continue
            s = t[c].astype(float).groupby(k).mean()
            common = grate.index.intersection(s.index)
            if len(common) < 0.9 * len(grate):
                continue
            v, g = s.loc[common].to_numpy(), grate.loc[common].to_numpy()
            hp = hpe.reindex(common).to_numpy()
            for sn, f in SCALES.items():
                ok = (np.abs(v / f - g) <= np.maximum(0.05 * g, 0.6))
                share = float(ok.sum() / len(grate))
        # a copy of the rounded HomelessPe field already in the input shapefile (e.g. a renamed
        # copy) is not the agent's join + calculation
                if share >= 0.95 and not read_xlsx and np.nanmean(np.abs(v / f - hp) < 1e-6) >= 0.95:
                    copies.append(f"{c} in {name}")
                    continue
                if share >= 0.95:
                    acc_rate.add(c)
                if share > best[0]:
                    best = (share, f"{c} in {name} [{sn}]")
            gc = gchg.loc[common].to_numpy()
            for f in (1.0, 100.0):
                share = float((np.abs(v / f - gc) <= 0.01 + 0.02 * np.abs(gc)).sum() / len(gchg))
                if share >= 0.95 and read_xlsx:
                    acc_chg.add(c)
                if share > bestc[0]:
                    bestc = (share, f"{c} in {name}")
    # The instruction asks for the JOIN (homeless_data.xlsx -> states layer on the state
    # abbreviation) and the maps; it names no calculated field. Required: the agent's own join
    # (xlsx read; a table of the states layer - geometry or the layer's own attribute columns -
    # carrying an xlsx column, Homeless13 / Pop13 / Change, or a value derived from them, aligned
    # per state for >= 95 % of states). The pre-joined Sheet1__* / HomelessPe fields of the input
    # shapefile are not the agent's join. The calculated field (rate / change) is optional.
    gref = {"Homeless13": gold["Homeless13"].astype(float), "Pop13": gold["Pop13"].astype(float)}
    jbest, acc_cnt = (0.0, None), set()
    for name, t in _tables(ctx):
        if not ("geometry" in t.columns or set(t.columns) & STATE_COLS):
            continue
        k = _keys(t, states)
        if k is None:
            continue
        for c in t.columns:
            if c == "geometry" or c in PREJOINED or t[c].dtype.kind not in "fiu":
                continue
            s = t[c].astype(float).groupby(k).mean()
            common = grate.index.intersection(s.index)
            if len(common) < 0.9 * len(grate):
                continue
            v = s.loc[common].to_numpy()
            cands = [(gn, g.loc[common].to_numpy(), lambda v_, g_: np.abs(v_ - g_) <= 0.005 * np.abs(g_) + 0.5)
                     for gn, g in gref.items()]
            cands += [(f"Change x{f:g}", gchg.loc[common].to_numpy() * f,
                       lambda v_, g_, f=f: np.abs(v_ - g_) <= f * (0.01 + 0.02 * np.abs(g_ / f))) for f in (1.0, 100.0)]
            cands += [(f"rate {sn}", grate.loc[common].to_numpy() * f,
                       lambda v_, g_, f=f: np.abs(v_ - g_) <= f * np.maximum(0.05 * g_ / f, 0.6)) for sn, f in SCALES.items()]
            for gn, g, fn in cands:
                share = float(fn(v, g).sum() / len(grate))
                if share >= 0.95 and gn == "Homeless13":
                    acc_cnt.add(c)
                if share > jbest[0]:
                    jbest = (share, f"{c} in {name} ~ {gn}")
    out.append(result("join: homeless_data.xlsx joined to the states layer on the abbreviation (values aligned per state)",
                      read_xlsx and jbest[0] >= 0.95, value=round(jbest[0], 3), threshold=">= 0.95 of states; code reads homeless_data.xlsx",
                      note=(jbest[1] or "no states-layer table carrying an xlsx column") + f" (xlsx read: {read_xlsx})"))
    # The instruction mentions "homelessness trends" and names no calculated field; a per-capita
    # rate (as in the benchmark's script) or the 2012-13 relative change joined from the xlsx
    # (the trend) are both accepted as the calculated field. Optional.
    ok_rate = best[0] >= 0.95
    ok_chg = bool(acc_chg)
    out.append(result("calculated field per state (rate, or 2012-13 change joined from the xlsx)", ok_rate or ok_chg,
                      required=False, value=[round(best[0], 3), round(bestc[0], 3)], threshold=0.95,
                      note=(best[1] or "no per-state rate column") + f"; change: {bestc[1]} (xlsx read: {read_xlsx})"
                      + (f"; copies of HomelessPe rejected: {copies[:2]}" if copies else "")))
    out.append(result("2012-13 change carried per state", bestc[0] >= 0.95, value=round(bestc[0], 3), required=False,
                      note=bestc[1] or ""))

    # The map must show the joined homelessness data - the rate, the 2012-13 change, or the
    # 2013 homeless count (the instruction asks for no rate); population alone fails
    acc = acc_rate | acc_chg | (acc_cnt if read_xlsx else set())
    drawn = [(t_, m, np.asarray(a, float).ravel()) for t_, m, a in drawn_arrays(ctx)
             if m["name"].endswith("_values") and 40 <= np.asarray(a).size <= 400]
    chits = []
    try:
        from checklib import drawn_color_column_matches
        chits = [h for h in drawn_color_column_matches(ctx, min_rho=0.9) if h[1] in acc]
    except Exception:
        pass
        # colours are inverted locally as well, with |rho| (reversed colormaps) and the
        # class-IoU criterion for classified maps, collapsing polygon parts (checklib's helper uses a
        # signed rho and a fixed rho threshold, which classified schemes cannot reach)
    fcs = []
    if not chits:
        try:
            from checklib import _cmap_positions
            for t_, m, a in drawn_arrays(ctx):
                a = np.asarray(a, float)
                if m["name"].endswith("_facecolors") and a.ndim == 2 and 40 <= len(a) <= 400 \
                        and len(np.unique(np.round(a[:, :3], 4), axis=0)) >= 3:
                    pos, cname, res = _cmap_positions(a)
                    if pos is not None and res <= 0.02:
                        fcs.append((t_, m, pos, cname))
        except Exception:
            fcs = []
    if chits:
        h = chits[0]
        out.append(result("map shows the joined homelessness data (drawn colour values rank-match it, rho >= 0.9)", True,
                          value=f"colours of {h[4]} ~ {h[1]} in {h[0]} (rho {h[2]:.3f}, {h[3]})"))
    elif not drawn and not any(m["name"].endswith("_facecolors") for _, m, _ in drawn_arrays(ctx)):
        out.append(unverifiable("map shows the joined homelessness data",
                                "no colour values captured from any figure (choropleth not drawn from data?)"))
    else:
        hit = None
        drawn = drawn or []
        for name, t in _tables(ctx):
            for c in sorted(acc & set(t.columns)):
                vals = t[c].astype(float).to_numpy()
                parts = None
                if "geometry" in t.columns:  # geopandas draws one patch per polygon part
                    try:
                        parts = np.array([len(getattr(g_, "geoms", [g_])) if g_ is not None else 0
                                          for g_ in t.geometry])
                    except Exception:
                        parts = None
                fin = np.isfinite(vals)
                for tg, m, d in [(tg, m, d) for tg, m, d in drawn] + [(tg, m, p_) for tg, m, p_, _ in fcs]:
                    for vv, pp in ((vals, parts), (vals[fin], parts[fin] if parts is not None else None)):
                        r_ = _drawn_match(vv, d, pp)
                        if r_ is not None:
                            hit = f"{m['name']}@{tg} ~ {c} in {name} (rho {r_[0]:.3f}, class IoU {r_[1]:.2f}, {r_[2]} classes)"
                            break
                    if not hit and m['name'].endswith("_values"):   # one axes of a multi-axes map
                        r_ = _subset_match(vals, d, parts)
                        if r_ is not None:
                            hit = (f"{m['name']}@{tg} ~ {c} in {name}, subset of the rows drawn in this axes "
                                   f"(rho {r_[0]:.3f}, {r_[2]} classes)")
                    if hit:
                        break
                if hit:
                    break
            if hit:
                break
        out.append(result("map shows the joined homelessness data (drawn colour values rank-match it, rho >= 0.9)",
                          hit is not None, value=hit, note="" if hit else
                          f"drawn: {[m['name'] for _, m, _ in drawn][:4]}; accepted columns: {sorted(acc)[:4]}"))
    out.append(result("tabular file read and joined (code uses homeless_data.xlsx)", "homeless_data" in code,
                      required=False))
    return out


def _strip_comments(code):
    """Code text without comments (a method named only in a comment is not used)."""
    import io
    import tokenize
    try:
        return tokenize.untokenize(t for t in tokenize.generate_tokens(io.StringIO(code).readline)
                                   if t.type != tokenize.COMMENT)
    except Exception:
        return "\n".join(l for l in code.splitlines() if not l.lstrip().startswith("#"))
