"""Task 11 - slope and aspect at coral/sponge observations, Catalina Island.

The gold per-point slope/aspect are recomputed here from the inputs (Horn 3x3, true 88 m cell
size, elevation surface, points projected to EPSG:3310).

Correctness (required):
  1. CoralandSponge.png exists and is not blank;
  2. database updated: a captured point table (~7 658 rows) carries a slope column (degrees or
     percent) with Spearman >= 0.9 against the gold per-point slope, and an aspect column within
     25 degrees of the gold aspect for >= 80 % of points (aligned on OBJECTID or row order).
     Wrong cell size, unprojected points or aspect of the depth surface (180 degrees off) fail;
  3. per-category means: a captured table/series indexed by VernacularNameCategory (or
     ScientificName) with a mean-slope column within max(0.5 deg, 20 %) of the gold for >= 80 %
     of categories with >= 20 observations, and a mean-aspect column within 25 degrees of the gold
     circular mean OR of the gold arithmetic mean for >= 80 % of those categories.
     When no per-category table exists (means computed inline in the plotting call, e.g.
     groupby().mean().plot.barh(); bar heights are not captured) and the per-point check passed,
     the group means of the agent's own per-point slope/aspect columns are tested instead,
     provided the code names the category column, groups (groupby / pivot_table / crosstab /
     a loop) and averages (mean / aggfunc); both category keys are tried, the better one kept.
  4. figure shows per-category mean slope and mean aspect (the instruction asks to visualize the
     mean slope and aspect distribution of each species): among the arrays
     drawn in the figure saved as CoralandSponge.png (bar heights / bar widths, scatter offsets,
     line x/y; on polar axes also the bar/point angle converted to degrees), one series must match
     the gold per-category mean slope and one the gold mean aspect, each one-to-one (optimal
     assignment) for >= 80 % of the categories with n >= 20, same tolerances as check 3.
     Candidate series need 0.8*n_categories .. 2.5*n_categories (<= 250) finite values and must
     not be constant or evenly spaced (bar positions / bar thickness), so raw per-point scatters,
     histograms of all points and charts of slope alone fail. A series whose matched values are
     mostly (> 50 %) exact per-point values (gold or the agent's own per-point columns) is a plot
     of raw observations (one category's points resemble the means of its taxa) and is ignored.
     When a drawn series can be tied to category labels - >= 80 %
     of its values equal exactly one per-category number the agent computed (numeric column of a
     captured category-labelled table / series, or a group mean of the agent's own per-point
     columns), per key - each category's drawn value is compared with that category's gold mean
     (share over all gold categories with n >= 20); the value-set assignment is kept only for
     series that cannot be tied to labels.
Reported, not required:
  5. aspect averaged as a circular quantity (matches the circular mean better than the arithmetic).
"""
import numpy as np

from checklib import drawn_arrays, file_exists, png_not_blank, result

CRS = "EPSG:3310"


def _gold(ctx):
    import geopandas as gpd
    import rasterio
    with rasterio.open(f"{ctx.inputs_dir}/CatalinaBathymetry.tif") as s:
        z, tr = s.read(1).astype(float), s.transform
    zp = np.pad(z, 1, mode="edge")
    a, b, c = zp[:-2, :-2], zp[:-2, 1:-1], zp[:-2, 2:]
    d, f = zp[1:-1, :-2], zp[1:-1, 2:]
    g, h, i = zp[2:, :-2], zp[2:, 1:-1], zp[2:, 2:]
    dx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * tr.a)
    dy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * tr.a)
    slope = np.degrees(np.arctan(np.hypot(dx, dy)))
    asp = (np.degrees(np.arctan2(dx, -dy)) + 180) % 360
    asp[(dx == 0) & (dy == 0)] = -1
    pts = gpd.read_file(f"{ctx.inputs_dir}/CoralandSpongeCatalina.geojson")
    p = pts.to_crs(CRS)
    inv = ~tr
    cc, rr = inv * (p.geometry.x.values, p.geometry.y.values)
    rr, cc = np.floor(rr).astype(int), np.floor(cc).astype(int)
    pts["g_slope"] = slope[rr, cc]
    pts["g_aspect"] = asp[rr, cc]
    return pts


def _angdiff(a, b):
    d = np.abs((np.asarray(a, float) - np.asarray(b, float) + 180) % 360 - 180)
    return d


def _circ(x):
    x = x[x >= 0]
    if not len(x):
        return np.nan
    r = np.radians(x)
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360)


def pd_series(v, cat):
    """Arithmetic group means of per-point values by category (aspect: ignore flat -1)."""
    import pandas as pd
    s = pd.Series(v)
    s[s < 0] = np.nan
    return s.groupby(pd.Series(cat)).mean()


def checks(ctx):
    from scipy.stats import spearmanr
    out = []
    names = ["CoralandSponge.png"]
    r = file_exists(ctx, names)
    out.append(r)
    if r["passed"]:
        nb, std = png_not_blank(ctx.pred(names[0]))
        out.append(result("figure not blank", nb, value=std, threshold=">=3 grey-level std"))
    else:
        out.append(result("figure not blank", False, note="no figure"))
    try:
        gold = _gold(ctx)
    except Exception as e:
        out.append(result("points updated with correct slope and aspect", False, note=repr(e)[:200]))
        out.append(result("per-category mean slope and aspect match gold", False))
        out.append(result("figure shows per-category mean slope and mean aspect", False, note="no gold"))
        return out
    n = len(gold)
    gs, ga = gold["g_slope"].to_numpy(), gold["g_aspect"].to_numpy()

    # --- 2. per-point -------------------------------------------------------------------------
    best = (0.0, 0.0, None)
    point_tab = None
    seen = set()
    for kind in ("gdf", "df"):
        for tag, meta, load in ctx.objects("pred", kind):
            rows = meta.get("rows", 0)
            sig = (meta.get("name"), rows, tuple(meta.get("columns", [])))
            if sig in seen or abs(rows - n) > 0.05 * n:
                continue
            seen.add(sig)
            try:
                o = load()
                if "OBJECTID" in o.columns and o["OBJECTID"].isin(gold["OBJECTID"]).mean() > 0.95:
                    pos = gold.reset_index().set_index("OBJECTID").loc[o["OBJECTID"], "index"].to_numpy()
                elif rows == n:
                    pos = np.arange(n)
                else:
                    continue
                s_best, a_best = 0.0, 0.0
                s_col, a_col = None, None
                for c in o.columns:
                    if c in ("geometry", "OBJECTID", "latitude", "longitude", "DepthInMeters", "index") or o[c].dtype.kind not in "fi":
                        continue
                    v = o[c].astype(float).to_numpy()
                    m = np.isfinite(v)
                    if m.sum() < 0.8 * len(v):
                        continue
                    rho = spearmanr(v[m], gs[pos][m]).correlation
                    if rho == rho and float(rho) > s_best:
                        s_best, s_col = float(rho), c
                    ok = m & (ga[pos] >= 0) & (v >= -1) & (v <= 360)
                    if ok.sum() > 0.8 * len(v):
                        sh = float(np.mean(_angdiff(v[ok], ga[pos][ok]) <= 25))
                        if sh > a_best:
                            a_best, a_col = sh, c
                if min(s_best / 0.9, a_best / 0.8) > min(best[0] / 0.9, best[1] / 0.8):
                    best = (s_best, a_best, f"{meta['name']}@{tag}")
                    point_tab = (o, pos, s_col, a_col)
            except Exception:
                continue
    out.append(result("points updated with correct slope and aspect", best[0] >= 0.9 and best[1] >= 0.8,
                      value={"slope_spearman": round(best[0], 3), "aspect_within_25deg": round(best[1], 3)},
                      threshold="spearman >= 0.9, >= 80 % within 25 deg", note=best[2] or "no point table"))

    # --- 3. per-category means -----------------------------------------------------------------
    golds = {}
    for key in ("VernacularNameCategory", "ScientificName"):
        grp = gold.groupby(key)
        t = grp.agg(n=("g_slope", "size"), slope=("g_slope", "mean"))
        t["slope_pct"] = grp["g_slope"].apply(lambda s: np.mean(np.tan(np.radians(s)) * 100))
        t["asp_circ"] = grp["g_aspect"].apply(lambda s: _circ(s.to_numpy()))
        t["asp_arith"] = grp["g_aspect"].apply(lambda s: s[s >= 0].mean())
        golds[key] = t[t.n >= 20]
    bs, ba = (0.0, None), (0.0, None, None)
    for kind in ("df", "series", "gdf"):
        for tag, meta, load in ctx.objects("pred", kind):
            rows = meta.get("rows", 0)
            if rows < 5 or rows > 200:
                continue
            try:
                o = load()
                strcols = [c for c in o.columns if o[c].dtype == object]
                for key, gt in golds.items():
                    for sc in strcols:
                        names_ = o[sc].astype(str)
                        common = [x for x in gt.index if x in set(names_)]
                        if len(common) < 0.8 * len(gt):
                            continue
                        oo = o.assign(_k=names_).groupby("_k").first().loc[common]
                        g = gt.loc[common]
                        s_ok, a_ok, circ_pref = 0.0, 0.0, None
                        for c in oo.columns:
                            if oo[c].dtype.kind not in "fi":
                                continue
                            v = oo[c].astype(float).to_numpy()
                            for ref in ("slope", "slope_pct"):
                                tol = np.maximum(0.5 if ref == "slope" else 0.9, 0.2 * g[ref].abs().to_numpy())
                                s_ok = max(s_ok, float(np.mean(np.abs(v - g[ref].to_numpy()) <= tol)))
                            dc = float(np.mean(_angdiff(v, g["asp_circ"].to_numpy()) <= 25))
                            da = float(np.mean(_angdiff(v, g["asp_arith"].to_numpy()) <= 25))
                            if max(dc, da) > a_ok:
                                a_ok, circ_pref = max(dc, da), dc >= da
                        # slope and aspect means may live in separate tables/series (e.g. two dicts)
                        if s_ok > bs[0]:
                            bs = (s_ok, f"slope: {meta['name']}@{tag} by {key}")
                        if a_ok > ba[0]:
                            ba = (a_ok, f"aspect: {meta['name']}@{tag} by {key}", circ_pref)
            except Exception:
                continue
    if not (bs[0] >= 0.8 and ba[0] >= 0.8) and point_tab is not None and best[0] >= 0.9 and best[1] >= 0.8:
        # per-category means computed inline in the plotting call (bar heights are not captured):
        # recompute them from the agent's own verified per-point columns, provided the code groups
        # by the category and averages.
        import re
        code = re.sub(r"#[^\n]*", "", ctx.code_text())     # comments are not evidence
        # Any grouping idiom counts (groupby(by=...), a variable holding the
        # column name, pivot_table, crosstab, a per-category loop), as long as the category column
        # is named in the code and an average is taken; both keys are tried, the better one kept.
        groups = re.search(r"groupby\s*\(|pivot_table\s*\(|crosstab\s*\(|\bfor\b", code)
        averages = re.search(r"\bmean\b|aggfunc|\baverage\b", code)
        keys = [k for k in ("VernacularNameCategory", "ScientificName") if k in code]
        if groups and averages and keys:
            o, pos, s_col, a_col = point_tab
            for key in keys:
                cat = gold[key].to_numpy()[pos]
                gt = golds[key]
                sv = pd_series(o[s_col].astype(float).to_numpy(), cat)
                av = pd_series(o[a_col].astype(float).to_numpy(), cat)
                common = [x for x in gt.index if x in sv.index]
                if len(common) < 0.8 * len(gt):
                    continue
                g = gt.loc[common]
                s_ok = 0.0
                for ref in ("slope", "slope_pct"):
                    tol = np.maximum(0.5 if ref == "slope" else 0.9, 0.2 * g[ref].abs().to_numpy())
                    s_ok = max(s_ok, float(np.mean(np.abs(sv.loc[common].to_numpy() - g[ref].to_numpy()) <= tol)))
                v = av.loc[common].to_numpy()
                dc = float(np.mean(_angdiff(v, g["asp_circ"].to_numpy()) <= 25))
                da = float(np.mean(_angdiff(v, g["asp_arith"].to_numpy()) <= 25))
                if s_ok > bs[0]:
                    bs = (s_ok, f"slope: group means of {s_col} by {key} (inline grouping in code)")
                if max(dc, da) > ba[0]:
                    ba = (max(dc, da), f"aspect: group means of {a_col} by {key} (inline grouping in code)", dc >= da)
    out.append(result("per-category mean slope and aspect match gold", bs[0] >= 0.8 and ba[0] >= 0.8,
                      value={"slope_share_ok": round(bs[0], 3), "aspect_share_ok": round(ba[0], 3)},
                      threshold=">= 80 % of categories (n >= 20) within tolerance",
                      note="; ".join(x for x in (bs[1], ba[1]) if x) or "no per-category table"))
    out.append(_figure_means_check(ctx, gold, golds, point_tab))
    out.append(result("aspect averaged circularly", bool(ba[2]), required=False))
    return out


# --- 4. what the figure draws -----------------------------------------------------------------
def _drawn_series(ctx, fname="CoralandSponge.png"):
    """Candidate 1-D value series drawn in the figure that ended up in `fname`."""
    arrs = drawn_arrays(ctx, "pred", file=fname)
    polar = set()
    for tag, m, a in arrs:
        if m.get("artist") == "axes_limits":
            lim = np.asarray(a, float).ravel()
            if len(lim) >= 2 and abs(lim[0]) < 1e-6 and abs(lim[1] - 2 * np.pi) < 1e-3:
                polar.add((tag, m["name"].rsplit("_", 1)[0]))       # (tag, "fig_ax<i>")
    series = []
    for tag, m, a in arrs:
        nm, art = m["name"], m.get("artist")
        ax = (tag, nm.split("_")[0] + "_" + nm.split("_")[1]) if nm.count("_") >= 2 else None
        a = np.asarray(a, float)
        if a.ndim != 2:
            continue
        cols = []
        if art == "bars" and a.shape[1] == 4:
            cols += [("bar heights", a[:, 3]), ("bar widths", a[:, 2])]
            if ax in polar:
                cols.append(("polar bar angle", np.degrees(a[:, 0] + a[:, 2] / 2) % 360))
        elif (art == "line" or nm.endswith("_offsets")) and a.shape[1] == 2:
            cols += [("x", a[:, 0]), ("y", a[:, 1])]
            if ax in polar:
                cols.append(("polar angle", np.degrees(a[:, 0]) % 360))
        for what, v in cols:
            v = v[np.isfinite(v)]
            if len(v) < 3 or np.ptp(v) <= 1e-9:
                continue
            dv = np.diff(np.sort(v))
            if np.ptp(dv) <= 1e-6 * max(1.0, np.ptp(v)):      # evenly spaced: positions, not values
                continue
            series.append((f"{nm}@{tag} {what}", np.abs(v) if "bar" in what else v))
    return series


def _one_to_one_share(cost):
    """Share of gold rows matched to a distinct drawn value with cost <= 1 (optimal assignment),
    and the indices of those drawn values."""
    from scipy.optimize import linear_sum_assignment
    c = np.where(cost <= 1, 0.0, 1.0) + 1e-3 * np.minimum(cost, 1.0)
    r, k = linear_sum_assignment(c)
    ok = cost[r, k] <= 1
    return float(np.sum(ok)) / cost.shape[0], k[ok]


def _exact_share(v, pool):
    """Share of values v that are (to float32 precision) one observation's per-point value."""
    if not len(v) or not len(pool):
        return 0.0
    pool = np.sort(pool)
    j = np.clip(np.searchsorted(pool, v), 1, len(pool) - 1)
    d = np.minimum(np.abs(pool[j - 1] - v), np.abs(pool[j] - v))
    return float(np.mean(d <= 1e-5 * np.maximum(1.0, np.abs(v))))


def _label_pairs(ctx, gold, golds, point_tab):
    """(key, label, value) triples: every per-category number the agent computed that we can see -
    numeric columns of captured category-labelled tables/series (labels in a string column or the
    index) and group means (arithmetic / flat-excluded / circular) of the agent's own per-point
    columns. Used to tie each drawn value to its category label."""
    import pandas as pd
    names = {k: set(map(str, gt.index)) | set(map(str, gold[k].unique())) for k, gt in golds.items()}
    out = []

    def add(labels, vals):
        labels = pd.Series(labels).astype(str).to_numpy()
        for key, nm in names.items():
            hit = np.isin(labels, list(nm))
            if hit.sum() >= 3 and hit.mean() >= 0.5:
                for c, v in vals:
                    v = np.asarray(v, float)
                    m = hit & np.isfinite(v)
                    out.extend((key, l, x) for l, x in zip(labels[m], v[m]))
    for kind in ("df", "series", "gdf"):
        for tag, meta, load in ctx.objects("pred", kind):
            if not (3 <= meta.get("rows", 0) <= 300):
                continue
            try:
                o = load()
                num = [(c, o[c].to_numpy()) for c in o.columns if o[c].dtype.kind in "fi"]
                if not num:
                    continue
                labs = [o[c] for c in o.columns if o[c].dtype == object]
                if o.index.dtype == object:
                    labs.append(o.index.to_series())
                for l in labs:
                    add(l.to_numpy(), num)
            except Exception:
                continue
    if point_tab is not None:
        o, pos, _, _ = point_tab
        for key in golds:
            cat = gold[key].to_numpy()[pos]
            for c in o.columns:
                try:
                    if o[c].dtype.kind not in "fi" or c in ("OBJECTID",):
                        continue
                    v = pd.Series(o[c].astype(float).to_numpy())
                    g = v.groupby(cat)
                    pos_v = v.where(v >= 0)
                    variants = [g.mean(), pos_v.groupby(cat).mean(),
                                pos_v.groupby(cat).apply(lambda s: _circ(s.dropna().to_numpy())),
                                v.groupby(cat).apply(lambda s: _circ(np.mod(s.dropna().to_numpy(), 360)))]
                    for r in variants:
                        out.extend((key, str(l), float(x)) for l, x in r.items() if np.isfinite(x))
                except Exception:
                    continue
    return out


def _bind(v, pairs, key):
    """Label (category of `key`) of each drawn value: exact match to one agent-computed
    per-category number; None if absent or ambiguous. Returns {label: (key, value)} or None if
    < 80 % of the values bind. Keys are bound separately (a category holding one taxon has the
    same mean under both keys)."""
    pairs = [p for p in pairs if p[0] == key]
    if not pairs:
        return None
    keys = np.array([p[0] for p in pairs]); labs = np.array([p[1] for p in pairs])
    vals = np.array([p[2] for p in pairs], float)
    bound, n_ok = {}, 0
    for x in v:
        d = np.minimum(np.abs(vals - x), _angdiff(vals, x))
        m = d <= 1e-5 * max(1.0, abs(x))
        ls = set(zip(keys[m], labs[m]))
        if len({l for _, l in ls}) == 1:
            (k, l), = list(ls)[:1]
            n_ok += 1
            bound[l] = (k, float(x))
    return bound if v.size and n_ok >= 0.8 * v.size else None


def _figure_means_check(ctx, gold, golds, point_tab=None):
    name = "figure shows per-category mean slope and mean aspect"
    try:
        series = _drawn_series(ctx)
    except Exception as e:
        return result(name, False, note=repr(e)[:200])
    if not series:
        return result(name, False, note="no drawn bars / points / lines captured for CoralandSponge.png")
    # per-point values (gold, and the agent's own per-point columns): a drawn series whose matched
    # values are mostly single observations' values is a raw-point plot, not a chart of means
    # (a category's raw points look like the means of its taxa).
    s_pool = [gold["g_slope"].to_numpy(), np.tan(np.radians(gold["g_slope"].to_numpy())) * 100]
    a_pool = [gold["g_aspect"].to_numpy()]
    if point_tab is not None:
        o, _, s_col, a_col = point_tab
        for col, pool in ((s_col, s_pool), (a_col, a_pool)):
            if col is not None:
                pool.append(o[col].astype(float).to_numpy())
    s_pool = np.concatenate(s_pool); s_pool = s_pool[np.isfinite(s_pool)]
    a_pool = np.concatenate(a_pool); a_pool = a_pool[np.isfinite(a_pool)]
    bs, ba = (0.0, None), (0.0, None)
    raw = []
    try:
        pairs = _label_pairs(ctx, gold, golds, point_tab)
    except Exception:
        pairs = []
    bindings = {(lab, key): _bind(v, pairs, key) for lab, v in series for key in golds}
    for key, gt in golds.items():
        n_all = gold[key].nunique()
        lo, hi = 0.8 * len(gt), min(250, 2.5 * n_all)
        for lab, v in series:
            if not (lo <= len(v) <= hi):
                continue
            bd = bindings.get((lab, key))
            if bd is not None:
                # the drawn values are tied to their category
                # labels, so each category's drawn mean is compared with that category's gold mean
                # (a value-set match would let wrong per-taxon means pass).
                lv = {l: x for l, (k, x) in bd.items() if k == key}
                cats = [c for c in gt.index if str(c) in lv]
                if not cats:
                    continue
                dv = np.array([lv[str(c)] for c in cats])
                for ref in ("slope", "slope_pct"):
                    g = gt.loc[cats, ref].to_numpy()
                    tol = np.maximum(0.5 if ref == "slope" else 0.9, 0.2 * np.abs(g))
                    sh = float(np.sum(np.abs(dv - g) <= tol)) / len(gt)
                    if sh > bs[0]:
                        bs = (sh, f"slope: {lab} by {key} ({ref}, labels bound)")
                for ref in ("asp_circ", "asp_arith"):
                    g = gt.loc[cats, ref].to_numpy()
                    sh = float(np.sum(_angdiff(dv, g) <= 25)) / len(gt)
                    if sh > ba[0]:
                        ba = (sh, f"aspect: {lab} by {key} ({ref}, labels bound)")
                continue
            for ref in ("slope", "slope_pct"):
                g = gt[ref].to_numpy()
                tol = np.maximum(0.5 if ref == "slope" else 0.9, 0.2 * np.abs(g))
                sh, k = _one_to_one_share(np.abs(v[None, :] - g[:, None]) / tol[:, None])
                if len(k) and _exact_share(v[k], s_pool) > 0.5:
                    raw.append(lab); continue
                if sh > bs[0]:
                    bs = (sh, f"slope: {lab} by {key} ({ref})")
            for ref in ("asp_circ", "asp_arith"):
                g = gt[ref].to_numpy()
                sh, k = _one_to_one_share(_angdiff(v[None, :], g[:, None]) / 25.0)
                if len(k) and _exact_share(v[k], a_pool) > 0.5:
                    raw.append(lab); continue
                if sh > ba[0]:
                    ba = (sh, f"aspect: {lab} by {key} ({ref})")
    return result(name, bs[0] >= 0.8 and ba[0] >= 0.8,
                  value={"slope_share_drawn": round(bs[0], 3), "aspect_share_drawn": round(ba[0], 3)},
                  threshold=">= 80 % of categories (n >= 20) drawn within tolerance, one-to-one",
                  note="; ".join(x for x in (bs[1], ba[1]) if x) or f"{len(series)} drawn series, none matches"
                  + (f"; ignored as per-point values: {sorted(set(raw))[:3]}" if raw else ""))
