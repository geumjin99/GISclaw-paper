"""Checks for task 46: peak-hour crash hot spots (Getis-Ord Gi*) on the Brevard County road network.

Task core: weekday 3-5 pm crashes -> snapped (0.25 mi) to road segments -> crash count/rate per
segment (all 69 531 segments kept) -> Gi* hot-spot analysis with the supplied NETWORK spatial
weights (nwswm360ft.swm; the domain knowledge stresses network distance) -> hot-spot map.

Correctness (required):
  1. a map (PNG) exists and is not blank;
  2. peak selection: a captured crash table has exactly the weekday crashes with hour 15-17
     (24 746, original SQL) or 15-16 ("3 pm to 5 pm" read as [15:00, 17:00), 16 515) or the closed
     interval [15:00, 17:00] (hours 15-16 plus crashes timed exactly 17:00, 16 978),
     +/-0.5 %, and contains only weekday rows in that window;
  3. per-segment crash measure: a table with one row per road segment (69 531 +/- 2 %,
     aligned on UniqID / row order) has a column whose Pearson r with the gold count or rate
     (either hour reading) is >= 0.9;
  4. Gi* on the network weights: some per-segment numeric column / array is a Gi(*) z-score of
     that crash measure under the supplied SWM: Pearson >= 0.9 with at least one of the
     variants {inverse-distance SWM weights with self weight 0 / 1 / row max, binary SWM
     neighbours with self weight 0 / 1} x {no / row standardisation} x {count / rate} x
     {15-17 / 15-16}, all computed by the checker. NB: with the SWM's inverse-distance weights
     (~0.003) a self weight of 1 (ArcGIS Gi* default, esda star=True) makes Gi* == the
     standardised rate (r = 0.9999). Those variants are excluded, and any candidate with
     r >= 0.95 to the plain count/rate is rejected (a plain z-score of
     the rate is not a spatial statistic). If the code used Gi* + the SWM in that degenerate convention the check
     is reported UNVERIFIABLE (not passed). Gi* on crash points, Local Moran's I, or
     Euclidean/KNN weights (r 0.26-0.63 with the network results) fail.
     Candidates must look like z-scores (centred, some negative values).
Reported, not required:
  5. FDR correction mentioned in code (the original applies it);
  6. hot spots found: >= 0.5 % of segments with z > 1.96 in the best-matching column.
"""
import numpy as np

from checklib import file_exists, png_not_blank, result, find_object, unverifiable

N = 69531
WEEKDAYS = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday"}
_CACHE = {}


def _inputs(ctx):
    if "roads" in _CACHE and _CACHE.get("dir") == ctx.inputs_dir:
        return _CACHE
    import geopandas as gpd
    import libpysal
    import scipy.sparse as sp
    roads = gpd.read_file(f"{ctx.inputs_dir}/roads.shp")
    cr = gpd.read_file(f"{ctx.inputs_dir}/crashes.shp", columns=["CRASH_DOW", "CRASH_HOD", "CRASH_TM", "CrashID"])
    wd = cr.CRASH_DOW.isin(WEEKDAYS)
    meas = {}
    for lab, hi in (("15-17", 17), ("15-16", 16)):
        sel = cr[wd & cr.CRASH_HOD.between(15, hi)]
        j = gpd.sjoin_nearest(sel[["geometry"]], roads[["UniqID", "geometry"]], max_distance=1320.0)
        j = j[~j.index.duplicated()]
        cnt = roads.UniqID.map(j.groupby("UniqID").size()).fillna(0).to_numpy(float)
        meas["count " + lab] = cnt
        meas["rate " + lab] = cnt / roads.Shape_Leng.to_numpy(float)
    w = libpysal.io.open(f"{ctx.inputs_dir}/nwswm360ft.swm", "r").read()
    pos = {u: i for i, u in enumerate(roads.UniqID.tolist())}
    ri, ci, vi = [], [], []
    for k, nb in w.neighbors.items():
        for b, v in zip(nb, w.weights[k]):
            ri.append(pos[k]); ci.append(pos[b]); vi.append(v)
    W = sp.csr_matrix((vi, (ri, ci)), shape=(len(roads), len(roads)))
    # closed interval [15:00, 17:00]: hours 15-16 plus crashes timed exactly 17:00
    closed = wd & (cr.CRASH_HOD.between(15, 16) | (cr.CRASH_TM.astype(str).str.strip() == "17:00"))
    _CACHE.update(dir=ctx.inputs_dir, roads=roads, meas=meas, W=W, n_sel={"15-17": int((wd & cr.CRASH_HOD.between(15, 17)).sum()),
                                                                          "15-16": int((wd & cr.CRASH_HOD.between(15, 16)).sum()),
                                                                          "15-16+17:00": int(closed.sum())})
    return _CACHE


def _gi_variants(W, meas):
    import scipy.sparse as sp
    n = W.shape[0]
    rowmax = np.asarray(W.max(axis=1).todense()).ravel()
    out = {}
    B = (W > 0).astype(float)
    combos = [("inv/self0", W, np.zeros(n)), ("inv/self1", W, np.ones(n)),
              ("inv/selfmax", W, np.where(rowmax > 0, rowmax, 1.0)),
              ("bin/self0", B, np.zeros(n)), ("bin/self1", B, np.ones(n))]
    for sname, W0, sw_ in combos:
        Ws = W0 + sp.diags(sw_)
        for std in ("none", "row"):
            M = Ws
            if std == "row":
                rs = np.asarray(Ws.sum(axis=1)).ravel()
                M = sp.diags(np.where(rs > 0, 1 / np.where(rs > 0, rs, 1), 0)) @ Ws
            s1 = np.asarray(M.sum(axis=1)).ravel()
            s2 = np.asarray(M.multiply(M).sum(axis=1)).ravel()
            for mname, x in meas.items():
                xb, S = x.mean(), x.std()
                with np.errstate(divide="ignore", invalid="ignore"):
                    z = (M @ x - xb * s1) / (S * np.sqrt((n * s2 - s1 ** 2) / (n - 1)))
                out[f"{mname}/{sname}/{std}"] = z
    return out


def _pearson(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 0.8 * len(a) or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _segment_columns(ctx, roads):
    """(name, aligned numpy vector of length N) for numeric per-segment columns/arrays."""
    out = []
    uid = roads.UniqID.to_numpy()
    for tag, meta, load in ctx.objects("pred"):
        k = meta.get("kind")
        if k in ("gdf", "df") and abs(meta.get("rows", 0) - N) <= 0.02 * N:
            try:
                obj = load()
            except Exception:
                continue
            key = None
            for c in ("UniqID", "UniqID_left", "UniqID_right", "UNIQID", "uniqid"):
                if c in obj.columns:
                    key = obj[c].to_numpy()
                    break
            if key is None:
                if len(obj) == N:
                    key = uid
                else:
                    continue
            idx = {u: i for i, u in enumerate(uid)}
            pos = np.array([idx.get(u, -1) for u in key])
            okp = pos >= 0
            for c in obj.columns:
                if c in ("geometry", "UniqID", "index", "level_0", "LengthMete", "Shape_Leng", "index_right") or \
                        obj[c].dtype.kind not in "fiu":
                    continue
                v = np.full(N, np.nan)
                vals = obj[c].to_numpy(float)
                # aggregate duplicates by mean
                s = np.bincount(pos[okp], weights=np.nan_to_num(vals[okp]), minlength=N)
                cnt = np.bincount(pos[okp], minlength=N)
                v[cnt > 0] = s[cnt > 0] / cnt[cnt > 0]
                out.append((f"{c} in {meta['name']}@{tag}", v))
        elif k == "array" and meta.get("shape") == [N]:
            try:
                out.append((f"{meta['name']}@{tag}", load().astype(float)))
            except Exception:
                pass
    return out


def checks(ctx):
    out = []
    pngs = ctx.pred_files("*.png")
    out.append(file_exists(ctx, pngs or ["<any>.png"], name="map present"))
    if pngs:
        nb, std = png_not_blank(ctx.pred(pngs[0]))
        out.append(result("map not blank", nb, value=std, threshold=">=3 grey-level std"))
    else:
        out.append(result("map not blank", False, note="no PNG"))

    try:
        I = _inputs(ctx)
    except Exception as e:
        out.append(result("inputs readable", False, note=repr(e)[:200]))
        return out
    roads, meas, W, nsel = I["roads"], I["meas"], I["W"], I["n_sel"]

    # 2 peak selection
    sel_ok, note = False, "no captured crash table with the peak-hour row count"
    for tag, meta, load in ctx.objects("pred"):
        if meta.get("kind") not in ("gdf", "df"):
            continue
        rows = meta.get("rows", 0)
        hit = [lab for lab, v in nsel.items() if abs(rows - v) <= 0.005 * v]
        if not hit:
            continue
        try:
            obj = load()
        except Exception:
            continue
        if "CRASH_DOW" in obj.columns and "CRASH_HOD" in obj.columns:
            good = obj.CRASH_DOW.isin(WEEKDAYS).all() and obj.CRASH_HOD.between(15, 17).all()
            if good and hit == ["15-16+17:00"]:  # the closed-interval reading: hour 17 only at 17:00 exactly
                h17 = obj[obj.CRASH_HOD == 17]
                good = "CRASH_TM" in obj.columns and bool((h17.CRASH_TM.astype(str).str.strip() == "17:00").all())
            if good:
                sel_ok, note = True, f"{meta['name']}@{tag}: {rows} rows ({hit[0]})"
                break
            note = f"{meta['name']}@{tag} has {rows} rows but contains non-peak rows"
    if not sel_ok:  # selection never stored as a table: accept per-segment counts summing to the peak total
        for name, v in _segment_columns(ctx, roads):
            tot = np.nansum(v)
            for lab, nv in nsel.items():
                if abs(tot - nv) <= 0.01 * nv and np.allclose(v[np.isfinite(v)], np.round(v[np.isfinite(v)])):
                    gm = meas["count " + ("15-16" if lab == "15-16+17:00" else lab)]
                    if _pearson(v, gm) >= 0.95:
                        sel_ok, note = True, f"per-segment counts {name} sum to {tot:.0f} ({lab})"
    out.append(result("peak selection: weekday 15-17 h (or 15-16 h) crashes", sel_ok, value=note,
                      threshold=nsel))

    # 3 per-segment measure
    cols = _segment_columns(ctx, roads)
    best = (-1.0, None)
    for name, v in cols:
        for mname, g in meas.items():
            r = _pearson(v, g)
            if np.isfinite(r) and r > best[0]:
                best = (r, f"{name} ~ {mname}")
    out.append(result("per-segment crash count/rate matches gold (Pearson >= 0.9)", best[0] >= 0.9,
                      value=round(best[0], 3), threshold=0.9, note=best[1] or "no per-segment table (69 531 rows)"))

    # 4 Gi* on network weights (spatially meaningful variants only)
    # Variants in which the self weight dominates (inverse SWM weights ~0.003 + self weight 1)
    # are identical to the standardised count/rate (r >= 0.9999): they carry no spatial
    # information and are NOT accepted. Every remaining variant correlates <= 0.71 with the
    # plain measure, so a candidate must also stay clearly away from the plain measure.
    variants = _gi_variants(W, meas)
    plain = list(meas.values())
    good_v = {k: z for k, z in variants.items() if max(_pearson(z, y) for y in plain) < 0.95}
    bestg = (-1.0, None, None)
    degenerate = None
    for name, v in cols:
        f = v[np.isfinite(v)]
        # a z-score: centred, with negative values (rules out raw counts/rates)
        if len(f) == 0 or np.std(f) == 0 or f.min() >= 0 or abs(f.mean()) > 0.5 * f.std():
            continue
        rp = max(_pearson(v, y) for y in plain)
        if np.isfinite(rp) and rp >= 0.95:
            degenerate = degenerate or (name, rp)
            continue  # standardised measure, no spatial statistic visible
        for vname, z in good_v.items():
            r = _pearson(v, z)
            if np.isfinite(r) and r > bestg[0]:
                bestg = (r, f"{name} ~ Gi[{vname}]", v)
    gname = "Gi* z-scores on the network SWM (Pearson >= 0.9 with a spatially meaningful Gi* variant)"
    if bestg[0] >= 0.9:
        out.append(result(gname, True, value=round(bestg[0], 3), threshold=0.9, note=bestg[1]))
    elif degenerate is not None:
        code = ctx.code_text().lower()
        used = "swm" in code and any(k in code for k in ("g_local", "getis", "gi*", "gistar", "gi_star"))
        if used:
            out.append(unverifiable(gname, f"{degenerate[0]} equals the standardised crash measure "
                                    f"(r={degenerate[1]:.4f}): Gi* with self weight 1 on the inverse-distance "
                                    "SWM collapses to a non-spatial z-score; no hot-spot clustering is shown"))
        else:
            out.append(result(gname, False, value=round(degenerate[1], 4), threshold="<0.95 with plain measure",
                              note=f"{degenerate[0]} is the standardised crash count/rate - no spatial statistic"))
    else:
        out.append(result(gname, False, value=round(bestg[0], 3), threshold=0.9,
                          note=bestg[1] or "no per-segment z column"))

    code = ctx.code_text()
    out.append(result("FDR correction applied (code)", any(k in code.lower() for k in ("fdr", "benjamini", "multipletests")),
                      required=False))
    try:
        from checklib import drawn_column_matches
        hits = drawn_column_matches(ctx)
        col = bestg[1].split(" in ")[0] if bestg[1] else None
        drew = bool(col) and any(h[1] == col for h in hits)
        out.append(result("map colours drawn from the Gi* column (data-coloured maps only)", drew,
                          value=[h[1] for h in hits][:5], required=False,
                          note="maps coloured class-by-class (colour=) leave no drawn values; reported only"))
    except Exception as e:
        out.append(result("map colours drawn from the Gi* column (data-coloured maps only)", False, required=False, note=repr(e)[:120]))
    if bestg[2] is not None:
        frac = float(np.nanmean(bestg[2] > 1.96))
        out.append(result("hot spots found (>= 0.5 % of segments with z > 1.96)", frac >= 0.005, value=round(frac, 4),
                          required=False))
    return out
