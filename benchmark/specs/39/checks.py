"""T39 - GWR of PctBach ~ PctFB + PctBlack + PctRural for 159 Georgia counties.

Correctness (required):
  1. GA_GWR.png exists and is not blank;
  2. an optimal bandwidth was searched: some captured number equals a bandwidth that mgwr's
     Sel_BW returns on these data for a standard kernel / criterion (adaptive bisquare AICc
     117, AIC 116, BIC 141, CV 112; adaptive gaussian/exponential 50 or 78; fixed kernels
     65.9-342.2 km), within 2 % - searched in scalars, dict / Series values, small arrays and
     small tables (<= 60 values) and numbers in .txt/.json/.md files; OR a captured local
     PctRural / PctBlack surface reproduces a gold *optimum* surface (AICc, fixed-bisquare or
     adaptive-gaussian optimum) at Pearson >= 0.998 (these surfaces are bandwidth-sensitive:
     100 NN gives 0.977 / 0.996, 125 NN 0.9965 / 0.998 for PctRural / PctBlack), which covers
     bandwidths kept only in Python lists or figure titles; OR the code
     shows a search (Sel_BW, or GWR fitted in a loop / >= 3 times plus a min / argmin / sort
     selection) and a captured scalar that is not a for-loop variable, or a number in a
     .txt/.json/.md output, is an adaptive bandwidth whose gold AICc (adaptive bisquare
     profile, 30..155 NN) is within 2 of the minimum (the AICc curve is flat: 95..135 NN). A
     hand grid 25..150 step 25 picking 125 passes; a hard-coded bw=100 (no search) fails;
  3. local GWR coefficients of the right model were estimated: some captured 159-long numeric
     vector (table column, array column, or values drawn in the figure) correlates
     (Pearson >= 0.97) with a gold local slope surface (PctFB / PctBlack / PctRural) at one of
     the gold bandwidths (25..150, AICc optimum, fixed-bisquare and adaptive-gaussian optima).
     Raw input attributes are excluded. A wrong
     dependent variable or regressor set, or a global OLS (constant coefficients), fails;
     and the model has no extra regressor: a captured local-coefficient array (e.g. the mgwr
     results' params, recorded from the result object) whose columns match >= 2 different gold
     slopes (r >= 0.9) has at most 4 columns (intercept + 3). A model with PctPov added keeps a
     PctRural surface at r 0.98 but has 5 columns and fails. (The slopes are judged one by
     one: lat/lon coordinates legitimately lower the PctFB surface to 0.84.)
  4. the figure shows the GWR result: a drawn polygon / point collection (values, or face
     colours inverted to colormap positions; located per county through captured offsets or
     path bounding boxes, else in input order) has |Spearman| >= 0.90 with a gold local slope
     surface or local intercept surface, or a drawn line, scatter or bar chart (numeric
     bandwidth on x, >= 4 points) reproduces the gold AICc-vs-bandwidth profile
     (Spearman >= 0.90).
  5. the figure compares >= 2 GWR-family models ("plot their differences in the same graph"),
     judged from what is drawn: (a) the same coefficient surface (intercept or a slope) of two
     different models - different bandwidths / kernels / criteria, or GWR vs MGWR - each
     matching its gold surface at |Spearman| >= 0.90 and not the same map (|Spearman| < 0.99
     between them), or a difference map matching a gold difference surface; (b) a
     criterion-vs-bandwidth profile over >= 4 bandwidths (as in check 4); (c) a bar / line /
     scatter whose drawn values equal diagnostics (AICc, AIC, BIC, R2, adj. R2) of two
     different GWR-family models. Gold surfaces beyond the gold run (MGWR raw / standardised,
     standardised GWR intercept, fixed-gaussian optimum, distractor t-value / local R2 maps
     of the optimum) are computed here once per process. A global OLS is not a GWR-family
     model. A figure with one model's maps only fails.
The AICc-bandwidth profile in checks 4 and 5(b) is picked by the drawn x axis - adaptive
nearest neighbours (x <= 159), or a fixed-kernel bandwidth in metres (x > 1000) or km (159 < x <= 1000), compared with gold
fixed-bisquare / fixed-gaussian AICc profiles computed here (30-620 km); for a fixed profile
the drawn curve must be the AICc (minimum within 2 gold AICc of the best drawn bandwidth, and
dAICc values within a median 1 of gold near the optimum). Check 5(c)
counts only artists with <= 20 points.
The GWR/MGWR fits computed here run with n_jobs=1 (the checks may be evaluated from parallel
threads), and a crash of check 5 is recorded as a scorer error (no verdict), not as a failed
check.
"""
import os
import re

import numpy as np

from checklib import file_exists, png_not_blank, result, scorer_error_result

N = 159
BW_SET = [117, 116, 141, 112, 50, 78,
          288296.23, 245739.65, 342190.58, 233055.17, 105939.8, 97036.59, 144841.95, 95021.86,
          90498.92, 196705.15, 65934.27]
SLOPES = ("PctFB", "PctBlack", "PctRural")
SURFACES = SLOPES + ("Intercept",)   # check 4 (figure) also accepts local intercept surfaces


def _search_evidence(code):
    """Code shows a bandwidth search: Sel_BW, or GWR fitted in a loop / >= 3 times plus a selection."""
    if re.search(r"Sel_BW\s*\(", code):
        return True
    lines = code.splitlines()
    loop = any(re.match(r"\s*for\s", l) and "GWR(" in "\n".join(lines[i:i + 8]) for i, l in enumerate(lines)) or \
        len(re.findall(r"GWR\(", code)) >= 3
    return loop and bool(re.search(r"\bmin\s*\(|argmin|idxmin|nsmallest|sort", code))


def _first_part_index(inp):
    """geopandas draws one patch per polygon part; index of each row's first patch."""
    if inp is None:
        return None, 0
    n = [len(g.geoms) if g is not None and g.geom_type.startswith("Multi") else 1 for g in inp.geometry]
    return np.r_[0, np.cumsum(n)[:-1]], int(sum(n))


def _vectors(ctx, side, inp=None):
    """All 159-long numeric vectors captured: (label, vector, keys or None). Values drawn in
    a choropleth (one per polygon part) are folded back to one value per county."""
    out = []
    first, n_parts = _first_part_index(inp)
    for tag, meta, load in ctx.objects(side):
        k = meta.get("kind")
        try:
            if k in ("gdf", "df", "series") and abs(meta.get("rows", 0) - N) <= 2:
                obj = load()
                keys = obj["AreaKey"].values if "AreaKey" in obj.columns else None
                for c in obj.columns:
                    if c != "geometry" and obj[c].dtype.kind in "fi" and c not in ("index", "level_0"):
                        out.append((f"{meta['name']}.{c}@{tag}", obj[c].to_numpy(float), keys))
            elif k == "array":
                shp = meta.get("shape") or []
                if len(shp) == 1 and shp[0] == N:
                    out.append((f"{meta['name']}@{tag}", np.asarray(load(), float), None))
                elif len(shp) == 1 and first is not None and shp[0] == n_parts and n_parts != N:
                    out.append((f"{meta['name']}@{tag}", np.asarray(load(), float)[first], None))
                elif len(shp) == 2 and shp[0] == N and shp[1] <= 20:
                    a = np.asarray(load(), float)
                    for j in range(shp[1]):
                        out.append((f"{meta['name']}[:,{j}]@{tag}", a[:, j], None))
        except Exception:
            continue
    return out


def _drawn_county_vectors(ctx, inp):
    """(label, 159-vector in input order) for every drawn collection with values / colours."""
    from checklib import drawn_arrays, _cmap_positions
    import geopandas as gpd
    first, n_parts = _first_part_index(inp)
    arrs = {}
    for tag, meta, a in drawn_arrays(ctx):
        n = meta["name"]
        for suf in ("_values", "_facecolors", "_offsets", "_paths_bbox"):
            if n.endswith(suf):
                arrs.setdefault((tag, n[: -len(suf)]), {})[suf] = np.asarray(a, float)
    poly = gpd.GeoDataFrame({"i": np.arange(len(inp))}, geometry=inp.geometry.values)
    out = []
    for (tag, base), d in arrs.items():
        cands = []
        if "_values" in d:
            cands.append(("values", d["_values"].ravel()))
        fc = d.get("_facecolors")
        if fc is not None and fc.ndim == 2 and len(np.unique(np.round(fc[:, :3], 4), axis=0)) >= 3:
            pos, cname, res = _cmap_positions(fc)
            if pos is not None and res <= 0.03:
                cands.append((f"colours ({cname})", pos))
        locs = None
        if "_offsets" in d and len(d["_offsets"].reshape(-1, 2)) > 2:
            locs = d["_offsets"].reshape(-1, 2)
        elif "_paths_bbox" in d:
            bb = d["_paths_bbox"].reshape(-1, 4)
            locs = np.c_[(bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2]
        for kind, v in cands:
            vec = None
            if locs is not None and len(locs) == len(v):
                try:
                    pts = gpd.GeoDataFrame({"v": v}, geometry=gpd.points_from_xy(locs[:, 0], locs[:, 1]))
                    j = gpd.sjoin(pts, poly, predicate="within")
                    if len(j) >= 0.8 * len(v):
                        vec = j.groupby("i")["v"].mean().reindex(np.arange(len(inp))).to_numpy()
                except Exception:
                    vec = None
            if vec is None and len(v) == N:
                vec = v
            elif vec is None and first is not None and len(v) == n_parts:
                vec = v[first]
            if vec is not None:
                out.append((f"{base}@{tag} {kind}", np.asarray(vec, float)))
    return out


def _corr(a, b):
    if len(a) != len(b):  # vector of another table length: not comparable
        return np.nan
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 0.9 * N or a[m].std() == 0 or b[m].std() == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def checks(ctx):
    out = []
    names = ["GA_GWR.png", "ga_gwr.png", "GA_GWR.PNG"]
    r = file_exists(ctx, names)
    out.append(r)
    if r["passed"]:
        nb, std = png_not_blank(ctx.pred(r["value"][0]))
        out.append(result("figure not blank", nb, value=std, threshold=">=3 grey-level std"))
    else:
        out.append(result("figure not blank", False, note="no GA_GWR.png"))

    # 2. optimal bandwidth (numbers); the surface-based alternative is decided after check 3
    hit = None

    def _match(v):
        if v is None or not np.isfinite(v):
            return None
        for b in BW_SET:
            if abs(v - b) <= 0.02 * b:
                return b
        return None
    for tag, meta, load in ctx.objects("pred"):
        k = meta.get("kind")
        vals = []
        try:
            if k == "scalar":
                vals = [meta.get("value")]
            elif k == "array" and 0 < np.prod(meta.get("shape") or [0]) <= 60:
                vals = np.asarray(load(), float).ravel().tolist()
            elif k in ("series", "df", "gdf") and meta.get("rows", 99) <= 60:
                obj = load()
                for c in obj.columns:
                    if c != "geometry" and getattr(obj[c], "dtype", None) is not None and obj[c].dtype.kind in "fi":
                        vals += obj[c].astype(float).tolist()
        except Exception:
            vals = []
        for v in vals:
            bm = _match(v)
            if bm:
                hit = (meta["name"] + "@" + tag, v, bm)
                break
        if hit:
            break
    if not hit and os.path.isdir(ctx.pred_dir):
        for f in sorted(os.listdir(ctx.pred_dir)):
            if f.lower().endswith((".txt", ".json", ".md")) and os.path.getsize(os.path.join(ctx.pred_dir, f)) < 200_000:
                for m in re.findall(r"(?<![\d.])\d+(?:\.\d+)?", open(os.path.join(ctx.pred_dir, f), errors="ignore").read()):
                    bm = _match(float(m))
                    if bm:
                        hit = (f"number in {f}", float(m), bm)
                        break
            if hit:
                break

    # 3. local coefficient surfaces
    import geopandas as gpd
    try:
        inp = gpd.read_file(f"{ctx.inputs_dir}/G_utm.shp")
    except Exception:
        inp = None
    gold, gold_fig = {}, {}
    for lab, v, keys in _vectors(ctx, "gold"):
        if lab.startswith("coef_table.") and any(lab.split("@")[0].endswith("_" + s) for s in SURFACES):
            gold_fig[lab.split("@")[0].split(".", 1)[1]] = v
            if any(lab.split("@")[0].endswith("_" + s) for s in SLOPES):
                gold[lab.split("@")[0].split(".", 1)[1]] = v
    if not gold or inp is None:
        out.append(result("optimal bandwidth searched (Sel_BW result present)", bool(hit), value=hit[1] if hit else None,
                          note="gold coef_table missing"))
        out.append(result("local GWR slope surfaces match gold (Pearson >= 0.97) and no extra regressor", False,
                          note="gold coef_table missing"))
        return out
    gkeys = inp["AreaKey"].values
    raw = [inp[c].to_numpy(float) for c in inp.columns if c != "geometry" and inp[c].dtype.kind in "fi"]
    best = (None, -2.0, None)
    opt_hit = (None, -2.0)
    groups = {}   # source (table / array @ tag) -> list of (label, vector)
    for lab, v, keys in _vectors(ctx, "pred", inp):
        if keys is not None and set(keys) == set(gkeys) and not np.array_equal(keys, gkeys):
            import pandas as pd
            v = pd.Series(v, index=keys).groupby(level=0).mean().reindex(gkeys).to_numpy()
        if np.nanstd(v) == 0 or any(abs(_corr(v, x)) > 0.95 for x in raw if np.isfinite(_corr(v, x))):
            continue  # constant, or simply an input attribute
        src = re.sub(r"(\.[^@]*|\[:,\d+\])@", "@", lab)
        groups.setdefault(src, []).append((lab, v))
        for gl, gv in gold.items():
            c = _corr(v, gv)
            if np.isfinite(c) and c > best[1]:
                best = (lab, c, gl)
            if np.isfinite(c) and gl.split("_")[0] in ("opt", "fixedbisq", "adaptgauss") and \
                    gl.endswith(("_PctRural", "_PctBlack")) and c > opt_hit[1]:
                opt_hit = (f"{lab} ~ gold {gl}", c)
    # model specification: a captured local-coefficient array (e.g. results.params, which the
    # capture records from mgwr result objects) whose columns match >= 2 different gold slopes
    # (r >= 0.9) must have <= 4 columns (intercept + PctFB, PctBlack, PctRural)
    wide = None
    for tag, meta, load in ctx.objects("pred", "array", include_files=False):
        shp = meta.get("shape") or []
        if len(shp) != 2 or shp[0] != N or not (2 <= shp[1] <= 20):
            continue
        try:
            A = np.asarray(load(), float)
        except Exception:
            continue
        vars_ = set()
        for j in range(A.shape[1]):
            rs = [(_corr(A[:, j], gv), gl.rsplit("_", 1)[1]) for gl, gv in gold.items()]
            rs = [t for t in rs if np.isfinite(t[0])]
            if rs and max(rs)[0] >= 0.9:
                vars_.add(max(rs)[1])
        if len(vars_) >= 2 and (wide is None or shp[1] > wide[1]):
            wide = (f"{meta['name']}@{tag}", shp[1])
    single_ok = best[1] >= 0.97
    spec_ok = wide is None or wide[1] <= 4
    ok3 = single_ok and spec_ok
    val = round(best[1], 3) if best[0] else None
    note = (f"{best[0]} ~ gold {best[2]}" if best[0] else "no 159-long local coefficient vector captured") + \
        (f"; coefficient array {wide[0]} has {wide[1]} columns" if wide else "; no coefficient array captured")
    res3 = result("local GWR slope surfaces match gold (Pearson >= 0.97) and no extra regressor", ok3,
                  value=val, threshold=0.97, note=note)
    prof = None
    for tag, meta, load in ctx.objects("gold", "df", include_files=False):
        if meta["name"] == "bw_profile":
            prof = load()
            break
    # 2c. search evidence in the code + a selected adaptive bandwidth within 2 AICc of the minimum
    aicc_hit = None
    code = ctx.code_text()
    if not hit and prof is not None and _search_evidence(code):
        px, py = prof["bandwidth"].to_numpy(float), prof["AICc"].to_numpy(float)
        loopvars = set(re.findall(r"^\s*for\s+(\w+)\s+in\b", code, re.M))
        cv = [(m["name"] + "@" + t, m.get("value")) for t, m, _ in ctx.objects("pred", "scalar")
              if m["name"].split(".")[-1] not in loopvars]
        if os.path.isdir(ctx.pred_dir):
            for f in sorted(os.listdir(ctx.pred_dir)):
                if f.lower().endswith((".txt", ".json", ".md")) and os.path.getsize(os.path.join(ctx.pred_dir, f)) < 200_000:
                    cv += [(f"number in {f}", float(m)) for m in
                           re.findall(r"(?<![\d.])\d+(?:\.\d+)?", open(os.path.join(ctx.pred_dir, f), errors="ignore").read())]
        for lab, v in cv:
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(v) or abs(v - round(v)) > 1e-6 or not (px.min() <= v <= px.max()):
                continue
            d = float(np.interp(v, px, py) - py.min())
            if d < 2 and (aicc_hit is None or d < aicc_hit[2]):
                aicc_hit = (lab, v, d)
    # 2. bandwidth check (number found, or an optimum surface reproduced)
    if hit:
        out.append(result("optimal bandwidth searched (Sel_BW result present)", True, value=hit[1],
                          threshold="Sel_BW optimum +/-2 % or optimum surface r >= 0.998", note=f"{hit[0]} ~ {hit[2]}"))
    else:
        ok2 = opt_hit[1] >= 0.998
        if not ok2 and aicc_hit:
            out.append(result("optimal bandwidth searched (Sel_BW result present)", True, value=aicc_hit[1],
                              threshold="Sel_BW optimum +/-2 %, optimum surface r >= 0.998, or searched bandwidth with dAICc < 2",
                              note=f"search in code; {aicc_hit[0]} dAICc={aicc_hit[2]:.2f}"))
        else:
            out.append(result("optimal bandwidth searched (Sel_BW result present)", ok2,
                              value=round(opt_hit[1], 4) if opt_hit[0] else None,
                              threshold="Sel_BW optimum +/-2 %, optimum surface r >= 0.998, or searched bandwidth with dAICc < 2",
                              note=(opt_hit[0] or "") + ("" if ok2 else "; no captured number equals a Sel_BW optimum")))
    out.append(res3)

    # 4. what the figure shows
    from scipy.stats import spearmanr
    from checklib import drawn_arrays
    bestd = (None, -1.0)
    try:
        for lab, v in _drawn_county_vectors(ctx, inp):
            m0 = np.isfinite(v)
            if m0.sum() < 0.8 * N or np.nanstd(v) == 0:
                continue
            for gl, gv in gold_fig.items():
                rho = abs(float(spearmanr(v[m0], gv[m0]).correlation))
                if rho > bestd[1]:
                    bestd = (f"{lab} ~ gold {gl}", rho)
        if True:   # profile picked by the drawn x axis (_profile_rho)
            curves = []
            for tag, meta, a in drawn_arrays(ctx):
                art = meta.get("artist")
                try:
                    if art == "line":
                        curves.append((tag, meta, np.asarray(a, float).reshape(-1, 2)))
                    elif art == "bars":       # x, y, width, height per bar -> (centre x, top)
                        b4 = np.asarray(a, float).reshape(-1, 4)
                        curves.append((tag, meta, np.c_[b4[:, 0] + b4[:, 2] / 2, b4[:, 1] + b4[:, 3]]))
                    elif meta["name"].endswith("_offsets") and art == "PathCollection":   # scatter
                        curves.append((tag, meta, np.asarray(a, float).reshape(-1, 2)))
                except Exception:
                    continue
            for tag, meta, xy in curves:
                pr_ = _profile_rho(xy, prof, inp)
                if pr_ and np.isfinite(pr_[0]) and pr_[0] > bestd[1]:
                    bestd = (f"{meta['name']}@{tag} ~ gold AICc profile, {pr_[1]}", pr_[0])
    except Exception as e:
        bestd = (None, -1.0)
    if bestd[0] is None:
        out.append(result("figure shows GWR surfaces or the AICc-bandwidth profile", False,
                          note="UNVERIFIABLE: no drawn collection / line located on the counties"))
        out[-1]["unverifiable"] = True
    else:
        out.append(result("figure shows GWR surfaces or the AICc-bandwidth profile", bestd[1] >= 0.90,
                          value=round(bestd[1], 3), threshold="|Spearman| >= 0.90", note=bestd[0]))

    # 5. the figure compares >= 2 GWR-family models (required)
    try:
        out.append(_comparison_check(ctx, inp, gold_fig, prof))
    except Exception as e:   # a scorer crash is no verdict, not a failed check
        out.append(scorer_error_result(f"{CMP_NAME}: comparison check crashed: {e!r}"[:300]))
    return out


def _fixed_profiles(inp):
    """Gold AICc profiles of FIXED-kernel GWR (bisquare and
    gaussian; bandwidth in metres on the UTM X/Y coordinates, 30-620 km (below the smallest
    county X of 627 km, so drawn map outlines in UTM metres never overlap it), geometric grid of
    40), computed once per process. The gold run keeps only the adaptive (nearest-neighbour)
    profile; a fixed-kernel profile drawn in m or km is compared with these.
    Bandwidths where the fit is singular are dropped."""
    import sys as _s
    cache = _s.__dict__.setdefault("_t39_fixed_profile_cache", {})
    if cache:
        return cache["profiles"]
    import contextlib
    import io
    from mgwr.gwr import GWR
    y = inp[["PctBach"]].to_numpy(float)
    X = inp[list(SLOPES)].to_numpy(float)
    uv = np.c_[inp["X"].to_numpy(float), inp["Y"].to_numpy(float)]
    profs = {}
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        for kern in ("bisquare", "gaussian"):
            bx, by = [], []
            for b in np.geomspace(30_000, 620_000, 40):
                try:
                    a = float(GWR(uv, y, X, bw=float(b), fixed=True, kernel=kern, n_jobs=1).fit().aicc)
                except Exception:
                    continue
                if np.isfinite(a) and abs(a) < 1e6:
                    bx.append(b); by.append(a)
            if len(bx) >= 4:
                profs[f"fixed {kern}"] = (np.array(bx), np.array(by))
    cache["profiles"] = profs
    return profs


def _profile_rho(xy, prof, inp):
    """Spearman of a drawn (bandwidth, criterion) curve with the gold AICc profile on the
    curve's own axis. x > 1000 -> fixed bandwidth in metres;
    N < x <= 1000 -> fixed bandwidth in km (an adaptive bandwidth cannot exceed N = 159
    neighbours); otherwise the adaptive nearest-neighbour profile. For a fixed
    profile the drawn curve must be the AICc itself: its minimum a bandwidth whose gold AICc is
    within 2 of the best gold AICc among the drawn bandwidths, and its values relative to its
    own minimum equal the gold dAICc (median |difference| <= 1 where gold dAICc <= 20; absolute
    or shifted AICc, e.g. standardised y, both pass). A mis-computed AICc, or another
    criterion (CV, AIC, BIC) on a fixed-km axis, is not credited (control
    wrong_nonstandard_aicc: a non-standard AICc and a CV-RMSE curve).
    Returns (rho, label) or None."""
    fin = np.isfinite(xy).all(1)
    if fin.sum() < 4:
        return None
    xmax = float(np.abs(xy[fin, 0]).max())
    if xmax <= N:
        if prof is None:
            return None
        cands = [("adaptive NN", prof["bandwidth"].to_numpy(float), prof["AICc"].to_numpy(float), 1.0)]
    else:
        sc = 1.0 if xmax > 1000 else 1000.0
        cands = [(f"{k} ({'m' if sc == 1 else 'km'})", px, py, sc) for k, (px, py) in _fixed_profiles(inp).items()]
    best = None
    for lab, px, py, sc in cands:
        x = xy[:, 0] * sc
        m0 = fin & (x >= px.min()) & (x <= px.max())
        if m0.sum() < 4 or xy[m0, 1].std() == 0:
            continue
        rho = float(spearmanr_(xy[m0, 1], np.interp(x[m0], px, py)))
        if sc != 1.0 or lab.startswith("fixed"):
            xa = x[m0][int(np.argmin(xy[m0, 1]))]
            ga = np.interp(x[m0], px, py)
            dy, dg = xy[m0, 1] - xy[m0, 1].min(), ga - ga.min()
            near = dg <= 20
            dev = float(np.median(np.abs(dy[near] - dg[near]))) if near.sum() >= 3 else np.inf
            if np.interp(xa, px, py) - ga.min() >= 2:
                rho = min(rho, 0.0)
                lab += f" (drawn minimum at {xa:.0f} m: gold AICc >= 2 above the best drawn bandwidth)"
            elif dev > 1.0:
                rho = min(rho, 0.0)
                lab += f" (not the AICc: median |dAICc drawn - gold| {dev:.2f} > 1 near the optimum)"
        if best is None or rho > best[0]:
            best = (rho, lab)
    return best


def spearmanr_(a, b):
    from scipy.stats import spearmanr
    return spearmanr(a, b).correlation


CMP_NAME = "figure compares >= 2 GWR-family models"
# the harness re-imports checks.py for every evaluation: keep the cache on the sys module
import sys as _sys
_EXTRA = _sys.__dict__.setdefault("_t39_extra_gold_cache", {})


def _extra_gold(inp):
    """Surfaces and diagnostics that the gold run does not keep, computed once per process
    (about 1 min): MGWR on raw and on standardised data, GWR at the optimum on standardised
    data (its intercept differs), the fixed-gaussian optimum, distractor surfaces of the
    optimal model (local t-values, local R2) and model diagnostics for the bar / line rule."""
    if _EXTRA:
        return _EXTRA
    import contextlib
    import io
    from mgwr.gwr import GWR, MGWR
    from mgwr.sel_bw import Sel_BW
    names = ("Intercept",) + SLOPES
    y = inp[["PctBach"]].to_numpy(float)
    X = inp[list(SLOPES)].to_numpy(float)
    uv = np.c_[inp["X"].to_numpy(float), inp["Y"].to_numpy(float)]
    surf, diag = {}, []

    def keep(lab, r, shift=()):
        for j, n in enumerate(names):
            surf[f"{lab}_{n}"] = np.asarray(r.params[:, j], float)
        for m in ("aicc", "aic", "bic", "R2", "adj_R2"):
            try:
                diag.append((lab, m, float(getattr(r, m))))
            except Exception:
                pass
        for sh in shift:   # standardising y shifts the information criteria by -n ln var(y)
            for m in ("aicc", "aic", "bic"):
                diag.append((lab + "-std", m, float(getattr(r, m)) - sh))
    shifts = [N * np.log(y.var(ddof=d)) for d in (0, 1)]
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        opt = float(Sel_BW(uv, y, X, n_jobs=1).search())
        r_opt = GWR(uv, y, X, bw=opt, n_jobs=1).fit()
        keep("opt", r_opt, shifts)
        for j, n in enumerate(SLOPES):            # distractors: same model, not a coefficient
            surf[f"optdist_t{n}"] = np.asarray(r_opt.tvalues[:, j + 1], float)
        surf["optdist_localR2"] = np.asarray(r_opt.localR2, float).ravel()
        for bw in (25, 50, 75, 100, 125, 150):
            keep(f"bw{bw}", GWR(uv, y, X, bw=bw, n_jobs=1).fit(), shifts)
        for lab, kw in (("fixedbisq", dict(fixed=True, kernel="bisquare")),
                        ("adaptgauss", dict(kernel="gaussian")),
                        ("fixedgauss", dict(fixed=True, kernel="gaussian"))):
            b = float(Sel_BW(uv, y, X, n_jobs=1, **kw).search())
            keep(lab, GWR(uv, y, X, bw=b, n_jobs=1, **kw).fit(), shifts)
        for k in [k for k in surf if k.startswith(("bw", "opt_", "fixedbisq_", "adaptgauss_"))]:
            del surf[k]        # these surfaces come from the gold run (gold_fig)
        ys = (y - y.mean()) / y.std()
        Xs = (X - X.mean(0)) / X.std(0)
        keep("optstd", GWR(uv, ys, Xs, bw=float(Sel_BW(uv, ys, Xs, n_jobs=1).search()), n_jobs=1).fit())
        for n in SLOPES:       # standardised slopes = rescaled opt slopes (same model): keep only the intercept
            del surf[f"optstd_{n}"]
        for lab, (yy, XX) in (("mgwrraw", (y, X)), ("mgwrstd", (ys, Xs))):
            sel = Sel_BW(uv, yy, XX, multi=True, n_jobs=1)
            sel.search()
            keep(lab, MGWR(uv, yy, XX, sel, n_jobs=1).fit())
    _EXTRA.update(surf=surf, diag=diag)
    return _EXTRA


def _comparison_check(ctx, inp, gold_fig, prof):
    """Required: "plot their differences in the same graph" - the figure must
    compare >= 2 GWR-family models (different bandwidths, kernels / criteria, or GWR vs MGWR).
    Accepted, from what is DRAWN:
      (a) two drawn county surfaces (one per drawn collection) whose best-matching gold
          surface (|Spearman| >= 0.90 over GWR at 25..150 NN, the AICc / fixed-bisquare /
          adaptive-gaussian / fixed-gaussian optima, standardised GWR, MGWR raw and
          standardised; distractors: local t-values and local R2 of the optimum) is the same
          coefficient (intercept or a slope) of two different models, and which are not the
          same map (|Spearman| between them < 0.99); a difference map (model A - model B)
          matching a gold difference surface at |Spearman| >= 0.90 and better than any single
          gold surface also counts;
      (b) a drawn line / scatter / bar profile of the criterion over >= 4 bandwidths
          (Spearman >= 0.90 with the gold AICc profile, as in check 4);
      (c) a drawn bar / line / scatter with >= 2 distinct values that each equal a model
          diagnostic (AICc, AIC, BIC within 0.06; R2, adj. R2 within 0.0006; raw or
          standardised y) of a GWR-family model, two of them from different models; only
          artists with <= 20 points (bars / markers) count.
    Profiles in (b) and check 4 are compared on the drawn x axis (adaptive NN, fixed m or km;
    see _profile_rho).
    A global OLS is not a GWR-family model (its coefficient map is constant)."""
    from scipy.stats import spearmanr
    from checklib import drawn_arrays
    ex = _extra_gold(inp)
    fam = dict(gold_fig)
    fam.update(ex["surf"])

    def model_var(gl):
        m, v = gl.rsplit("_", 1)
        return m, v

    # (a) one vector per drawn artist: the candidate (values / colours) with the best match
    per_art = {}
    for lab, v in _drawn_county_vectors(ctx, inp):
        m0 = np.isfinite(v)
        if m0.sum() < 0.8 * N or np.nanstd(v[m0]) == 0:
            continue
        rs = [(abs(float(spearmanr(v[m0], gv[m0]).correlation)), gl) for gl, gv in fam.items()]
        rs = [t for t in rs if np.isfinite(t[0])]
        if not rs:
            continue
        rho, gl = max(rs)
        art = lab.split(" ")[0]
        if art not in per_art or rho > per_art[art][1]:
            per_art[art] = (v, rho, gl)
    arts = [(a, v, rho, gl) for a, (v, rho, gl) in per_art.items() if rho >= 0.90]
    pair = None
    for i in range(len(arts)):
        for j in range(i + 1, len(arts)):
            a1, v1, r1, g1 = arts[i]
            a2, v2, r2, g2 = arts[j]
            (m1, c1), (m2, c2) = model_var(g1), model_var(g2)
            if c1 != c2 or c1 not in SURFACES or m1 == m2 or m1.endswith("dist") or m2.endswith("dist"):
                continue
            mm = np.isfinite(v1) & np.isfinite(v2)
            mut = abs(float(spearmanr(v1[mm], v2[mm]).correlation))
            if mut < 0.99:
                pair = f"{a1} ~ {g1} ({r1:.3f}) vs {a2} ~ {g2} ({r2:.3f}); between maps {mut:.3f}"
                break
        if pair:
            break
    if pair:
        return result(CMP_NAME, True, value="surfaces of 2 models", note=pair)
    # (a') difference maps
    for a, (v, rho, gl) in per_art.items():
        m0 = np.isfinite(v)
        for c in SURFACES:
            ks = [k for k in fam if k.endswith("_" + c) and not k.split("_")[0].endswith("dist")]
            for i in range(len(ks)):
                for j in range(i + 1, len(ks)):
                    d = fam[ks[i]] - fam[ks[j]]
                    if np.nanstd(d) == 0:
                        continue
                    rd = abs(float(spearmanr(v[m0], d[m0]).correlation))
                    if rd >= 0.90 and rd > rho + 0.02:
                        return result(CMP_NAME, True, value=round(rd, 3),
                                      note=f"difference map {a} ~ gold {ks[i]} - {ks[j]} (best single {gl} {rho:.3f})")
    # (b) / (c) curves
    curves = []
    for tag, meta, a in drawn_arrays(ctx):
        art = meta.get("artist")
        try:
            if art == "line":
                curves.append((f"{meta['name']}@{tag}", np.asarray(a, float).reshape(-1, 2)))
            elif art == "bars":
                b4 = np.asarray(a, float).reshape(-1, 4)
                curves.append((f"{meta['name']}@{tag}", np.c_[b4[:, 0] + b4[:, 2] / 2, b4[:, 1] + b4[:, 3]]))
            elif meta["name"].endswith("_offsets") and art == "PathCollection":
                curves.append((f"{meta['name']}@{tag}", np.asarray(a, float).reshape(-1, 2)))
        except Exception:
            continue
    best_prof = (None, -1.0)
    for lab, xy in curves:   # profile picked by the drawn x axis
        pr_ = _profile_rho(xy, prof, inp)
        if pr_ and np.isfinite(pr_[0]) and pr_[0] > best_prof[1]:
            best_prof = (f"{lab} ({pr_[1]})", pr_[0])
    if best_prof[1] >= 0.90:
        return result(CMP_NAME, True, value=round(best_prof[1], 3),
                      note=f"criterion-vs-bandwidth profile {best_prof[0]} ~ gold AICc profile")
    diag = list(ex["diag"])
    if prof is not None:
        diag += [(f"bw{int(b)}", "aicc", float(v)) for b, v in zip(prof["bandwidth"], prof["AICc"])]
    dv = np.array([d[2] for d in diag])
    tol = np.array([0.06 if d[1] in ("aicc", "aic", "bic") else 6e-4 for d in diag])   # rounding to 0.1 / 0.001
    dmod = [d[0].replace("-std", "") for d in diag]       # standardised y = same model
    for lab, xy in curves:
        if len(xy) > 20:   # a diagnostics chart has few bars / points;
            continue       # one point of a long (e.g. 159-county) line can equal a diagnostic by chance
        ys = np.unique(np.round(xy[np.isfinite(xy[:, 1]), 1], 9))
        hits = {}
        for yv in ys:
            k = np.where(np.abs(dv - yv) <= tol)[0]
            if len(k):
                hits[yv] = {dmod[i] for i in k}
        hv = list(hits)
        two = next(((hv[i], hv[j], m1, m2) for i in range(len(hv)) for j in range(i + 1, len(hv))
                    for m1 in hits[hv[i]] for m2 in hits[hv[j]] if m1 != m2), None)
        if two:
            return result(CMP_NAME, True, value=len(hits),
                          note=f"{lab}: drawn values {two[0]:.4g} ~ {two[2]} and {two[1]:.4g} ~ {two[3]} "
                               f"(model diagnostics; {len(hits)} drawn values matched)")
    note = "single model only: " + "; ".join(f"{a} ~ {gl} ({rho:.3f})" for a, _, rho, gl in arts[:6]) if arts \
        else "no drawn GWR surface, profile or diagnostic chart"
    return result(CMP_NAME, False, note=note[:400],
                  threshold="2 models' surfaces (|Spearman| >= 0.90, maps not identical), profile >= 4 bandwidths, or >= 2 model diagnostics")
