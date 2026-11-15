"""Task 20 - random-forest prospectivity map for Sn-W deposits, NE Tasmania.

Required:
  1. mineral_prospectivity.png exists and is not blank;
  2. a probability map was produced: a captured 2-D array on the evidence grid (2633 x 1876,
     or same aspect, resampled) with values in [0, 1] whose ranks agree with the gold RF map
     over the valid cells at Spearman >= 0.60. Other RF seeds 0.99, unbalanced RF 0.95,
     deep RF 0.77, logistic regression 0.77; P(class 0) instead of P(deposit) -1.0;
     a single evidence layer is not a probability (and the best one, gravity, reaches 0.68);
  3. the map discriminates the known deposits: ROC AUC of the map values, cells within 1 km
     of an occurrence vs. all other valid cells, >= 0.85 (gold 0.945). Misaligned labels
     (rasterising without the raster transform, reprojected points) fail here;
     Probabilities in percent (0-100) are accepted (rescaled);
  4. the figure shows that map: a drawn image reaches rho >= 0.60 (either row order: the
     orientation on the map is presentation); no raster artist ->
     UNVERIFIABLE (not passed);
  4b. all 7 evidence layers are model features (the task prescribes them): a captured feature
     matrix whose columns contain every layer's values (exact float32 membership; a rescaled
     matrix with >= 7 columns also counts). The gravity-only RF (rho 0.685, AUC 0.944 - above
     the map-level thresholds) fails here; no feature matrix -> UNVERIFIABLE.
  6. (required: the task text says "The final auc should be at
     least 0.9") an AUC was computed - captured scalar named *auc* in [0.5, 1], or a number in
     [0.5, 1] printed on a stdout line mentioning AUC; also a figure text
     mentioning AUC, the mean of a 1-D array named *auc*, or a table column named *auc* - and
     the largest one is >= 0.9;
  5. (a method the task text prescribes is
     required; "performing a random forest prospectivity analysis ... build a random forest
     classifier") a random-forest classifier is constructed in the executed code
     (RandomForestClassifier / imblearn BalancedRandomForestClassifier, aliases followed,
     comments ignored), or a fitted RF object is in the captured namespace;
     other models (gradient boosting, extra trees, logistic regression) fail.
     Undersampling and 1 km buffers are only described in the Domain Knowledge (definitions),
     not prescribed, and stay optional;
Reported, not required:
  4f. (kind="figure") the drawn probability image, as displayed (origin respected), reaches rho >= 0.60;
  7b. labels are the occurrences buffered by 1 km (label raster positives ~ gold);
  7. spatially blocked / grouped validation used (random pixel splits leak: deep RF 0.995 random
     split vs 0.926 spatial CV).
"""
import os
import re

import numpy as np

from checklib import file_exists, png_not_blank, result, resample_to, unverifiable

NAME = "mineral_prospectivity.png"
SHAPE = (2633, 1876)
SUB = (658, 469)


LAYERS = ["tasgrav_IR", "tasgrav_IR_1VD", "tasmag_TMI", "tasmag_TMI_1VD", "tasrad_K_pct", "tasrad_Th_ppm", "tasrad_U_ppm"]
_LAYER_CACHE = {}
MIN_ROWS = 200      # feature-matrix row floor


def _layer_values(inputs_dir):
    """Sorted unique float32 values of each evidence layer (valid cells)."""
    if inputs_dir in _LAYER_CACHE:
        return _LAYER_CACHE[inputs_dir]
    import glob
    import rasterio
    vals = {}
    for f in sorted(glob.glob(os.path.join(inputs_dir, "*.tif"))):
        with rasterio.open(f) as r:
            a = r.read(1).astype("float32")
            if r.nodata is not None:
                a = a[a != np.float32(r.nodata)]
        vals[os.path.basename(f)[:-4]] = np.unique(a)
    _LAYER_CACHE[inputs_dir] = vals
    return vals


def _features_check(ctx):
    """The task prescribes all 7 evidence rasters as features. Evidence: a captured feature
    matrix (2-D array or table, >= 200 rows, 2-40 columns) whose columns contain each
    layer's values (exact float32 membership of a 400-row sample). A matrix whose columns
    were rescaled (no raw layer found) counts when it has >= 7 columns. Fewer layers (e.g.
    gravity only) fail; no feature matrix captured, or none containing any layer ->
    UNVERIFIABLE.
    The row floor is 200 (>= the 199 occurrences of one class), so that balanced training
    matrices (occurrences + 1:1 background, ~400 rows) are seen; integer arrays with <= 3
    columns (row/col indices from np.argwhere, coordinates) are not feature matrices; a
    matrix set in which no column is any layer gives UNVERIFIABLE, not FAIL (fewer than 7
    layers found still fails)."""
    name = "model uses all 7 evidence layers"
    try:
        lv = _layer_values(ctx.inputs_dir)
    except Exception as e:
        return result(name, False, note=repr(e)[:150])
    rng = np.random.default_rng(1)
    best_found, best_where, scaled_ok, n_mats = -1, None, None, 0
    for tag, meta, load in ctx.objects("pred", include_files=False):
        k = meta.get("kind")
        sh = meta.get("shape") or []
        if k == "array" and len(sh) == 2 and sh[0] >= MIN_ROWS and 2 <= sh[1] <= 40 and meta.get("source") != "figure":
            if str(meta.get("dtype", "")).startswith(("int", "uint")) and sh[1] <= 3:
                continue                       # argwhere / rowcol index or coordinate array
            try:
                M = np.asarray(load(), float)
            except Exception:
                continue
        elif k == "df" and meta.get("rows", 0) >= MIN_ROWS and 2 <= len(meta.get("columns", [])) <= 40:
            try:
                d = load()
                d = d[[c for c in d.columns if c != "index" and d[c].dtype.kind in "fiu"]]
                M = d.to_numpy(float)
            except Exception:
                continue
            if M.ndim != 2 or M.shape[1] < 2:
                continue
        else:
            continue
        n_mats += 1
        idx = rng.choice(len(M), min(400, len(M)), replace=False)
        S = M[idx]
        found, ncols = set(), 0
        for j in range(S.shape[1]):
            col = S[:, j]
            col = col[np.isfinite(col)].astype("float32")
            if not len(col):
                continue
            hit = False
            for ln, u in lv.items():
                pos = np.clip(np.searchsorted(u, col), 0, len(u) - 1)
                if (u[pos] == col).mean() >= 0.95:
                    found.add(ln)
                    hit = True
            ncols += hit
        n_found = min(len(found), ncols)      # one column cannot stand for several layers
        if n_found > best_found:
            best_found, best_where = n_found, f"{meta['name']}@{tag} ({ncols} layer columns: {sorted(found)})"
        if not found and M.shape[1] >= 7:
            scaled_ok = f"{meta['name']}@{tag} ({M.shape[1]} rescaled columns)"
    if best_found >= 7:
        return result(name, True, value=best_found, threshold=7, note=best_where)
    if best_found <= 0 and scaled_ok:
        return result(name, True, value=None, threshold=7, note=scaled_ok)
    if n_mats == 0:
        return unverifiable(name, "no feature matrix captured")
    if best_found <= 0:
        return unverifiable(name, f"{n_mats} candidate matrices, none contains an evidence layer")
    return result(name, False, value=best_found, threshold=7, note=best_where or "feature matrix has < 7 evidence columns")


def _buffer_check(ctx):
    """Reported: the labels are the occurrences buffered by 1 km (a captured 0/1 label raster
    on the evidence grid with roughly the gold number of positive cells)."""
    name = "labels = occurrences buffered by 1 km (label raster positives ~ gold)"
    gold_pos = None
    for tag, meta, load in ctx.objects("gold", "array", include_files=False):
        if meta["name"] == "labels":
            gold_pos = float(np.nansum(load()))
    if not gold_pos:
        return result(name, False, required=False, note="gold labels unavailable")
    best = None
    for tag, meta, load in ctx.objects("pred", "array", include_files=False):
        if list(meta.get("shape") or [])[-2:] != list(SHAPE) or meta.get("source") == "figure":
            continue
        try:
            a = np.asarray(load(), float)
        except Exception:
            continue
        u = np.unique(a[np.isfinite(a)][::997])
        if not np.isin(u, [0, 1]).all() or len(u) != 2:
            continue
        n = float(np.nansum(a))
        if best is None or abs(np.log(max(n, 1) / gold_pos)) < abs(np.log(max(best, 1) / gold_pos)):
            best = n
    ok = best is not None and 0.7 <= best / gold_pos <= 1.4
    return result(name, ok, value=best, threshold=f"~{gold_pos:.0f}", required=False)


def _labels(inputs_dir):
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    with rasterio.open(os.path.join(inputs_dir, "tasgrav_IR.tif")) as r:
        a = r.read(1)
        valid = a != r.nodata
        tr = r.transform
    occ = gpd.read_file(os.path.join(inputs_dir, "sn_w_minoccs.gpkg"))
    lab = rasterize(((g, 1) for g in occ.buffer(1000).geometry), out_shape=SHAPE, transform=tr, fill=0)
    return lab.astype(bool), valid


def _cands(ctx, side):
    for tag, meta, load in ctx.objects(side, "array"):
        sh = list(meta.get("shape") or [])
        if len(sh) == 3 and sh[0] == 1:
            sh = sh[1:]
        if len(sh) != 2 or min(sh) < 100 or abs(sh[1] / sh[0] - SHAPE[1] / SHAPE[0]) > 0.05:
            continue
        yield tag, meta, load, sh


def _oriented(cands):
    """Candidates with their array; a drawn image is yielded as displayed (origin respected,
    displayed=True) and upside down (displayed=False) - the orientation on the map is
    presentation and is scored by a separate kind="figure" check."""
    for tag, meta, load, sh in cands:
        try:
            a = np.asarray(load(), float).reshape(sh)
        except Exception:
            continue
        if meta.get("source") == "figure":
            if meta.get("origin") == "lower":
                a = a[::-1]                        # as displayed (north up = row 0 on top)
            yield tag, meta, load, sh, a, True
            yield tag, meta, load, sh, a[::-1], False
        else:
            yield tag, meta, load, sh, a, True


def _stdout_aucs(ctx):
    """AUC values printed in stdout: a number in [0.5, 1] on a line that mentions AUC."""
    try:
        so = getattr(ctx, "stdout", "")
        txt = so() if callable(so) else (so if isinstance(so, str) else "")
    except Exception:
        txt = ""
    out = []
    for line in (txt or "").splitlines():
        if re.search(r"\bauc\b|roc[ _-]?auc|area under", line, re.I):
            for m in re.finditer(r"(?<![\w.])(0?\.\d+|1\.0+|1)(?![\w.])", line):
                v = float(m.group(1))
                if 0.5 <= v <= 1:
                    out.append((v, "stdout: " + line.strip()[:60]))
    return out


def _auc_line_values(text, where):
    out = []
    for line in (text or "").splitlines():
        if re.search(r"\bauc\b|roc[ _-]?auc|area under", line, re.I):
            for m in re.finditer(r"(?<![\w.])(0?\.\d+|1\.0+|1)(?![\w.])", line):
                v = float(m.group(1))
                if 0.5 <= v <= 1:
                    out.append((v, where + line.strip()[:60]))
    return out


def _more_aucs(ctx):
    """AUCs reported elsewhere than an *auc* scalar or a
    labelled stdout line - print(roc_auc_score(...)) without a label, CV results in tuples or
    tables. Accepted, with the same [0.5, 1] range: (i) a figure text (title, axis label,
    legend) mentioning AUC with the number, e.g. 'RF, AUC=0.999'; (ii) a 1-D array / list named
    *auc* (fold AUCs: their mean); (iii) a table column named *auc* (each row one estimate)."""
    out = []
    for tag, meta, load in ctx.objects("pred", include_files=True):
        k, nm = meta.get("kind"), str(meta.get("name", ""))
        try:
            if k == "text" and meta.get("source") == "figure":
                tx = meta.get("texts") or {}
                parts = []
                for v in tx.values():
                    if isinstance(v, dict):
                        for w in v.values():
                            parts += w if isinstance(w, list) else [w]
                    else:
                        parts.append(v)
                for p in parts:
                    out += _auc_line_values(str(p), "figure text: ")
            elif k == "array" and "auc" in nm.lower() and len(meta.get("shape") or []) == 1 \
                    and meta.get("source") != "figure":
                a = np.asarray(load(), float)
                a = a[np.isfinite(a)]
                if 1 <= len(a) <= 100 and a.min() >= 0.5 and a.max() <= 1:
                    out.append((float(a.mean()), f"mean of {nm}@{tag}"))
            elif k in ("df", "series"):
                cols = [c for c in (meta.get("columns") or []) if "auc" in str(c).lower()]
                if k == "series" or meta.get("source") == "file" or cols:
                    d = load()
                    for c in d.columns:
                        if "auc" not in str(c).lower() and not (k == "series" and "auc" in nm.lower()):
                            continue
                        if c == "index" or d[c].dtype.kind not in "f":
                            continue
                        v = d[c].to_numpy(float)
                        v = v[np.isfinite(v)]
                        if 1 <= len(v) <= 100 and v.min() >= 0.5 and v.max() <= 1:
                            out += [(float(x), f"{nm}@{tag}[{c}]") for x in v]
        except Exception:
            continue
    return out


def checks(ctx):
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    out = []
    r = file_exists(ctx, [NAME])
    out.append(r)
    if r["passed"]:
        nb, std = png_not_blank(ctx.pred(NAME))
        out.append(result("map not blank", nb, value=std, threshold=">=3 grey-level std"))
    else:
        out.append(result("map not blank", False, note="no map"))
    gold = None
    for tag, meta, load, sh in _cands(ctx, "gold"):
        if meta["name"] == "prob_map":
            gold = resample_to(np.asarray(load(), float).reshape(sh), SUB)
    try:
        lab, valid = _labels(ctx.inputs_dir)
    except Exception as e:
        lab = None
    if gold is None or lab is None:
        for n in ("probability map agrees with gold RF map (Spearman >= 0.60)", "map discriminates known deposits (AUC >= 0.85)",
                  "figure shows the probability map", "model uses all 7 evidence layers"):
            out.append(result(n, False, note="gold or labels unavailable"))
        return out
    lab_s = resample_to(lab, SUB).astype(bool)
    val_s = resample_to(valid, SUB).astype(bool) & np.isfinite(gold)
    rng = np.random.default_rng(0)
    best = (-2.0, None, None)
    fig_best, fig_disp, n_fig = -2.0, -2.0, 0
    for tag, meta, load, sh, a, displayed in _oriented(_cands(ctx, "pred")):
        p = resample_to(a, SUB)
        m = val_s & np.isfinite(p)
        if m.sum() < 0.8 * val_s.sum():
            continue
        lo, hi = np.nanpercentile(p[m], [0.5, 99.5])
        if 1 + 1e-6 < hi <= 100 + 1e-6 and lo >= -1e-6:
            p = p / 100.0                          # probability expressed in percent
            lo, hi = lo / 100.0, hi / 100.0
        if lo < -1e-6 or hi > 1 + 1e-6 or np.nanstd(p[m]) == 0:
            continue
        if len(np.unique(p[m][:: max(1, m.sum() // 20000)])) <= 10:   # label / class rasters are not probabilities
            continue
        idx = np.flatnonzero(m.ravel())
        if len(idx) > 200000:
            idx = rng.choice(idx, 200000, replace=False)
        rho = float(spearmanr(p.ravel()[idx], gold.ravel()[idx]).correlation)
        if meta.get("source") == "figure":
            n_fig += int(displayed)
            fig_best = max(fig_best, rho)
            if displayed:
                fig_disp = max(fig_disp, rho)
        if rho > best[0]:
            auc = float(roc_auc_score(lab_s[m], p[m])) if lab_s[m].any() else float("nan")
            best = (rho, auc, f"{meta['name']}@{tag}" + ("" if displayed else " (drawn image read upside down)"))
    rho, auc, where = best
    out.append(result("probability map agrees with gold RF map (Spearman >= 0.60)", rho >= 0.60,
                      value=round(rho, 3), threshold=0.60, note=where or "no [0,1] map on the evidence grid captured"))
    out.append(result("map discriminates known deposits (AUC >= 0.85)", auc is not None and auc >= 0.85,
                      value=None if auc is None else round(auc, 3), threshold=0.85))
    if n_fig:
        out.append(result("figure shows the probability map", fig_best >= 0.60, value=round(fig_best, 3), threshold=0.60,
                          note="either row order (orientation is the figure check below)"))
        out.append(result("probability map drawn north-up (figure)", fig_disp >= 0.60 or fig_best < 0.60,
                          value=round(fig_disp, 3), threshold="rho >= 0.60 as displayed (origin respected)",
                          note="" if fig_best >= 0.60 else "not assessed (map not drawn)", kind="figure"))
    else:
        out.append(unverifiable("figure shows the probability map", "no raster artist with the probability map captured"))
    out.append(_features_check(ctx))
    code = ctx.code_text() if hasattr(ctx, "code_text") else ""
    # the text prescribes the model ("a random forest prospectivity analysis ...
    # build a random forest classifier") -> a random-forest estimator must be constructed in the
    # executed code (comments ignored; sklearn / imblearn BalancedRandomForest / aliases followed)
    code_nc = "\n".join("" if re.match(r"\s*#", ln) else ln.split("#")[0] for ln in code.splitlines())
    rf_names = {"RandomForestClassifier", "BalancedRandomForestClassifier"}
    for m in re.finditer(r"import\s+[^\n]*?\b((?:Balanced)?RandomForestClassifier)\s+as\s+(\w+)", code_nc):
        rf_names.add(m.group(2))
    rf = re.search(r"(?<![\w])(?:[\w.]+\.)?(" + "|".join(sorted(rf_names)) + r")\s*\(", code_nc)
    rf_val = rf.group(0) if rf else None
    if not rf:
        # The code text holds only successful blocks; a block that builds and fits the RF and
        # raises further down (its namespace effects persist, later blocks predict with it)
        # is not in it. A fitted RF in the captured namespace counts:
        # an sklearn/imblearn RandomForestClassifier object with its n_features_in_ attribute.
        objs = ctx.objects("pred", include_files=False)
        fitted = {(t, m["name"]) for t, m, _ in objs if m.get("kind") == "scalar" and m.get("source") == "object_attr"}
        for t, m, _ in objs:
            ot = str(m.get("object_type", ""))
            if m.get("kind") == "object" and re.search(r"\.(Balanced)?RandomForestClassifier$", ot) \
                    and (t, m["name"] + ".n_features_in_") in fitted:
                rf_val = f"fitted {ot.rsplit('.', 1)[-1]} '{m['name']}'@{t} (captured namespace)"
                break
    out.append(result("random forest classifier used (task: 'build a random forest classifier')", bool(rf_val),
                      value=rf_val,
                      note="" if rf_val else "no RandomForestClassifier construction in the code or fitted RF captured"))
    # the task text asks for it ("The final auc should be at least 0.9")
    aucs = []
    for tag, meta, load in ctx.objects("pred", "scalar"):
        try:
            if "auc" in meta["name"].lower() and 0.5 <= float(meta.get("value", 0)) <= 1:
                aucs.append((float(meta["value"]), meta["name"]))
        except Exception:
            pass
    aucs += _stdout_aucs(ctx)
    aucs += _more_aucs(ctx)
    hit = max(aucs) if aucs else None
    out.append(result("AUC computed and >= 0.9 (task text)", bool(hit) and hit[0] >= 0.9,
                      value=round(hit[0], 4) if hit else None, threshold=0.9,
                      note=hit[1] if hit else "no AUC scalar, printed AUC, AUC in figure text or AUC table found"))
    out.append(_buffer_check(ctx))
    out.append(result("spatially blocked / grouped validation",
                      bool(re.search(r"GroupKFold|GroupShuffleSplit|LeaveOneGroupOut|spatial|block", code, re.I)), required=False))
    return out
