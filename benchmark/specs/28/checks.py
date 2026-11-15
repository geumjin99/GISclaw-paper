"""T28 - Interaction strength between subregions with a GridSearchCV-tuned Random Forest.

Task core: sum the 5000 OD records to 2139 (origin, destination) pairs, attach the origin's and
the destination's socio-economic attributes, split train/test, tune a Random Forest with
GridSearchCV and report the test-set MSE.

The supplied data are synthetic and carry no signal (flows are uniform random 1-99; R2 on
the test set is ~0, below a mean-only baseline). A correct test MSE therefore lies close to
the variance of the aggregated flows (5757): 0.8-1.5 x var(TotalFlow). MSE measured on the
training data (~3700 for the tuned forest, ~860 untuned) or on non-aggregated records (~1050)
falls outside.

Correctness (required):
  1. aggregation: a captured table with one row per pair (2139, or 2095 without intra-subregion
     pairs) whose flow column equals the gold pair sums on >= 99 % of pairs (key columns found
     by name or by their subregion-name values, e.g. orig/dest);
  2. merge: a pair table carrying origin AND destination attributes (>= 3 of the 6 attributes
     matched for each side, aligned on Origin/Destination);
  3. evaluation: a reported MSE (variable/file entry named *mse*; RMSE squared; or, if it
     was only printed, recomputed from captured test targets/predictions) within
     0.8-1.5 x var(aggregated flows).
  4. model features: a captured train/test feature matrix (10-95 % of
     the pairs as rows) holds >= 3 attributes for both origin and destination (>= 2 columns
     drawn from the same attribute's values) - an origin-only model fails;
  5. method (required, as nothing else can reveal it on this
     signal-free data): RandomForestRegressor(...) and GridSearchCV(...) called in the code
     (comments stripped).
Reported, not required:
  6. a result file written to pred_results.
Known limitation: an MSE that is only printed, computed inline from predictions that are never
stored, has no captured evidence unless the harness exposes the run's stdout as ctx.stdout
(used when present); otherwise the MSE check is reported as UNVERIFIABLE (not passed).
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from checklib import result, unverifiable

ATTRS = ["Population", "GDP", "EmploymentRate", "AverageIncome", "EducationIndex", "HealthcareAccess"]


def _key_pairs(df, names):
    """Ordered (origin, destination) key-column candidates: by name first, then any two text
    columns whose values are subregion names (orig/dest, o_id/d_id, src/dst ...), both orders."""
    cols = {str(c).lower(): c for c in df.columns}
    o = next((cols[k] for k in ("origin", "origin_subregion", "o", "from") if k in cols), None)
    d = next((cols[k] for k in ("destination", "destination_subregion", "d", "to") if k in cols), None)
    out = [(o, d)] if o is not None and d is not None else []
    txt = []
    for c in df.columns:
        if c == "geometry" or getattr(df[c], "dtype", None) is None or df[c].dtype.kind not in "OSU":
            continue
        try:
            if df[c].astype(str).isin(names).mean() >= 0.99:
                txt.append(c)
        except Exception:
            continue
    for a in txt:
        for b in txt:
            if a != b and (a, b) not in out:
                out.append((a, b))
    return out


def _mse_values(ctx):
    vals = []
    for tag, meta, load in ctx.objects("pred", "scalar"):
        n = meta["name"].lower()
        if "mse" in n and "baseline" not in n and "train" not in n:
            v = float(meta["value"])
            vals.append((meta["name"], v * v if "rmse" in n else abs(v)))
    # small captured tables / dicts / series and metric files (wide: test_mse column; long:
    # metric,value rows)
    for tag, meta, load in ctx.objects("pred"):
        if meta.get("kind") not in ("df", "series") or not 0 < meta.get("rows", 0) <= 20:
            continue
        try:
            t = load()
        except Exception:
            continue
        for c in t.columns:
            cl = str(c).lower()
            if "mse" in cl and "baseline" not in cl and "train" not in cl and "cv" not in cl:
                for v in pd.to_numeric(t[c], errors="coerce").dropna():
                    vals.append((f"{meta['name']}.{c}@{tag}", v * v if "rmse" in cl else abs(v)))
        txtc = [c for c in t.columns if t[c].dtype.kind in "OSU"]
        numc = [c for c in t.columns if t[c].dtype.kind in "fi" and str(c) != "index"]
        for c in txtc:
            for i, lab in t[c].astype(str).items():
                ll = lab.lower()
                if "mse" in ll and "baseline" not in ll and "train" not in ll and "cv" not in ll:
                    for nc in numc:
                        v = t.at[i, nc]
                        if pd.notna(v):
                            vals.append((f"{meta['name']}[{lab}]@{tag}", float(v) ** 2 if "rmse" in ll else abs(float(v))))
    for f in glob.glob(os.path.join(ctx.pred_dir, "*.json")) + glob.glob(os.path.join(ctx.pred_dir, "*.txt")) \
            + glob.glob(os.path.join(ctx.pred_dir, "*.csv")):
        try:
            txt = open(f, errors="ignore").read()
        except Exception:
            continue
        if f.endswith(".json"):
            try:
                d = json.loads(txt)
                stack = [d]
                while stack:
                    x = stack.pop()
                    if isinstance(x, dict):
                        for k, v in x.items():
                            kl = str(k).lower()
                            if isinstance(v, (int, float)) and "mse" in kl and "baseline" not in kl and "train" not in kl:
                                vals.append((os.path.basename(f) + ":" + k, v * v if "rmse" in kl else abs(v)))
                            elif isinstance(v, (dict, list)):
                                stack.append(v)
                    elif isinstance(x, list):
                        stack.extend(x)
            except Exception:
                pass
        else:
            import re
            for m in re.finditer(r"(r?mse)[^0-9\n]{0,40}?([0-9]+\.?[0-9]*(?:e[+-]?\d+)?)", txt, re.I):
                if "train" in txt[max(0, m.start() - 20):m.start()].lower():
                    continue
                v = float(m.group(2))
                vals.append((os.path.basename(f), v * v if m.group(1).lower() == "rmse" else v))
    # printed output, if the harness provides it (ctx.stdout)
    txt = getattr(ctx, "stdout", None)
    if isinstance(txt, str) and txt:
        import re
        for m in re.finditer(r"(r?mse)[^0-9\n]{0,40}?([0-9]+\.?[0-9]*(?:e[+-]?\d+)?)", txt, re.I):
            pre = txt[max(0, m.start() - 20):m.start()].lower()
            if "train" in pre or "baseline" in pre or "cv" in pre:
                continue
            v = float(m.group(2))
            vals.append(("stdout", v * v if m.group(1).lower() == "rmse" else v))
    return vals


def _mse_from_arrays(ctx, flow_values):
    """Fallback when the MSE was only printed: recompute it from captured test targets and
    predictions (1-D, 200-860 values, i.e. a 10-40 % test split; targets must be pair sums)."""
    ys, ps = [], []
    for tag, meta, load in ctx.objects("pred"):
        k = meta.get("kind")
        n = meta.get("rows") if k == "series" else (meta.get("shape") or [0])[0]
        if k not in ("series", "array") or not 200 <= (n or 0) <= 860:
            continue
        if k == "array" and len(meta.get("shape", [])) != 1:
            continue
        try:
            v = load()
            v = v.iloc[:, -1].to_numpy(float) if k == "series" else np.asarray(v, float)
        except Exception:
            continue
        if np.isin(v, flow_values).all():
            ys.append((meta["name"], v))
        else:
            ps.append((meta["name"], v))
    best = None
    for yn, y in ys:
        for pn, p in ps:
            if len(p) == len(y):
                m = float(np.mean((y - p) ** 2))
                if best is None or m > best[1]:
                    best = (f"mse({yn},{pn})", m)
    return [best] if best else []


def checks(ctx):
    out = []
    od = pd.read_csv(os.path.join(ctx.inputs_dir, "od_data.csv"))
    socio = pd.read_csv(os.path.join(ctx.inputs_dir, "socioeconomic_data.csv")).set_index("Subregion")
    gold = od.groupby(["Origin", "Destination"])["FlowVolume"].sum()
    var = float(gold.var())
    names = set(socio.index.astype(str))
    n_all, n_inter = len(gold), int((gold.index.get_level_values(0) != gold.index.get_level_values(1)).sum())

    agg_ok, agg_note, merge_best = False, "no pair table", (0, 0, None)
    # key arrays of captured pair tables, to align key-less attribute tables positionally
    # (e.g. X built with se.loc[pairs.Origin] ... reset_index)
    key_arrays = {}
    objs = list(ctx.objects("pred"))
    for tag, meta, load in objs:
        if meta.get("kind") in ("df", "gdf") and n_inter - 5 <= meta.get("rows", 0) <= n_all + 5:
            try:
                df = load()
            except Exception:
                continue
            for o, d in _key_pairs(df, names):
                key_arrays.setdefault(len(df), []).append((df[o].astype(str).to_numpy(), df[d].astype(str).to_numpy()))
    for tag, meta, load in objs:
        if meta.get("kind") not in ("df", "gdf"):
            continue
        if not (n_inter - 5 <= meta.get("rows", 0) <= n_all + 5):
            continue
        try:
            df = load()
        except Exception:
            continue
        kp = [(df[o].astype(str).to_numpy(), df[d].astype(str).to_numpy(), o, d) for o, d in _key_pairs(df, names)]
        if not kp:
            kp = [(ko, kd, None, None) for ko, kd in key_arrays.get(len(df), [])]
        for ko, kd, o, d in kp:
            key = pd.MultiIndex.from_arrays([ko, kd])
            g = gold.reindex(key)
            if g.isna().mean() > 0.01:
                continue
            for c in df.columns:
                if c in (o, d) or df[c].dtype.kind not in "fi":
                    continue
                eq = np.isclose(df[c].to_numpy(float), g.to_numpy(float)).mean()
                if eq >= 0.99:
                    agg_ok, agg_note = True, f"{c} in {meta['name']}@{tag}"
                    break
            so = socio.reindex(ko)
            sd = socio.reindex(kd)
            mo = md = 0
            for a in ATTRS:
                ao, ad = so[a].to_numpy(float), sd[a].to_numpy(float)
                cols = [c for c in df.columns if df[c].dtype.kind in "fi"]
                if any(np.isclose(df[c].to_numpy(float), ao, rtol=1e-6).mean() >= 0.99 for c in cols):
                    mo += 1
                if any(np.isclose(df[c].to_numpy(float), ad, rtol=1e-6).mean() >= 0.99 for c in cols):
                    md += 1
            if min(mo, md) > min(merge_best[0], merge_best[1]):
                merge_best = (mo, md, f"{meta['name']}@{tag}")
    out.append(result("OD records aggregated to pair sums (match gold)", agg_ok, note=agg_note,
                      threshold=f"{n_all} pairs (or {n_inter} without intra-subregion), >= 99 % equal"))
    mo, md, where = merge_best
    out.append(result("origin and destination attributes merged", mo >= 3 and md >= 3,
                      value={"origin_attrs": mo, "destination_attrs": md}, threshold=">=3 each", note=where or ""))

    vals = _mse_values(ctx)
    if not any(0.8 * var <= v <= 1.5 * var for _, v in vals):
        vals += _mse_from_arrays(ctx, np.unique(gold.to_numpy(float)))
    lo, hi = 0.8 * var, 1.5 * var
    good = [(n, v) for n, v in vals if lo <= v <= hi]
    if not vals:
        out.append(unverifiable("test-set MSE reported and plausible (0.8-1.5 x var of aggregated flows)",
                                "no MSE value, metrics file or stored test predictions captured (printed-only "
                                "MSE computed inline cannot be read: stdout is not available to checks)"))
    else:
        out.append(result("test-set MSE reported and plausible (0.8-1.5 x var of aggregated flows)", bool(good),
                          value=(good or vals or [None])[0], threshold=f"[{lo:.0f}, {hi:.0f}]",
                          note=f"{len(vals)} MSE-named values found"))
    # model features: with signal-free data any honest test MSE is
    # plausible, so the MSE cannot reveal what the model was fed. The training/test feature
    # matrix (a numeric table or 2-D array with 10-95 % of the pairs as rows, i.e. one split
    # part) must contain >= 3 socio-economic attributes on BOTH sides: >= 2 columns whose
    # values all come from the same attribute (origin copy + destination copy).
    sets = {a: np.unique(np.round(socio[a].to_numpy(float), 4)) for a in ATTRS}
    feat_best = (0, None)
    for tag, meta, load in ctx.objects("pred"):
        k = meta.get("kind")
        if k == "array":
            shp = meta.get("shape") or []
            if len(shp) != 2 or meta.get("source") == "figure":
                continue
            rows = shp[0]
        elif k in ("df", "gdf"):
            rows = meta.get("rows", 0)
        else:
            continue
        if not 0.10 * n_all <= rows <= 0.95 * n_inter:
            continue
        try:
            X = load()
            if k == "array":
                M = np.asarray(X, float)
            else:
                M = X[[c for c in X.columns if str(c) not in ("index", "level_0") and X[c].dtype.kind in "fiub"]].to_numpy(float)
        except Exception:
            continue
        if M.ndim != 2 or M.shape[1] < 2:
            continue
        both = 0
        for a in ATTRS:
            n = sum(np.isin(np.round(M[:, j], 4), sets[a]).mean() >= 0.99 for j in range(M.shape[1]))
            both += n >= 2
        if both > feat_best[0]:
            feat_best = (both, f"{meta['name']}@{tag}")
    out.append(result("model features include origin AND destination attributes", feat_best[0] >= 3,
                      value=feat_best[0], threshold=">=3 attributes present for both sides in a train/test matrix",
                      note=feat_best[1] or "no train/test feature matrix captured"))

    # method fidelity is required here: on signal-free data no output
    # can distinguish a tuned Random Forest from any other regressor, so the instructed method
    # is checked in the executed code (comments removed): RandomForestRegressor + GridSearchCV.
    import re
    code = "\n".join(l.split("#", 1)[0] for l in ctx.code_text().splitlines())
    rf = re.search(r"RandomForestRegressor\s*\(", code) is not None
    gsv = re.search(r"GridSearchCV\s*\(", code) is not None
    out.append(result("method: GridSearchCV-tuned RandomForestRegressor", rf and gsv,
                      value={"RandomForestRegressor": rf, "GridSearchCV": gsv},
                      note="instructed method; only the code can show it on this signal-free data"))
    files = ctx.pred_files("*")
    out.append(result("results written to pred_results", bool(files), value=files[:5], required=False))
    return out
