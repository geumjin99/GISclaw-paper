"""T33 - 2050 flood damage to buildings from land subsidence (RD New, elevation in cm).

Gold: flooded = elevation <= -200 cm; depth = Int(-(elev + 200)) cm; 3 928 of 9 068 buildings
intersect the flood area; mean depth per building (area-weighted over the flooded part);
DamageCosts = (0.298 ln(0.01 depth) + 1.4502) * 271 * Shape_Area if depth > 1 cm; total
~4.11e8.

Correctness (required):
  1. some result file was written to pred_results (the task names none);
  2. a per-building damage result exists: a captured building table (aligned to the input
     buildings by row index / geometry) with a numeric column whose damaged set (value > 0)
     overlaps the gold damaged set (Jaccard >= 0.80, or precision >= 0.90 and gold-damage-weighted
     recall >= 0.90, which admits the cell-centre zonal rule: J 0.64, P 1.00, weighted R 0.99)
     and whose values agree with gold (Spearman >= 0.90 on common buildings);
  3. per-building damage VALUES follow the given function: on buildings damaged in both,
     the median ratio pred/gold lies within [0.95, 1.05] (all_touched zonal mean 1.000;
     log10 instead of ln 1.68; MAX instead of mean depth 1.12);
  4. total damage within 10 % of gold (captured scalar, column sum, or a number written to a
     .txt/.json/.md summary file, or a summary line of stdout). A depth sign error, a wrong flood threshold (e.g. 0 cm)
     or depth kept in cm inside ln(0.01*d) all fail.
Tables are taken from captured objects, function locals/returns and output files (vector or
CSV); rows are aligned to the input buildings by an id column, geometry, or a full 0..n-1 index.
Reported, not required:
  5. per-building mean flood depth agrees with gold (Spearman >= 0.9).
"""
import os
import re

import numpy as np
import pandas as pd

from checklib import result, key_values


def _gold(ctx):
    for tag, meta, load in ctx.objects("gold", "gdf"):
        if meta["name"] == "flooded_buildings":
            return load().set_index("bid")
    return None


def _numbers_in_text_files(d):
    """Numbers >= 1e6 written in small text/json/markdown/csv summary files (e.g. a total); commas only as thousands separators."""
    out = []
    for f in sorted(os.listdir(d)) if d and os.path.isdir(d) else []:
        p = os.path.join(d, f)
        if not f.lower().endswith((".txt", ".json", ".md")) or os.path.getsize(p) > 200_000:
            continue
        try:
            txt = open(p, errors="ignore").read()
        except OSError:
            continue
        for m in re.findall(r"(?<![\d.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?", txt):
            try:
                v = float(m.replace(",", ""))
            except ValueError:
                continue
            if v >= 1e6:
                out.append((f"number in {f}", v))
    return out


def checks(ctx):
    out = []
    files = ctx.nonempty_pred_files()   # 0-byte files are not results
    out.append(result("result file written to pred_results", bool(files), value=files[:6], threshold="any non-empty file"))
    gold = _gold(ctx)
    nm = ["per-building damage matches gold (Jaccard>=0.80, Spearman>=0.90)",  # the set rule also admits precision / weighted recall (see docstring)
          "per-building damage values follow the given function (median pred/gold ratio 0.95-1.05)",
          "total damage within 10 % of gold"]
    if gold is None:
        for n in nm:
            out.append(result(n, False, note="gold missing"))
        return out
    import geopandas as gpd
    b = gpd.read_file(f"{ctx.inputs_dir}/StudyAreaBuildings.shp")
    b["bid"] = np.arange(len(b))
    gdam = gold["DamageCosts"].astype(float)
    gset = set(gdam.index[gdam > 0])
    gmean = gold["MEAN_gridcode"].astype(float)
    gtotal = float(gdam.sum())

    best = None
    best_depth = None
    for tag, meta, load in ctx.objects("pred"):
        if meta.get("kind") not in ("gdf", "df") or not (1000 <= meta.get("rows", 0) <= 12000):
            continue
        try:
            obj = load()
            keys = key_values(obj, b, "bid")
        except Exception:
            continue
        if keys is None or keys.isna().mean() > 0.2:
            continue
        for c in obj.columns:
            if c in ("geometry", "bid", "Shape_Leng", "Shape_Area", "index", "index_right", "index_left", "level_0",
                     "FID", "OBJECTID") or obj[c].dtype.kind not in "fi":
                continue
            s = obj[c].astype(float).groupby(keys.values).mean()
            pset = set(s.index[(s > 0) & np.isfinite(s)])
            jac = len(pset & gset) / max(1, len(pset | gset))
            prec = len(pset & gset) / max(1, len(pset))
            wrec = float(gdam.loc[list(pset & gset)].sum()) / max(1e-9, float(gdam.loc[list(gset)].sum()))
            setok = jac >= 0.8 or (prec >= 0.9 and wrec >= 0.9)
            common = list(pset & gset)
            rho = float(s.loc[common].rank().corr(gdam.loc[common].rank())) if len(common) > 10 else -1.0
            tot = float(s[np.isfinite(s)].sum())
            both = [i for i in common if s.loc[i] > 0 and gdam.loc[i] > 0]
            ratio = float(np.median(s.loc[both].to_numpy() / gdam.loc[both].to_numpy())) if len(both) > 10 else float("nan")
            sc = (setok and rho >= 0.9, -abs(ratio - 1) if ratio == ratio else -9, rho)
            if best is None or sc > best[0]:
                best = (sc, f"{meta['name']}.{c}@{tag}", jac, rho, tot, ratio, prec, wrec, setok)
            cm = list(set(s.index) & set(gmean.index))
            if len(cm) > 10:
                rd = float(s.loc[cm].rank().corr(gmean.loc[cm].rank()))
                if best_depth is None or rd > best_depth[1]:
                    best_depth = (f"{meta['name']}.{c}@{tag}", rd)
    if best is None:
        out.append(result(nm[0], False, note="no building-level table found"))
    else:
        _, lab, jac, rho, tot, ratio, prec, wrec, setok = best
        out.append(result(nm[0], setok and rho >= 0.9,
                          value=f"Jaccard={jac:.3f}, precision={prec:.3f}, weighted recall={wrec:.3f}, rho={rho:.3f}",
                          threshold="(Jaccard>=0.80 or (precision>=0.90 and weighted recall>=0.90)) and rho>=0.90",
                          note=lab))
    if best is None:
        out.append(result(nm[1], False, note="no building-level table found"))
    else:
        out.append(result(nm[1], ratio == ratio and abs(ratio - 1) <= 0.05, value=round(ratio, 3) if ratio == ratio else None,
                          threshold="0.95-1.05", note=best[1]))
    cands = [(m["name"], m["value"]) for _, m, _ in ctx.objects("pred", "scalar") if m.get("value")]
    if best is not None:
        cands.append(("sum of " + best[1], best[4]))
    cands += _numbers_in_text_files(ctx.pred_dir)
    try:
        # ctx.stdout may be a property (str) or a callable
        _so = getattr(ctx, "stdout", "")
        so = _so() if callable(_so) else (_so if isinstance(_so, str) else "")
    except Exception:
        so = ""
    for line in so.splitlines():
        if re.search(r"total|damage|sum|cost", line, re.I):
            nums = re.findall(r"(?<![\d.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?", line)
            if 0 < len(nums) <= 3:
                for m in nums:
                    v = float(m.replace(",", ""))
                    if v >= 1e6:
                        cands.append(("stdout: " + line.strip()[:60], v))
    cands = [c for c in cands if c[1] and np.isfinite(c[1])]
    hit = min(cands, key=lambda t: abs(t[1] / gtotal - 1)) if cands else None
    out.append(result(nm[2], bool(hit) and abs(hit[1] / gtotal - 1) <= 0.10, value=round(hit[1]) if hit else None,
                      threshold=f"{gtotal:.4g} +/- 10 %", note=hit[0] if hit else "no total found"))
    out.append(result("per-building mean flood depth matches gold", bool(best_depth and best_depth[1] >= 0.9),
                      value=round(best_depth[1], 3) if best_depth else None, threshold=0.9, required=False,
                      note=best_depth[0] if best_depth else ""))
    return out
