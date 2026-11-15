"""T37 - County heat-exposure sentiment from tweets, USA.

Gold (authors' method with a case-insensitive vocabulary lookup, unit fragments such as 'oF'
dropped): 294 located weather-related tweets, 92 US counties; heat exposure = mean tweet value
per county, min-max normalised; natural-breaks choropleth. Two other lookup conventions are
accepted as gold variants (`county_variants`): the authors' literal lookup (lower-cased text in
a cased dictionary, so 'Hot'/'Cold' never match; 75 counties) and an exact-case lookup
(65 counties). A case-insensitive lookup that keeps 'oF' -> 'of' (514 ordinary tweets become
"weather" tweets, 174 counties) is not accepted.

Correctness (required), each against the best-matching variant:
  1. US_heat_exposure_map.png exists and is not blank;
  2. a per-county heat-exposure result exists (table aligned by GEOID / row index / geometry)
     covering >= 60 % of the variant's counties with Spearman >= 0.80;
  3. the county value is the MEAN of the tweet values, not the sum: regressing the result on
     [mean, (n - 1) * mean] (n = tweets per county) gives a (n-1)*mean coefficient below 0.4 of
     the mean coefficient (means 0.00-0.13, sums 0.75-1.4 across conventions and text fields);
     UNVERIFIABLE (never passed) when check 2 fails: the regression is noise unless the result
     covers the variant's tweet set. Count-like
     columns (integer-valued or <= 2 distinct values) are not candidates for checks 2-5 (a 0/1
     tweet-count column could outscore the real result column on coverage);
  4. only weather-related tweets were used: counties with a non-zero value <= 1.5 x the
     variant's count (keeping all tweets gives 511 counties). A fill value is not counted:
     when the column covers >= 90 % of all US counties and one value is
     shared by > 50 % of them, counties with that value that have no tweets in any variant are
     ignored (filling tweet-less counties with 0 and then min-max normalising turns the fill
     into a constant non-zero value). Known gap: skipping the weather filter AND filling every
     county the same way is indistinguishable by this count;
  5. values were min-max normalised: the column spans [0, 1] (min <= 0.02, max >= 0.98);
  6. the map shows the result: a drawn polygon collection, located county by county through
     the captured per-path bounding boxes (geographic, Albers, Web Mercator or National Atlas
     coordinates), has face colours whose colormap positions rank-correlate with the variant's
     county means (|Spearman| >= 0.70 over >= 60 % of its counties). If the colours do not fit a
     colormap (mean residual > 0.03), achromatic (grey) faces are dropped as no-data and the fit
     is retried (grey no-data counties may be drawn in the same
     collection with color=...). The collections of one axes
     are also read merged, so a natural-breaks map drawn class by class (one colour per
     collection) is located as one map. The CRS is chosen by the number of
     distinct counties located (counting located paths would let a wrong CRS, which piles
     all paths into one county, win when a few paths fall outside every county).
  7. (the text says "choropleth map using natural break") natural
     breaks (Jenks / Fisher-Jenks: mapclassify NaturalBreaks / FisherJenks, geopandas
     scheme='natural_breaks' / 'fisherjenks', jenkspy, a hand-written jenks) used for the
     classification - code inspection with comments stripped.
Reported, not required:
  8. post counts per county computed (a column with Spearman >= 0.8 vs gold counts).
"""
import re

import numpy as np
import pandas as pd

from checklib import drawn_arrays, file_exists, png_not_blank, result, key_values, unverifiable
from checklib import _cmap_positions


def _strip_comments(src):
    """Source without '#' comments (tokenize; falls back to the raw text)."""
    import io
    import tokenize
    try:
        toks = [t for t in tokenize.generate_tokens(io.StringIO(src).readline) if t.type != tokenize.COMMENT]
        return tokenize.untokenize(toks)
    except Exception:
        return src

SUM_RATIO_MAX = 0.4


def _gold(ctx):
    g, v = None, None
    for tag, meta, load in ctx.objects("gold", "df", include_files=False):
        if tag != "final":
            continue
        if meta["name"] == "county_stats":
            g = load()
            g["GEOID"] = g["GEOID"].astype(str)
            g = g.set_index("GEOID")
        if meta["name"] == "county_variants":
            v = load()
            v["GEOID"] = v["GEOID"].astype(str)
    if v is None:
        return g, {}
    return g, {k: d.set_index("GEOID") for k, d in v.groupby("variant")}


def _sum_ratio(s, gv):
    """Coefficient of (n-1)*mean relative to mean when regressing s on both (gv: mean, count)."""
    c = s.index.intersection(gv.index)
    if len(c) < 10:
        return float("nan")
    m, n = gv.loc[c, "mean"].astype(float).to_numpy(), gv.loc[c, "count"].astype(float).to_numpy()
    X = np.c_[np.ones(len(c)), m, (n - 1) * m]
    b = np.linalg.lstsq(X, s.loc[c].astype(float).to_numpy(), rcond=None)[0]
    return float(b[2] / b[1]) if b[1] != 0 else float("nan")


def _drawn_county_positions(ctx, US):
    """Per drawn polygon collection: Series GEOID -> colormap position of its face colour.
    Paths are located by the centre of their bounding box (several CRSs are tried)."""
    from pyproj import Transformer
    import geopandas as gpd
    arrs = {}
    for tag, meta, a in drawn_arrays(ctx):
        n = meta["name"]
        for suf in ("_facecolors", "_paths_bbox"):
            if n.endswith(suf):
                arrs.setdefault((tag, n[: -len(suf)]), {})[suf] = np.asarray(a, float)
    units = []
    per_ax = {}
    for (tag, base), d in arrs.items():
        if "_facecolors" not in d or "_paths_bbox" not in d:
            continue
        fc, bb = d["_facecolors"], d["_paths_bbox"].reshape(-1, 4)
        units.append((f"{base}@{tag}", fc, bb))
        # a natural-breaks map drawn class by class (one colour per
        # collection) is read by merging the collections of the same axes; one-row facecolors are
        # broadcast over the collection's paths, fully transparent faces are left out
        if len(fc) == 1 and len(bb) >= 1:
            fc = np.repeat(fc, len(bb), axis=0)
        if len(fc) == len(bb):
            vis = fc[:, 3] > 0 if fc.shape[1] > 3 else np.ones(len(fc), bool)
            per_ax.setdefault((tag, base.split("_coll")[0]), []).append((fc[vis], bb[vis]))
    for (tag, ax), parts in per_ax.items():
        if len(parts) >= 2:
            units.append((f"{ax} (all {len(parts)} collections merged)@{tag}",
                          np.concatenate([f for f, _ in parts]), np.concatenate([b for _, b in parts])))
    out = []
    for base, fc, bb in units:
        if len(fc) != len(bb) or len(bb) < 10 or len(np.unique(np.round(fc[:, :3], 4), axis=0)) < 3:
            continue
        pos, cname, res = _cmap_positions(fc)
        if pos is None or res > 0.03:
            # fallback: no-data counties drawn in grey in the same
            # collection (color=...) are not on the colormap. Drop achromatic faces (R=G=B) and
            # refit; only when the full fit fails, so Greys-type colormaps are unaffected.
            chroma = fc[:, :3].max(1) - fc[:, :3].min(1)
            keep = chroma > 0.02
            if keep.sum() < 10 or len(np.unique(np.round(fc[keep, :3], 4), axis=0)) < 3:
                continue
            pos, cname, res = _cmap_positions(fc[keep])
            if pos is None or res > 0.03:
                continue
            bb = bb[keep]
            cname = f"{cname}; {int((~keep).sum())} grey no-data faces dropped"
        cx, cy = (bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2
        best = None
        for crs in ("EPSG:4269", "EPSG:5070", "EPSG:3857", "EPSG:9311", "ESRI:102003"):
            try:
                if crs == "EPSG:4269":
                    x, y = cx, cy
                else:
                    x, y = Transformer.from_crs(crs, "EPSG:4269", always_xy=True).transform(cx, cy)
                pts = gpd.GeoDataFrame({"pos": pos}, geometry=gpd.points_from_xy(x, y), crs="EPSG:4269")
                j = gpd.sjoin(pts, US[["GEOID", "geometry"]].to_crs("EPSG:4269"), predicate="within")
            except Exception:
                continue
            # rank CRSs by distinct counties hit: degrees read as projected metres collapse every
            # path into one county near the projection origin, which could out-count the right CRS
            if best is None or j["GEOID"].nunique() > best["GEOID"].nunique():
                best = j
        if best is not None and len(best) >= 0.5 * len(bb):
            out.append((f"{base} ({cname})", best.groupby("GEOID")["pos"].mean()))
    return out


def _columns(ctx, US):
    for tag, meta, load in ctx.objects("pred"):
        if meta.get("kind") not in ("gdf", "df") or not (20 <= meta.get("rows", 0) <= 4000):
            continue
        try:
            obj = load()
            keys = key_values(obj, US, "GEOID")
        except Exception:
            continue
        if keys is None:
            continue
        keys = keys.astype(str)
        if keys.duplicated().mean() > 0.05:
            continue  # per-tweet tables (e.g. the spatial join) are not a county result
        for c in obj.columns:
            if c in ("geometry", "GEOID", "index", "index_right", "index_left", "level_0", "ALAND", "AWATER",
                     "lat", "lon", "latitude", "longitude", "x", "y") or obj[c].dtype.kind not in "fi":
                continue
            s = obj[c].astype(float).groupby(keys.values).mean()
            yield f"{meta['name']}.{c}@{tag}", s


def checks(ctx):
    out = []
    names = ["US_heat_exposure_map.png", "us_heat_exposure_map.png", "US_heat_exposure_map.PNG"]
    r = file_exists(ctx, names)
    out.append(r)
    if r["passed"]:
        nb, std = png_not_blank(ctx.pred(r["value"][0]))
        out.append(result("map not blank", nb, value=std, threshold=">=3 grey-level std"))
    else:
        out.append(result("map not blank", False, note="no US_heat_exposure_map.png"))

    gold, variants = _gold(ctx)
    nm = ["per-county heat exposure matches gold (coverage>=60 %, Spearman>=0.80)",
          "county value is the mean (not the sum) of tweet values",
          "only weather-related tweets used (counties with value <= 1.5x gold)",
          "min-max normalised to [0, 1]",
          "map shows the county heat exposure (drawn colours vs gold, |rho| >= 0.70)"]
    if gold is None or not variants:
        for n in nm:
            out.append(result(n, False, note="gold table missing"))
        return out
    import geopandas as gpd
    US = gpd.read_file(f"{ctx.inputs_dir}/cb_2020_us_county_20m.shp")
    US["GEOID"] = US["GEOID"].astype(str)
    gcnt = gold["number_of_twitter"].astype(float)
    best, best_cnt = None, None
    tweeted = set().union(*[set(gv.index) for gv in variants.values()])
    for lab, s in _columns(ctx, US):
        s = s[np.isfinite(s)]
        if len(s) < 5 or s.std() == 0:
            continue
        c2 = gcnt.index.intersection(s.index)
        rc = float(s.loc[c2].rank().corr(gcnt.loc[c2].rank())) if len(c2) > 5 and s.loc[c2].std() else -1
        if best_cnt is None or rc > best_cnt[1]:
            best_cnt = (lab, rc)
        # count-like columns (integer-valued, or <= 2 distinct values, e.g.
        # a 0/1 tweet count over all counties) are not candidates for the exposure result; they
        # could outscore the real result column on coverage. Post counts (check 8) still see them.
        if s.round(9).nunique() <= 2 or np.allclose(s.to_numpy(), np.round(s.to_numpy()), rtol=0):
            continue
        sr = s.round(9)
        mode = sr.mode().iloc[0]
        at_mode = (sr == mode).to_numpy()
        # only for a table filled over (nearly) all counties: a no-weather-filter result over the
        # ~511 counties with posts has the same constant (mean 0 shifted by min-max) and stays counted
        full = len(s) >= 0.9 * len(US)
        fill = at_mode & ~s.index.isin(list(tweeted)) if (full and at_mode.mean() > 0.5) else np.zeros(len(s), bool)
        lo, hi = float(s.min()), float(s.max())
        normed = -1e-9 <= lo <= 0.02 and 0.98 <= hi <= 1.0 + 1e-9
        for vname, gv in variants.items():
            gm = gv["mean"].astype(float)
            common = gm.index.intersection(s.index)
            if len(common) < 5 or s.loc[common].std() == 0:
                continue
            cov = len(common) / len(gm)
            rho = float(s.loc[common].rank().corr(gm.loc[common].rank()))
            ratio = _sum_ratio(s, gv)
            nz = int(((s != 0).to_numpy() & ~fill).sum())
            ok = cov >= 0.6 and rho >= 0.8
            score = (ok, ok and abs(ratio) <= SUM_RATIO_MAX, ok and nz <= 1.5 * len(gm), normed,
                     rho if cov >= 0.6 else rho - 2)
            if best is None or score > best[0]:
                best = (score, lab, cov, rho, s, vname, ratio, nz)
    if best is None:
        for n in nm[:4]:
            out.append(result(n, False, note="no per-county numeric result aligned to GEOID"))
        vname = "case_insensitive"
    else:
        _, lab, cov, rho, s, vname, ratio, nz = best
        nv = len(variants[vname])
        out.append(result(nm[0], cov >= 0.6 and rho >= 0.8, value=f"coverage={cov:.2f}, rho={rho:.3f}",
                          threshold="coverage>=0.6, rho>=0.80", note=f"{lab} vs {vname}"))
        if cov >= 0.6 and rho >= 0.8:
            out.append(result(nm[1], ratio == ratio and abs(ratio) <= SUM_RATIO_MAX,
                              value=round(ratio, 3) if ratio == ratio else None,
                              threshold=f"|coef((n-1)*mean)/coef(mean)| <= {SUM_RATIO_MAX}", note=f"{lab} vs {vname}"))
        else:
            # the regression on the variant's mean and tweet
            # counts is only meaningful when the result matches the variant's tweet set (check 1)
            out.append(unverifiable(nm[1], f"check 1 failed (coverage={cov:.2f}, rho={rho:.3f}); "
                                           f"mean-vs-sum regression not meaningful ({lab} vs {vname})"))
        out.append(result(nm[2], nz <= 1.5 * nv, value=nz, threshold=f"<= {1.5 * nv:.0f}", note=f"{lab} vs {vname}"))
        lo, hi = float(s.min()), float(s.max())
        out.append(result(nm[3], lo <= 0.02 and lo >= -1e-9 and 0.98 <= hi <= 1.0 + 1e-9,
                          value=f"[{lo:.3f}, {hi:.3f}]", threshold="[0, 1]", note=lab))
    # drawn map
    try:
        drawn = _drawn_county_positions(ctx, US)
    except Exception as e:
        drawn = []
    bestd = None
    for dlab, pos in drawn:
        for vn in [vname] + [k for k in variants if k != vname]:
            gm = variants[vn]["mean"].astype(float)
            c = gm.index.intersection(pos.index)
            if len(c) < 5 or pos.loc[c].std() == 0:
                continue
            cov = len(c) / len(gm)
            r = abs(float(pos.loc[c].rank().corr(gm.loc[c].rank())))
            sc = (cov >= 0.6 and r >= 0.7, r)
            if bestd is None or sc > bestd[0]:
                bestd = (sc, f"{dlab} vs {vn}", cov, r)
    if bestd is None:
        out.append(result(nm[4], False, note="UNVERIFIABLE: no drawn polygon collection with colormap colours located on counties"))
        out[-1]["unverifiable"] = True
    else:
        _, dl, cov, r = bestd
        out.append(result(nm[4], cov >= 0.6 and r >= 0.7, value=f"coverage={cov:.2f}, |rho|={r:.3f}",
                          threshold="coverage>=0.6, |rho|>=0.70", note=dl))
    code = _strip_comments(ctx.code_text())
    jenks = bool(re.search(r"natural_?breaks|fisher_?jenks|jenks", code, re.I))
    # the text prescribes the classification ("choropleth map using natural break"), so a
    # method named in the task text is required. Comments are stripped so a "# natural breaks"
    # remark alone does not pass.
    out.append(result("natural breaks classification used", jenks,
                      note="code inspection (comments stripped); the text prescribes natural breaks"))
    out.append(result("post counts per county computed", bool(best_cnt and best_cnt[1] >= 0.8),
                      value=round(best_cnt[1], 3) if best_cnt else None, threshold=0.8, required=False,
                      note=best_cnt[0] if best_cnt else ""))
    return out
