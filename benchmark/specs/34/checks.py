"""Checks for task 34: rural population within 2 km of all-season roads, Shikoku.

Gold: areal-weighted population of the 48 RURAL municipalities, 2 km road buffer computed in
JGD2011 / UTM 53N; 87.7 % of the rural population (814 k) has access; per-municipality
shares 55-100 %.

Correctness (required):
  1. accessibility.png exists and is not blank;
  2. a per-area access result restricted to the rural areas exists: a captured table whose
     rows map (by JISCODE, row index or geometry) to municipalities, >= 90 % of them RURAL,
     either per municipality (~48 rows) or per population small area (aggregated by mean);
  3. its pattern agrees with gold per-municipality shares: Spearman >= 0.70 or mean absolute
     difference <= 7 percentage points (areal weighting vs centroid allocation differ by
     rho 0.76 / MAE 5.9 pp; share of *area* instead of population gives rho 0.55 / MAE 24);
  4. the choropleth shows the access share: a column identified as drawn (colour values, or
     face colours inverted to colormap positions) belongs to a per-area table whose pattern
     passes check 3;
  5. the overall rural share the solution itself reports is right: within 3 pp of 87.7 %
     (fractions accepted). Candidates are numbers written to .txt/.json/.md files, summary lines
     of stdout (keyword + <= 3 numbers; '2' as in '2 km' ignored; an integer needs '%' or share
     wording - 'Rural count: 48' is not a share) and
     captured scalars named like a share (share/pct/percent/ratio/frac/prop/access/overall/
     rate/cover), and a number printed on one stdout line together with the two
     totals it is the ratio of (only when exactly one stdout line has such a ratio - a
     per-area table satisfies it on every row); if none is present, any captured scalar. The
     overall share is not derived from the solution's per-area table (an unweighted mean of
     municipal shares, 82.4 %, must fail).
"""
import os
import re

import numpy as np
import pandas as pd

from checklib import (drawn_color_column_matches, drawn_column_matches, file_exists, png_not_blank,
                      result, key_values)

SHARE_NAME = re.compile(r"share|pct|perc|ratio|frac|prop|access|overall|rate|cover", re.I)
SHARE_WORD = re.compile(r"%|percent|share|proportion|ratio|fraction|pct|rate|coverage", re.I)


def _stdout_numbers(ctx):
    """Shares printed in a summary line of stdout: a line mentioning overall/total/percent/share/
    access/proportion/rural with at most 3 numbers (table rows are ignored). Digits inside
    identifiers are not numbers ('D0001' is not 1), thousands
    separators are parsed ('805,227' is one number), and in a line mentioning overall/total a
    number written directly with '%' is a candidate whatever the count of other numbers
    ('rural pop within 2km: 705918.8 of 805226.6 -> 87.67 %'). On a
    keyword line with <= 3 numbers an integer counts only with '%' or share wording
    ('Rural count: 48' is not a share)."""
    out = []
    try:
        # ctx.stdout is a property (str); a callable is accepted as well
        _so = getattr(ctx, "stdout", "")
        txt = _so() if callable(_so) else (_so if isinstance(_so, str) else "")
    except Exception:
        txt = ""
    num = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+(?!\d)|\d+)(\.\d+)?(\s*%)?")
    ratio_out = []
    for line in txt.splitlines():
        nums = [(float((m.group(1) + (m.group(2) or "")).replace(",", "")), bool(m.group(3)), bool(m.group(2)))
                for m in num.finditer(line)]
        # A share printed together with the two totals it
        # is computed from, on a line without a keyword (e.g. appended to the last row of a
        # describe() table: 'max ... 805226.65 705918.85 87.667'): a number v (0 < v <= 100)
        # that equals 100 * a / b (or a / b) for two other numbers a < b (both > 100, i.e.
        # totals) printed on the same line, exactly up to the rounding of v as printed (half a
        # unit of its last decimal), so that describe() rows do not match by coincidence.
        if 3 <= len(nums) <= 12:
            vals = [v for v, _, _ in nums]
            decs = [len(m.group(2)) - 1 if m.group(2) else 0 for m in num.finditer(line)]
            for i, v in enumerate(vals):
                if not 0 < v <= 100 or v == 2 or decs[i] < 1:
                    continue
                tol = 0.5 * 10 ** -decs[i] + 1e-12
                if any(100 < a < b and (abs(100 * a / b - v) <= tol or abs(a / b - v) <= tol)
                       for ia, a in enumerate(vals) for ib, b in enumerate(vals) if len({i, ia, ib}) == 3):
                    ratio_out.append(("stdout (ratio of printed totals): " + line.strip()[-60:], v * 100 if v <= 1 else v))
                    break
        if not re.search(r"overall|total|percent|%|share|access|proportion|rural", line, re.I):
            continue
        # A keyword line alone ('Rural count: 48') does
        # not make its numbers shares. A number counts only when written with '%', when the
        # line uses share wording, or when it is non-integer (87.67, 0.877) - counts are integers.
        share_word = bool(SHARE_WORD.search(line))
        if 0 < len(nums) <= 3:
            picked = [v for v, pct, dec in nums if pct or share_word or dec]
        elif re.search(r"overall|total", line, re.I):
            picked = [v for v, pct, dec in nums if pct]
        else:
            picked = []
        for v in picked:
            if 0 < v <= 100 and v != 2:
                out.append(("stdout: " + line.strip()[:60], v * 100 if v <= 1 else v))
    # A per-area table printed with population, population within the buffer and the share
    # satisfies the ratio on every row, and a single municipality's share is not the overall
    # share. Ratio candidates count only when exactly
    # one stdout line has one (the overall share printed next to the two totals).
    if len(ratio_out) == 1:
        out += ratio_out
    return out


def _text_numbers(d):
    out = []
    for f in sorted(os.listdir(d)) if d and os.path.isdir(d) else []:
        p = os.path.join(d, f)
        if not f.lower().endswith((".txt", ".json", ".md")) or os.path.getsize(p) > 200_000:
            continue
        try:
            txt = open(p, errors="ignore").read()
        except OSError:
            continue
        for m in re.findall(r"(?<![\d.])\d+(?:\.\d+)?", txt):
            v = float(m)
            if 0 < v <= 100:
                out.append((f"number in {f}", v * 100 if v <= 1 else v))
    return out


def _gold(ctx):
    for tag, meta, load in ctx.objects("gold", "gdf"):
        if meta["name"] == "rural_access":
            g = load()
            return g.set_index("JISCODE")
    return None


def _scale(v):
    """Return values in percent (fractions in [0,1] are scaled)."""
    v = np.asarray(v, float)
    return v * 100 if np.nanmax(v) <= 1.0001 else v


def checks(ctx):
    out = []
    names = ["accessibility.png", "Accessibility.png", "accessibility.PNG"]
    r = file_exists(ctx, names)
    out.append(r)
    if r["passed"]:
        nb, std = png_not_blank(ctx.pred(r["value"][0]))
        out.append(result("map not blank", nb, value=std, threshold=">=3 grey-level std"))
    else:
        out.append(result("map not blank", False, note="no accessibility.png"))

    gold = _gold(ctx)
    names3 = ["rural-only per-area access result present", "per-area pattern matches gold (Spearman>=0.70 or MAE<=7pp)",
              "choropleth draws the per-area access share", "overall rural share within 3 pp of gold"]
    if gold is None:
        for nm in names3:
            out.append(result(nm, False, note="gold table missing"))
        return out
    import geopandas as gpd
    metro = gpd.read_file(f"{ctx.inputs_dir}/ShikokuMetropolitan.geojson")
    rural_codes = set(metro.loc[metro["AREATYPE"] == "RURAL", "JISCODE"].astype(str))
    gpct = gold["pct_access"].astype(float)
    gpct.index = gpct.index.astype(str)
    gw = gold["pop_total"].astype(float)
    gw.index = gw.index.astype(str)
    goverall = float((gpct * gw).sum() / gw.sum())

    best = None  # (score tuple, label, rho, mae, series, rural_frac)
    good_cols = set()   # (table name, column, tag) passing the pattern test
    exclude = {"OBJECTID", "MEACODE", "Shape__Area", "Shape__Length", "ID", "D0001", "index", "index_right",
               "index_left", "level_0", "JISCODE", "orig_area", "area"}
    for tag, meta, load in ctx.objects("pred"):
        if meta.get("kind") not in ("gdf", "df") or not (40 <= meta.get("rows", 0) <= 8000):
            continue
        try:
            obj = load()
            keys = key_values(obj, metro, "JISCODE")
        except Exception:
            continue
        if keys is None:
            continue
        keys = keys.astype(str).str.replace(r"\.0$", "", regex=True)
        rural_frac = float(keys.isin(rural_codes).mean())
        for c in obj.columns:
            if c in exclude or c == "geometry" or obj[c].dtype.kind not in "fi":
                continue
            s = obj[c].astype(float).groupby(keys.values).mean()
            common = gpct.index.intersection(s.index)
            if len(common) < 0.8 * len(gpct) or s.loc[common].std() == 0:
                continue
            x = pd.Series(_scale(s.loc[common].values), index=common)
            y = gpct.loc[common]
            rho = float(x.rank().corr(y.rank()))
            mae = float((x - y).abs().mean())
            ok = rho >= 0.70 or mae <= 7.0
            if ok and rural_frac >= 0.9:
                good_cols.add((meta["name"], c, tag))
            score = (rural_frac >= 0.9, ok, rho)
            if best is None or score > best[0]:
                best = (score, f"{meta['name']}.{c}@{tag}", rho, mae, x, rural_frac)
    if best is None:
        out.append(result(names3[0], False, note="no per-area table mapping to the rural municipalities"))
        out.append(result(names3[1], False))
    else:
        _, lab, rho, mae, x, rf = best
        out.append(result(names3[0], rf >= 0.9, value=round(rf, 3), threshold=">=0.9 of rows RURAL", note=lab))
        out.append(result(names3[1], rho >= 0.70 or mae <= 7.0, value=f"rho={rho:.3f}, MAE={mae:.1f}",
                          threshold="rho>=0.70 or MAE<=7", note=lab))
    # drawn choropleth
    drawn = [(t, c, tg) for t, c, tg in drawn_column_matches(ctx)] + \
            [(t, c, tg) for t, c, _, _, tg in drawn_color_column_matches(ctx, min_rho=0.9)]
    ok_drawn = [d for d in drawn if d in good_cols]
    if ok_drawn:
        out.append(result(names3[2], True, value=f"{ok_drawn[0][0]}.{ok_drawn[0][1]}@{ok_drawn[0][2]}"))
    elif drawn:
        out.append(result(names3[2], False, value=[f"{t}.{c}" for t, c, _ in drawn[:5]],
                          note="drawn columns do not carry the per-area access share"))
    else:
        out.append(result(names3[2], False, note="UNVERIFIABLE: no drawn column identified (colours not invertible)"))
        out[-1]["unverifiable"] = True
    # overall share reported by the solution itself
    scal = [(m["name"], m.get("value")) for _, m, _ in ctx.objects("pred", "scalar")]
    scal = [(n, v * 100 if v <= 1 else v) for n, v in scal if v is not None and 0 < v <= 100]
    so = _stdout_numbers(ctx)
    # ratio-of-printed-totals candidates are added on their own, so that they
    # do not switch off the "any captured scalar" fallback when nothing else is reported
    ratio_c = [c for c in so if c[0].startswith("stdout (ratio")]
    reported = _text_numbers(ctx.pred_dir) + [(n, v) for n, v in scal if SHARE_NAME.search(n)] + \
        [c for c in so if not c[0].startswith("stdout (ratio")]
    cands = (reported or scal) + ratio_c
    hit = min(cands, key=lambda t: abs(t[1] - goverall)) if cands else None
    out.append(result(names3[3], bool(hit) and abs(hit[1] - goverall) <= 3.0,
                      value=round(hit[1], 2) if hit else None, threshold=f"{goverall:.2f} +/- 3",
                      note=(hit[0] + (" (reported)" if reported or hit in ratio_c else " (any scalar)"))
                      if hit else "no overall share found"))
    return out
