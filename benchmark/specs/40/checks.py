"""Checks for task 40: malaria prevalence change in Sub-Saharan Africa, 2000 -> 2015.

Task core: change = PR(2015) - PR(2000) with the Minus function (1741 x 1681 grid, ~70 %
nodata), then a diverging map that separates increases from decreases. The instruction names
no output file, so any PNG figure is accepted.

Correctness (required):
  1. a figure (any .png) exists and is not blank;
  2. a continuous change field was computed: some captured 2-D array or written GeoTIFF on
     the input grid agrees cell-by-cell with the gold change (Pearson >= 0.95 on cells valid
     in both). Reversed subtraction (2000 - 2015) gives r = -1, plotting a single year gives
     |r| <= 0.76, a ratio gives r ~ 0;
  3. the field has the gold's scale and sign: its mean equals the gold mean (-0.171) within
     25 % - either as a fraction or in percentage points (x100);
  4. the map shows the change: an image (imshow, or pcolormesh with orientation from its mesh
     coordinates) drawn in a saved figure (a continuous or a classified change map) has
     Spearman >= 0.85 with the gold change (the 2000 or 2015 surface, a ratio or a reversed
     difference fail). The drawn image is accepted as
     drawn or flipped N-S (which way up is presentation; kind="figure" check 9 keeps "as drawn");
  5. that drawn change map masks nodata: >= 90 % of the gold nodata cells (ocean / outside the
     endemic area, 70 % of the grid) are NaN / masked in the drawn image, not drawn as
     "no change" (an unmasked -9999 - -9999 = 0 ocean fails).
Subtraction order: the instruction's "difference between the
prevalence rates in 2000 and 2015 using the Minus raster function" can be read literally as
Minus(2000, 2015). The task asks for the change from 2000 to 2015, i.e. 2015 - 2000, so the
reversed order is NOT accepted: a 2000 - 2015 field fails checks 2-5 even when the direction
is stated and drawn consistently. A "reversed order ... stated" row is reported (not
required) for the sensitivity analysis. It is true when (a) the
code or a figure text states the direction ("2000 minus 2015", or with a U+2212 minus) AND
(b) the saved map reads correctly: with 92 % of gold cells decreasing, strongly blue pixels
outnumber strongly red ones at least 2:1 (red = increase, blue = decrease). With
REV_ACCEPT = True such a field is scored sign-flipped in checks 2-5; a reversed field
drawn red = decrease, or one without a direction statement, fails under either setting.
  8. (the text says "visualize the change ... with a diverging color
     scheme") a diverging colour scheme is used - code inspection
     with comments stripped: a diverging colormap name (matplotlib, seaborn vlag/icefire,
     cmocean), TwoSlopeNorm / CenteredNorm / vcenter, or a custom ListedColormap /
     LinearSegmentedColormap.
Reported, not required:
  6. nodata masked in the best change array;
  7. a categorical change map (<= 10 classes, ordered from decrease to increase) was derived;
  9. (kind="figure") the change map is drawn north-up (check 4 without the flip).
"""
import glob
import os

import numpy as np

import re

from checklib import drawn_arrays, raster_agreement, result, png_not_blank, read_raster, find_object

GOLD_TIF = "Malaria_Change.tif"


def _strip_comments(src):
    """Source without '#' comments (tokenize; falls back to the raw text)."""
    import io
    import tokenize
    try:
        toks = [t for t in tokenize.generate_tokens(io.StringIO(src).readline) if t.type != tokenize.COMMENT]
        return tokenize.untokenize(toks)
    except Exception:
        return src


def _gold(ctx):
    try:
        a, _ = read_raster(ctx.gold(GOLD_TIF))
        return a
    except Exception:
        return None


def _candidates(ctx, shape):
    """(label, array) for every 2-D captured array / written GeoTIFF on the input grid."""
    out = []
    for tag, meta, load in ctx.objects("pred", "array"):
        shp = meta.get("shape") or []
        if len(shp) == 3 and shp[0] == 1:
            shp = shp[1:]
        if list(shp) != list(shape):
            continue
        try:
            a = np.asarray(load(), dtype=float).reshape(shape)
        except Exception:
            continue
        a[a <= -9998] = np.nan
        out.append((f"{meta['name']}@{tag}", a))
    for p in glob.glob(os.path.join(ctx.pred_dir, "*.tif*")):
        try:
            a, m = read_raster(p)
        except Exception:
            continue
        if a.shape == tuple(shape):
            a[a <= -9998] = np.nan
            out.append((os.path.basename(p), a))
    return out


_DIR_RE = re.compile(r"2000\s*(?:minus|\u2212)\s*2015", re.I)


def _direction_stated(ctx):
    """The 2000-minus-2015 direction is stated in the code or in a figure text."""
    texts = [ctx.code_text()]
    try:
        for tag, meta, load in ctx.objects("pred", "text", include_files=False):
            if meta.get("source") == "figure":
                texts.append(repr(meta.get("texts")))
    except Exception:
        pass
    return any(_DIR_RE.search(t or "") for t in texts)


def _decrease_drawn_blue(ctx):
    """Visual reading of a 2000-minus-2015 map: most of the endemic area decreased (gold: 92 % of
    cells), so with red = increase / blue = decrease the strongly coloured map pixels are mostly
    blue. Returns (ok, note) for the PNG with the clearest blue majority."""
    import matplotlib.image as mpimg
    best = (False, "no PNG")
    for p in ctx.pred_files("*.png"):
        try:
            im = mpimg.imread(ctx.pred(p))[..., :3].astype(float)
        except Exception:
            continue
        if im.max() > 1.5:
            im = im / 255.0
        R, G, B = im[..., 0], im[..., 1], im[..., 2]
        red = int((((R - B) > 0.25) & ((R - G) > 0.15)).sum())
        blue = int(((B - R) > 0.25).sum())
        ok = blue >= 2 * red and blue >= 0.01 * R.size
        note = f"{p}: blue px {blue}, red px {red}"
        if ok or not best[0]:
            best = (ok, note)
        if ok:
            break
    return best


# Reversed subtraction is reported but not accepted (see docstring).
REV_ACCEPT = False


def checks(ctx):
    out = []
    pngs = ctx.pred_files("*.png")
    out.append(result("a figure (PNG) is present", bool(pngs), value=pngs, threshold="any *.png"))
    if pngs:
        best = max((png_not_blank(ctx.pred(p)) for p in pngs), key=lambda t: t[1] if isinstance(t[1], float) else -1)
        out.append(result("figure not blank", best[0], value=best[1], threshold=">=3 grey-level std"))
    else:
        out.append(result("figure not blank", False, note="no PNG"))

    gold = _gold(ctx)
    if gold is None:
        out.append(result("change field matches gold (Pearson)", False, note="gold raster missing"))
        out.append(result("change has gold scale and sign (mean within 25 %)", False))
        out.append(result("map shows the change (drawn image Spearman >= 0.85)", False, note="gold raster missing"))
        out.append(result("drawn change map masks nodata", False, note="gold raster missing"))
        return out
    gv = np.isfinite(gold)
    cands = _candidates(ctx, gold.shape)
    # the literal "Minus(2000, 2015)" order: reported when the
    # direction is stated (code / figure text) and the map reads correctly (decrease = blue).
    rev_ok, rev_note = False, ""
    if _direction_stated(ctx):
        rev_ok, rev_note = _decrease_drawn_blue(ctx)
    out.append(result("reversed order (2000 minus 2015) stated and drawn with red = increase", rev_ok,
                      required=False, value=rev_note or None,
                      note=("when true, the sign-flipped field / drawn image is scored by checks 2-5" if REV_ACCEPT
                            else "reported only; reversed order not accepted")))
    if rev_ok and REV_ACCEPT:
        cands = cands + [(f"-({lab}) [2000 minus 2015, stated]", -a) for lab, a in cands]
    best = (None, -2.0, None)
    for lab, a in cands:
        m = gv & np.isfinite(a)
        if m.sum() < 0.8 * gv.sum() or np.nanstd(a[m]) == 0:
            continue
        r = float(np.corrcoef(a[m], gold[m])[0, 1])
        if r > best[1]:
            best = (lab, r, a)
    lab, r, arr = best
    out.append(result("change field matches gold (Pearson)", r >= 0.95, value=round(r, 4) if lab else None,
                      threshold=0.95, note=f"{lab}" if lab else f"no 2-D field on the {gold.shape} grid ({len(cands)} candidates)"))
    gmean = float(np.nanmean(gold))
    if lab and r >= 0.95:
        m = gv & np.isfinite(arr)
        pm = float(arr[m].mean())
        ok = any(abs(pm / s - gmean) <= 0.25 * abs(gmean) for s in (1.0, 100.0))
        out.append(result("change has gold scale and sign (mean within 25 %)", ok, value=round(pm, 4),
                          threshold=f"{gmean:.4f} (or x100)"))
        nod = ~gv
        masked = float((~np.isfinite(arr[nod])).mean()) if nod.any() else 1.0
        out.append(result("nodata masked (not shown as zero change)", masked >= 0.9, value=round(masked, 3),
                          threshold=0.9, required=False))
    else:
        out.append(result("change has gold scale and sign (mean within 25 %)", False, note="no matching change field"))

    # drawn change map
    bestd = (None, -2.0, None)
    bestd_or = (None, -2.0)
    N_OR = "figure: change map drawn north-up (drawn image Spearman >= 0.85 as drawn)"
    drawn = list(drawn_arrays(ctx, artist="image"))
    mesh = {(t, m["name"][: -len("_mesh_coords")]): a for t, m, a in drawn_arrays(ctx, artist="QuadMesh")
            if m["name"].endswith("_mesh_coords")}
    for t, m, a in drawn_arrays(ctx, artist="QuadMesh"):
        if m["name"].endswith("_values") and np.asarray(a).ndim == 2:
            mc = mesh.get((t, m["name"][: -len("_values")]))
            m = dict(m)
            # pcolormesh: rows run in the direction of the y coordinates; flip when y increases
            if mc is not None and np.asarray(mc).ndim == 3:
                mc = np.asarray(mc, float)
                m["origin"] = "lower" if mc[-1, 0, 1] > mc[0, 0, 1] else "upper"
            else:
                m["origin"] = "upper"
            drawn.append((t, m, a))
    if rev_ok and REV_ACCEPT:
        drawn = drawn + [(t, dict(m, name=f"-({m['name']}) [2000 minus 2015, stated]"), -np.asarray(a, float))
                         for t, m, a in drawn]
    for tag, meta, img in drawn:
        img = np.asarray(img, float)
        if img.ndim != 2 or min(img.shape) < 50:
            continue
        if meta.get("origin") == "lower":
            img = img[::-1]
        img = img.copy()
        img[img <= -9998] = np.nan
        try:
            ag = raster_agreement(img, gold)
        except Exception:
            continue
        sp = ag.get("spearman", float("nan"))
        if sp == sp and sp > bestd_or[1]:
            bestd_or = (f"{meta['name']}@{tag}", sp)
        if sp == sp and sp > bestd[1]:
            bestd = (f"{meta['name']}@{tag}", sp, img)
        # which way up the map is drawn is presentation; the required
        # check also accepts the drawn image flipped N-S (kind="figure" row keeps "as drawn")
        try:
            agf = raster_agreement(img[::-1].copy(), gold)
        except Exception:
            continue
        spf = agf.get("spearman", float("nan"))
        if spf == spf and spf > bestd[1]:
            bestd = (f"{meta['name']}@{tag} [drawn upside down]", spf, img[::-1].copy())
    dl, dsp, dimg = bestd
    if dl is None:
        out.append(result("map shows the change (drawn image Spearman >= 0.85)", False,
                          note="UNVERIFIABLE: no 2-D image drawn in a saved figure"))
        out[-1]["unverifiable"] = True
        out.append(result("drawn change map masks nodata", False, note="no drawn change image"))
        out.append(result(N_OR, False, note="no drawn image", kind="figure"))
    else:
        out.append(result("map shows the change (drawn image Spearman >= 0.85)", dsp >= 0.85, value=round(dsp, 3),
                          threshold=0.85, note=dl))
        from checklib import resample_to
        p = resample_to(dimg, gold.shape)
        nod = ~gv
        masked = float((~np.isfinite(p[nod])).mean()) if nod.any() else 1.0
        out.append(result("drawn change map masks nodata", dsp >= 0.85 and masked >= 0.9, value=round(masked, 3),
                          threshold=">= 0.9 of nodata cells masked", note=dl))
        out.append(result(N_OR, bestd_or[1] >= 0.85, value=round(bestd_or[1], 3), threshold=0.85,
                          note=bestd_or[0] or "", kind="figure"))
    code = _strip_comments(ctx.code_text())
    div = bool(re.search(r"RdBu|RdYlBu|RdYlGn|PuOr|BrBG|PiYG|PRGn|RdGy|coolwarm|bwr|seismic|Spectral|TwoSlopeNorm|"
                         r"CenteredNorm|ListedColormap|LinearSegmentedColormap|vcenter|vlag|icefire|"
                         r"cmo\.(?:balance|curl|delta|diff|tarn)", code))
    # the text prescribes the colour scheme ("with a diverging
    # color scheme"), so it is required
    out.append(result("diverging colour scheme (code inspection)", div,
                      note="comments stripped; the text prescribes a diverging colour scheme"))

    # optional: categorical change map ordered decrease -> increase
    found = None
    for lab2, a in cands:
        m = gv & np.isfinite(a)
        u = np.unique(a[m])
        if 2 < len(u) <= 10 and m.sum() > 0.8 * gv.sum():
            x, y = a[m], gold[m]
            from scipy.stats import spearmanr
            idx = np.random.default_rng(0).choice(len(x), min(len(x), 200000), replace=False)
            rho = float(spearmanr(x[idx], y[idx]).correlation)
            if found is None or rho > found[1]:
                found = (lab2, rho)
    out.append(result("categorical change map derived (ordered classes)", bool(found and found[1] >= 0.7),
                      value=round(found[1], 3) if found else None, threshold="Spearman >= 0.7", required=False,
                      note=found[0] if found else "no classified raster/array found"))
    return out
