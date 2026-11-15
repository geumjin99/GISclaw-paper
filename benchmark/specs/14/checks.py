"""Task 14 - count of 5-consecutive-year spells above 280 K per grid cell, mapped.

"Number of occurrences where the temperature exceeds 280K for five consecutive years" has
several defensible counting rules; all are recomputed from the input and accepted:
  (a) overlapping 5-year windows entirely > 280 K (original Iris example, reference);
  (b) distinct runs of >= 5 consecutive years;
  (c) non-overlapping 5-year blocks within runs (sum of floor(run/5));
  (d) number of years that belong to a run of >= 5 years.
Required:
  1. the figure exists and is not blank;
  2. a captured 37 x 49 grid (any orientation: transposed / flipped) equals one of (a)-(d)
     on >= 98 % of cells. Counting single hot years, a wrong threshold (e.g. Celsius
     confusion), a wrong spell length or a mean-temperature map all fail;
  3. the figure shows such a count (the drawn grid, any flip; cells drawn as polygons -
     GeoDataFrame.plot of cell boxes - are placed by their centres).
     contour/contourf: the Z argument
     of the plotting call is identified in the code and must be a captured count grid whose
     range the captured contour levels fit. No identifiable drawn grid -> UNVERIFIABLE.
Reported, not required:
  4. the count follows the reference rule (a).
Presentation (kind="figure"; secondary analysis, never required):
  5. the map is drawn north up: imshow as displayed (origin respected), pcolormesh in index
     space by its y-axis direction, in coordinate space both vertical orientations (mesh y
     direction is not captured). Contour plots: UNVERIFIABLE (orientation set by coordinates).
"""
import os
import re

import numpy as np

from checklib import drawn_arrays, file_exists, png_not_blank, result, unverifiable

NAME = "temperature_statistic_vis.png"
SHAPE = (37, 49)


def _definitions(inputs_dir):
    import xarray as xr
    t = xr.open_dataset(os.path.join(inputs_dir, "E1_north_america.nc"))["air_temperature"].values
    hits = t > 280.0
    win = np.lib.stride_tricks.sliding_window_view(hits, 5, axis=0).all(-1).sum(0)
    n_runs = np.zeros(SHAPE, int)
    blocks = np.zeros(SHAPE, int)
    years = np.zeros(SHAPE, int)
    for i in range(SHAPE[0]):
        for j in range(SHAPE[1]):
            h = np.r_[0, hits[:, i, j].astype(int), 0]
            d = np.diff(h)
            runs = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)
            long = runs[runs >= 5]
            n_runs[i, j] = len(long)
            blocks[i, j] = int((long // 5).sum())
            years[i, j] = int(long.sum())
    return {"overlapping windows": win, "distinct runs>=5": n_runs, "non-overlapping blocks": blocks,
            "years in runs>=5": years}


def _orientations(a):
    a = np.asarray(a, float)
    if a.ndim == 1 and a.size == SHAPE[0] * SHAPE[1]:
        a = a.reshape(SHAPE)
    if a.ndim != 2:
        return []
    outs = []
    for b in (a, a.T):
        if b.shape == SHAPE:
            outs += [b, b[::-1], b[:, ::-1], b[::-1, ::-1]]
    return outs


def _best(arr, defs):
    best = (0.0, None)
    for b in _orientations(arr):
        for name, d in defs.items():
            m = np.isfinite(b)
            if m.mean() < 0.9:
                continue
            acc = float((np.round(b[m]) == d[m]).mean()) * m.mean()
            if acc > best[0]:
                best = (acc, name)
    return best


def _limits(ctx, tag, ax):
    for t, m, a in drawn_arrays(ctx, "pred", "axes_limits"):
        if t == tag and m["name"] == f"fig_ax{ax}_limits":
            return np.asarray(a, float)
    return None


def _score_grid(b, defs):
    """Best share of cells equal to an accepted rule, for an array in the file's own
    (latitude-ascending) orientation, no flips."""
    best = (0.0, None)
    b = np.asarray(b, float)
    if b.shape != SHAPE:
        return best
    m = np.isfinite(b)
    if m.mean() < 0.9:
        return best
    for name, d in defs.items():
        acc = float((np.round(b[m]) == d[m]).mean()) * m.mean()
        if acc > best[0]:
            best = (acc, name)
    return best


def _split_args(s):
    depth, cur, out = 0, "", []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            if depth == 0:
                break
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


_SKIP = {"np", "numpy", "ma", "flipud", "fliplr", "T", "values", "data", "filled", "nan_to_num", "plt", "ax",
         "ccrs", "PlateCarree", "transform", "iplt", "qplt"}


def _contour_z_names(code):
    """Identifiers used as the Z argument of contour/contourf calls (incl. da.plot.contourf)."""
    names = []
    for m in re.finditer(r"([A-Za-z_][\w\.]*?)\.plot\.contourf?\s*\(", code):
        names.append(m.group(1).split(".")[0])
    for m in re.finditer(r"(?<!plot\.)\bcontourf?\s*\(", code):
        args = [x for x in _split_args(code[m.end():]) if not re.match(r"^\w+\s*=", x)]
        if not args:
            continue
        z = args[0] if len(args) <= 2 else args[2]
        names.extend(i for i in re.findall(r"[A-Za-z_]\w*", z) if i not in _SKIP)
    return names


def _polygon_grids(ctx):
    """Grids drawn as one polygon per cell (GeoDataFrame.plot(column=...) of cell boxes, a
    PatchCollection with drawn values). The drawn values are placed on
    the 37 x 49 grid by the ranks of the cell centres (49 distinct x, 37 distinct y; cells not
    drawn count as 0), top row = largest y. Returns [(tag, meta, grid)] with artist
    'polygon_grid'."""
    out = []
    arrs = {(t, m["name"]): np.asarray(a, float) for t, m, a in drawn_arrays(ctx, "pred")
            if m.get("artist") == "PatchCollection"}
    for (t, nm), v in arrs.items():
        if not nm.endswith("_values"):
            continue
        bb = arrs.get((t, nm[:-len("_values")] + "_paths_bbox"))
        v = v.ravel()
        if bb is None or bb.ndim != 2 or len(bb) != len(v) or not 0.5 * SHAPE[0] * SHAPE[1] <= len(v) <= SHAPE[0] * SHAPE[1]:
            continue
        cx, cy = (bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2
        ux, uy = np.unique(np.round(cx, 6)), np.unique(np.round(cy, 6))
        if len(ux) != SHAPE[1] or len(uy) != SHAPE[0]:
            continue
        g = np.zeros(SHAPE)
        col = np.searchsorted(ux, np.round(cx, 6))
        row = SHAPE[0] - 1 - np.searchsorted(uy, np.round(cy, 6))
        g[row, col] = np.where(np.isfinite(v), v, 0)
        out.append((t, {"name": nm, "artist": "polygon_grid"}, g))
    return out


def _figure_check(ctx, defs, name, oname):
    """Score what the figure shows. Returns (values result, orientation result):
    - values (required, main metric): the drawn grid equals an accepted count in
      any vertical/horizontal flip (the analysis result drawn, whatever the map orientation);
    - orientation (kind="figure"): the same, in the orientation it was drawn (north up = last
      latitude row on top)."""
    north_up = {k: v[::-1] for k, v in defs.items()}   # top row first
    best, where, seen = (0.0, None), None, False
    anyb, anyw = (0.0, None), None
    for tag, meta, a in drawn_arrays(ctx, "pred") + _polygon_grids(ctx):
        art = meta.get("artist")
        a = np.asarray(a, float)
        if a.ndim != 2 or a.shape != SHAPE:
            continue
        if art == "polygon_grid":   # cells drawn as polygons, placed by their centres (top row first)
            seen = True
            b = _score_grid(a, north_up)
            if b[0] > best[0]:
                best, where = b, f"{meta['name']}@{tag} (polygon cells, placed by coordinates)"
            for c in (a, a[::-1], a[:, ::-1], a[::-1, ::-1]):
                b = _score_grid(c, defs)
                if b[0] > anyb[0]:
                    anyb, anyw = b, f"{meta['name']}@{tag} (polygon cells, orientation free)"
            continue
        if art == "image":
            cands = [a[::-1] if meta.get("origin") == "lower" else a]       # as displayed, top row first
            how = f"imshow origin={meta.get('origin')}"
        elif "QuadMesh" in str(art) and meta["name"].endswith("_values"):
            ax = int(re.match(r"fig_ax(\d+)_", meta["name"]).group(1))
            lim = _limits(ctx, tag, ax)
            if lim is not None and max(abs(lim[0]), abs(lim[1])) <= SHAPE[1] + 2 \
                    and max(abs(lim[2]), abs(lim[3])) <= SHAPE[0] + 2:
                cands = [a[::-1] if lim[2] < lim[3] else a]
                how = "pcolormesh index space"
            else:
                cands = [a[::-1], a]   # coordinate space: mesh y direction not captured
                how = "pcolormesh coordinate space (both vertical orientations)"
        else:
            continue
        seen = True
        for c in cands:
            b = _score_grid(c, north_up)
            if b[0] > best[0]:
                best, where = b, f"{meta['name']}@{tag} ({how})"
        for c in (a, a[::-1], a[:, ::-1], a[::-1, ::-1]):
            b = _score_grid(c, defs)
            if b[0] > anyb[0]:
                anyb, anyw = b, f"{meta['name']}@{tag} ({art}, orientation free)"
    if seen:
        return (result(name, anyb[0] >= 0.98, value=round(anyb[0], 4), threshold=0.98,
                       note=f"{anyw}: {anyb[1]}" if anyw else "drawn grid is not a spell count"),
                result(oname, best[0] >= 0.98, value=round(best[0], 4), threshold=0.98, kind="figure",
                       note=f"{where}: {best[1]}" if where else "drawn grid is not a spell count as oriented"))
    o_na = result(oname, False, kind="figure",
                  note="UNVERIFIABLE: no image/mesh artist; contour orientation is set by its coordinates (not captured)")
    o_na["unverifiable"] = True
    levels = [np.asarray(a, float).ravel() for t, m, a in drawn_arrays(ctx, "pred")
              if "Contour" in str(m.get("artist")) and m["name"].endswith("_values")]
    if not levels:
        return unverifiable(name, "no image/mesh/contour artist of the 37x49 grid captured in the figure"), o_na
    lv = np.concatenate(levels)
    lv = lv[np.isfinite(lv) & (np.abs(lv) < 1e100)]   # extend='min'/'max' layers are +-huge
    zn = set(_contour_z_names(ctx.code_text()))
    best, where = (0.0, None), None
    for tag, meta, load in ctx.objects("pred", "array", include_files=False):
        if meta.get("source") == "figure" or meta["name"] not in zn:
            continue
        try:
            arr = np.asarray(load(), float)
        except Exception:
            continue
        for b in _orientations(arr):   # orientation of a contour Z is set by its coordinates
            lo, hi = np.nanmin(b), np.nanmax(b)
            span = hi - lo if hi > lo else 1.0
            if len(lv) < 2 or lv.min() < lo - 0.25 * span or lv.max() > hi + 0.25 * span \
                    or (lv.max() - lv.min()) < 0.5 * span:
                continue   # contour levels do not belong to this variable
            sc = _score_grid(b, defs)
            if sc[0] > best[0]:
                best, where = sc, f"contour of {meta['name']}@{tag}"
    if where is None:
        return unverifiable(name, f"contour plot; drawn Z ({sorted(zn) or 'unparsed'}) not identified as a captured "
                                  "37x49 array consistent with the contour levels"), o_na
    return result(name, best[0] >= 0.98, value=round(best[0], 4), threshold=0.98, note=f"{where}: {best[1]}"), o_na


def checks(ctx):
    out = []
    r = file_exists(ctx, [NAME])
    out.append(r)
    if r["passed"]:
        nb, std = png_not_blank(ctx.pred(NAME))
        out.append(result("map not blank", nb, value=std, threshold=">=3 grey-level std"))
    else:
        out.append(result("map not blank", False, note="no map"))
    try:
        defs = _definitions(ctx.inputs_dir)
    except Exception as e:
        out.append(result("spell-count grid equals an accepted counting rule (>=98 % cells)", False, note=repr(e)[:200]))
        return out
    best, where = (0.0, None), None
    for tag, meta, load in ctx.objects("pred", "array"):
        sh = meta.get("shape") or []
        size = int(np.prod(sh)) if sh else 0
        if size != SHAPE[0] * SHAPE[1]:
            continue
        try:
            b = _best(load(), defs)
        except Exception:
            continue
        if b[0] > best[0]:
            best, where = b, f"{meta['name']}@{tag}"
    out.append(result("spell-count grid equals an accepted counting rule (>=98 % cells)", best[0] >= 0.98,
                      value=round(best[0], 4), threshold=0.98,
                      note=f"{where}: {best[1]}" if where else "no 37x49 array captured"))
    fv, fo = "figure shows the spell count", "figure drawn north up (presentation)"
    try:
        out.extend(_figure_check(ctx, defs, fv, fo))
    except Exception as e:
        out.append(result(fv, False, note=repr(e)[:200]))
        out.append(result(fo, False, kind="figure", note=repr(e)[:200]))
    out.append(result("reference counting rule (overlapping windows)",
                      best[0] >= 0.98 and best[1] == "overlapping windows", required=False, note=str(best[1])))
    return out
