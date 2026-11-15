"""Checks for task 13: vertical temperature-salinity profile (South Atlantic), ocean_profiles_vis.png.

The instruction names no lat/lon/depth numbers (the original solution uses 330.5E,
-10<lat<-9, depth<=1000 m), so any legitimate vertical profile from this file is accepted:
a single water column or the mean over any rectangular lat/lon box, over any contiguous
depth range of >= 5 levels. Temperature may be in K or degC.

Required:
  1. the figure exists and is not blank;
  2. a potential-temperature PROFILE was extracted: a captured 1-D array (or a drawn line)
     equals theta over >= 5 consecutive depth levels of some column/box (|diff| <= 0.02);
  3. a salinity profile was extracted from the SAME column/box (pairing check);
  4. the figure shows both against depth, from the same column/box: among the drawn lines
     and scatter markers (offsets), one is the temperature and one the salinity profile, each
     with depth as the other coordinate, and they share a column/box (a salinity line from
     another column fails). No line/scatter artist captured -> UNVERIFIABLE (not passed).
Reported, not required:
  5. the canonical selection (330.5E, lat -9.83, depth <= 1000 m) was used.
  6. (kind="figure") each drawn
     profile is legible: its value range spans >= 3 % of its axis (salinity drawn on the
     temperature-in-K axis is a flat line). Presentation, not the analysis result.
"""
import os
import re

import numpy as np

from checklib import drawn_arrays, file_exists, png_not_blank, result, unverifiable

NAME = "ocean_profiles_vis.png"
TOL = 0.02
LEGIBLE = 0.03   # a drawn profile must span >= 3 % of its value axis


def _axis_limits(ctx):
    """{(tag, axis index): [x0, x1, y0, y1]} of the captured figure axes."""
    out = {}
    for tag, meta, a in drawn_arrays(ctx, "pred", "axes_limits"):
        m = re.match(r"fig_ax(\d+)_limits", meta["name"])
        if m:
            out[(tag, int(m.group(1)))] = np.asarray(a, float).ravel()
    return out


def _legible(v, which, meta, tag, limits):
    """Does the drawn value vector occupy a readable share of its axis? (None = unknown)."""
    m = re.match(r"fig_ax(\d+)_", meta["name"])
    lim = limits.get((tag, int(m.group(1)))) if m else None
    if lim is None or len(lim) != 4:
        return None
    lo, hi = (lim[0], lim[1]) if which == 0 else (lim[2], lim[3])
    span = abs(hi - lo)
    if not np.isfinite(span) or span == 0:
        return None
    v = np.asarray(v, float)
    return bool(float(np.nanmax(v) - np.nanmin(v)) / float(span) >= LEGIBLE)


def _boxes(inputs_dir):
    import xarray as xr
    ds = xr.open_dataset(os.path.join(inputs_dir, "atlantic_profiles.nc"))
    th, sa = ds["theta"].values.astype(float), ds["salinity"].values.astype(float)
    depth = ds["depth"].values.astype(float)
    ny, nx = th.shape[1:]
    boxes = []
    for i0 in range(ny):
        for i1 in range(i0 + 1, ny + 1):
            for j0 in range(nx):
                for j1 in range(j0 + 1, nx + 1):
                    with np.errstate(all="ignore"):
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            boxes.append(((i0, i1, j0, j1), np.nanmean(th[:, i0:i1, j0:j1], (1, 2)),
                                          np.nanmean(sa[:, i0:i1, j0:j1], (1, 2))))
    return boxes, depth, ds["lat"].values, ds["lon"].values


def _match(v, profiles, depth):
    """Return list of (box, start, k) where v equals profile[start:start+k]."""
    v = np.asarray(v, float).ravel()
    k = len(v)
    hits = []
    if k < 5 or k > len(depth) or np.isfinite(v).sum() < 5:
        return hits
    fv = np.isfinite(v)
    for box, prof in profiles:
        for s in range(0, len(depth) - k + 1):
            p = prof[s:s + k]
            fp = np.isfinite(p)
            if (fv & ~fp).any() or (fv & fp).sum() < 5:
                continue
            for off in (0.0, -273.15):
                if np.nanmax(np.abs(v[fv] - (p[fv] + off))) <= TOL:
                    hits.append((box, s, k))
                    break
    return hits


def _vectors(meta, a):
    """Candidate 1-D vectors from a captured array, plus (vector, partner, value-axis index)
    for N x 2 lines (0 = the vector is drawn along x, 1 = along y)."""
    a = np.asarray(a, float)
    if a.ndim == 1:
        return [(a, None, None)]
    if a.ndim == 2 and a.shape[1] == 2:
        return [(a[:, 0], a[:, 1], 0), (a[:, 1], a[:, 0], 1)]
    if a.ndim == 2 and min(a.shape) == 1:
        return [(a.ravel(), None, None)]
    return []


def checks(ctx):
    out = []
    r = file_exists(ctx, [NAME])
    out.append(r)
    if r["passed"]:
        nb, std = png_not_blank(ctx.pred(NAME))
        out.append(result("figure not blank", nb, value=std, threshold=">=3 grey-level std"))
    else:
        out.append(result("figure not blank", False, note="no figure"))
    try:
        boxes, depth, lat, lon = _boxes(ctx.inputs_dir)
    except Exception as e:
        for n in ("temperature profile extracted", "salinity profile from the same column/box",
                  "figure shows both profiles against depth, from the same column/box"):
            out.append(result(n, False, note=repr(e)[:200]))
        return out
    tprof = [(b, t) for b, t, s in boxes]
    sprof = [(b, s) for b, t, s in boxes]
    t_hits, s_hits = set(), set()
    fig_artists, fig_t, fig_s = 0, set(), set()   # boxes of profiles drawn against depth
    leg_t, leg_s = set(), set()                    # ... and drawn legibly (figure check)
    illegible = []
    try:
        limits = _axis_limits(ctx)
    except Exception:
        limits = {}
    for tag, meta, load in ctx.objects("pred", "array"):
        sh = meta.get("shape") or []
        if not sh or int(np.prod(sh)) > 200:
            continue
        try:
            arr = load()
        except Exception:
            continue
        # drawn data: line x/y and scatter offsets (marker profiles are as valid as lines)
        is_drawn = meta.get("source") == "figure" and (
            meta.get("artist") == "line" or (meta["name"].endswith("_offsets") and meta.get("artist") == "PathCollection"))
        fig_artists += int(is_drawn)
        for v, partner, which in _vectors(meta, arr):
            th = _match(v, tprof, depth)
            sh_ = _match(v, sprof, depth)
            t_hits.update((b, s, k) for b, s, k in th)
            s_hits.update((b, s, k) for b, s, k in sh_)
            if is_drawn and partner is not None and len(partner) == len(v):
                def _vs_depth(s, k):
                    return np.allclose(partner, depth[s:s + k], atol=0.5) or np.allclose(partner, -depth[s:s + k], atol=0.5)
                th_d = [b for b, s, k in th if _vs_depth(s, k)]
                sh_d = [b for b, s, k in sh_ if _vs_depth(s, k)]
                # A profile drawn on an axis scaled for the other variable (salinity
                # 34-36 on the same axis as temperature 275-300 K) is a flat, unreadable line.
                # Legibility is presentation -> separate kind="figure" check.
                fig_t.update(th_d)
                fig_s.update(sh_d)
                if (th_d or sh_d) and _legible(v, which, meta, tag, limits) is False:
                    illegible.append(meta["name"])
                    continue
                leg_t.update(th_d)
                leg_s.update(sh_d)
    out.append(result("temperature profile extracted", bool(t_hits), value=len(t_hits),
                      note="theta over >=5 consecutive levels of a column/box" if t_hits else "no captured theta profile"))
    common = {b for b, s, k in t_hits} & {b for b, s, k in s_hits}
    out.append(result("salinity profile from the same column/box", bool(common), value=len(common),
                      note="" if common else ("no salinity profile" if not s_hits else "temperature and salinity from different locations")))
    N_FIG = "figure shows both profiles against depth, from the same column/box"
    if fig_artists:
        drawn_common = fig_t & fig_s
        out.append(result(N_FIG, bool(drawn_common),
                          value={"temperature_drawn": bool(fig_t), "salinity_drawn": bool(fig_s),
                                 "same_location": bool(drawn_common)},
                          note="" if drawn_common else ("drawn T and S profiles come from different locations"
                                                        if fig_t and fig_s else "a profile is not drawn against depth")))
        leg_common = leg_t & leg_s
        out.append(result("drawn profiles legible (each spans >= 3 % of its axis) (figure)",
                          bool(leg_common) or not drawn_common,
                          value={"illegible": sorted(set(illegible))},
                          note="" if leg_common else ("not assessed (profiles not drawn)" if not drawn_common else
                                                      f"illegible (< {LEGIBLE:.0%} of its axis): {sorted(set(illegible))}"),
                          kind="figure"))
    else:
        out.append(unverifiable(N_FIG, "no line or scatter artists captured in the figure"))
    i = int(np.argmin(np.abs(lat - (-9.834))))
    j = int(np.argmin(np.abs(lon - 330.5)))
    k1000 = int((depth <= 1000).sum())
    canon = any(b == (i, i + 1, j, j + 1) and s == 0 and k == k1000 for b, s, k in t_hits)
    out.append(result("canonical selection (330.5E, 9.83S, <=1000 m)", canon, required=False))
    return out
