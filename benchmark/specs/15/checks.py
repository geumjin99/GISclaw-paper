"""Checks for task 15: Total Electron Content filled-contour map with coastline background.

Masking the 192 negative cells is not required (the task text does not ask for it):
masked, unmasked or clipped TEC are all accepted provided the field is drawn at its true
geographic location. The masking check is a reported item.

Required:
  1. TEC_vis.png exists and is not blank;
  2. the figure draws TEC at its TRUE geographic location, cell by cell.
     TEC is on a rotated-pole grid (pole 45N/180E, centre 45N 0E); plotting rLon/rLat as if
     they were geographic lon/lat shifts the map 45 degrees south.
     Contour artists keep only their levels, so the drawn field is reconstructed from
     the plotting call itself: each contour/contourf/pcolormesh/pcolor call in the code is
     parsed, its X / Y / Z arguments are evaluated (no builtins) against the run's captured
     arrays (falling back to the input file's variables, e.g. ds.rLon), and its CRS is read
     from transform= (or the GeoAxes projection when transform is omitted): a RotatedPole /
     as_cartopy_crs / ob_tran CRS means X, Y are rotated coordinates, PlateCarree/Geodetic
     (or plain axes) means geographic lon/lat. The drawn points are converted to true
     lon/lat and must coincide with the TEC cells (>= 99 % of drawn points within 0.3 deg of
     their cell, values on the non-negative cells equal to TEC up to a constant unit factor,
     >= 95 % of non-negative cells drawn; negative cells may be blank, raw or clipped). This catches a RotatedPole object defined but not
     used as transform, a transposed field, and plotting Ne. Iris quickplot / iplt of the
     cube (which reads the coordinate system itself) is accepted when the cube data equal TEC
     on the non-negative cells. If no plotting call can be evaluated -> UNVERIFIABLE (not passed).
  3. (the task text asks for "a geographical and coastline
     background") a coastline / geographic background was drawn: code inspection (cartopy
     coastlines / COASTLINE / LAND / OCEAN / NaturalEarthFeature / stock_img / basemap tiles /
     Basemap drawcoastlines) or a polygon/line GeoDataFrame layer drawn in the saved map.
  3b. (a method the task text prescribes is required; the text says
     "visualize it as a filled contour map") the field of check 2 is drawn with contourf
     (matplotlib / cartopy, Iris qplt/iplt.contourf, DataArray.plot.contourf): a contourf call
     must itself pass the check-2 evaluation; pcolormesh / imshow / line contours fail.
Reported, not required:
  4. negative TEC cells masked: a captured 31 x 31 array equals TEC on >= 99 % of the valid
     cells and leaves all 192 negative cells masked/NaN (the task text does not ask for it);
  5. contour levels span ~0 .. max TEC - kind="figure" (colour-scale range vs data range).
"""
import os
import re

import numpy as np

from checklib import drawn_arrays, file_exists, png_not_blank, result, unverifiable

NAME = "TEC_vis.png"
N_FIG = "figure draws TEC at its true geographic location (plotting call evaluated)"
N_MASK = "negative TEC cells masked"
N_CF = "TEC drawn as filled contours (task: 'filled contour map')"


# ----------------------------------------------------------------------------- evaluation
class _Arr(np.ndarray):
    """Captured array that tolerates .values / .data / .filled() as used on xarray/masked."""

    @property
    def values(self):
        return self

    @property
    def data(self):
        return self

    def filled(self, *a, **k):
        return self

    def to_numpy(self):
        return self


def _wrap(a):
    return np.asarray(a, float).view(_Arr)


class _NS:
    """Dotted captured names (ds.TEC) with fallback to the input dataset (ds.rLon)."""

    def __init__(self, items, fallback):
        self._items, self._fb = items, fallback

    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        if k in self._items:
            return self._items[k]
        return getattr(self._fb, k)

    def __getitem__(self, k):
        if k in self._items:
            return self._items[k]
        return self._fb[k]


def _namespace(ctx, tag, inputs_ds):
    import numpy.ma as ma
    try:
        import cartopy.crs as ccrs
    except Exception:
        ccrs = None
    ns = {"np": np, "numpy": np, "ma": ma, "ccrs": ccrs}
    dotted = {}
    for t, meta, load in ctx.objects("pred", "array", include_files=False):
        if t != tag or meta.get("source") == "figure":
            continue
        try:
            a = _wrap(load())
        except Exception:
            continue
        nm = meta["name"]
        if "." in nm:
            base, attr = nm.split(".", 1)
            dotted.setdefault(base, {})[attr] = a
        else:
            ns[nm] = a
    for base, items in dotted.items():
        if base not in ns:
            ns[base] = _NS(items, inputs_ds)
    return ns


class _Fallback(dict):
    """Unknown names resolve to the input dataset (solutions name it ds, data, nc, ...)."""

    def __init__(self, ns, inputs_ds):
        super().__init__(ns)
        self._ds = inputs_ds

    def __missing__(self, k):
        return self._ds


def _safe_eval(expr, ns, inputs_ds):
    if "__" in expr or re.search(r"\b(import|open|exec|eval|compile|globals|locals|getattr|setattr|lambda)\b", expr):
        raise ValueError("unsafe expression")
    return eval(expr, {"__builtins__": {}}, _Fallback(ns, inputs_ds))


def _split_args(s):
    depth, cur, out, quote = 0, "", [], None
    for ch in s:
        if quote:
            cur += ch
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
        elif ch in "([{":
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


def _crs_kind(expr, code):
    """'rotated', 'geo' or None, from a CRS expression (resolving one level of assignment)."""
    e = expr or ""
    m = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*", e)
    if m:
        a = re.findall(r"^\s*" + re.escape(m.group(1)) + r"\s*=\s*(.+)$", code, re.M)
        if a:
            e = a[-1]
    if re.search(r"RotatedPole|as_cartopy_crs|RotatedGeogCS|ob_tran", e):
        return "rotated"
    if re.search(r"PlateCarree|Geodetic|4326", e):
        return "geo"
    return None


def _plot_calls(code):
    """(func, positional args, kwargs) of contour/contourf/pcolormesh/pcolor calls."""
    calls = []
    for m in re.finditer(r"(?:([A-Za-z_][\w\.\[\]]*)\.)?\b(contourf|contour|pcolormesh|pcolor)\s*\(", code):
        args = _split_args(code[m.end():])
        pos = [a for a in args if not re.match(r"^\w+\s*=[^=]", a)]
        kw = dict(re.match(r"^(\w+)\s*=\s*(.*)$", a, re.S).groups() for a in args if re.match(r"^\w+\s*=[^=]", a))
        calls.append((m.group(1) or "", m.group(2), pos, kw))
    return calls


def _to_geo(x, y, kind, rp):
    if kind == "geo":
        return x, y
    import cartopy.crs as ccrs
    p = ccrs.PlateCarree().transform_points(rp, x, y)
    return p[:, 0], p[:, 1]


def _unit(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def _judge_drawn(X, Y, Z, kind, truth):
    """Compare a drawn (X, Y, Z) in CRS `kind` with the true geographic TEC cells."""
    from scipy.spatial import cKDTree
    geo_lon, geo_lat, tec, rp = truth
    Z = np.asarray(Z, float)
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    if Z.ndim != 2:
        return None, "Z is not 2-D"
    if X.ndim == 1 and Y.ndim == 1:
        if (len(Y), len(X)) != Z.shape:
            return None, "X/Y lengths do not match Z"
        X, Y = np.meshgrid(X, Y)
    if X.shape != Z.shape or Y.shape != Z.shape:
        return None, "X/Y/Z shapes differ"
    lon, lat = _to_geo(X.ravel(), Y.ravel(), kind, rp)
    z = Z.ravel()
    ok = np.isfinite(lon) & np.isfinite(lat)
    tree = cKDTree(_unit(geo_lon.ravel(), geo_lat.ravel()))
    d, idx = tree.query(_unit(lon[ok], lat[ok]))
    near = np.degrees(d) <= 0.3
    pos_share = float(near.mean()) if len(near) else 0.0
    t = tec.ravel()[idx]
    zz = z[ok]
    neg = t < 0
    drawn_neg = bool(np.isfinite(zz[near & neg]).any())
    use = near & ~neg & np.isfinite(zz)
    coverage = use.sum() / max(1, (tec >= 0).sum())
    if use.sum() < 10:
        # a field drawn nowhere near the TEC cells (e.g. on the fill-valued 2-D lat/lon) fails
        return False, {"position_match": round(pos_share, 3), "coverage": round(float(coverage), 3),
                       "note": "drawn points are not at the TEC cells' geographic location"}
    ratio = np.median(zz[use] / np.where(t[use] == 0, np.nan, t[use]))
    if not np.isfinite(ratio) or ratio <= 0:
        ratio = 1.0
    val_share = float((np.abs(zz[use] / ratio - t[use]) <= 1e-3 + 1e-4 * np.abs(t[use])).mean())
    info = {"position_match": round(pos_share, 3), "value_match": round(val_share, 4),
            "coverage": round(float(coverage), 3), "negatives_drawn": drawn_neg, "crs": kind}
    good = pos_share >= 0.99 and val_share >= 0.99 and coverage >= 0.95   # negatives: any treatment
    return good, info


def _figure_check(ctx, truth, inputs_ds, only=None):
    code = ctx.code_text()
    calls = _plot_calls(code)
    if only is not None:      # restrict to the prescribed plotting function(s)
        calls = [c for c in calls if c[1] in only]
    if not calls:
        return unverifiable(N_FIG, "no contour/contourf/pcolormesh call found in the code")
    tags = sorted({t for t, m, a in drawn_arrays(ctx, "pred")}) + ["final"]
    proj = re.findall(r"projection\s*=\s*([^,\)\n]+(?:\([^\)]*\))?)", code)
    best_info, notes = None, []
    for tag in tags:
        ns = _namespace(ctx, tag, inputs_ds)
        for owner, fn, pos, kw in calls:
            try:
                if re.search(r"\b(qplt|iplt|quickplot)\b", owner) or owner.endswith("iris.plot"):
                    Z = np.asarray(_safe_eval(pos[0], ns, inputs_ds), float)
                    tec = truth[2]
                    if Z.shape != tec.shape:
                        continue
                    valid = tec >= 0
                    agree = float((np.abs(Z - tec) <= 1e-3)[valid].mean())
                    masked = not np.isfinite(Z[~valid]).any()
                    info = {"iris_plot": True, "value_match": round(agree, 4), "negatives_drawn": not masked}
                    if agree >= 0.99:
                        return result(N_FIG, True, value=info, note=f"iris {fn} of the TEC cube @{tag}")
                    best_info = best_info or info
                    continue
                if owner.endswith(".plot"):          # xarray DataArray.plot.contourf(...)
                    obj = _safe_eval(owner[:-5], ns, inputs_ds)
                    if hasattr(obj, "dims") and len(obj.dims) == 2:
                        Y = np.asarray(obj[obj.dims[0]].values, float)
                        X = np.asarray(obj[obj.dims[1]].values, float)
                        Z = np.asarray(obj.values, float)
                    else:   # captured as plain array: dims as stored in the file (rLat, rLon)
                        Z = np.asarray(obj, float)
                        X, Y = np.asarray(inputs_ds["rLon"].values), np.asarray(inputs_ds["rLat"].values)
                else:
                    if len(pos) < 3:
                        notes.append(f"{fn}: no X/Y coordinates (index space, not georeferenced)")
                        continue
                    X, Y, Z = (_safe_eval(p, ns, inputs_ds) for p in pos[:3])
                    if hasattr(Z, "filled") and not isinstance(Z, _Arr):
                        Z = np.ma.filled(np.ma.asarray(Z, float), np.nan)
                kind = _crs_kind(kw.get("transform"), code) if "transform" in kw else \
                    (_crs_kind(proj[-1], code) if proj else "geo")
                if kind is None:
                    notes.append(f"{fn}: CRS of transform={kw.get('transform')} not recognised")
                    continue
                good, info = _judge_drawn(np.asarray(X, float), np.asarray(Y, float),
                                          np.ma.filled(np.ma.asarray(Z, float), np.nan), kind, truth)
                if good is None:
                    notes.append(f"{fn}: {info}")
                    continue
                if good:
                    return result(N_FIG, True, value=info, note=f"{fn} @{tag}")
                best_info = best_info or info
            except Exception as e:
                notes.append(f"{fn}: {type(e).__name__}: {str(e)[:60]}")
                continue
    if best_info is not None:
        return result(N_FIG, False, value=best_info, note="drawn field is misplaced, transposed or not TEC")
    return unverifiable(N_FIG, "; ".join(dict.fromkeys(notes))[:300] or "plotting call could not be evaluated")


def checks(ctx):
    import xarray as xr
    out = []
    r = file_exists(ctx, [NAME, "tec_vis.png"])
    out.append(r)
    if r["passed"]:
        nb, std = png_not_blank(ctx.pred(r["value"][0]))
        out.append(result("map not blank", nb, value=std, threshold=">=3 grey-level std"))
    else:
        out.append(result("map not blank", False, note="no map"))
    try:
        ds = xr.open_dataset(os.path.join(ctx.inputs_dir, "space_weather.nc"))
        tec = ds["TEC"].values.astype(float)
        import cartopy.crs as ccrs
        gm = ds["rotated_pole"].attrs
        rp = ccrs.RotatedPole(pole_longitude=float(gm.get("grid_north_pole_longitude", 180.0)),
                              pole_latitude=float(gm.get("grid_north_pole_latitude", 45.0)))
        RL, RT = np.meshgrid(ds["rLon"].values, ds["rLat"].values)
        geo = ccrs.PlateCarree().transform_points(rp, RL, RT)
        truth = (geo[..., 0], geo[..., 1], tec, rp)
    except Exception as e:
        out.append(result(N_MASK, False, note=repr(e)[:200], required=False))
        out.append(result(N_FIG, False, note=repr(e)[:200]))
        return out
    neg = tec < 0
    best, where = (0.0, False), None
    lv_note, lv_ok = "", False
    vmax = float(np.nanmax(tec))
    for tag, meta, load in ctx.objects("pred", "array"):
        sh = meta.get("shape") or []
        if meta.get("source") == "figure" and "Contour" in str(meta.get("artist", "")) \
                and meta["name"].endswith("_values"):
            try:
                lv = np.asarray(load(), float).ravel()
                lv = lv[np.isfinite(lv) & (np.abs(lv) < 1e100)]
                lv_ok |= bool(-1.0 <= np.nanmin(lv) <= 3.0 and 0.75 * vmax <= np.nanmax(lv) <= 1.4 * vmax)
                lv_note = lv_note or f"{meta['name']}: levels {np.nanmin(lv):.2f}..{np.nanmax(lv):.2f}"
            except Exception:
                pass
            continue
        if not sh or int(np.prod(sh)) != tec.size:
            continue
        try:
            a = np.asarray(load(), float).reshape(tec.shape)
        except Exception:
            continue
        for b in (a, a.T, a[::-1], a[::-1].T):
            valid = ~neg
            fin = np.isfinite(b)
            agree = float((fin[valid] & (np.abs(b - tec) <= 1e-3)[valid]).mean())
            masked_ok = not fin[neg].any()
            if (agree, masked_ok) > best:
                best, where = (agree, masked_ok), f"{meta['name']}@{tag}"
    mask_res = result(N_MASK, best[0] >= 0.99 and best[1], required=False,
                      value={"agreement_valid_cells": round(best[0], 4), "negatives_masked": best[1]},
                      threshold=">=0.99 and negatives masked", note=where or "no 31x31 array captured")
    try:
        fig_res = _figure_check(ctx, truth, ds)
        out.append(fig_res)
    except Exception as e:
        fig_res = result(N_FIG, False, note=repr(e)[:200])
        out.append(fig_res)
    # the text prescribes "a filled contour map" -> the TEC field placed correctly
    # (check 2) must be drawn by a contourf call (matplotlib/cartopy, Iris qplt/iplt.contourf or
    # DataArray.plot.contourf); pcolormesh / imshow / contour lines fail
    try:
        if fig_res.get("passed") and str(fig_res.get("note", "")).split(" @")[0].split()[-1] == "contourf":
            cf_ok, cf_note = True, fig_res.get("note", "")
        else:
            r_cf = _figure_check(ctx, truth, ds, only={"contourf"})
            cf_ok, cf_note = bool(r_cf.get("passed")), r_cf.get("note", "")
    except Exception as e:
        cf_ok, cf_note = False, repr(e)[:200]
    out.append(result(N_CF, cf_ok, note=cf_note or "no contourf call drawing TEC at its true location"))
    out.append(mask_res)
    out.append(result("contour levels span ~0 .. max TEC", lv_ok, value=lv_note or None, kind="figure"))
    code = ctx.code_text()
    code_nc = "\n".join(ln for ln in code.splitlines() if not ln.lstrip().startswith("#"))
    m = re.search(r"coastlines\s*\(|COASTLINE|feature\.(?:LAND|OCEAN)\b|NaturalEarthFeature|[Nn]atural_?[Ee]arth|"
                  r"stock_img\s*\(|add_basemap|contextily|GSHHS|shadedrelief\s*\(|bluemarble\s*\(", code_nc)
    coast_note = f"code: {m.group(0)}" if m else ""
    if not m:
        try:   # a land / coastline / world layer drawn with GeoDataFrame.plot in the saved map
            from checklib import drawn_geo_layers
            for tag, rec, b in drawn_geo_layers(ctx, "pred", file=r["value"][0] if r["passed"] else NAME):
                gt = " ".join(rec.get("geom_types") or [])
                if re.search(r"Polygon|LineString", gt):
                    coast_note = f"drawn layer {rec.get('name', '')} ({gt}) @{tag}"
                    break
        except Exception:
            pass
    out.append(result("coastline / geographic background drawn", bool(coast_note), note=coast_note or
                      "no coastline / land / geographic background found in code or drawn layers"))
    return out
