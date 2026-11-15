"""Checks for task 16: quadratic polynomial fit of the North America temperature time series,
by year.

The instruction asks for the regional mean over the whole grid, so the
fitted series must be a domain mean - unweighted or cos(latitude)-weighted (the reference).
Single grid points (incl. the first point, used by the benchmark's reference script) are
identified so the note can say which one was fitted, but they fail check 4. Kelvin or Celsius.

Required:
  1. polynomial_fit_pred.csv exists and is readable;
  2. one row per year 1860-2099 (240 years, a year column or date strings);
  3. some numeric column equals the least-squares QUADRATIC fit of one of those candidate
     series (RMSE <= 0.02 K after optional +273.15, widened by the rounding error of the
     output precision: values written to 0.1 K -> 0.039 K; precision coarser than 0.1 K is
     not credited). A linear (0.61) or cubic (>= 0.07) fit, cos(degrees) weights (0.12),
     the raw temperatures, or a fit over the wrong axis all fail.
  4. the fitted series is the domain-wide (area-weighted or plain) mean rather than a single
     grid point.
"""
import os
import re

import numpy as np

from checklib import result

NAME = "polynomial_fit_pred.csv"
TOL = 0.02


def _candidates(inputs_dir):
    import xarray as xr
    ds = xr.open_dataset(os.path.join(inputs_dir, "A1B_north_america.nc"))
    t = ds["air_temperature"].values.astype("float64")          # (240, 37, 49)
    lat = ds["latitude"].values
    n = t.shape[0]
    x = np.arange(n, dtype=float)
    V = np.vander(x, 3)
    P = V @ np.linalg.pinv(V)                                    # projection onto quadratics
    pts = P @ t.reshape(n, -1)                                   # fits at every grid point
    w = np.cos(np.deg2rad(lat))[None, :, None] * np.ones_like(t)
    wmean = (t * w).sum((1, 2)) / w.sum((1, 2))
    means = np.c_[P @ t.mean((1, 2)), P @ wmean]                 # unweighted, area-weighted
    return pts, means


def _decimals(v):
    """Smallest number of decimals (<= 6) to which all values are rounded."""
    for d in range(0, 7):
        if np.allclose(v, np.round(v, d), atol=1e-9, rtol=0):
            return d
    return 7


def _tol(v):
    """RMSE tolerance: 0.02 K plus the RMSE that rounding to the output precision adds
    (step / sqrt(12); 0.1 K -> 0.029). Precision coarser than 0.1 K is not credited."""
    step = 10.0 ** -max(1, _decimals(v))
    return float(np.sqrt(TOL ** 2 + step ** 2 / 12.0) * 1.1)


def _years(df):
    """Year per row. Every column that yields a year is considered and the one with the most
    distinct years wins (an iris-exported 'forecast_reference_time' column, constant 1859,
    does not hide a correct 'year' column)."""
    cands = []
    for c in df.columns:
        s = df[c]
        yrs = None
        if s.dtype.kind in "iuf" and s.notna().all() and s.between(1800, 2200).all():
            yrs = s.astype(int).to_numpy()
        elif s.dtype == object:
            yy = s.astype(str).str.extract(r"((?:18|19|20)\d\d)")[0]
            if yy.notna().all():
                yrs = yy.astype(int).to_numpy()
        if yrs is not None:
            cands.append((len(set(yrs.tolist())), str(c).lower() == "year", yrs))
    if not cands:
        return None
    return max(cands, key=lambda t: (t[0], t[1]))[2]


def checks(ctx):
    import pandas as pd
    out = []
    p = None
    for f in ctx.pred_files("*"):
        if f.lower() == NAME:
            p = ctx.pred(f)
    df = None
    if p:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            df = None
    out.append(result("polynomial_fit_pred.csv present and readable", df is not None and len(df) > 0,
                      value=os.path.basename(p) if p else None))
    if df is None or not len(df):
        out.append(result("one row per year 1860-2099", False))
        out.append(result("values are the quadratic LS fit of a legitimate series", False))
        return out
    yrs = _years(df)
    extra_note = ""
    if yrs is not None and len(yrs) > 240:
        # 1860-2099 complete plus at most 2 extrapolated years at either
        # end (e.g. 2100 from rounding up a fractional last year) -> score the 1860-2099 rows
        y = np.asarray(yrs)
        inside = (y >= 1860) & (y <= 2099)
        extra = y[~inside]
        if len(set(y[inside].tolist())) == 240 and inside.sum() == 240 and len(extra) <= 2 \
                and len(set(extra.tolist())) == len(extra) \
                and all((1858 <= e < 1860) or (2099 < e <= 2101) for e in extra.tolist()):
            df = df.loc[inside].reset_index(drop=True)
            yrs = _years(df)
            extra_note = f"extrapolated rows {sorted(int(e) for e in extra)} ignored"
    ok_years = yrs is not None and len(yrs) == 240 and yrs.min() == 1860 and yrs.max() == 2099 \
        and len(set(yrs.tolist())) == 240
    out.append(result("one row per year 1860-2099", ok_years,
                      value=None if yrs is None else [int(yrs.min()), int(yrs.max()), len(yrs)],
                      threshold=[1860, 2099, 240], note=extra_note))
    try:
        pts, means = _candidates(ctx.inputs_dir)
    except Exception as e:
        out.append(result("values are the quadratic LS fit of a legitimate series", False, note=repr(e)[:200]))
        return out
    best = (np.inf, None, None)
    tol_used = TOL
    if yrs is not None and len(yrs) == 240:
        order = np.argsort(yrs)
        for c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.isna().any() or s.between(1800, 2200).all() and s.dtype.kind in "iu":
                continue
            v = s.to_numpy(float)[order]
            tol_c = _tol(v)
            for off in (0.0, 273.15):
                vv = v + off
                rm = np.sqrt(((means - vv[:, None]) ** 2).mean(0))
                k = int(np.argmin(rm))
                if rm[k] / tol_c < best[0] / tol_used:
                    best = (float(rm[k]), c, ["domain mean", "area-weighted domain mean"][k])
                    tol_used = tol_c
                rp = np.sqrt(((pts - vv[:, None]) ** 2).mean(0))
                k = int(np.argmin(rp))
                if rp[k] / tol_c < best[0] / tol_used:
                    best = (float(rp[k]), c, f"grid point #{k} (lat idx {k // 49}, lon idx {k % 49})")
                    tol_used = tol_c
    out.append(result("values are the quadratic LS fit of a legitimate series", best[0] <= tol_used,
                      value=None if not np.isfinite(best[0]) else round(best[0], 4),
                      threshold=f"RMSE <= {tol_used:.3f} K (0.02 K + output rounding)",
                      note=f"column {best[1]!r} ~ {best[2]}" if best[1] else "no usable numeric column / years"))
    out.append(result("fit describes the domain mean (not a single grid point)",
                      best[0] <= tol_used and bool(best[2]) and "mean" in best[2], note=best[2] or ""))
    return out
