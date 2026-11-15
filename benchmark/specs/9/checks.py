"""Checks for task 9: deforestation rate within 5.5 km of roads, Rondonia.

Gold: 100 * area(deforested within the dissolved 5.5 km buffer of all roads) / area(buffer)
= 47.55 % (equal-area CRS; UTM 20S gives 47.51 %, UTM 23S 47.95 %).

Correctness (required):
  1. deforestation_rate.csv exists;
  2. it has a column named percentage_deforestation (case-insensitive) with one finite value;
     (lenient parse: a trailing "%" or decimal comma is accepted; a value written with "%"
     is never re-read as a fraction);
  3. the value equals the gold within 1.5 percentage points (a fraction <= 1 is read as x100).
     A "difference" overlay (non-deforested share, 52 %), buffers in degrees,
     only official roads, or the share of all deforestation lying near roads (99.5 %) fail.
"""
import os

import numpy as np

from checklib import result, parse_number

TOL = 1.5


def _gold(ctx):
    import pandas as pd
    v = float(pd.read_csv(ctx.gold("deforestation_rate.csv"))["percentage_deforestation"].iloc[0])
    return v * 100 if v <= 1 else v


def checks(ctx):
    import pandas as pd
    out = []
    files = [f for f in ctx.pred_files("*.csv") if f.lower() == "deforestation_rate.csv"]
    out.append(result("required output present", bool(files), value=files, threshold=["deforestation_rate.csv"]))
    if not files:
        out.append(result("column percentage_deforestation with one value", False))
        out.append(result("percentage within 1.5 pp of gold", False))
        return out
    try:
        df = pd.read_csv(ctx.pred(files[0]))
        cols = [c for c in df.columns if str(c).strip().lower() == "percentage_deforestation"]
        raw = df[cols[0]].dropna() if cols else pd.Series(dtype=object)
        # lenient parse: "47.53%", "47,53", " 47.53 " are legitimate ways to write a percentage
        vals = raw.map(parse_number)
        is_pct = raw.astype(str).str.contains("%")
        keep = vals.map(np.isfinite)
        vals, is_pct = vals[keep], is_pct[keep]
        ok = bool(cols) and len(vals) >= 1
        out.append(result("column percentage_deforestation with one value", ok,
                          value=list(map(str, df.columns))[:10], note=f"{len(vals)} numeric value(s)"))
        if not ok:
            out.append(result("percentage within 1.5 pp of gold", False))
            return out
        v = float(vals.iloc[0])
        frac = abs(v) <= 1 and not bool(is_pct.iloc[0])
        pct = v * 100 if frac else v
        g = _gold(ctx)
        out.append(result("percentage within 1.5 pp of gold", abs(pct - g) <= TOL, value=round(pct, 3),
                          threshold=f"{g:.2f} +/- {TOL}", note="fraction read as x100" if frac else ("percent string" if bool(is_pct.iloc[0]) else "")))
    except Exception as e:
        out.append(result("column percentage_deforestation with one value", False, note=repr(e)[:200]))
        out.append(result("percentage within 1.5 pp of gold", False))
    return out
