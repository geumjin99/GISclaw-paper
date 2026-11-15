"""Task 47 - impervious surface area per parcel (Tabulate Area, Add Join, Copy Features).

Task core: tabulate, for each of the 121 parcels, the area of each class of impervious.tif,
join the table to the parcels and save the joined layer. The raster's attribute table
(impervious.tif.vat.dbf) defines Value 0 = Impervious and Value 1 = Pervious - the natural
assumption "1 = impervious" is wrong (it sums the pervious cells).

Correctness (required):
  1. a joined parcel layer is saved: a vector file with one feature per parcel (121 +/- 2);
  2. it carries the impervious area: a numeric column proportional to the gold Value-0 area in
     a recognised unit (ft2, m2, acres, cell counts) - ratio within 5 % of the unit factor for
     >= 90 % of parcels (aligned on Parcel_ID or geometry). Pervious area, impervious share in
     %, or parcel area fail. A column whose name declares a unit
     (ft2/sqft, m2, ha, acres, km2) counts only in that unit, and a layer is rejected when a
     column named as impervious area (contains "imp") disagrees with gold in its declared unit;
     a parcel also matches when |v - gold|
     <= parcel perimeter x half the cell size (0.25 ft), in the matched unit - all_touched masks
     add a boundary ring of cells (on average half a cell outside the parcel), which on parcels
     with little impervious area exceeds 5 %; a full-cell ring would also admit a parcel layer
     buffered by 1 ft;
Reported, not required:
  3. pervious area (Value 1) also tabulated (Tabulate Area gives every class);
  4. the class meaning was read from the attribute table (code mentions vat / Impervious class 0).
"""
import glob
import os

import numpy as np

from checklib import result, key_values

N = 121
UNITS = {"ft2": 1.0, "m2": 0.09290341, "acres": 1 / 43560.0, "cells(0.25 ft2)": 4.0, "km2": 0.09290341e-6,
         "ha": 0.09290341e-4}


def _vectors(ctx):
    import geopandas as gpd
    out = []
    for pat in ("*.shp", "*.geojson", "*.gpkg", "*.json"):
        for p in sorted(glob.glob(os.path.join(ctx.pred_dir, "**", pat), recursive=True)):
            try:
                out.append((os.path.basename(p), gpd.read_file(p)))
            except Exception:
                continue
    return out


def _gold(ctx):
    import geopandas as gpd
    g = gpd.read_file(ctx.gold("parcels_joint.shp"))
    return g.set_index("Parcel_ID")


_UNIT_TOKENS = {"ft2": "ft2", "sqft": "ft2", "sft": "ft2", "ftsq": "ft2", "m2": "m2", "sqm": "m2", "sqmeters": "m2",
                "ha": "ha", "hectare": "ha", "hectares": "ha", "acre": "acres", "acres": "acres", "ac": "acres",
                "km2": "km2", "sqkm": "km2"}


def _declared_unit(col):
    """Unit declared by a column name (Imperv_ft2, imp_area_sqft, ImpHa, ...) or None."""
    import re
    t = re.sub(r"([a-z])([A-Z])", r"\1_\2", str(col)).lower().replace("²", "2")
    toks = [x for x in re.split(r"[^a-z0-9]+", t) if x]
    toks += ["".join(toks[i:i + 2]) for i in range(len(toks) - 1)]  # sq_ft -> sqft
    for x in toks:
        if x in _UNIT_TOKENS:
            return _UNIT_TOKENS[x]
    return None


CELL_FT = 0.5   # impervious.tif cell size (US ft)


def _match(v, g, units=None, perim=None):
    """best unit and share of parcels whose v/g is within 5 % of that unit factor - or, with
    `perim` (gold parcel perimeters in ft), within half a boundary
    ring of cells (perimeter x cell size / 2, ft2) of gold."""
    m = np.isfinite(v) & np.isfinite(g) & (g > 0)
    if m.sum() < 0.8 * N:
        return None, 0.0
    best = (None, 0.0)
    for u, f in UNITS.items():
        if units is not None and u not in units:
            continue
        ok = np.abs(v[m] / g[m] / f - 1) <= 0.05
        zero_ok = (g <= 0) & np.isfinite(v) & (np.abs(v) <= 1e-9 + 0.05 * f)
        if perim is not None:
            ring = perim * CELL_FT / 2
            ok |= np.abs(v[m] / f - g[m]) <= ring[m] + 1e-9
            zero_ok |= (g <= 0) & np.isfinite(v) & (np.abs(v / f) <= ring + 1e-9)
        share = (ok.sum() + zero_ok.sum()) / N
        if share > best[1]:
            best = (u, float(share))
    return best


def checks(ctx):
    import geopandas as gpd
    out = []
    try:
        gold = _gold(ctx)
    except Exception as e:
        return [result("gold readable", False, note=repr(e)[:200])]
    parcels = gpd.read_file(f"{ctx.inputs_dir}/parcels.shp")
    gimp = gold["VALUE_0"].astype(float)
    gper = gold["VALUE_1"].astype(float)
    perim = gold.geometry.length.reindex(gimp.index).to_numpy(float)   # US ft

    files = [(n, g) for n, g in _vectors(ctx) if abs(len(g) - N) <= 2]
    out.append(result("joined parcel layer saved (121 features)", bool(files), value=[n for n, _ in files]))

    best = (None, 0.0, None)
    per_best = 0.0
    mislabelled = []
    for n, g in files:
        keys = key_values(g, parcels, "Parcel_ID")
        if keys is None:
            continue
        try:
            keys = keys.astype(parcels["Parcel_ID"].dtype)
        except Exception:
            pass
        layer_best = (None, 0.0, None)
        layer_bad = []
        for c in g.columns:
            if c in ("geometry", "Parcel_ID") or str(c).lower() == "parcel_id" or g[c].dtype.kind not in "fiu":
                continue
            s = g[c].astype(float).groupby(keys.values).mean()
            v = s.reindex(gimp.index).to_numpy(float)
            du = _declared_unit(c)
            # a column whose name declares a unit is read in that unit only (otherwise
            # a double unit conversion is rescued by an intermediate column mislabelled m2)
            u, share = _match(v, gimp.to_numpy(float), units={du} if du else None, perim=perim)
            if du and "imp" in str(c).lower() and share < 0.9:
                layer_bad.append(f"{c} (declared {du}, {share:.2f} match)")
            if share > layer_best[1]:
                layer_best = (f"{c} in {n}", share, u)
            _, sp = _match(v, gper.to_numpy(float), units={du} if du else None)
            per_best = max(per_best, sp)
        if layer_bad:
            mislabelled.append(f"{n}: " + ", ".join(layer_bad))
            continue  # the layer presents a wrong impervious area in its declared unit
        if layer_best[1] > best[1]:
            best = layer_best
    out.append(result("impervious (Value 0) area per parcel matches gold (>= 90 % within 5 %)", best[1] >= 0.9,
                      value=round(best[1], 3), threshold="0.9 (per parcel: within 5 % or one boundary cell ring)",
                      note=(f"{best[0]} [unit {best[2]}]" if best[0] else "no matching numeric column in a saved parcel layer")
                      + ("; rejected (labelled impervious column wrong in its unit): " + "; ".join(mislabelled)
                         if mislabelled else "")))
    out.append(result("pervious (Value 1) area also tabulated", per_best >= 0.9, value=round(per_best, 3),
                      required=False))
    code = ctx.code_text()
    out.append(result("class meaning read from attribute table (vat.dbf / Class_name)",
                      any(k in code for k in ("vat.dbf", "Class_name", "Classvalue")), required=False))
    return out
