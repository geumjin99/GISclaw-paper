"""T24 - De-identification by rounding coordinates to two decimals (WGS 84).

373 child blood-lead points (EPSG:3857 in the file). Correct: project to WGS 84, round lon/lat
to 0.01 degree, rebuild points at the rounded coordinates, save, and document the reduction in
unique locations (372 -> 168).

Correctness (required):
  1. a point vector file (shp/gpkg/geojson) was written to pred_results;
  2. it holds the 373 records (or the 168 de-duplicated locations);
  3. its geometries, in WGS 84, lie on the 0.01-degree grid (>= 99 % within 1e-6 deg) - this
     fails when rounding was done in Web Mercator metres or the original geometry was kept;
  4. the set of unique rounded locations equals the gold set (>= 98 % overlap both ways).
  5. the reduction is documented (the instruction asks for it): a captured scalar,
     a summary file or a small table containing 168 (unique after) / 204 or 205 (reduction) /
     ~54.8-55.0 % (or 45.2 % retained);
  6. the published layer does not keep the exact coordinates as attribute fields.
Reported, not required:
  7. rounded coordinate fields added to the attribute table.
"""
import glob
import os

import numpy as np

from checklib import result

GOLD = "High_Blood_Level_Results_Rounded.shp"
N, N_UNIQUE = 373, 168


def _keyset(g):
    return set(zip(np.round(g.geometry.x.values, 4), np.round(g.geometry.y.values, 4)))


def _pred_layers(ctx):
    import geopandas as gpd
    files = []
    for ext in ("shp", "gpkg", "geojson", "json"):
        files += glob.glob(os.path.join(ctx.pred_dir, "*." + ext))
    out = []
    for f in sorted(files):
        try:
            g = gpd.read_file(f)
        except Exception:
            continue
        if len(g) and (g.geom_type == "Point").mean() > 0.95:
            if g.crs is None:
                # no CRS guessing for deliverable files: deliverable files must carry a CRS,
                # so the layer is not interpreted
                continue
            out.append((os.path.basename(f), g.to_crs("EPSG:4326")))
    return out


def checks(ctx):
    import geopandas as gpd
    out = []
    layers = _pred_layers(ctx)
    out.append(result("point vector file written", bool(layers), value=[n for n, _ in layers]))
    gold = gpd.read_file(ctx.gold(GOLD)).to_crs("EPSG:4326")
    gk = _keyset(gold)
    best = None
    for name, g in layers:
        on_grid = float(np.mean((np.abs(g.geometry.x * 100 - np.round(g.geometry.x * 100)) < 1e-4)
                                & (np.abs(g.geometry.y * 100 - np.round(g.geometry.y * 100)) < 1e-4)))
        pk = _keyset(g)
        ov = min(len(pk & gk) / max(1, len(gk)), len(pk & gk) / max(1, len(pk)))
        score = on_grid + ov
        if best is None or score > best[0]:
            best = (score, name, g, on_grid, ov, len(pk))
    if best is None:
        for nm in ("records preserved (373 or 168 unique)", "geometries on 0.01-degree WGS84 grid",
                   "unique rounded locations equal gold set"):
            out.append(result(nm, False, note="no point layer"))
    else:
        _, name, g, on_grid, ov, npk = best
        out.append(result("records preserved (373 or 168 unique)", len(g) in (N, N_UNIQUE), value=len(g),
                          threshold=f"{N} or {N_UNIQUE}", note=name))
        out.append(result("geometries on 0.01-degree WGS84 grid", on_grid >= 0.99, value=round(on_grid, 4),
                          threshold=0.99))
        out.append(result("unique rounded locations equal gold set", ov >= 0.98, value=round(ov, 4),
                          threshold=0.98, note=f"{npk} unique pred locations vs {len(gk)} gold"))

    # documentation of the reduction - required (the instruction's stated goal).
    # Positive evidence: 168 unique rounded locations, a reduction of 204/205,
    # or ~54.8-55.0 % reduction (45.0-45.2 % retained) in a captured scalar (top level or
    # function locals), a small captured table/series/dict, or a text/csv/json/md output.
    def _hit(v):
        try:
            v = float(v)
        except Exception:
            return False
        return (v in (168.0, 204.0, 205.0) or 54.6 <= v <= 55.1 or 0.546 <= v <= 0.551
                or 45.0 <= v <= 45.2 or 0.450 <= v <= 0.452)
    found = None
    for tag, meta, load in ctx.objects("pred"):
        k = meta.get("kind")
        if k == "scalar":
            if _hit(meta.get("value")):
                found = f"{meta['name']}={meta.get('value')}@{tag}"
        elif k in ("df", "series") and 0 < meta.get("rows", 0) <= 20:
            try:
                t = load()
                vals = [x for c in t.columns if c != "geometry" for x in t[c].tolist()]
            except Exception:
                continue
            if any(_hit(x) for x in vals if not isinstance(x, str)):
                found = f"{meta['name']}@{tag}"
        if found:
            break
    if not found:
        import re
        for f in glob.glob(os.path.join(ctx.pred_dir, "*")):
            if f.lower().endswith((".csv", ".txt", ".json", ".md")):
                try:
                    txt = open(f, errors="ignore").read()
                except Exception:
                    continue
                if any(_hit(x) for x in re.findall(r"\d+(?:\.\d+)?", txt)):
                    found = os.path.basename(f)
                    break
    if not found:
        # a solution may compute and print
        # "unique locations before rounding: 372 / after rounding: 168" - the analysis log is
        # the documentation of a run. Accept a printed line that names the quantity
        # (unique/distinct/reduction/locations/fewer) together with a correct value; a
        # misreported value (e.g. "before 372 after 372") still fails.
        import re
        for line in ctx.stdout.splitlines():
            if re.search(r"unique|distinct|reduc|location|fewer|decreas", line, re.I) and \
                    any(_hit(x) for x in re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", line)):
                found = "stdout: " + line.strip()[:80]
                break
    out.append(result("reduction in unique locations documented (168 / 204-205 / ~55 %)", bool(found),
                      value=found, note="searched captured scalars/tables, text outputs and labelled stdout lines"))

    # exact coordinates must not be published: no attribute of the
    # written layer may reproduce the original lon/lat (or Web Mercator x/y) beyond 0.01 deg.
    leak = None
    if best is not None:
        try:
            orig = gpd.read_file(os.path.join(ctx.inputs_dir, "High_Blood_Level_Results.shp"))
            o84 = orig.to_crs("EPSG:4326")
            refs = {"lon": o84.geometry.x.values, "lat": o84.geometry.y.values,
                    "x3857": orig.to_crs("EPSG:3857").geometry.x.values,
                    "y3857": orig.to_crs("EPSG:3857").geometry.y.values}
            g = best[2]
            for c in g.columns:
                if c == "geometry" or g[c].dtype.kind != "f":
                    continue
                vals = g[c].dropna().values
                if not len(vals):
                    continue
                for rn, rv in refs.items():
                    tol = 1e-5 if rn in ("lon", "lat") else 1.0
                    d = np.abs(vals[:, None] - rv[None, :]).min(axis=1)
                    on_grid = np.abs(vals * 100 - np.round(vals * 100)) < 1e-6
                    if np.mean((d <= tol) & ~on_grid) >= 0.5:
                        leak = f"{c} ~ original {rn}"
                        break
                if leak:
                    break
        except Exception as e:
            leak = None
    out.append(result("exact original coordinates not kept as attributes", best is not None and leak is None,
                      value=leak, note="published layer must not carry the exact coordinates" if leak else ""))
    fields = False
    if best is not None:
        g = best[2]
        for c in g.columns:
            if c != "geometry" and g[c].dtype.kind == "f":
                vals = g[c].dropna().values
                if len(vals) and np.all(np.abs(vals * 100 - np.round(vals * 100)) < 1e-6) and np.abs(vals).max() <= 180:
                    fields = True
    out.append(result("rounded coordinate fields added", fields, required=False))
    return out
