"""Task 29 - tsunami travel time to Virginia Beach (Learn ArcGIS 'Calculate travel time for a tsunami').

Task core: invert the bathymetry, speed = sqrt(g*d) (m/s), follow the three source -> Virginia
Beach paths (geodesic), sample speed along them and sum length/speed per route. Gold (geodesic
densification, Int() speeds, NoData parts skipped): Currituck Slide 2.21 h, La Palma 9.45 h,
Puerto Rico Trench 6.25 h. Planar (Albers) segment lengths or an undensified straight line
change these by < 1 %.
Note: the GeoJSON layers hold Atlantic-Albers metres but declare no CRS (read as EPSG:4326).

Correctness (required) - either form of evidence is accepted:
  A. per-route travel times (captured table, CSV/JSON/TXT in pred_results) for all three routes
     consistent with one reading (straight path +/-6 % of gold, or least-time arrival; units
     h, min or s detected), or
  B. an arrival-time surface on the DEM grid (captured array or GeoTIFF band) that matches a
     gold least-time surface - one source, or the first-arrival surface
     of all three sources, i.e. their cell-wise minimum - (Spearman >= 0.95 on shared ocean cells) and whose value at
     Virginia Beach is within +/-15 % of the gold least-time arrival for that source (in h,
     min or s). A constant-speed distance surface fails the Virginia Beach test (shelf).
  plus: a non-blank map PNG (the task asks to visualise the result), and
  the figure shows the analysis: a drawn image/mesh rank-correlating (>= 0.9) with the gold
  speed or an arrival surface, or a tsunami route drawn as a line / collection path whose end
  points match a path (15 km / 0.15 deg).
  A drawn surface is accepted as drawn or flipped N-S
  (which way up is presentation; transposes / E-W mirrors still fail).
Reported, not required:
  - (kind="figure") the drawn surface is north-up: as drawn it fits the references at least as
    well (Spearman) as its N-S flip;
  - (kind="figure") the colour scale covers the drawn data: 5-95 % span of the drawn values
    >= 0.1 of the colour range (vmin..vmax) and >= 50 % of values inside it (a colour
    scale in seconds over hour values paints the whole map one colour);
  - speed raster sqrt(g*d): >= 95 % of ocean cells within 2 m/s of gold (captured or GeoTIFF);
  - both A and B present.
"""
import glob
import os
import re

import numpy as np
import pandas as pd

from checklib import result, png_not_blank


def _fix_proj():
    """Point PROJ at the data directory bundled with rasterio before its first import, so that
    GeoTIFF CRSs are read with a proj.db that matches rasterio's GDAL."""
    import importlib.util
    import sys
    if "rasterio" in sys.modules:
        return
    spec = importlib.util.find_spec("rasterio")
    if spec and spec.origin:
        d = os.path.join(os.path.dirname(spec.origin), "proj_data")
        if os.path.exists(os.path.join(d, "proj.db")):
            os.environ["PROJ_DATA"] = d


_fix_proj()

GOLD_TT = {"currituck": 2.208, "palma": 9.453, "puerto": 6.247}
KEYS = {"currituck": ("currituck",), "palma": ("palma",), "puerto": ("puerto", "rico")}
UNITS = {"h": 1.0, "min": 60.0, "s": 3600.0}
DEM_SHAPE = (2109, 3852)
STEP = 4


def _route_key(text):
    t = str(text).lower()
    for k, words in KEYS.items():
        if any(w in t for w in words):
            return k
    return None


def _tables(ctx):
    for tag, meta, load in ctx.objects("pred"):
        if meta.get("kind") in ("df", "gdf") and 3 <= meta.get("rows", 0) <= 50:
            try:
                yield meta["name"] + "@" + tag, load()
            except Exception:
                continue
    for f in glob.glob(os.path.join(ctx.pred_dir, "*.csv")):
        try:
            yield os.path.basename(f), pd.read_csv(f)
        except Exception:
            continue


def _route_times_from_table(df):
    """{route_key: [numeric values in that row]}"""
    best = {}
    for c in df.columns:
        if df[c].dtype == object:
            keys = df[c].map(_route_key)
            if keys.notna().sum() >= 3:
                for i, k in keys.items():
                    if k is None:
                        continue
                    vals = [float(df.at[i, cc]) for cc in df.columns
                            if cc != c and pd.api.types.is_numeric_dtype(df[cc]) and pd.notna(df.at[i, cc])]
                    best.setdefault(k, []).extend(vals)
    # route names as index
    if not best and df.index.dtype == object:
        for i in df.index:
            k = _route_key(i)
            if k:
                best.setdefault(k, []).extend(float(v) for v in df.loc[i].values if isinstance(v, (int, float, np.number)))
    return best


def _route_times_from_text(txt):
    out = {}
    for line in txt.splitlines():
        k = _route_key(line)
        if k:
            out.setdefault(k, []).extend(float(x) for x in re.findall(r"(?<![A-Za-z_])\d+\.?\d*(?:e[+-]?\d+)?", line))
    return out


LT_RANGE = {}   # least-time readings: filled in checks() from the gold arrival surfaces


def _match(cands):
    """All three routes consistent with ONE reading in one unit: straight-path times (+/-6 %:
    planar/undensified variants differ by <= 4.2 %, abs(DEM)
    by +8.6 %) or least-time (fastest-path) arrivals at Virginia Beach (gold surface value at
    VB +10 % down to the minimum within +/-5 cells (VB snapped to the ocean) -10 %)."""
    for u, f in UNITS.items():
        if all(any(abs(v - GOLD_TT[k] * f) <= 0.06 * GOLD_TT[k] * f for v in cands.get(k, [])) for k in GOLD_TT):
            return u + " straight path"
        if LT_RANGE and all(any(LT_RANGE[k][0] * f <= v <= LT_RANGE[k][1] * f for v in cands.get(k, []))
                            for k in GOLD_TT):
            return u + " least-time"
    return None


def _evidence_a(ctx):
    for name, df in _tables(ctx):
        try:
            u = _match(_route_times_from_table(df))
        except Exception:
            u = None
        if u:
            return f"{name} ({u})"
    for f in glob.glob(os.path.join(ctx.pred_dir, "*.txt")) + glob.glob(os.path.join(ctx.pred_dir, "*.json")) \
            + glob.glob(os.path.join(ctx.pred_dir, "*.md")):
        try:
            u = _match(_route_times_from_text(open(f, errors="ignore").read()))
        except Exception:
            u = None
        if u:
            return f"{os.path.basename(f)} ({u})"
    # scalars named after the routes
    cands = {}
    for tag, meta, load in ctx.objects("pred", "scalar"):
        k = _route_key(meta["name"].replace("pr_", "puerto_"))
        if k:
            cands.setdefault(k, []).append(meta["value"])
    u = _match(cands)
    return f"scalars ({u})" if u else None


def _gold_surfaces(ctx):
    import rasterio
    with rasterio.open(ctx.gold("tsunami_arrival_time_hours.tif")) as g:
        return {g.descriptions[i - 1] or str(i): g.read(i).astype(float) for i in range(1, g.count + 1)}, g.transform


def _pred_surfaces(ctx):
    """2-D arrays / GeoTIFF bands on the DEM grid or a coarser/finer grid of the same extent
    (same aspect ratio; resampled to the DEM shape) - coarser grids are accepted."""
    from checklib import resample_to
    asp = DEM_SHAPE[0] / DEM_SHAPE[1]

    def ok(shp):
        return len(shp) == 2 and min(shp) >= 200 and abs(shp[0] / shp[1] - asp) <= 0.01 * asp

    for tag, meta, load in ctx.objects("pred", "array", include_files=False):
        shp = tuple(meta.get("shape", []))
        if not ok(shp) or meta.get("source") == "figure":
            continue
        try:
            yield meta["name"] + "@" + tag, resample_to(np.asarray(load(), float), DEM_SHAPE)
        except Exception:
            continue
    import rasterio
    for f in glob.glob(os.path.join(ctx.pred_dir, "*.tif*")):
        try:
            with rasterio.open(f) as r:
                if not ok((r.height, r.width)):
                    continue
                for i in range(1, r.count + 1):
                    a = r.read(i).astype(float)
                    if r.nodata is not None:
                        a[a == r.nodata] = np.nan
                    yield f"{os.path.basename(f)}[{i}]", resample_to(a, DEM_SHAPE)
        except Exception:
            continue


N_UP = "surface drawn north-up (tie as drawn)"
N_CLIM = "colour scale covers the drawn data range"


def _drawn_tie(ctx):
    """The figure must show the analysis, not just anything. Accepted:
    (i) a drawn image/mesh (same extent aspect) whose values rank-correlate (>= 0.9) with the
    gold speed raster or a gold arrival surface; (ii) at least one tsunami route drawn as a
    line (ax.plot vertices or a collection path's bbox) matching a path's end points within
    15 km (Albers) or 0.15 deg (lon/lat)."""
    name = "figure shows the analysis (speed/arrival surface or the tsunami routes)"
    from checklib import drawn_arrays, resample_to
    drawn = drawn_arrays(ctx, "pred")
    if not drawn:
        from checklib import unverifiable
        return [unverifiable(name, "no drawn data captured from any figure"),
                result(N_UP, False, kind="figure", note="no drawn data"),
                result(N_CLIM, False, kind="figure", note="no drawn data")]
    try:
        import rasterio
        import geopandas as gpd
        from pyproj import Transformer
        with rasterio.open(ctx.gold("tsunami_speed.tif")) as s_:
            refs = [s_.read(1).astype(float)]
            crs = s_.crs
        with rasterio.open(ctx.gold("tsunami_arrival_time_hours.tif")) as g:
            refs += [g.read(i).astype(float) for i in range(1, g.count + 1)]
        refs.append(np.fmin.reduce(refs[1:]))   # first-arrival map of all sources
        asp = DEM_SHAPE[0] / DEM_SHAPE[1]
        # Which way up the surface is drawn and the colour range used are
        # presentation. The required tie accepts the drawn image as drawn or flipped N-S (never
        # transposed / mirrored E-W); kind="figure" rows report the orientation as drawn and
        # whether the colour scale covers the drawn data (e.g. a seconds scale on hour values).
        hit_any, hit_up = None, None
        for tag, meta, arr in drawn:
            if meta.get("artist") not in ("image", "QuadMesh"):
                continue
            a0 = np.asarray(arr, float)
            if a0.ndim != 2 or min(a0.shape) < 100 or abs(a0.shape[0] / a0.shape[1] - asp) > 0.01 * asp:
                continue
            if meta.get("origin") == "lower":
                a0 = a0[::-1]
            best_or = {}
            for flipped in (False, True):
                a = resample_to(a0[::-1] if flipped else a0, DEM_SHAPE)[::STEP, ::STEP]
                for ref in refs:
                    rr = ref[::STEP, ::STEP]
                    m = np.isfinite(a) & np.isfinite(rr)
                    if m.sum() > 1000 and a[m].std() > 0:
                        rho = float(pd.Series(a[m]).rank().corr(pd.Series(rr[m]).rank()))
                        best_or[flipped] = max(best_or.get(flipped, -1.0), rho)
            r_up, r_fl = best_or.get(False, -1.0), best_or.get(True, -1.0)
            if max(r_up, r_fl) >= 0.9:
                # north-up as drawn: the drawn orientation fits the references at least as well
                # as its N-S flip (a flipped arrival surface can still exceed 0.9 against some
                # reference, so the comparison, not the threshold, decides the orientation)
                up = r_up >= r_fl
                lab = f"{meta['name']}@{tag} rho as drawn {r_up:.3f} / flipped {r_fl:.3f}"
                hit_any = hit_any or (lab, meta, a0, up)
                if up and hit_up is None:
                    hit_up = (lab, meta, a0, up)
        if hit_any:
            lab, meta, a0, _ = hit_up or hit_any
            fin = a0[np.isfinite(a0)]
            clim = meta.get("clim")
            if clim and len(fin) and clim[1] > clim[0]:
                lo, hi = np.percentile(fin, [5, 95])
                span = (hi - lo) / (clim[1] - clim[0])
                inside = float(np.mean((fin >= clim[0]) & (fin <= clim[1])))
                cs = result(N_CLIM, span >= 0.1 and inside >= 0.5, kind="figure",
                            value={"data_5_95": [round(float(lo), 3), round(float(hi), 3)], "clim": clim,
                                   "span_share": round(float(span), 4), "inside_share": round(inside, 3)},
                            threshold="5-95 % data span >= 0.1 of the colour range; >= 50 % of values inside it",
                            note=lab)
            else:
                cs = result(N_CLIM, True, kind="figure", note=f"colour range not recorded ({lab})")
            return [result(name, True, value=lab),
                    result(N_UP, hit_up is not None, kind="figure", note=(hit_up or hit_any)[0]),
                    cs]
        paths = gpd.read_file(os.path.join(ctx.inputs_dir, "TsuniamiPaths.gpkg"))
        to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        ends = []
        for _, pr in paths.iterrows():
            ln = pr.geometry.geoms[0] if pr.geometry.geom_type == "MultiLineString" else pr.geometry
            (x0, y0), (x1, y1) = ln.coords[0], ln.coords[-1]
            ends.append((np.array([x0, y0]), np.array([x1, y1]), 15000.0))
            ends.append((np.array(to_ll.transform(x0, y0)), np.array(to_ll.transform(x1, y1)), 0.15))
        for tag, meta, arr in drawn:
            a = np.asarray(arr, float)
            nm = meta["name"]
            if meta.get("artist") == "line" and a.ndim == 2 and a.shape[1] == 2 and len(a) >= 2:
                cand = [(a[0], a[-1])]
            elif nm.endswith("_paths_bbox") and a.ndim == 2 and a.shape[1] == 4:
                cand = [(b[:2], b[2:]) for b in a if np.isfinite(b).all()]
            else:
                continue
            for p0, p1 in cand:
                for e0, e1, tol in ends:
                    lo, hi = np.minimum(e0, e1), np.maximum(e0, e1)
                    if nm.endswith("_paths_bbox"):
                        hit = np.all(np.abs(p0 - lo) <= tol) and np.all(np.abs(p1 - hi) <= tol)
                    else:
                        hit = (np.all(np.abs(p0 - e0) <= tol) and np.all(np.abs(p1 - e1) <= tol)) or \
                              (np.all(np.abs(p0 - e1) <= tol) and np.all(np.abs(p1 - e0) <= tol))
                    if hit:
                        return [result(name, True, value=f"route drawn: {nm}@{tag}"),
                                result(N_UP, True, kind="figure", note="route lines drawn in map coordinates"),
                                result(N_CLIM, True, kind="figure", note="no surface image matched; routes only")]
    except Exception as e:
        return [result(name, False, note=repr(e)[:200]), result(N_UP, False, kind="figure", note="error"),
                result(N_CLIM, False, kind="figure", note="error")]
    return [result(name, False, note="no drawn surface correlates with speed/arrival and no route line drawn"),
            result(N_UP, False, kind="figure", note="no drawn analysis"),
            result(N_CLIM, False, kind="figure", note="no drawn analysis")]


def checks(ctx):
    out = []
    pngs = ctx.pred_files("*.png")
    nb = any(png_not_blank(ctx.pred(p))[0] for p in pngs)
    out.append(result("map/figure written (non-blank PNG)", nb, value=pngs))

    b = None
    a = None
    speed_ok, speed_note = False, "no speed raster found"
    try:
        gold, gt = _gold_surfaces(ctx)
        import rasterio
        with rasterio.open(ctx.gold("tsunami_speed.tif")) as s:
            gspeed = s.read(1).astype(float)
        inv = ~gt
        vb = gpd_vb = None
        import geopandas as gpd
        src = gpd.read_file(os.path.join(ctx.inputs_dir, "TsunamiSources.gpkg"))
        p = src[src["Name"] == "Virginia Beach"].geometry.iloc[0]
        c, r = inv * (p.x, p.y)
        vr, vc = int(r), int(c)
        LT_RANGE.clear()
        for k, g in gold.items():
            rk = _route_key(k)
            if rk:
                LT_RANGE[rk] = (0.9 * float(np.nanmin(g[vr - 5:vr + 6, vc - 5:vc + 6])), 1.1 * float(g[vr, vc]))
        # path start cells (tsunami origins) for surfaces computed FROM Virginia Beach
        paths = gpd.read_file(os.path.join(ctx.inputs_dir, "TsuniamiPaths.gpkg"))
        starts = {}
        for _, pr in paths.iterrows():
            ln = pr.geometry.geoms[0] if pr.geometry.geom_type == "MultiLineString" else pr.geometry
            cc, rr = inv * ln.coords[0]
            starts[_route_key(pr["Route"])] = (int(rr), int(cc))
        for name, arr in _pred_surfaces(ctx):
            sub = arr[::STEP, ::STEP]
            # speed check
            gs = gspeed[::STEP, ::STEP]
            m = np.isfinite(gs) & np.isfinite(sub)
            if m.sum() > 0.5 * np.isfinite(gs).sum():
                frac = float((np.abs(sub[m] - gs[m]) <= 2.0).mean())
                if frac >= 0.95:
                    speed_ok, speed_note = True, f"{name}: {frac:.3f} within 2 m/s"
            if b:
                continue
            # B2: least-time surface computed FROM Virginia Beach
            # (reciprocity): ~0 at VB and, at the three tsunami origins, values within the
            # least-time ranges of all three routes in one unit.
            try:
                vals = {k: float(np.nanmin(arr[r0 - 1:r0 + 2, c0 - 1:c0 + 2])) for k, (r0, c0) in starts.items()}
                near = np.nanmin(arr[vr - 5:vr + 6, vc - 5:vc + 6])
                for u, f in UNITS.items():
                    if near <= 0.3 * f and all(LT_RANGE[k][0] * f <= vals[k] <= LT_RANGE[k][1] * f for k in GOLD_TT):
                        b = f"{name}: surface from Virginia Beach, at origins " + \
                            ", ".join(f"{k} {vals[k]:.2f}" for k in GOLD_TT) + f" {u}"
                        break
            except Exception:
                pass
            if b:
                continue
            win = arr[vr - 2:vr + 3, vc - 2:vc + 3]
            pv = np.nanmin(win) if np.isfinite(win).any() else np.nan
            # A combined first-arrival surface of the three sources (cell-wise
            # minimum) is accepted as a reference surface as well.
            for k, g in list(gold.items()) + [("first arrival (min of 3 sources)", np.fmin.reduce(list(gold.values())))]:
                gg = g[::STEP, ::STEP]
                mm = np.isfinite(gg) & np.isfinite(sub)
                if mm.sum() < 0.2 * np.isfinite(gg).sum():
                    continue
                x, y = sub[mm], gg[mm]
                if x.std() == 0:
                    continue
                rho = float(pd.Series(x).rank().corr(pd.Series(y).rank()))
                gw = g[vr - 2:vr + 3, vc - 2:vc + 3]
                gv = np.nanmin(gw)
                unit = next((u for u, f in UNITS.items() if np.isfinite(pv) and abs(pv - gv * f) <= 0.15 * gv * f), None)
                if rho >= 0.95 and unit:
                    b = f"{name} ~ {k}: rho {rho:.3f}, VB {pv:.2f} {unit} (gold {gv:.2f} h)"
                    break
    except Exception as e:
        speed_note = f"error: {e!r}"[:200]
    a = _evidence_a(ctx)
    out.extend(_drawn_tie(ctx))
    out.append(result("travel time to Virginia Beach correct (route times A or arrival surface B)", bool(a or b),
                      value={"A_route_times": a, "B_arrival_surface": b},
                      threshold="A: all 3 routes +/-6 % of 2.21/9.45/6.25 h or in the least-time ranges; B: rho>=0.95 & VB +/-15 %, or a surface from VB matching at the origins"))
    out.append(result("speed raster sqrt(g*d) matches gold (>=95 % cells within 2 m/s)", speed_ok, required=False,
                      note=speed_note))
    out.append(result("both route times and an arrival-time surface", bool(a and b), required=False))
    return out
