"""Task 44 - Fire-station coverage gaps in Etobicoke (Toronto).

Task core: buffer the fire stations, overlay with Etobicoke, map the areas without fire
service and save that no-service area. Both inputs are in EPSG:4326, so the buffer must be
built in metres (the classic trap: buffer(2500) in degrees covers the whole city and leaves
no gap). The instruction gives no buffer distance (the original code uses 2 500 m), so the
checks infer the distance the solution used and verify geometric consistency with it.

Correctness (required):
  1. a map (any PNG) exists and is not blank;
  2. a saved polygon vector file holds a non-empty no-service area that lies inside Etobicoke
     (>= 98 % of its area);
  3. it is the true gap for a plausible metric service radius: the radius d is searched over
     400-5000 m (50 m steps, refined to 5 m) for the best match of the stored gap with
     Etobicoke minus the union of ALL station buffers of radius d (area IoU >= 0.85).
     The radius is searched and not taken as the distance from the nearest station to the
     stored gap, because a few tiny spurious pieces of the gap can pull that distance far
     below the true radius. Subset constraint: the share of the stored gap
     lying within 0.9 d of ANY station must be <= 0.5 % - a gap built from a subset of the
     stations (e.g. only those inside the district) leaves areas served by the omitted
     stations in the gap (1-12 % for 0.8-2.6 km radii) and fails; where a subset gives the
     same gap (< 0.5 % differs) it is accepted, since the result is the same.
     Saving the covered area, buffering in degrees or another layer fails.
     A radius given in Web-Mercator "metres" (code uses EPSG:3857, the inferred
     radius is not round but equals a round distance x cos 43.7 deg, e.g. 2 500 -> 1 807 m)
     fails.
Reported, not required:
  4. buffered stations saved as their own file (workflow step 3);
  5. gap area equals the gold 2 500 m gap (IoU >= 0.85) - only meaningful if 2 500 m was used.
"""
import glob
import os

import numpy as np

from checklib import file_exists, png_not_blank, result

METRIC = "EPSG:26917"
VEC = ("*.geojson", "*.json", "*.shp", "*.gpkg")
DS = np.arange(400, 5001, 50)
SUBSET_SHARE = 0.005


def _search_radius(GE, gap, ga):
    """Best (IoU, d) of GE against Etobicoke minus all-station buffers of radius d: area
    pre-screen on a 200 m grid, 50 m search around it, then 5 m refinement."""
    ks = sorted(ga)
    k = int(np.argmin([abs(ga[x] - GE.area) for x in ks]))
    cand = [int(x) for x in DS if ks[max(0, k - 2)] <= x <= ks[min(len(ks) - 1, k + 2)]]
    best = (0.0, float(cand[0]))
    for d in cand:
        v = _iou(GE, gap(d))
        if v > best[0]:
            best = (v, float(d))
    lo, hi = max(400, int(best[1]) - 45), min(5000, int(best[1]) + 45)
    for d in range(lo, hi + 1, 5):
        v = _iou(GE, gap(d))
        if v > best[0]:
            best = (v, float(d))
    return best


def _vectors(ctx):
    import geopandas as gpd
    out = []
    for pat in VEC:
        for p in sorted(glob.glob(os.path.join(ctx.pred_dir, "**", pat), recursive=True)):
            try:
                if p.endswith(".gpkg"):
                    import pyogrio
                    for lyr in pyogrio.list_layers(p)[:, 0]:
                        out.append((os.path.basename(p) + ":" + lyr, gpd.read_file(p, layer=lyr)))
                else:
                    out.append((os.path.basename(p), gpd.read_file(p)))
            except Exception:
                continue
    return out


def _round(x, tol):
    """x within tol (relative) of a multiple of 250 m or of a half mile."""
    for step in (250.0, 804.672):
        k = round(x / step)
        if k >= 1 and abs(x - k * step) <= tol * x:
            return k * step
    return None


def _mercator_radius(code, d):
    """Buffering 'R metres' in EPSG:3857 gives a ground radius of
    R * cos(43.7 deg) ~ 0.723 R (2 500 -> 1 807 m), which still looks plausible. Flag it when
    the code uses Web Mercator, the inferred radius is not itself round, and d / 0.715-0.731
    hits a round distance."""
    low = code.lower()
    if not ("3857" in low or "mercator" in low or "900913" in low):
        return None
    if _round(d, 0.01):
        return None
    for f in (0.715, 0.719, 0.723, 0.727, 0.731):
        r = _round(d / f, 0.006)
        if r:
            return f"inferred {d:.0f} m = {r:.0f} m x cos(lat) - distance taken in Web Mercator units"
    return None


def _iou(a, b):
    u = a.union(b).area
    return float(a.intersection(b).area / u) if u > 0 else float("nan")


def checks(ctx):
    import geopandas as gpd
    out = []
    pngs = ctx.pred_files("*.png")
    r = file_exists(ctx, pngs or ["<any>.png"], name="map present")
    out.append(r)
    if pngs:
        nb, std = png_not_blank(ctx.pred(pngs[0]))
        out.append(result("map not blank", nb, value=std, threshold=">=3 grey-level std"))
    else:
        out.append(result("map not blank", False, note="no PNG"))

    E = gpd.read_file(f"{ctx.inputs_dir}/etobicoke.geojson").to_crs(METRIC).union_all()
    st = gpd.read_file(f"{ctx.inputs_dir}/fire_station.geojson").to_crs(METRIC)
    gold_gap = None
    try:
        gold_gap = gpd.read_file(ctx.gold("no_service.geojson")).to_crs(METRIC).union_all()
    except Exception:
        pass

    refs, covs = {}, {}

    def gap(d):
        if d not in refs:
            refs[d] = E.difference(st.buffer(d, resolution=32).union_all())
        return refs[d]

    def cover(d):
        d = round(d, 1)
        if d not in covs:
            covs[d] = st.buffer(d, resolution=32).union_all()
        return covs[d]

    ga = {int(d): gap(int(d)).area for d in DS[::4]}   # gap area is monotone in d
    best = None  # (iou, name, inside, d, area, nearest-station distance, share within 0.9 d)
    buffers_saved = False
    for name, g in _vectors(ctx):
        try:
            if g.crs is None or len(g) == 0:
                continue
            g = g[g.geometry.notna() & ~g.geometry.is_empty]
            if len(g) == 0 or not g.geom_type.isin(["Polygon", "MultiPolygon"]).all():
                continue
            G = g.to_crs(METRIC).geometry.buffer(0).union_all()
            if G.is_empty or G.area <= 0:
                continue
            inside = G.intersection(E).area / G.area
            if len(g) >= 0.8 * len(st) and inside < 0.9:
                buffers_saved = True
            d0 = float(st.distance(G).min())   # nearest-station distance, reported only
            GE = G.intersection(E)
            if GE.is_empty or GE.area <= 0:
                cand = (0.0, name, inside, d0, G.area, d0, 1.0)
            else:
                v, d = _search_radius(GE, gap, ga)
                share = float(GE.intersection(cover(0.9 * d)).area / GE.area)
                cand = (v, name, inside, d, G.area, d0, share)
            if inside < 0.98:
                cand = (min(cand[0], 0.0),) + cand[1:]
            if best is None or cand[0] > best[0] or (cand[0] == best[0] and cand[2] > best[2]):
                best = cand
        except Exception:
            continue

    if best is None:
        out.append(result("no-service area saved, non-empty and inside Etobicoke", False,
                          note="no polygon vector file in pred_results"))
        out.append(result("gap = Etobicoke minus station buffers of a plausible radius (IoU >= 0.85)", False))
    else:
        iou, name, inside, d, area, d0, share = best
        out.append(result("no-service area saved, non-empty and inside Etobicoke", inside >= 0.98 and area > 0,
                          value=round(inside, 3), threshold=">=0.98 inside", note=f"{name}, {area / 1e6:.2f} km2"))
        subset_ok = share <= SUBSET_SHARE
        ok = 400 <= d <= 5000 and iou >= 0.85 and subset_ok
        merc = _mercator_radius(ctx.code_text(), d)
        if merc:
            ok = False
        out.append(result("gap = Etobicoke minus station buffers of a plausible radius (IoU >= 0.85)", ok,
                          value=[round(iou, 3), round(d, 1)],
                          threshold="IoU>=0.85, radius 400-5000 m, <=0.5 % of the gap within 0.9 d of a station",
                          note=f"{name}; best radius {d:.0f} m (nearest station {d0:.0f} m); "
                               f"{100 * share:.2f} % of the gap within 0.9 d of a station"
                               + ("" if subset_ok else " -> areas served by some stations left in the gap "
                                  "(subset of stations buffered)")
                               + (f"; MERCATOR: {merc}" if merc else "")))
        if gold_gap is not None:
            try:
                G = [g for n, g in _vectors(ctx) if n == name][0].to_crs(METRIC).geometry.buffer(0).union_all()
                gi = _iou(G.intersection(E), gold_gap)
            except Exception:
                gi = float("nan")
            out.append(result("gap equals the gold 2500 m gap (IoU >= 0.85)", gi >= 0.85, value=round(gi, 3),
                              threshold=0.85, required=False))
    out.append(result("buffered stations saved as a file", buffers_saved, required=False))
    return out
