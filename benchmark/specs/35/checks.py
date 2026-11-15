"""T35 - Post-fire landslide risk (weighted overlay of burn severity, slope, land cover).

Gold: all layers on the Landsat UTM 10N 30 m grid; dNBR = NBR_pre - NBR_post with USGS
classes; slope classes 5/15/25/35 deg; the task's land-cover table; risk = 0.30 burn +
0.55 slope + 0.15 land cover.

Correctness (required):
  1. landslide_map.png exists and is not blank;
  2. a risk surface agrees with gold: some captured 2-D array (or written GeoTIFF) on the
     Landsat, DEM or NLCD grid (gold is reprojected to that grid) has Pearson >= 0.85 with the
     gold risk (>= 0.80). Alternative breaks (authors' min-max burn bins 0.93, quantile slope
     0.90, slope in Web-Mercator units 0.92; all combined on the DEM grid 0.85) pass;
     misregistered layers (zoom-resampling that ignores the CRS, 0.56), equal weights (0.74) or swapped weights
     (0.14) fail. The candidate is chosen jointly with check 3, so a lone input layer
     (e.g. the slope classes, r = 0.87) cannot stand in for the overlay;
  3. all three layers enter with the right sign and roughly the instructed weights:
     regressing the surface on burn / slope / land-cover classes (coefficients normalised by
     the slope coefficient; gold 0.55 / 1 / 0.27) gives burn/slope >= 0.35,
     land cover/slope >= 0.10 and slope the largest. If a captured dNBR layer matches the gold
     dNBR at |r| >= 0.97 (right bands and index), other burn class breaks are accepted and
     burn/slope >= 0.20 suffices (the authors' min-max bins give 0.25 on gold USGS classes). A reversed burn-severity sign (post - pre
     remapped low->high) gives a negative burn coefficient; a dropped land-cover layer ~0;
     NBR from the wrong Landsat bands (B6/B7, 0-based index slip) burn/slope ~0.21.
     Land cover = the gold table classes; burn and slope = gold breaks, 5 equal-interval or 5
     quantile classes of the gold dNBR / slope, whichever fits the surface best (the
     text gives no burn/slope breaks, and on gold breaks alone quantile classes pull
     land cover/slope to ~0.05 although land cover matches gold 100 %);
  4. the map shows the risk surface: an image drawn in the figure (a 2-D array written with
     plt.imsave counts as drawn) itself passes
     checks 2 and 3, in the displayed or the vertically mirrored orientation (the drawn
     values are the result, the map orientation is presentation).
  5. (the text prescribes NBR) burn severity is the NBR difference: a captured
     layer matches gold dNBR at |r| >= 0.97 (dNDVI 0.92 and NBR from bands 6/7 0.94 fail).
Presentation (kind="figure"; secondary analysis, never required):
  6. the map is drawn north up: check 4 on the image as displayed (imshow by its origin;
     pcolormesh flipped when its mesh y coordinates increase with the row index).
"""
import glob
import os

import numpy as np

from checklib import file_exists, png_not_blank, result

LANDSAT = (1602, 1346)
DEM = (2561, 1830)
NLCD = (2333, 1923)
R_MIN = 0.80
# Lower bounds of the coefficient ratios. Gold ratios are land cover/slope 0.27, burn/slope 0.55;
# the lowest legitimate variants are the quantile control (land cover 0.14) and the authors'
# min-max burn bins (burn 0.25), so the bounds sit below them rather than at 0.5x gold
# (0.135 / 0.275).
LC_MIN = 0.10
BURN_MIN_EXACT = 0.20


def _fix_proj():
    """Point PROJ at the data directory bundled with rasterio if the EPSG database cannot be read."""
    try:
        import rasterio
        from rasterio.crs import CRS
        CRS.from_epsg(3857)
    except Exception:
        try:
            from rasterio._env import set_proj_data_search_path
            set_proj_data_search_path(os.path.join(os.path.dirname(rasterio.__file__), "proj_data"))
        except Exception:
            pass


def _grids(ctx):
    _fix_proj()
    import rasterio
    from rasterio.crs import CRS
    out = {}
    with rasterio.open(f"{ctx.inputs_dir}/Before_L8.tif") as s:
        out[LANDSAT] = (s.crs, s.transform)
    with rasterio.open(f"{ctx.inputs_dir}/DEM_30m.tif") as s:
        out[DEM] = (CRS.from_epsg(3857), s.transform)
    with rasterio.open(f"{ctx.inputs_dir}/Sonoma_NLCD2011.tif") as s:
        out[NLCD] = (s.crs, s.transform)
    return out


def _to(arr, src, dst_shape, dst, nearest=False):
    from rasterio.warp import reproject, Resampling
    out = np.full(dst_shape, np.nan, dtype="float32")
    reproject(arr.astype("float32"), out, src_transform=src[1], src_crs=src[0], dst_transform=dst[1],
              dst_crs=dst[0], resampling=Resampling.nearest if nearest else Resampling.bilinear,
              src_nodata=np.nan, dst_nodata=np.nan)
    return out


def _gold_arrays(ctx):
    want = {"landslide_risk", "burn_class", "slope_class", "lc_class", "dnbr", "slope"}
    got = {}
    for tag, meta, load in ctx.objects("gold", "array"):
        if meta["name"] in want and meta["name"] not in got:
            got[meta["name"]] = np.asarray(load(), float)
    for k in ("burn_class", "slope_class", "lc_class"):
        if k in got:
            got[k][got[k] == 0] = np.nan
    return got


def _bins(v, mask, how):
    """5 classes (1-5) of a continuous gold layer: equal-interval over min-max or quantiles."""
    x = v[mask]
    if how == "quantile":
        br = np.nanquantile(x, [0.2, 0.4, 0.6, 0.8])
    else:
        lo, hi = np.nanmin(x), np.nanmax(x)
        br = lo + (hi - lo) * np.array([0.2, 0.4, 0.6, 0.8])
    return (np.digitize(v, br) + 1).astype(float)


def _best_basis_fit(a, mm, gg):
    """Regression of the surface on the burn / slope / land-cover classes. The text fixes the
    land-cover table and the weights but not the burn or slope class breaks (benchmark defect),
    so the burn and slope classes are taken in the scheme the surface was built with: gold
    breaks, 5 equal-interval or 5 quantile classes of the gold dNBR / slope, whichever fits the
    surface best (largest R^2). Land cover always uses the gold table classes. The bounds are
    then applied to that fit (on gold breaks alone, quantile burn/slope classes pull the
    land-cover coefficient to ~0.05 although land cover matches gold 100 %)."""
    y = a[mm]
    lc = gg["lc_class"][mm]
    burns = {"gold": gg["burn_class"][mm]}
    slopes = {"gold": gg["slope_class"][mm]}
    for how in ("equal", "quantile"):
        if "dnbr" in gg and np.isfinite(gg["dnbr"][mm]).all():
            burns[how] = _bins(gg["dnbr"], mm, how)[mm]
        if "slope" in gg and np.isfinite(gg["slope"][mm]).all():
            slopes[how] = _bins(gg["slope"], mm, how)[mm]
    best = None
    for bn, b in burns.items():
        for sn, sl in slopes.items():
            X = np.c_[b, sl, lc, np.ones(len(y))]
            c, res, *_ = np.linalg.lstsq(X, y, rcond=None)
            sse = float(res[0]) if len(res) else float(((X @ c - y) ** 2).sum())
            if best is None or sse < best[0] - 1e-9 * abs(best[0]):
                best = (sse, c[:3], f"{bn}/{sn}")
    return best[1], best[2]


def checks(ctx):
    out = []
    names = ["landslide_map.png", "Landslide_map.png", "landslide_map.PNG"]
    r = file_exists(ctx, names)
    out.append(r)
    if r["passed"]:
        nb, std = png_not_blank(ctx.pred(r["value"][0]))
        out.append(result("map not blank", nb, value=std, threshold=">=3 grey-level std"))
    else:
        out.append(result("map not blank", False, note="no landslide_map.png"))
    nm = ["risk surface matches gold (Pearson >= 0.80)",
          "burn, slope and land cover enter with instructed weights (regression on gold classes)",
          "map shows the risk surface (drawn image passes the surface checks)"]
    try:
        grids = _grids(ctx)
        g = _gold_arrays(ctx)
        assert "landslide_risk" in g
    except Exception as e:
        for n in nm:
            out.append(result(n, False, note=f"gold/grid unavailable: {e!r}"[:200]))
        return out

    gold_on = {LANDSAT: {k: g[k] for k in ("landslide_risk", "burn_class", "slope_class", "lc_class", "dnbr", "slope")
                         if k in g}}

    def gold_for(shape, grid=None):
        key = shape if grid is None else ("tif", shape, tuple(grid[1])[:6])
        if key not in gold_on:
            dst = grid if grid is not None else grids[shape]
            gold_on[key] = {k: _to(v, grids[LANDSAT], shape, dst, nearest=k.endswith("class"))
                            for k, v in gold_on[LANDSAT].items()}
        return gold_on[key]

    cands = []
    mesh = {}
    for tag, meta, load in ctx.objects("pred", "array", include_files=False):
        if meta.get("source") == "figure" and meta["name"].endswith("_mesh_coords"):
            try:
                a = np.asarray(load(), float)
                if a.ndim == 3:
                    mesh[(tag, meta["name"][: -len("_mesh_coords")])] = a
            except Exception:
                pass
    for tag, meta, load in ctx.objects("pred", "array"):
        shp = tuple(meta.get("shape") or [])
        if len(shp) == 3 and shp[0] == 1:
            shp = shp[1:]
        if shp in (LANDSAT, DEM, NLCD):
            drawn = meta.get("source") == "figure" and meta.get("artist") in ("image", "QuadMesh")
            if drawn and meta.get("artist") == "QuadMesh" and not meta["name"].endswith("_values"):
                continue
            if meta.get("artist") == "QuadMesh":
                mc = mesh.get((tag, meta["name"][: -len("_values")]))
                fl0 = bool(mc is not None and mc[-1, 0, 1] > mc[0, 0, 1])
            else:
                fl0 = meta.get("origin") == "lower"
            # drawn images: the as-displayed orientation, plus the vertically
            # mirrored one - the drawn VALUES are the analysis result, the map orientation is
            # presentation and is reported separately (kind="figure")
            for fl in ([fl0, not fl0] if drawn else [fl0]):
                ld = (lambda load=load, fl=fl: np.asarray(load(), float)[::-1] if fl else load())
                as_drawn = fl == fl0
                cands.append((f"{meta['name']}@{tag}" + (" (flipped)" if fl else "")
                              + ("" if as_drawn else " [mirrored vs display]"), ld, shp, None, drawn, as_drawn))
    for p in glob.glob(os.path.join(ctx.pred_dir, "*.tif*")):
        try:
            import rasterio
            with rasterio.open(p) as s:
                if s.crs is None:
                    continue
                grid = (s.crs, s.transform)
                shp = (s.height, s.width)
            cands.append((os.path.basename(p), p, shp, grid, False, True))
        except Exception:
            continue

    best = (None, None, -2.0, None, False)
    best_drawn = (None, None, -2.0, None, False)
    best_shown = (None, None, -2.0, None, False)
    burn_best = (None, 0.0)
    evals = []
    for lab, src, shp, grid, drawn, as_drawn in cands:
        try:
            if grid is None:
                a = np.asarray(src(), float).reshape(shp)
            else:
                import rasterio
                with rasterio.open(src) as s:
                    a = s.read(1).astype(float)
                    if s.nodata is not None:
                        a[a == s.nodata] = np.nan
            gg = gold_for(shp, grid)
        except Exception:
            continue
        gr = gg["landslide_risk"]
        m = np.isfinite(a) & np.isfinite(gr)
        if m.sum() < 0.5 * np.isfinite(gr).sum() or np.nanstd(a[m]) == 0:
            continue
        rr = float(np.corrcoef(a[m], gr[m])[0, 1])
        mm = m & np.isfinite(gg["burn_class"]) & np.isfinite(gg["slope_class"]) & np.isfinite(gg["lc_class"])
        coef, basis = _best_basis_fit(a, mm, gg)
        rel = coef / coef[1] if coef[1] > 0 else np.array([-9.0, 1.0, -9.0])
        evals.append((lab + (f" [basis {basis}]" if basis != "gold/gold" else ""), rr, rel, coef, drawn, as_drawn))
        if "dnbr" in gg:
            dm = m & np.isfinite(gg["dnbr"])
            if dm.sum() > 1000:
                rb = float(np.corrcoef(a[dm], gg["dnbr"][dm])[0, 1])
                if abs(rb) > abs(burn_best[1]):
                    burn_best = (lab, rb)
    # burn layer evidence: a captured dNBR-like layer matching gold dNBR (|r| >= 0.97) shows the
    # right bands / index; then other burn class breaks (e.g. the authors' min-max bins, which
    # attenuate the burn coefficient on gold USGS classes to ~0.25) are accepted with burn/slope
    # >= BURN_MIN_EXACT (0.20). Without it burn/slope >= 0.35 is needed (NBR from bands 6/7: dNBR r 0.94, 0.21).
    burn_exact = abs(burn_best[1]) >= 0.97
    burn_min = BURN_MIN_EXACT if burn_exact else 0.35
    for lab_, rr_, rel_, coef_, drawn_, as_drawn_ in evals:
        reg_ok_ = bool(coef_[1] > 0 and rel_[0] >= burn_min and rel_[2] >= LC_MIN and coef_[1] >= coef_[0]
                       and coef_[1] >= coef_[2])
        score = (rr_ >= R_MIN and reg_ok_, rr_)
        if best[0] is None or score > best[0]:
            best = (score, lab_, rr_, rel_, reg_ok_)
        if drawn_ and (best_drawn[0] is None or score > best_drawn[0]):
            best_drawn = (score, lab_, rr_, rel_, reg_ok_)
        if drawn_ and as_drawn_ and (best_shown[0] is None or score > best_shown[0]):
            best_shown = (score, lab_, rr_, rel_, reg_ok_)
    _, lab, rr, coef, reg_ok = best
    out.append(result(nm[0], lab is not None and rr >= R_MIN, value=round(rr, 3) if lab else None, threshold=R_MIN,
                      note=lab or "no 2-D surface on the Landsat/DEM/NLCD grid or georeferenced GeoTIFF"))
    out.append(result(nm[1], bool(lab) and reg_ok,
                      value="burn/slope=%.2f, landcover/slope=%.2f" % (coef[0], coef[2]) if lab else None,
                      threshold=f"burn/slope >= {burn_min} ({BURN_MIN_EXACT} if a dNBR layer matches gold at |r|>=0.97), landcover/slope >= {LC_MIN}, slope largest (gold 0.55/0.27)",
                      note=lab or "no risk surface"))
    sc, dlab, drr, drel, dok = best_drawn
    if dlab is None:
        out.append(result(nm[2], False, note="UNVERIFIABLE: no 2-D image on a native grid drawn in the figure"))
        out[-1]["unverifiable"] = True
    else:
        out.append(result(nm[2], drr >= R_MIN and dok, value=f"r={drr:.3f}, burn/slope={drel[0]:.2f}, lc/slope={drel[2]:.2f}",
                          threshold="checks 2 and 3 on the drawn image (vertical orientation free)", note=dlab))
    fo = "map drawn north up (presentation)"
    sc, slab, srr, srel, sok = best_shown
    if slab is None:
        out.append(result(fo, False, kind="figure", note="UNVERIFIABLE: no 2-D image on a native grid drawn in the figure"))
        out[-1]["unverifiable"] = True
    else:
        out.append(result(fo, srr >= R_MIN and sok, kind="figure", value=f"r={srr:.3f}",
                          threshold="checks 2 and 3 on the image as displayed", note=slab))
    # the text prescribes the NBR (B5/B7) difference as burn severity, so a captured dNBR layer
    # is required. |r| >= 0.97 with gold dNBR separates it from other indices that also track
    # the burn (dNDVI r 0.92, NBR from bands 6/7 r 0.94).
    out.append(result("burn severity = NBR difference (captured layer |r| >= 0.97 with gold dNBR)", burn_exact,
                      value=round(burn_best[1], 3) if burn_best[0] else None, threshold=0.97,
                      note=(burn_best[0] or "no dNBR-like layer captured") + "; the text prescribes NBR"))
    return out
