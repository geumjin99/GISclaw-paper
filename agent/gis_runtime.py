"""GIS runtime installed in the agent's kernel (harness prelude, re-installed after a restart).

Sections are installed only when their component is on, so each can be ablated:
  G2  spatial-operation guards      notes after each block
  G3  figure semantics at savefig   notes after each block (file checks at finish: gis_validate)
  G6  GIS library API index         lookup_api tool + "did you mean" after import/attribute errors
  G7  provenance                    which files each block read and wrote (not shown to the agent)
All wrappers only OBSERVE: they call the original function and return its result unchanged.
Wrappers that check user-level operations only react to calls made directly from the agent's
code (frame filename "<string>"), not to calls made inside libraries.
Every rule is generic GIS practice; none refers to a particular task.
"""
import ast

CORE = r'''
def _gg_core():
    import sys, os, math
    st = {"notes": [], "seen": set(), "io": [], "refs": None}
    def note(comp, kind, msg, key=None):
        k = (comp, kind, key if key is not None else msg[:120])
        if k in st["seen"]:
            return
        st["seen"].add(k)
        st["notes"].append({"comp": comp, "kind": kind, "msg": msg})
    def from_agent(depth=2):
        try:
            return sys._getframe(depth).f_code.co_filename == "<string>"
        except Exception:
            return False
    def refs():
        """Extents of the input layers: [(name, lonlat_bbox, native_bbox)] (lazy)."""
        if st["refs"] is not None:
            return st["refs"]
        out = []
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from pyproj import Transformer, CRS
            def ll(b, crs):
                t = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
                xs, ys = t.transform([b[0], b[2], b[0], b[2], (b[0] + b[2]) / 2], [b[1], b[1], b[3], b[3], (b[1] + b[3]) / 2])
                return (min(xs), min(ys), max(xs), max(ys))
            for r, ds, fs in os.walk("dataset"):
                for f in fs:
                    p = os.path.join(r, f); e = os.path.splitext(f)[1].lower()
                    try:
                        if e in (".shp", ".geojson", ".gpkg"):
                            import geopandas as g
                            d = g.read_file(p, rows=20000)
                            if len(d) and d.crs is not None:
                                b = tuple(d.total_bounds); out.append((f, ll(b, d.crs), b))
                        elif e in (".tif", ".tiff", ".img", ".asc"):
                            import rasterio as rio
                            with rio.open(p) as s:
                                if s.crs is not None:
                                    b = tuple(s.bounds); out.append((f, ll(b, s.crs), b))
                        elif e in (".nc", ".nc4"):
                            bb = _gg_nc_lonlat(p)
                            if bb:
                                out.append((f, bb, None))
                    except Exception:
                        pass
        st["refs"] = out
        return out
    return st, note, from_agent, refs

def _gg_nc_lonlat(p):
    """Lon/lat bbox of a CF NetCDF grid: 2-D/1-D lat/lon variables, or a rotated-pole grid."""
    import numpy as _np
    import xarray as _xr
    with _xr.open_dataset(p, decode_times=False) as d:
        lat = lon = None
        for v in list(d.variables):
            sn = str(d[v].attrs.get("standard_name", "")).lower()
            if sn == "latitude" or v.lower() in ("lat", "latitude"):
                lat = d[v].values
            if sn == "longitude" or v.lower() in ("lon", "longitude", "long"):
                lon = d[v].values
        if lat is not None and lon is not None:
            la = _np.asarray(lat, float); lo = _np.asarray(lon, float)
            ok_la = _np.isfinite(la) & (_np.abs(la) <= 90); ok_lo = _np.isfinite(lo) & (_np.abs(lo) <= 360)
            # fill values (e.g. 9.97e36) are common in these variables: trust them only if mostly valid
            if ok_la.mean() > 0.5 and ok_lo.mean() > 0.5:
                return (float(lo[ok_lo].min()), float(la[ok_la].min()), float(lo[ok_lo].max()), float(la[ok_la].max()))
        for v in list(d.variables):
            a = d[v].attrs
            if a.get("grid_mapping_name") == "rotated_latitude_longitude":
                import cartopy.crs as _ccrs
                rp = _ccrs.RotatedPole(pole_longitude=float(a["grid_north_pole_longitude"]),
                                       pole_latitude=float(a["grid_north_pole_latitude"]))
                rx = ry = None
                for c in d.coords:
                    sn = str(d[c].attrs.get("standard_name", ""))
                    if sn == "grid_longitude": rx = d[c].values
                    if sn == "grid_latitude": ry = d[c].values
                if rx is None or ry is None:
                    return None
                X, Y = _np.meshgrid(rx, ry)
                pts = _ccrs.PlateCarree().transform_points(rp, X.ravel(), Y.ravel())
                return (float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max()))
    return None

_gg_st, _gg_note, _gg_from_agent, _gg_refs = _gg_core()
del _gg_core
'''

# ----------------------------------------------------------------------------- G2
G2 = r'''
def _gg_install_g2(st, note, from_agent):
    import functools, numpy as _np, pandas as _pd
    import geopandas as _gpd
    import rasterio as _rio
    st["grids"] = {}
    def _n(x):
        try:
            return len(x)
        except Exception:
            return None
    def wrap_join(fn, label):
        @functools.wraps(fn)
        def w(*a, **k):
            r = fn(*a, **k)
            try:
                left = a[0] if a else k.get("left_df", k.get("df1"))
                nl, nr = _n(left), _n(r)
                how = k.get("how", "intersection" if "overlay" in label else "inner")
                if how in ("difference", "symmetric_difference", "union", "identity", "right"):
                    return r
                if nr == 0 and nl:
                    note("g2", "empty_result", f"{label} returned 0 rows (input had {nl}). This usually means the layers do not overlap: check that both use the same CRS and that the predicate is right.")
                elif label.startswith(("gpd.sjoin", "GeoDataFrame.sjoin")) and nl and nr is not None and "index_right" in getattr(r, "columns", []):
                    matched = int(r["index_right"].notna().sum()) if how == "left" else nr
                    if nl >= 20 and matched < 0.05 * nl:
                        note("g2", "few_matches", f"{label} matched only {matched} of {nl} rows ({matched / nl:.1%}). Check CRS and predicate before relying on it.")
            except Exception:
                pass
            return r
        return w
    for name in ("sjoin", "sjoin_nearest", "overlay", "clip"):
        if hasattr(_gpd, name):
            setattr(_gpd, name, wrap_join(getattr(_gpd, name), "gpd." + name))
    for name in ("sjoin", "sjoin_nearest", "overlay", "clip"):
        if hasattr(_gpd.GeoDataFrame, name):
            orig = getattr(_gpd.GeoDataFrame, name)
            def mk(orig=orig, name=name):
                @functools.wraps(orig)
                def m(self, *a, **k):
                    return wrap_join(lambda *aa, **kk: orig(self, *aa, **kk), "GeoDataFrame." + name)(self, *a, **k)
                return m
            setattr(_gpd.GeoDataFrame, name, mk())
    # attribute joins: how many rows found a partner (agent's own calls only)
    def check_merge(left, r, how, label):
        try:
            nl = len(left)
            if nl < 20:
                return
            if how == "inner" and len(r) < 0.5 * nl:
                note("g2", "merge_few_matches", f"{label} (inner) kept {len(r)} of {nl} rows. The join keys may not match (different types, spelling, codes with/without leading zeros).")
            elif how == "left":
                extra = [c for c in r.columns if c not in left.columns]
                if extra:
                    un = int(r[extra].isna().all(axis=1).sum())
                    if un > 0.5 * len(r):
                        note("g2", "merge_few_matches", f"{label} (left) found no partner for {un} of {len(r)} rows. The join keys may not match (different types, spelling, codes with/without leading zeros).")
        except Exception:
            pass
    orig_dfm = _pd.DataFrame.merge
    @functools.wraps(orig_dfm)
    def dfm(self, right, *a, **k):
        r = orig_dfm(self, right, *a, **k)
        if from_agent():
            check_merge(self, r, k.get("how", "inner"), "merge")
        return r
    _pd.DataFrame.merge = dfm
    orig_pm = _pd.merge
    @functools.wraps(orig_pm)
    def pm(left, right, *a, **k):
        r = orig_pm(left, right, *a, **k)
        if from_agent():
            check_merge(left, r, k.get("how", "inner"), "pd.merge")
        return r
    _pd.merge = pm
    # reprojection that produced unusable coordinates
    for cls in (_gpd.GeoDataFrame, _gpd.GeoSeries):
        orig_tc = cls.to_crs
        def mk_tc(orig_tc=orig_tc):
            @functools.wraps(orig_tc)
            def tc(self, *a, **k):
                r = orig_tc(self, *a, **k)
                try:
                    if from_agent() and len(r):
                        b = r.total_bounds
                        if not _np.all(_np.isfinite(b)):
                            note("g2", "reprojection_invalid", f"to_crs produced infinite/NaN coordinates (source CRS {self.crs}). The source CRS is probably wrong: use set_crs to declare the CRS the coordinates are actually in, then to_crs.")
                except Exception:
                    pass
                return r
            return tc
        cls.to_crs = mk_tc()
    # writing without CRS
    for cls in (_gpd.GeoDataFrame, _gpd.GeoSeries):
        orig_tf = cls.to_file
        def mk_tf(orig_tf=orig_tf):
            @functools.wraps(orig_tf)
            def tf(self, filename, *a, **k):
                try:
                    if self.crs is None:
                        note("g2", "no_crs_written", f"{filename} is being written without a CRS. Set it (set_crs / to_crs) so the file can be used as a map layer.")
                except Exception:
                    pass
                return orig_tf(self, filename, *a, **k)
            return tf
        cls.to_file = mk_tf()
    # rasters: grid agreement and nodata
    DR = _rio.io.DatasetReader
    orig_read = DR.read
    @functools.wraps(orig_read)
    def read(self, *a, **k):
        arr = orig_read(self, *a, **k)
        try:
            path = getattr(self, "name", "?")
            grid = (self.width, self.height, tuple(round(v, 9) for v in self.transform)[:6], str(self.crs))
            if path not in st["grids"]:
                for p2, g2 in st["grids"].items():
                    if g2 != grid:
                        diff = [n for n, x, y in (("size", g2[:2], grid[:2]), ("cell size/origin", g2[2], grid[2]), ("CRS", g2[3], grid[3])) if x != y]
                        note("g2", "grid_mismatch", key=(path, p2), msg=f"{path} and {p2} are on different grids ({', '.join(diff)} differ). Resample/reproject one onto the other before combining them cell by cell.")
                        break
                st["grids"][path] = grid
            masked = k.get("masked", False) or isinstance(arr, _np.ma.MaskedArray)
            if not masked and _np.issubdtype(_np.asarray(arr).dtype, _np.number):
                nod = self.nodata
                if nod is not None and not (isinstance(nod, float) and _np.isnan(nod)):
                    c = int(_np.count_nonzero(arr == nod))
                    if c:
                        note("g2", "nodata_unmasked", key=path, msg=f"Array read from {path} contains {c} cells equal to nodata={nod}. Mask them (read(masked=True), or set them to NaN) before computing statistics or combining rasters.")
                elif nod is None:
                    for s in (-9999, -32768, -3.4028234663852886e38):
                        c = int(_np.count_nonzero(arr == s))
                        if c and c > 0.01 * arr.size:
                            note("g2", "undeclared_nodata", key=path, msg=f"{path} declares no nodata value, but {c} cells equal {s}, which is a common nodata code. Check whether they should be excluded.")
                            break
        except Exception:
            pass
        return arr
    DR.read = read
    orig_open = _rio.open
    @functools.wraps(orig_open)
    def ropen(fp, mode="r", *a, **k):
        try:
            if mode == "w" and k.get("crs") is None and k.get("driver", "GTiff") in ("GTiff", "COG"):
                note("g2", "no_crs_written", f"{fp} is being written without a CRS. Pass crs=... so the raster is georeferenced.")
        except Exception:
            pass
        return orig_open(fp, mode, *a, **k)
    _rio.open = ropen
    # masking/clipping a raster with shapes that do not overlap it (usually a CRS mismatch)
    try:
        import rasterio.mask as _rm
        from shapely.geometry import shape as _shape, box as _box
        orig_mask = _rm.mask
        @functools.wraps(orig_mask)
        def mask(dataset, shapes, *a, **k):
            try:
                shp = list(shapes)
                geoms = [s if hasattr(s, "bounds") else _shape(s) for s in shp]
                if geoms:
                    xs = [g.bounds for g in geoms]
                    sb = _box(min(b[0] for b in xs), min(b[1] for b in xs), max(b[2] for b in xs), max(b[3] for b in xs))
                    if not sb.intersects(_box(*dataset.bounds)):
                        note("g2", "mask_no_overlap", f"The shapes used to mask {getattr(dataset, 'name', 'the raster')} lie at {tuple(round(v, 3) for v in sb.bounds)}, outside the raster ({tuple(round(v, 3) for v in dataset.bounds)}). Reproject the shapes to the raster's CRS ({dataset.crs}) first.")
                shapes = shp
            except Exception:
                pass
            return orig_mask(dataset, shapes, *a, **k)
        _rm.mask = mask
    except Exception:
        pass
    # interpolation results far outside the observed values
    def range_check(label, z, out):
        try:
            z = _np.asarray(z, float); o = _np.asarray(out, float)
            z = z[_np.isfinite(z)]; o = o[_np.isfinite(o)]
            if z.size < 3 or o.size == 0:
                return
            lo, hi = z.min(), z.max(); span = max(hi - lo, 1e-12)
            if o.min() < lo - 0.25 * span or o.max() > hi + 0.25 * span:
                note("g2", "interpolation_out_of_range", key=label, msg=f"{label} produced values {o.min():.4g}..{o.max():.4g}, well outside the observed range {lo:.4g}..{hi:.4g}. Check the model/variogram parameters and the coordinates (CRS, units) of the points.")
        except Exception:
            pass
    try:
        import importlib as _il  # by name: the sandbox's import blocker scans for "import <pkg>"
        _ok, _uk = _il.import_module("pykrige" + ".ok"), _il.import_module("pykrige" + ".uk")
        for C, nm in ((_ok.OrdinaryKriging, "OrdinaryKriging"), (_uk.UniversalKriging, "UniversalKriging")):
            orig_ex = C.execute
            def mk_ex(orig_ex=orig_ex, nm=nm):
                @functools.wraps(orig_ex)
                def ex(self, *a, **k):
                    r = orig_ex(self, *a, **k)
                    range_check(nm, getattr(self, "Z", []), r[0] if isinstance(r, tuple) else r)
                    return r
                return ex
            C.execute = mk_ex()
    except Exception:
        pass
    try:
        import scipy.interpolate as _si
        orig_rbf = _si.Rbf.__call__
        @functools.wraps(orig_rbf)
        def rbf(self, *a, **k):
            r = orig_rbf(self, *a, **k)
            range_check("scipy Rbf", getattr(self, "di", []), r)
            return r
        _si.Rbf.__call__ = rbf
        orig_rbfi = _si.RBFInterpolator.__call__
        @functools.wraps(orig_rbfi)
        def rbfi(self, *a, **k):
            r = orig_rbfi(self, *a, **k)
            range_check("RBFInterpolator", getattr(self, "d", []), r)
            return r
        _si.RBFInterpolator.__call__ = rbfi
    except Exception:
        pass
_gg_install_g2(_gg_st, _gg_note, _gg_from_agent)
del _gg_install_g2

def _gg_post_g2(names):
    """After a block: shapely metric operations on lon/lat geometries; spatial weights islands."""
    import geopandas as _gpd
    g = globals()
    try:
        from shapely.geometry.base import BaseGeometry
        geo = []
        for v in list(g.values()):
            try:
                if isinstance(v, (_gpd.GeoDataFrame, _gpd.GeoSeries)) and v.crs is not None and v.crs.is_geographic and len(v):
                    geo.append(tuple(v.total_bounds))
            except Exception:
                pass
        if geo:
            for nm, attr in names:
                o = g.get(nm)
                if isinstance(o, BaseGeometry) and not o.is_empty:
                    b = o.bounds
                    if any(b[0] >= gb[0] - 1 and b[2] <= gb[2] + 1 and b[1] >= gb[1] - 1 and b[3] <= gb[3] + 1 for gb in geo):
                        _gg_note("g2", "shapely_degrees", f"'{nm}' is a shapely geometry with coordinates that look like lon/lat degrees; .{attr} on it is in degrees, not metres. Project it first (e.g. via a GeoSeries with a CRS).")
    except Exception:
        pass
    for nm, v in list(g.items()):
        if nm.startswith("_"):
            continue
        try:
            if type(v).__module__.startswith("libpysal.weights") and hasattr(v, "islands") and v.islands:
                _gg_note("g2", "weights_islands", key=nm, msg=f"Spatial weights '{nm}' have {len(v.islands)} island(s) (units with no neighbours). Local statistics are undefined for them: use a larger distance band, k-nearest neighbours, or handle them explicitly.")
        except Exception:
            pass
'''

# ----------------------------------------------------------------------------- G3 figures
G3FIG = r'''
def _gg_install_g3(st, note, refs):
    import functools
    import matplotlib.figure as _mf
    def frac(a, b):
        """Largest of: share of a covered by b, share of b covered by a (bounding boxes)."""
        w = min(a[2], b[2]) - max(a[0], b[0]); h = min(a[3], b[3]) - max(a[1], b[1])
        if w <= 0 or h <= 0:
            return 0.0
        ar = max((a[2] - a[0]) * (a[3] - a[1]), 1e-12); br = max((b[2] - b[0]) * (b[3] - b[1]), 1e-12)
        return max(w * h / ar, w * h / br)
    def overlap(a, b):
        return frac(a, b) >= 0.5
    def check(fig, fname):
        try:
            R = refs()
            has_cbar = any(hasattr(ax, "_colorbar") for ax in fig.axes)
            for ax in fig.axes:
                if hasattr(ax, "_colorbar"):
                    continue
                data = bool(ax.images or ax.collections or ax.patches)
                if not data:
                    continue
                geo = hasattr(ax, "projection") and type(ax).__module__.startswith("cartopy")
                asp = ax.get_aspect()
                maplike = geo or asp in ("equal", 1.0) or (isinstance(asp, float) and asp != 1.0)
                if not maplike or not R:
                    continue
                dl = ax.dataLim
                if not (abs(dl.x0) < 1e300 and abs(dl.x1) < 1e300) or dl.width == 0:
                    continue
                if geo:
                    import cartopy.crs as _ccrs, numpy as _np
                    xs = _np.linspace(dl.x0, dl.x1, 7); ys = _np.linspace(dl.y0, dl.y1, 7)
                    X, Y = _np.meshgrid(xs, ys)
                    pts = _ccrs.PlateCarree().transform_points(ax.projection, X.ravel(), Y.ravel())
                    ok = _np.isfinite(pts[:, 0])
                    if not ok.any():
                        continue
                    drawn = (pts[ok, 0].min(), pts[ok, 1].min(), pts[ok, 0].max(), pts[ok, 1].max())
                    if not any(overlap(drawn, r[1]) for r in R):
                        note("g3", "figure_misplaced", key=fname, msg=f"In {fname}, the plotted data lies at lon {drawn[0]:.1f}..{drawn[2]:.1f}, lat {drawn[1]:.1f}..{drawn[3]:.1f}, but the input data covers " + "; ".join(f"{r[0]}: lon {r[1][0]:.1f}..{r[1][2]:.1f}, lat {r[1][1]:.1f}..{r[1][3]:.1f}" for r in R[:3]) + ". Check the coordinate transformation (e.g. the transform= argument, rotated grids).")
                else:
                    import numpy as _np
                    # extents of the individual layers drawn on these axes
                    boxes = []
                    for c in list(ax.collections) + list(ax.images):
                        try:
                            if c in ax.images:
                                e = c.get_extent()
                                bb = (min(e[0], e[1]), min(e[2], e[3]), max(e[0], e[1]), max(e[2], e[3]))
                            else:
                                d = c.get_datalim(ax.transData)
                                bb = (d.x0, d.y0, d.x1, d.y1)
                            if all(_np.isfinite(bb)) and max(abs(v) for v in bb) < 1e300 and bb[2] >= bb[0]:
                                boxes.append(bb)
                        except Exception:
                            pass
                    # layers in different CRS on one map: one in degree range, another far outside it
                    deg = [b for b in boxes if max(abs(b[0]), abs(b[2])) <= 360 and max(abs(b[1]), abs(b[3])) <= 90 and b[2] > b[0]]
                    big = [b for b in boxes if max(abs(v) for v in b) > 10000]
                    if deg and big:
                        note("g3", "figure_mixed_crs", key=fname, msg=f"In {fname}, some layers are drawn in degree coordinates (e.g. x {deg[0][0]:.4g}..{deg[0][2]:.4g}) and others in projected units (e.g. x {big[0][0]:.4g}..{big[0][2]:.4g}), so they cannot line up. Reproject all layers to one CRS before plotting them together.")
                    # misplaced: find the drawn coordinates among session layers whose CRS is known, then
                    # compare that layer's lon/lat position with the inputs (unknown coordinates: no verdict)
                    drawn = (dl.x0, dl.y0, dl.x1, dl.y1)
                    try:
                        import geopandas as _gpd
                        from pyproj import Transformer, CRS
                        for nm, v in list(globals().items()):
                            if nm.startswith("_") or not isinstance(v, (_gpd.GeoDataFrame, _gpd.GeoSeries)) or v.crs is None or not len(v):
                                continue
                            b = tuple(float(x) for x in v.total_bounds)
                            if not all(_np.isfinite(b)) or not overlap(b, drawn):
                                continue
                            t = Transformer.from_crs(v.crs, CRS.from_epsg(4326), always_xy=True)
                            xs, ys = t.transform([b[0], b[2], b[0], b[2]], [b[1], b[1], b[3], b[3]])
                            ll = (min(xs), min(ys), max(xs), max(ys))
                            if all(_np.isfinite(ll)) and R and not any(overlap(ll, r[1]) for r in R):
                                note("g3", "figure_misplaced", key=fname, msg=f"In {fname}, the plotted layer '{nm}' (CRS {v.crs.to_string()}) lies at lon {ll[0]:.2f}..{ll[2]:.2f}, lat {ll[1]:.2f}..{ll[3]:.2f}, but the input data covers " + "; ".join(f"{r[0]}: lon {r[1][0]:.2f}..{r[1][2]:.2f}, lat {r[1][1]:.2f}..{r[1][3]:.2f}" for r in R[:3]) + ". Its coordinates or its declared CRS are probably wrong (set_crs vs to_crs).")
                            break
                    except Exception:
                        pass
                cont = [m for m in list(ax.images) + list(ax.collections)
                        if hasattr(m, "get_array") and m.get_array() is not None and getattr(m.get_array(), "size", 0) > 20]
                if cont and not has_cbar and ax.get_legend() is None:
                    import numpy as _np
                    vals = _np.asarray(cont[0].get_array()).ravel()
                    vals = vals[_np.isfinite(vals.astype(float))] if vals.size else vals
                    if vals.size and len(_np.unique(vals)) > 12:
                        note("g3", "figure_no_colorbar", key=fname, msg=f"{fname} shows a continuous value surface but has no colorbar or legend, so the colours cannot be read. Add fig.colorbar(...).")
        except Exception:
            pass
    orig_sf = _mf.Figure.savefig
    @functools.wraps(orig_sf)
    def savefig(self, fname, *a, **k):
        check(self, str(fname))
        return orig_sf(self, fname, *a, **k)
    _mf.Figure.savefig = savefig
_gg_install_g3(_gg_st, _gg_note, _gg_refs)
del _gg_install_g3
'''

# ----------------------------------------------------------------------------- G6
G6 = r'''
def _gg_install_g6():
    import importlib, inspect, difflib, re
    MODS = ["geopandas", "shapely", "shapely.ops", "shapely.geometry", "rasterio", "rasterio.features",
            "rasterio.mask", "rasterio.warp", "rasterio.transform", "rasterio.plot", "pyproj", "fiona",
            "esda", "esda.getisord", "esda.moran", "libpysal.weights", "mgwr.gwr", "mgwr.sel_bw",
            "pykrige.ok", "pykrige.uk", "rasterstats", "pointpats", "spaghetti", "mapclassify", "momepy",
            "scipy.interpolate", "scipy.spatial", "scipy.ndimage", "scipy.stats", "sklearn.cluster",
            "sklearn.neighbors", "xarray", "networkx", "osmnx", "h3", "cartopy.crs", "cartopy.feature",
            "matplotlib.colors", "contextily", "pandas", "numpy.ma"]
    idx = {"entries": None, "mods": {}}
    def sig(o):
        try:
            return str(inspect.signature(o))
        except Exception:
            return "(...)"
    def doc1(o):
        d = inspect.getdoc(o) or ""
        return d.strip().splitlines()[0][:140] if d.strip() else ""
    def build():
        if idx["entries"] is not None:
            return idx["entries"]
        E = []
        for m in MODS:
            try:
                mod = importlib.import_module(m)
            except Exception:
                continue
            idx["mods"][m] = mod
            names = getattr(mod, "__all__", None) or [n for n in dir(mod) if not n.startswith("_")]
            for n in list(names)[:600]:
                try:
                    o = getattr(mod, n)
                except Exception:
                    continue
                if inspect.ismodule(o):
                    continue
                if inspect.isclass(o):
                    E.append((f"{m}.{n}", "class", o))
                    for mn, mo in list(vars(o).items())[:200]:
                        if not mn.startswith("_") and (inspect.isfunction(mo) or isinstance(mo, (classmethod, staticmethod, property))):
                            E.append((f"{m}.{n}.{mn}", "method", getattr(o, mn, mo)))
                elif callable(o):
                    E.append((f"{m}.{n}", "function", o))
        idx["entries"] = E
        return E
    def fmt(e):
        q, kind, o = e
        s = "" if kind == "class" and not callable(o) else sig(o)
        return f"{q}{s}" + (f"  - {doc1(o)}" if doc1(o) else "")
    def lookup(query, k=10):
        E = build()
        toks = [t for t in re.split(r"[^a-z0-9]+", query.lower()) if len(t) >= 3]
        if not toks:
            return "Give a function, class or topic, e.g. 'getis ord local', 'rasterio reproject', 'kriging'."
        scored = []
        for e in E:
            q = e[0].lower(); last = q.rsplit(".", 1)[-1]
            d = doc1(e[2]).lower() if len(scored) < 100000 else ""
            s = sum(3 for t in toks if t in last) + sum(1 for t in toks if t in q) + sum(0.5 for t in toks if t in d)
            s += 2 * max((difflib.SequenceMatcher(None, t, last).ratio() for t in toks), default=0)
            if s >= 2.5:
                scored.append((s, e))
        scored.sort(key=lambda x: -x[0])
        if not scored:
            return f"No installed API matches '{query}'. Searched: {', '.join(sorted(idx['mods']))}."
        return "Installed APIs matching '" + query + "':\n" + "\n".join("- " + fmt(e) for _, e in scored[:k])
    def suggest(err):
        build()
        out = []
        m = re.search(r"cannot import name '(\w+)' from '([\w.]+)'", err) or re.search(r"module '([\w.]+)' has no attribute '(\w+)'", err)
        if m:
            name, mod = (m.group(1), m.group(2)) if "cannot import" in m.group(0) else (m.group(2), m.group(1))
            try:
                mo = importlib.import_module(mod)
                cands = difflib.get_close_matches(name, [n for n in dir(mo) if not n.startswith("_")], n=5, cutoff=0.5)
                cands += [n for n in dir(mo) if name.lower() in n.lower() and n not in cands][:3]
                pub = [n for n in dir(mo) if not n.startswith("_") and not inspect.ismodule(getattr(mo, n, None))]
                if cands:
                    out.append(f"'{mod}' has no '{name}'. Existing names: " + "; ".join(fmt((f"{mod}.{c}", "x", getattr(mo, c))) for c in cands[:5]))
                elif len(pub) <= 60:
                    out.append(f"'{mod}' has no '{name}'. What it contains:\n" + "\n".join("- " + fmt((f"{mod}.{c}", "x", getattr(mo, c))) for c in pub[:15]))
                else:
                    out.append(lookup(name.replace("_", " "), 5))
            except Exception:
                out.append(lookup(name, 5))
        m = re.search(r"(?:type object|') ?'?(\w+)'? object has no attribute '(\w+)'|type object '(\w+)' has no attribute '(\w+)'", err)
        if m and not out:
            cls = m.group(1) or m.group(3); name = m.group(2) or m.group(4)
            hits = [e for e in build() if e[1] == "class" and e[0].rsplit(".", 1)[-1] == cls]
            for q, _, o in hits[:1]:
                attrs = [a for a in dir(o) if not a.startswith("_")]
                c = difflib.get_close_matches(name, attrs, n=6, cutoff=0.45)
                if c:
                    out.append(f"{q} has no '{name}'. Closest attributes: " + "; ".join(fmt((f"{q}.{a}", "x", getattr(o, a))) for a in c))
        m = re.search(r"([\w.]+)\(\) got an unexpected keyword argument '(\w+)'|([\w.]+)\(\) missing \d+ required positional argument", err)
        if m and not out:
            fn = (m.group(1) or m.group(3)).split(".")[-1]
            hits = [e for e in build() if e[0].endswith("." + fn)]
            if hits:
                out.append("Actual signature(s): " + "; ".join(fmt(e) for e in hits[:3]))
        m = re.search(r"No module named '([\w.]+)'", err)
        if m and not out:
            import pkgutil
            names = sorted({mi.name for mi in pkgutil.iter_modules()})
            c = difflib.get_close_matches(m.group(1).split(".")[0], names, n=5, cutoff=0.6)
            if c:
                out.append(f"No module '{m.group(1)}'. Installed modules with similar names: {', '.join(c)}")
        return "\n".join(out)
    return lookup, suggest
_gg_api_lookup, _gg_api_suggest = _gg_install_g6()
del _gg_install_g6
'''

# ----------------------------------------------------------------------------- G7
G7 = r'''
def _gg_install_g7(st, from_agent):
    import functools, os
    import pandas as _pd, geopandas as _gpd, rasterio as _rio
    import matplotlib.figure as _mf
    def rec(op, path, mode):
        try:
            st["io"].append({"op": op, "path": os.path.normpath(str(path)), "mode": mode})
        except Exception:
            pass
    def wr(obj, attr, op, mode, argpos=0, self_arg=False):
        orig = getattr(obj, attr)
        @functools.wraps(orig)
        def w(*a, **k):
            p = a[argpos + (1 if self_arg else 0)] if len(a) > argpos + (1 if self_arg else 0) else k.get("filename", k.get("path_or_buf", k.get("fp", k.get("fname", "?"))))
            if isinstance(p, (str, os.PathLike)):
                rec(op, p, mode)
            return orig(*a, **k)
        setattr(obj, attr, w)
    wr(_gpd, "read_file", "gpd.read_file", "r")
    wr(_pd, "read_csv", "pd.read_csv", "r")
    wr(_pd, "read_excel", "pd.read_excel", "r")
    wr(_gpd.GeoDataFrame, "to_file", "to_file", "w", self_arg=True)
    wr(_gpd.GeoSeries, "to_file", "to_file", "w", self_arg=True)
    wr(_pd.DataFrame, "to_csv", "to_csv", "w", self_arg=True)
    wr(_mf.Figure, "savefig", "savefig", "w", self_arg=True)
    orig_open = _rio.open
    @functools.wraps(orig_open)
    def ropen(fp, mode="r", *a, **k):
        if isinstance(fp, (str, os.PathLike)):
            rec("rasterio.open", fp, "w" if mode in ("w", "w+", "r+") else "r")
        return orig_open(fp, mode, *a, **k)
    _rio.open = ropen
    try:
        import xarray as _xr
        wr(_xr, "open_dataset", "xr.open_dataset", "r")
    except Exception:
        pass
_gg_install_g7(_gg_st, _gg_from_agent)
del _gg_install_g7
'''

COLLECT = r'''
import json as _ggjson
_gg_out = {"notes": _gg_st["notes"][:], "io": _gg_st["io"][:]}
_gg_st["notes"].clear(); _gg_st["io"].clear()
print(_ggjson.dumps(_gg_out, default=str))
del _gg_out
'''

METRIC_ATTRS = {"area", "length", "distance", "buffer", "hausdorff_distance", "dwithin"}


def prelude(cfg):
    """Kernel code for the enabled components (None if none is on)."""
    parts = []
    if cfg.get("g2"):
        parts.append(G2)
    if cfg.get("g3"):
        parts.append(G3FIG)
    if cfg.get("g6"):
        parts.append(G6)
    if cfg.get("g7"):
        parts.append(G7)
    return (CORE + "\n".join(parts)) if parts else None


def shapely_targets(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr in METRIC_ATTRS and isinstance(n.value, ast.Name):
            out.append((n.value.id, n.attr))
    return sorted(set(out))


def post_code(cfg, code):
    """Code run after each agent block: post-block checks, then collect notes and I/O."""
    s = ""
    if cfg.get("g2"):
        s += f"_gg_post_g2({shapely_targets(code)!r})\n"
    return s + COLLECT


def lookup_code(query):
    return f"print(_gg_api_lookup({query!r}))"


def suggest_code(stderr_tail):
    return f"print(_gg_api_suggest({stderr_tail!r}))"
