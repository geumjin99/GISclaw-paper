"""G3 GIS deliverable validation, run inside the container when the agent calls finish.

Generic rules only (nothing per task): geographic files must have a CRS, geometries must be
valid and non-empty, results should lie within the extent of the inputs, rasters must not
be constant or empty, figures must not be blank, tables must not be empty.
Returns a JSON list of {"file", "level": "error"|"warn", "msg"}.
"""

VALIDATE_CODE = r'''
def _gg_validate(_pred="pred_results", _data="dataset"):
    import os, json, warnings
    import numpy as _np
    warnings.simplefilter("ignore")
    issues = []
    def add(f, lvl, m):
        issues.append({"file": f, "level": lvl, "msg": m})
    # union of input extents in lon/lat (for the "outside the study area" check)
    ext = None
    try:
        import geopandas as _gpd, rasterio as _rio
        from pyproj import Transformer, CRS
        boxes = []
        for r, ds, fs in os.walk(_data):
            for f in fs:
                p = os.path.join(r, f); e = os.path.splitext(f)[1].lower()
                try:
                    if e in (".shp", ".geojson", ".gpkg"):
                        g = _gpd.read_file(p, rows=5000)
                        if g.crs is not None and len(g):
                            b = g.total_bounds; c = g.crs
                        else:
                            continue
                    elif e in (".tif", ".tiff"):
                        with _rio.open(p) as s:
                            if s.crs is None:
                                continue
                            b = s.bounds; c = s.crs
                    else:
                        continue
                    t = Transformer.from_crs(c, CRS.from_epsg(4326), always_xy=True)
                    xs, ys = t.transform([b[0], b[2], b[0], b[2]], [b[1], b[1], b[3], b[3]])
                    boxes.append((min(xs), min(ys), max(xs), max(ys)))
                except Exception:
                    pass
        if boxes:
            ext = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
    except Exception:
        pass
    def outside(b, c):
        if ext is None or c is None:
            return False
        try:
            from pyproj import Transformer, CRS
            t = Transformer.from_crs(c, CRS.from_epsg(4326), always_xy=True)
            xs, ys = t.transform([b[0], b[2]], [b[1], b[3]])
            w, h = ext[2] - ext[0], ext[3] - ext[1]
            pad_x, pad_y = max(w * 0.5, 0.05), max(h * 0.5, 0.05)
            return xs[1] < ext[0] - pad_x or xs[0] > ext[2] + pad_x or ys[1] < ext[1] - pad_y or ys[0] > ext[3] + pad_y
        except Exception:
            return False
    if not os.path.isdir(_pred):
        return issues
    for f in sorted(os.listdir(_pred)):
        p = os.path.join(_pred, f); e = os.path.splitext(f)[1].lower()
        if f.startswith(".") or os.path.isdir(p) or e in (".dbf", ".shx", ".prj", ".cpg", ".qmd", ".xml", ".sbn", ".sbx"):
            continue
        try:
            if os.path.getsize(p) == 0:
                add(f, "error", "file is empty (0 bytes)"); continue
            if e in (".shp", ".geojson", ".gpkg"):
                import geopandas as _gpd
                g = _gpd.read_file(p)
                if len(g) == 0:
                    add(f, "error", "has no features")
                    continue
                if g.crs is None:
                    add(f, "error", "has no CRS; set it before saving (set_crs if the coordinates are already in a known CRS)")
                elif g.crs.is_geographic:
                    b = g.total_bounds
                    if max(abs(b[0]), abs(b[2])) > 180 or max(abs(b[1]), abs(b[3])) > 90:
                        add(f, "error", "is labelled as lon/lat degrees but its coordinates are far outside +-180/+-90: "
                            "projected coordinates were probably saved without their CRS (GeoJSON assumes WGS84 when none is set)")
                inv = int((~g.geometry.is_valid).sum()); emp = int((g.geometry.is_empty | g.geometry.isna()).sum())
                if emp == len(g):
                    add(f, "error", "all geometries are empty")
                elif emp:
                    add(f, "warn", f"{emp} of {len(g)} geometries are empty")
                if inv:
                    add(f, "warn", f"{inv} of {len(g)} geometries are invalid (shapely make_valid or buffer(0) can repair them)")
                if g.crs is not None and outside(g.total_bounds, g.crs):
                    add(f, "error", "lies far outside the extent of the input data; check the CRS / coordinate order")
                cols = [c for c in g.columns if c != g.geometry.name]
                allnull = [c for c in cols if g[c].isna().all()]
                if allnull:
                    add(f, "warn", f"columns entirely empty: {allnull[:8]}")
            elif e in (".tif", ".tiff"):
                import rasterio as _rio
                with _rio.open(p) as s:
                    if s.crs is None:
                        add(f, "error", "has no CRS; pass crs=... when writing")
                    a = s.read(1, masked=True)
                    v = a.compressed() if hasattr(a, "compressed") else _np.asarray(a).ravel()
                    v = v[_np.isfinite(v.astype(float))] if v.size else v
                    if v.size == 0:
                        add(f, "error", "contains no valid cells (all nodata/NaN)")
                    elif float(v.min()) == float(v.max()):
                        add(f, "error", f"every valid cell has the same value ({float(v.min()):.4g})")
                    if s.crs is not None and outside(s.bounds, s.crs):
                        add(f, "error", "lies far outside the extent of the input data; check the transform / CRS")
            elif e in (".png", ".jpg", ".jpeg"):
                import matplotlib.image as _mi
                im = _mi.imread(p)
                a = _np.asarray(im, dtype=float)
                if a.ndim == 3:
                    a = a[..., :3].mean(axis=2)
                if a.size and float(a.std()) < 1e-3:
                    add(f, "error", "image is blank (a single colour)")
            elif e == ".csv":
                import pandas as _pd
                d = _pd.read_csv(p)
                if len(d) == 0:
                    add(f, "error", "table has a header but no rows")
                else:
                    num = d.select_dtypes("number")
                    dead = [c for c in num.columns if num[c].isna().all()]
                    if dead:
                        add(f, "warn", f"numeric columns entirely empty: {dead[:8]}")
        except Exception as ex:
            add(f, "error", f"could not be opened ({type(ex).__name__}: {str(ex)[:150]})")
    return issues
import json as _ggj
print(_ggj.dumps(_gg_validate()))
del _gg_validate
'''
