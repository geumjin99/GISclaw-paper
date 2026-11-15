"""G1 data profile card: run once at the start, inside the container, on the harness channel.

Generic: describes whatever is in dataset/ (vector, raster, table) and checks the layers
against each other (CRS agreement, extent overlap). Nothing task-specific.
"""

PROFILE_CODE = r'''
def _gg_profile(_root="dataset", _limit=6500):
    import os, warnings, math
    import numpy as _np, pandas as _pd
    warnings.simplefilter("ignore")
    out, layers = [], []
    VEC = (".shp", ".geojson", ".gpkg", ".kml", ".gml")
    RAS = (".tif", ".tiff", ".img", ".asc", ".vrt")
    TAB = (".csv", ".xlsx", ".xls")
    def unit(crs):
        try:
            return crs.axis_info[0].unit_name
        except Exception:
            return "?"
    def crs_s(crs):
        if crs is None:
            return "NONE (no CRS defined)"
        try:
            e = crs.to_epsg()
        except Exception:
            e = None
        kind = "geographic" if crs.is_geographic else "projected"
        name = crs.name if hasattr(crs, "name") else str(crs)
        return f"{('EPSG:%s ' % e) if e else ''}{name} [{kind}, units: {unit(crs)}]"
    def fld(s):
        n = len(s); miss = int(s.isna().sum())
        d = str(s.dtype)
        if _pd.api.types.is_numeric_dtype(s) and s.notna().any():
            return f"{s.name} ({d}, missing {miss}/{n}, range {s.min():.4g}..{s.max():.4g})"
        u = s.dropna().unique()
        ex = ", ".join(map(lambda x: str(x)[:20], u[:3]))
        return f"{s.name} ({d}, missing {miss}/{n}, {len(u)} distinct, e.g. {ex})"
    files = []
    for r, ds, fs in os.walk(_root):
        ds.sort()
        for f in sorted(fs):
            files.append(os.path.join(r, f))
    for p in files:
        ext = os.path.splitext(p)[1].lower()
        mb = os.path.getsize(p) / 1048576
        try:
            if ext in VEC or (ext == ".json" and mb < 200):
                import geopandas as _gpd
                if ext == ".json":
                    with open(p) as fh:
                        head = fh.read(2000)
                    if '"FeatureCollection"' not in head and '"Feature"' not in head:
                        continue
                g = _gpd.read_file(p, rows=20000) if mb > 150 else _gpd.read_file(p)
                sampled = " (first 20000 rows)" if mb > 150 else ""
                cols = [c for c in g.columns if c != g.geometry.name]
                inval = int((~g.geometry.is_valid).sum()) if len(g) else 0
                empty = int(g.geometry.is_empty.sum() + g.geometry.isna().sum()) if len(g) else 0
                out.append(f"- {p}: vector, {len(g)} features{sampled}, { {k: int(v) for k, v in g.geom_type.value_counts().items()} }")
                out.append(f"    CRS: {crs_s(g.crs)}; bounds {[round(float(b), 4) for b in g.total_bounds]}"
                           + (f"; {inval} invalid geometries" if inval else "") + (f"; {empty} empty/missing geometries" if empty else ""))
                if cols:
                    out.append("    fields: " + "; ".join(fld(g[c]) for c in cols[:25]) + (f"; (+{len(cols) - 25} more)" if len(cols) > 25 else ""))
                layers.append((p, g.crs, tuple(g.total_bounds) if len(g) else None))
            elif ext in RAS:
                import rasterio as _rio
                with _rio.open(p) as s:
                    step = max(1, int(math.ceil(max(s.width, s.height) / 1500)))
                    a = s.read(1, out_shape=(max(1, s.height // step), max(1, s.width // step)))
                    nod = s.nodata
                    m = _np.isfinite(a.astype(float))
                    if nod is not None:
                        m &= a != nod
                    v = a[m]
                    sent = []
                    if nod is None and a.size:
                        for c in (-9999, -32768, -3.4028234663852886e38, 255, 65535):
                            frac = float((a == c).mean())
                            if frac > 0.01 and (v.size == 0 or c in (v.min(), v.max())):
                                sent.append(f"{c} ({frac:.0%} of cells)")
                    out.append(f"- {p}: raster {s.width}x{s.height}, {s.count} band(s), {s.dtypes[0]}, resolution {tuple(round(r, 6) for r in s.res)}")
                    out.append(f"    CRS: {crs_s(s.crs)}; bounds {[round(float(b), 4) for b in s.bounds]}; nodata={nod}"
                               + (f" ({1 - m.mean():.0%} of cells are nodata)" if a.size else "")
                               + (f"; band 1 valid range {v.min():.4g}..{v.max():.4g}, mean {v.mean():.4g}" if v.size else "; no valid cells")
                               + (" (sampled)" if step > 1 else ""))
                    if sent:
                        out.append(f"    note: no nodata value is declared, but these values are frequent and sit at the edge of the range: {', '.join(sent)}")
                    layers.append((p, s.crs, tuple(s.bounds)))
            elif ext in TAB:
                df = _pd.read_csv(p, nrows=200000) if ext == ".csv" else _pd.read_excel(p)
                out.append(f"- {p}: table, {len(df)} rows x {df.shape[1]} columns")
                out.append("    fields: " + "; ".join(fld(df[c]) for c in df.columns[:25]) + (f"; (+{df.shape[1] - 25} more)" if df.shape[1] > 25 else ""))
                xy = [c for c in df.columns if str(c).lower() in ("lat", "latitude", "lon", "lng", "long", "longitude", "x", "y", "easting", "northing")]
                if xy:
                    out.append(f"    coordinate-like columns: {xy} (no CRS is stored in a table; infer it from the value ranges and the description)")
            elif ext in (".nc", ".nc4"):
                import xarray as _xr
                with _xr.open_dataset(p, decode_times=False) as d:
                    out.append(f"- {p}: NetCDF, dims {dict(d.sizes)}")
                    gm = [v for v in d.variables if "grid_mapping_name" in d[v].attrs]
                    for v in list(d.data_vars)[:15]:
                        a = d[v]
                        if v in gm:
                            out.append(f"    {v}: grid mapping '{a.attrs.get('grid_mapping_name')}' "
                                       + ", ".join(f"{k}={a.attrs[k]}" for k in a.attrs if k != "grid_mapping_name")[:200])
                            continue
                        rng = ""
                        try:
                            vv = _np.asarray(a.values, dtype=float)
                            f = vv[_np.isfinite(vv)]
                            if f.size:
                                rng = f", range {f.min():.4g}..{f.max():.4g}" + (f", {int((f < 0).sum())} negative values" if (f < 0).any() else "")
                        except Exception:
                            pass
                        out.append(f"    {v}{tuple(a.dims)} units={a.attrs.get('units', '?')}{rng}"
                                   + (f", grid_mapping={a.attrs['grid_mapping']}" if 'grid_mapping' in a.attrs else ""))
                    try:
                        bb = _gg_nc_lonlat(p) if "_gg_nc_lonlat" in globals() else None
                    except Exception:
                        bb = None
                    if bb:
                        out.append(f"    geographic extent (lon/lat): lon {bb[0]:.2f}..{bb[2]:.2f}, lat {bb[1]:.2f}..{bb[3]:.2f}")
            elif ext in (".prj", ".dbf", ".shx", ".cpg", ".sbn", ".sbx", ".xml", ".qmd", ".aux", ".ovr", ".tfw"):
                continue
            else:
                out.append(f"- {p}: {ext or 'no extension'}, {mb:.1f} MB (not profiled)")
        except Exception as e:
            out.append(f"- {p}: could not be read automatically ({type(e).__name__}: {str(e)[:120]})")
    # cross-layer checks
    cross = []
    crss = {}
    for p, c, b in layers:
        crss.setdefault(crs_s(c), []).append(os.path.basename(p))
    if len(crss) > 1:
        cross.append("Layers use different CRS: " + "; ".join(f"{k}: {', '.join(v)}" for k, v in crss.items()))
    try:
        from pyproj import Transformer, CRS
        boxes = []
        for p, c, b in layers:
            if b is None or c is None:
                continue
            t = Transformer.from_crs(c, CRS.from_epsg(4326), always_xy=True)
            xs, ys = t.transform([b[0], b[2], b[0], b[2]], [b[1], b[1], b[3], b[3]])
            boxes.append((os.path.basename(p), (min(xs), min(ys), max(xs), max(ys))))
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                (n1, a), (n2, b2) = boxes[i], boxes[j]
                if a[2] < b2[0] or b2[2] < a[0] or a[3] < b2[1] or b2[3] < a[1]:
                    cross.append(f"{n1} and {n2} do not overlap in space (after converting both extents to lon/lat).")
    except Exception:
        pass
    if cross:
        out.append("Cross-layer checks:\n  " + "\n  ".join(cross))
    s = "\n".join(out) or "(no readable data files)"
    if len(s) > _limit:
        s = s[:_limit] + f"\n... ({len(s) - _limit} characters omitted)"
    return s
print(_gg_profile())
del _gg_profile
'''
