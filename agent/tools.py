"""Tool implementations on top of the container sandbox.

load_vector / load_raster run as ordinary agent code (they are part of the code history
and of the scoring replay). inspect runs on the harness channel: it only reads. Tables are
shown without the geometry column (geometry types and bounds are summarised instead), so
one large polygon cannot flood the context.
"""
import os

from observe import cap, format_exec

TYPE_MAP = {'.geojson': 'Vector', '.shp': 'Vector', '.gpkg': 'Vector', '.kml': 'Vector', '.gml': 'Vector',
            '.tif': 'Raster', '.tiff': 'Raster', '.img': 'Raster', '.asc': 'Raster', '.nc': 'NetCDF',
            '.csv': 'Table', '.xlsx': 'Table', '.xls': 'Table', '.txt': 'Text', '.json': 'JSON',
            '.lyrx': 'ArcGIS layer file (not readable here)', '.ppkx': 'ArcGIS package (not readable here)'}


def _pyname(s):
    return s.isidentifier() and not s.startswith("_")


def list_files(dataset_dir, rel="dataset"):
    lines = []
    for root, dirs, files in os.walk(dataset_dir):
        dirs.sort()
        for f in sorted(files):
            if f.startswith("."):
                continue
            p = os.path.join(root, f)
            size = os.path.getsize(p)
            ext = os.path.splitext(f)[1].lower()
            s = f"{size / 1048576:.1f} MB" if size > 1048576 else f"{size / 1024:.1f} KB"
            shown = os.path.join(rel, os.path.relpath(p, dataset_dir))
            lines.append(f"  - {shown} [{TYPE_MAP.get(ext, 'Other')}, {s}]")
    if not lines:
        return f"{rel}/ is empty"
    return cap(f"{rel}/ ({len(lines)} files):\n" + "\n".join(lines), 4000)


VECTOR_CODE = r'''{v} = gpd.read_file({path!r})
print(f"Rows: {{len({v})}}")
print(f"CRS: {{{v}.crs.to_string() if {v}.crs is not None else None}}")
print(f"Geometry types: {{ {{k: int(c) for k, c in {v}.geom_type.value_counts().items()}} }}")
print(f"Bounds (minx, miny, maxx, maxy): {{[round(float(_b), 4) for _b in {v}.total_bounds]}}")
_cols = [c for c in {v}.columns if c != {v}.geometry.name]
print(f"Columns ({{len(_cols)}}): {{_cols}}")
import pandas as _pd
with _pd.option_context('display.max_columns', 30, 'display.width', 200, 'display.max_colwidth', 40):
    print({v}[_cols].head(3).to_string())
'''

RASTER_CODE = r'''_src = rasterio.open({path!r})
{v} = _src.read().astype(float) if _src.count > 1 else _src.read(1).astype(float)
{v}_meta = dict(crs=_src.crs, transform=_src.transform, width=_src.width, height=_src.height,
               nodata=_src.nodata, dtype=_src.dtypes[0], count=_src.count, bounds=_src.bounds, res=_src.res)
if _src.nodata is not None:
    {v}[{v} == _src.nodata] = np.nan
print(f"Size: {{_src.width}} x {{_src.height}}, bands: {{_src.count}}, dtype: {{_src.dtypes[0]}}")
print(f"CRS: {{_src.crs.to_string() if _src.crs else None}}, resolution: {{_src.res}}")
print(f"Bounds: {{tuple(round(float(_b), 4) for _b in _src.bounds)}}")
print(f"NoData: {{_src.nodata}} (replaced by NaN in {v})")
_valid = {v}[~np.isnan({v})]
print(f"Valid values: {{_valid.size}} of {{{v}.size}}" + (f", min={{_valid.min():.4g}}, max={{_valid.max():.4g}}, mean={{_valid.mean():.4g}}" if _valid.size else ""))
print("Variables: {v} = float array" + (" (bands, rows, cols)" if _src.count > 1 else " (rows, cols)") + ", {v}_meta = dict(crs, transform, nodata, ...)")
_src.close()
'''

INSPECT_CODE = r'''
def _gg_inspect(_name):
    import numpy as _np, pandas as _pd
    _ns = globals()
    if _name not in _ns:
        _vs = [k for k in _ns if not k.startswith('_') and not callable(_ns[k]) and type(_ns[k]).__name__ != 'module']
        return f"Variable '{_name}' does not exist. Variables: {_vs[:40]}"
    o = _ns[_name]
    t = type(o).__name__
    try:
        import geopandas as _gpd
        if isinstance(o, _gpd.GeoDataFrame):
            cols = [c for c in o.columns if c != o.geometry.name]
            s = [f"GeoDataFrame: {len(o)} rows, CRS={o.crs.to_string() if o.crs is not None else None}",
                 f"Geometry types: { {k: int(c) for k, c in o.geom_type.value_counts().items()} }",
                 f"Bounds: {[round(float(b), 4) for b in o.total_bounds] if len(o) else None}",
                 f"Columns: {cols}"]
            with _pd.option_context('display.max_columns', 30, 'display.width', 200, 'display.max_colwidth', 40):
                s.append(o[cols].head(5).to_string())
                num = o[cols].select_dtypes('number')
                if num.shape[1]:
                    s.append(num.describe().T[['count', 'mean', 'min', 'max']].to_string())
            return "\n".join(s)
        if isinstance(o, _gpd.GeoSeries):
            return f"GeoSeries: {len(o)} geometries, CRS={o.crs}, types={ {k: int(c) for k, c in o.geom_type.value_counts().items()} }, bounds={list(o.total_bounds)}"
    except Exception as e:
        pass
    if isinstance(o, _pd.DataFrame):
        with _pd.option_context('display.max_columns', 30, 'display.width', 200, 'display.max_colwidth', 40):
            s = [f"DataFrame: {o.shape[0]} rows x {o.shape[1]} columns", f"Columns: {list(o.columns)[:60]}",
                 o.head(5).to_string()]
            num = o.select_dtypes('number')
            if num.shape[1]:
                s.append(num.describe().T[['count', 'mean', 'min', 'max']].to_string())
        return "\n".join(s)
    if isinstance(o, _np.ndarray):
        s = f"ndarray: shape={o.shape}, dtype={o.dtype}"
        if o.size and _np.issubdtype(o.dtype, _np.number):
            f = o[_np.isfinite(o)] if _np.issubdtype(o.dtype, _np.floating) else o
            s += f", finite={f.size}/{o.size}"
            if f.size:
                s += f", min={f.min():.4g}, max={f.max():.4g}, mean={f.mean():.4g}"
        return s
    if hasattr(o, 'geom_type') and hasattr(o, 'bounds'):
        return f"shapely {o.geom_type}: bounds={tuple(round(b, 4) for b in o.bounds)}, area={o.area:.6g}, length={o.length:.6g} (in the units of its coordinates)"
    r = repr(o)
    return f"{t}: " + (r[:1500] + f"... ({len(r)} chars)" if len(r) > 1500 else r)
print(_gg_inspect(__NAME__))
del _gg_inspect
'''


class Toolkit:
    def __init__(self, sandbox, dataset_dir, tools):
        self.sb = sandbox
        self.dataset_dir = dataset_dir
        self.tools = list(tools)

    def run(self, name, args):
        """Returns (observation, exec_result_or_None, meta)."""
        if name == "list_files":
            pred = os.path.join(self.sb.work_dir, "pred_results")
            outs = list_files(pred, rel="pred_results") if os.path.isdir(pred) else "pred_results/ is empty"
            return list_files(self.dataset_dir) + "\n" + outs, None, {}
        if name in ("load_vector", "load_raster"):
            v, path = args.get("var_name", ""), args.get("path", "")
            if not _pyname(v):
                return f"\u274c var_name must be a valid Python name not starting with '_' (got {v!r}).", None, {}
            code = (VECTOR_CODE if name == "load_vector" else RASTER_CODE).format(v=v, path=path)
            r = self.sb.execute(code)
            return format_exec(r), r, {"code": code, "loaded": [v] + ([v + "_meta"] if name == "load_raster" else [])}
        if name == "inspect":
            v = args.get("var_name", "")
            r = self.sb.system_exec(INSPECT_CODE.replace('__NAME__', repr(v)), timeout=60)
            if r is None:
                return "\u274c inspect did not answer (the environment may be busy or restarting).", None, {}
            return cap((r["stdout"] or r["stderr"] or "(no output)").strip(), 4000), None, {}
        if name == "lookup_api":
            import gis_runtime
            r = self.sb.system_exec(gis_runtime.lookup_code(args.get("query", "")), timeout=180)
            if r is None:
                return "\u274c lookup_api did not answer.", None, {}
            return cap((r["stdout"] or r["stderr"] or "(no output)").strip(), 4000), None, {}
        if name == "execute":
            code = args.get("code", "")
            r = self.sb.execute(code)
            return format_exec(r), r, {"code": code}
        raise KeyError(name)
