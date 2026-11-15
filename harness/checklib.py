"""Shared helpers for per-task correctness checks (benchmark/specs/<id>/checks.py).

A task's checks.py defines
    def checks(ctx) -> list[dict]
where ctx is a Ctx (below). Each returned dict comes from `result(...)`:
    {"name", "passed", "value", "threshold", "required", "note"}
A run is *analytically correct* when it has at least one required check and every check
with required=True passes. Checks with required=False are reported but do not affect the
verdict. A required check that cannot be decided from the captured evidence is returned by
`unverifiable(...)`: it does not pass. A crash of the checks themselves gives no verdict
(`scorer_error_result`, `verdict`).

Two rules apply to every task on top of its own checks (section "uniform rules" below):
geographic deliverable files must carry a CRS, and output files that merely copy an input
dataset are hidden from the checks.

Design rules for writing checks:
  * compare against the gold run (ctx.gold_*) rather than hard-coding numbers when possible;
  * tolerances must accept legitimate method variation (different libraries, resolution,
    classification breaks) and reject wrong analysis (wrong data, wrong operation, wrong area);
  * a check must never crash: missing inputs -> passed=False with a note.
"""
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

# Scoring must not depend on network access: with PROJ_NETWORK=ON and no network, pyproj
# returns inf for datum shifts that want a remote grid. Offline grids only, so that the
# result is the same on every machine.
os.environ["PROJ_NETWORK"] = "OFF"
# Checks run outside the container. When the environment's PROJ_DATA points to a proj.db
# older than the one rasterio was built with, CRS.from_epsg / reproject fail; rasterio (only)
# is pointed at the proj.db shipped in its wheel.
try:
    import importlib.util as _ilu
    _spec = _ilu.find_spec("rasterio")
    if _spec and _spec.origin:
        _pd0 = os.path.join(os.path.dirname(_spec.origin), "proj_data")
        if os.path.isdir(_pd0):
            os.environ["PROJ_DATA"] = _pd0  # must be set before rasterio/GDAL first opens a file
except Exception:
    pass
try:
    import rasterio as _rio
    from rasterio._env import set_proj_data_search_path as _set_proj
    _pd_dir = os.path.join(os.path.dirname(_rio.__file__), "proj_data")
    if os.path.isdir(_pd_dir):
        _set_proj(_pd_dir)
except Exception:
    pass


# ----------------------------------------------------------------------------- context
class Ctx:
    def __init__(self, pred_dir, gold_dir, inputs_dir, pred_snap=None, gold_snap=None, code_path=None,
                 stdout_path=None):
        self.pred_dir, self.gold_dir, self.inputs_dir = pred_dir, gold_dir, inputs_dir
        self.pred_snap, self.gold_snap = pred_snap, gold_snap
        self.code_path = code_path
        self.stdout_path = stdout_path

    @property
    def stdout(self):
        """Full stdout of the evaluated run (all executed blocks, in order), or ''."""
        try:
            return open(self.stdout_path, errors="ignore").read() if self.stdout_path else ""
        except OSError:
            return ""

    def code_text(self):
        """Source of the evaluated solution (the code the run executed, or the reference
        solution)."""
        try:
            return open(self.code_path, errors="ignore").read() if self.code_path else ""
        except OSError:
            return ""

    # files ------------------------------------------------------------------
    def pred(self, name):
        return os.path.join(self.pred_dir, name)

    def gold(self, name):
        return os.path.join(self.gold_dir, name)

    def pred_files(self, pattern="*"):
        return sorted(os.path.basename(p) for p in glob.glob(os.path.join(self.pred_dir, pattern)))

    def nonempty_pred_files(self, pattern="*"):
        """pred_files without 0-byte files and empty folders: an empty file is not a result."""
        def full(p):
            if os.path.isdir(p):
                return any(full(os.path.join(p, x)) for x in os.listdir(p))
            return os.path.getsize(p) > 0
        return [n for n in self.pred_files(pattern) if full(self.pred(n))]

    # snapshots ---------------------------------------------------------------
    def objects(self, side="pred", kind=None, include_files=True):
        """All captured objects across snapshots: list of (tag, meta, loader)."""
        root = self.pred_snap if side == "pred" else self.gold_snap
        out = []
        if not root or not os.path.isdir(root):
            return out
        for tag in sorted(os.listdir(root)):
            p = os.path.join(root, tag, "index.json")
            if not os.path.exists(p):
                continue
            for m in json.load(open(p)):
                if kind and m.get("kind") != kind or m.get("skipped") or m.get("kind") == "error":
                    continue
                out.append((tag, m, _loader(os.path.join(root, tag), m)))
        if include_files:
            out.extend(o for o in self.file_objects(side) if not kind or o[1]["kind"] == kind)
        return out

    def file_objects(self, side="pred"):
        """Output files of the run as candidate objects (tag 'file'): tables (.csv/.xlsx),
        vectors (.shp/.geojson/.gpkg) and rasters (.tif/.tiff, band 1, nodata -> NaN)."""
        d = self.pred_dir if side == "pred" else self.gold_dir
        out = []
        if not d or not os.path.isdir(d):
            return out
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            ext = f.lower().rsplit(".", 1)[-1] if "." in f else ""
            if ext in ("csv", "txt") and os.path.getsize(p) < 500 * 1024 * 1024:
                out.append(("file", {"name": f, "kind": "df", "source": "file"}, _file_loader(p, "csv")))
            elif ext in ("xlsx", "xls"):
                out.append(("file", {"name": f, "kind": "df", "source": "file"}, _file_loader(p, "xlsx")))
            elif ext in ("shp", "geojson", "gpkg", "json") and not (ext == "json" and os.path.getsize(p) > 2e8):
                out.append(("file", {"name": f, "kind": "gdf", "source": "file"}, _file_loader(p, "vector")))
            elif ext in ("tif", "tiff"):
                m = {"name": f, "kind": "array", "source": "file"}
                try:  # shape (band 1), so helpers that filter 2-D arrays by shape see output rasters
                    import rasterio
                    with rasterio.open(p) as r:
                        m["shape"] = [r.height, r.width]
                except Exception:
                    pass
                out.append(("file", m, _file_loader(p, "raster")))
        # rows for the size filters used by helpers
        for _, m, load in out:
            if m["kind"] in ("df", "gdf"):
                try:
                    m["rows"] = len(load())
                except Exception:
                    m["rows"] = -1
        return out


def _file_loader(p, how):
    cache = {}

    def load():
        if "v" in cache:
            return cache["v"]
        if how == "csv":
            import pandas as pd
            v = pd.read_csv(p, sep=None, engine="python")
        elif how == "xlsx":
            import pandas as pd
            v = pd.read_excel(p)
        elif how == "vector":
            import geopandas as gpd
            v = gpd.read_file(p)
        else:
            v = read_raster(p)[0]
        cache["v"] = v
        return v
    return load


def _loader(d, m):
    def load():
        k = m["kind"]
        if k == "gdf":
            import geopandas as gpd
            return gpd.read_parquet(os.path.join(d, m["name"] + ".parquet"))
        if k in ("df", "series"):
            import pandas as pd
            return pd.read_parquet(os.path.join(d, m["name"] + ".parquet"))
        if k == "array":
            return np.load(os.path.join(d, m["name"] + ".npy"))
        return m
    return load


# ----------------------------------------------------------------------------- results
def result(name, passed, value=None, threshold=None, required=True, note="", kind=None):
    """One check. `kind="figure"` marks a check of how a figure presents the data (colour direction,
    raster orientation/registration on the map, colour-scale range, readability): such checks
    are never required; they are reported separately from the verdict."""
    if kind == "figure":
        required = False
    def clean(x):
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return x
    return {"name": name, "passed": bool(passed), "value": clean(value), "threshold": clean(threshold),
            "required": bool(required), "note": note, **({"kind": kind} if kind else {})}


# A crash inside a task's checks (or the uniform rules) is a SCORER error, not a wrong answer:
# the run gets no verdict (None). The analysis scripts count such a run as not correct.
SCORER_ERROR_NAME = "checks.py crashed"


def scorer_error_result(note=""):
    r = result(SCORER_ERROR_NAME, False, required=True, note=note)
    r["scorer_error"] = True
    return r


def has_scorer_error(checks):
    """True when the check list records a scorer crash."""
    return any(c.get("scorer_error") for c in checks or [])


def verdict(checks):
    """True / False = analytically correct or not; None = no verdict (scorer crashed)."""
    if has_scorer_error(checks):
        return None
    req = [c for c in checks if c["required"]]
    return bool(req) and all(c["passed"] for c in req)


def finite_rows(a):
    """Rows (points) of a 2-D coordinate array with all-finite values, as float array (n, k)."""
    a = np.asarray(a, float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.size == 0:
        return a.reshape(0, a.shape[1] if a.ndim == 2 else 0)
    return a[np.isfinite(a).all(axis=1)]


def point_xy(gdf):
    """(n, 2) x/y of a point layer's geometries; missing / empty / non-point -> NaN rows."""
    out = np.full((len(gdf), 2), np.nan)
    for i, g in enumerate(gdf.geometry):
        try:
            if g is not None and not g.is_empty and g.geom_type == "Point":
                out[i] = (g.x, g.y)
        except Exception:
            pass
    return out


def file_exists(ctx, names, name="required output present", required=True):
    """names: list of acceptable file names (any one is enough)."""
    have = [n for n in names if os.path.exists(ctx.pred(n))]
    return result(name, bool(have), value=have, threshold=names, required=required)


def png_not_blank(path, min_std=3.0):
    """A saved figure with real content (not a blank canvas)."""
    from PIL import Image
    try:
        a = np.asarray(Image.open(path).convert("L"), float)
        return float(a.std()) >= min_std, float(a.std())
    except Exception as e:
        return False, repr(e)


# ----------------------------------------------------------------------------- rasters
def read_raster(path):
    import rasterio
    with rasterio.open(path) as r:
        a = r.read(1).astype(float)
        if r.nodata is not None:
            a[a == r.nodata] = np.nan
        return a, {"crs": r.crs.to_string() if r.crs else None, "transform": r.transform,
                   "bounds": tuple(r.bounds), "shape": a.shape, "res": r.res}


def resample_to(a, shape):
    """Nearest-neighbour resample of a 2-D array to `shape` (for different resolutions)."""
    if a.shape == tuple(shape):
        return a
    ri = (np.arange(shape[0]) * a.shape[0] / shape[0]).astype(int)
    ci = (np.arange(shape[1]) * a.shape[1] / shape[1]).astype(int)
    return a[np.ix_(ri, ci)]


def raster_agreement(pred, gold):
    """Cell-level agreement between two 2-D arrays (pred resampled onto gold's grid):
    pearson r, spearman rho, rmse, nrmse (by gold range), valid-mask IoU."""
    p = resample_to(np.asarray(pred, float), np.asarray(gold).shape)
    g = np.asarray(gold, float)
    vp, vg = np.isfinite(p), np.isfinite(g)
    both = vp & vg
    mask_iou = both.sum() / max(1, (vp | vg).sum())
    if both.sum() < 10:
        return {"n": int(both.sum()), "mask_iou": float(mask_iou)}
    x, y = p[both], g[both]
    r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    from scipy.stats import spearmanr
    rho = float(spearmanr(x, y).correlation) if x.std() > 0 and y.std() > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((x - y) ** 2)))
    rng = float(np.nanmax(y) - np.nanmin(y)) or 1.0
    return {"n": int(both.sum()), "pearson": r, "spearman": rho, "rmse": rmse, "nrmse": rmse / rng,
            "mask_iou": float(mask_iou), "pred_mean": float(x.mean()), "gold_mean": float(y.mean())}


def categorical_agreement(pred, gold):
    """Share of cells with the same class (after resampling), plus per-class IoU mean."""
    p = resample_to(np.asarray(pred), np.asarray(gold).shape)
    g = np.asarray(gold)
    valid = np.isfinite(p.astype(float)) & np.isfinite(g.astype(float))
    acc = float((p[valid] == g[valid]).mean()) if valid.any() else float("nan")
    ious = []
    for c in np.unique(g[valid]):
        a, b = p[valid] == c, g[valid] == c
        u = (a | b).sum()
        if u:
            ious.append((a & b).sum() / u)
    return {"accuracy": acc, "mean_iou": float(np.mean(ious)) if ious else float("nan")}


# ----------------------------------------------------------------------------- vectors
def read_vector(path):
    import geopandas as gpd
    return gpd.read_file(path)


def to_common_crs(a, b):
    if a.crs and b.crs and a.crs != b.crs:
        a = a.to_crs(b.crs)
    return a, b


def area_iou(a, b):
    """Area-weighted IoU of the union of polygons in a and b (same CRS assumed/handled)."""
    a, b = to_common_crs(a, b)
    ua, ub = a.geometry.union_all(), b.geometry.union_all()
    inter, uni = ua.intersection(ub).area, ua.union(ub).area
    return float(inter / uni) if uni > 0 else float("nan")


def point_set_distance(a, b):
    """Symmetric mean nearest-neighbour distance between two point sets (CRS units)."""
    from scipy.spatial import cKDTree
    a, b = to_common_crs(a, b)
    pa = finite_rows(point_xy(a))   # missing / empty geometries would crash cKDTree
    pb = finite_rows(point_xy(b))
    if len(pa) == 0 or len(pb) == 0:
        return float("nan")
    d1 = cKDTree(pb).query(pa)[0].mean()
    d2 = cKDTree(pa).query(pb)[0].mean()
    return float((d1 + d2) / 2)


def valid_fraction(gdf):
    return float(gdf.geometry.is_valid.mean()) if len(gdf) else float("nan")


def hausdorff(a, b):
    a, b = to_common_crs(a, b)
    return float(a.geometry.union_all().hausdorff_distance(b.geometry.union_all()))


# ----------------------------------------------------------------------------- tables / values
def rel_close(x, y, rtol):
    return abs(x - y) <= rtol * max(abs(y), 1e-12)


def find_object(ctx, side, kind, predicate):
    """First captured object of `kind` whose loaded value satisfies predicate(obj, meta)."""
    for tag, meta, load in reversed(ctx.objects(side, kind)):  # latest first
        try:
            obj = load()
            if predicate(obj, meta):
                return obj, meta, tag
        except Exception:
            continue
    return None, None, None


# ----------------------------------------------------------------------------- alignment
def key_values(pred, inputs_gdf, key):
    """Return a Series giving, for every row of `pred`, the value of `key` in the input layer,
    or None if the rows cannot be aligned. Order of preference:
      1. a column named like the key (case-insensitive; also key_left/key_right after joins),
         provided it covers the input ids (>= 80 % of rows hit an input id, and at least
         0.5 * min(rows, n) distinct values); otherwise it is ignored;
      2. any other column whose values match the input key values (>= 80 % of rows, e.g. an
         id column renamed by the agent);
      3. geometry: the representative point of each pred geometry inside an input polygon;
      4. positional sjoin columns (index_right / index_left);
      5. a plain 'index' column only when it is exactly 0..n-1 for the full layer (a complete
         reset_index of a per-feature table); partial or reordered tables are not trusted.
    An index NAMED like the key (e.g. after set_index(key)) counts as the key column (step 1)."""
    import pandas as pd
    ids = inputs_gdf[key]
    id_str = set(ids.astype(str))
    n = len(inputs_gdf)

    def covers_ids(v):
        # a key-named column is trusted only if it really holds the input ids: >= 80 % of rows
        # hit an input id (as text, or numerically, e.g. 5.0 vs 5) and enough distinct values.
        # A same-name column brought in from ANOTHER layer by overlay/sjoin (e.g. OBJECTID = 1 on
        # every row) fails this and falls through to the geometry join.
        v = pd.Series(np.asarray(v))
        if not len(v):
            return False
        try:
            hit = v.astype(str).isin(id_str).mean()
        except Exception:
            hit = 0.0
        if hit < 0.8:
            try:
                vn = pd.to_numeric(v, errors="coerce")
                idn = set(pd.to_numeric(ids, errors="coerce").dropna().tolist())
                if idn:
                    hit = max(hit, vn.isin(idn).mean())
            except Exception:
                pass
        return hit >= 0.8 and v.nunique() >= 0.5 * min(len(v), n)

    lower = {str(c).lower(): c for c in pred.columns}
    for c in (key, key + "_left", key + "_right"):
        if c.lower() in lower and covers_ids(pred[lower[c.lower()]]):
            return pred[lower[c.lower()]]
    if getattr(pred.index, "name", None) is not None and str(pred.index.name).lower() == key.lower() \
            and covers_ids(pred.index.to_numpy()):
        return pd.Series(pred.index.to_numpy(), index=pred.index)
    import re as _re
    back = dict(zip(ids.astype(str), ids))

    def id_overlap(numeric):
        for c in pred.columns:
            if c == "geometry" or c == "index":
                continue
            is_num = getattr(pred[c], "dtype", None) is not None and pred[c].dtype.kind in "fiub"
            if is_num != numeric:
                continue
            if numeric and not _re.search(r"(^|_)(id|fid|code|key|geoid|zip|zcta)($|_)|id$|code$|key$",
                                          str(c), _re.I):
                continue  # plain numeric columns (counts, values) are never identifiers
            try:
                v = pred[c].astype(str)
            except Exception:
                continue
            if len(v) and v.isin(id_str).mean() >= 0.8 and v.nunique() >= 0.5 * min(len(v), n):
                return v.map(back)
        return None

    r = id_overlap(numeric=False)          # text ids are unambiguous
    if r is not None:
        return r
    if "geometry" in pred.columns:         # geometry beats numeric ids (which may be off by one)
        try:
            import geopandas as gpd
            g = gpd.GeoDataFrame(pred[["geometry"]].copy(), geometry="geometry", crs=getattr(pred, "crs", None))
            if g.crs and inputs_gdf.crs and g.crs != inputs_gdf.crs:
                g = g.to_crs(inputs_gdf.crs)
            pts = g.copy()
            pts.index.name = None   # an index name equal to a right-hand column breaks the sjoin
            pts["geometry"] = g.geometry.representative_point()
            j = gpd.sjoin(pts, inputs_gdf[[key, "geometry"]], how="left", predicate="within")
            j = j[~j.index.duplicated()]
            if j[key].notna().mean() > 0.8:
                return j[key].reindex(pred.index)
        except Exception:
            pass
    else:
        r = id_overlap(numeric=True)
        if r is not None:
            return r
    arr = ids.to_numpy()
    for c in ("index_right", "index_left"):
        if c in pred.columns and pred[c].dtype.kind in "iu" and pred[c].between(0, n - 1).all():
            return pd.Series(arr[pred[c].to_numpy()], index=pred.index)
    if "index" in pred.columns and len(pred) == n and pred["index"].dtype.kind in "iu" \
            and (pred["index"].to_numpy() == np.arange(n)).all():
        return pd.Series(arr, index=pred.index)
    return None


def best_matching_column(ctx, gold_values, inputs_gdf, key, n_expected, exclude=(), tol=0.05):
    """Search every captured pred table with ~n_expected rows for the numeric column whose
    values, aligned on `key`, best rank-correlate with gold_values (Series indexed by key).
    Returns (column, rho, table_name, aligned_series) - rho = -1 if nothing usable."""
    best = (None, -1.0, None, None)
    for tag, meta, load in ctx.objects("pred"):
        if meta.get("kind") not in ("gdf", "df") or abs(meta.get("rows", 0) - n_expected) > tol * n_expected:
            continue
        try:
            obj = load()
        except Exception:
            continue
        keys = key_values(obj, inputs_gdf, key)
        if keys is None:
            continue
        for c in obj.columns:
            if c in exclude or c == "geometry" or c == key or obj[c].dtype.kind not in "fiu":
                continue
            s = obj[c].astype(float).groupby(keys.values).mean()
            common = gold_values.index.intersection(s.index)
            x, y = s.loc[common], gold_values.loc[common].astype(float)
            m = x.notna() & y.notna()
            if m.sum() < 0.8 * n_expected or x[m].std() == 0:
                continue
            rho = float(x[m].rank().corr(y[m].rank()))
            if rho > best[1]:
                best = (c, rho, meta["name"] + "@" + tag, s)
    return best


# ----------------------------------------------------------------------------- unverifiable
def unverifiable(name, note=""):
    """A required check that cannot be decided from the captured evidence. It does NOT pass:
    'analytically correct' needs positive evidence. Counted separately in reports."""
    r = result(name, False, required=True, note="UNVERIFIABLE: " + note)
    r["unverifiable"] = True
    return r


# ----------------------------------------------------------------------------- what was drawn
def saved_figures(ctx, side="pred"):
    """Every savefig call of the run, in order: list of {"i", "fname", "tag"} (tag = the data
    snapshot taken at that call, None beyond the first 20). [] when the run saved no figure."""
    root = ctx.pred_snap if side == "pred" else ctx.gold_snap
    p = os.path.join(root, "savefig.json") if root else None
    try:
        return json.load(open(p)) if p and os.path.exists(p) else []
    except (OSError, ValueError):
        return []


def figure_record_for_file(ctx, filename, side="pred"):
    """The LAST savefig record that wrote `filename` (basename, case-insensitive) - the figure
    that ended up in that file - or None if no recorded savefig wrote it. Its "tag" is None
    when that call was beyond the 20 data snapshots."""
    want = os.path.basename(filename).lower()
    hits = [s for s in saved_figures(ctx, side) if (s.get("fname") or "").lower() == want]
    return hits[-1] if hits else None


def figure_tag_for_file(ctx, filename, side="pred"):
    """Snapshot tag of the figure that ended up in `filename`, or None (see above)."""
    rec = figure_record_for_file(ctx, filename, side)
    return rec.get("tag") if rec else None


def drawn_geo_layers(ctx, side="pred", file=None):
    """Layers drawn with GeoDataFrame.plot / GeoSeries.plot, recorded per figure at savefig
    (capture.py): list of (tag, record, per-feature bounds array or None). A record has rows,
    crs, type, geom_types, total_bounds. With `file`, only the figure that ended up in that
    file."""
    root = ctx.pred_snap if side == "pred" else ctx.gold_snap
    if not root or not os.path.isdir(root):
        return []
    want = figure_tag_for_file(ctx, file, side) if file else None
    if file and want is None:
        return []
    out = []
    for tag in sorted(os.listdir(root)):
        if want is not None and tag != want:
            continue
        p = os.path.join(root, tag, "drawn_layers.json")
        if not os.path.exists(p):
            continue
        try:
            recs = json.load(open(p))
        except (OSError, ValueError):
            continue
        for r in recs:
            b = None
            if r.get("bounds_file"):
                try:
                    b = np.load(os.path.join(root, tag, r["bounds_file"]))
                except Exception:
                    b = None
            out.append((tag, r, b))
    return out


def drawn_arrays(ctx, side="pred", artist=None, file=None):
    """Arrays captured from figures at savefig time: list of (tag, meta, array).
    artist: 'image', 'line', 'image_extent', 'axes_limits' or a collection class name
    (e.g. 'PatchCollection', 'PathCollection', 'QuadMesh').
    file: only the figure that ended up in this output file (see figure_tag_for_file). The
    filter is not applied when the run recorded no savefig call, nor when the file was written
    by a savefig beyond the 20 data snapshots."""
    only = None
    if file and saved_figures(ctx, side):
        rec = figure_record_for_file(ctx, file, side)
        if rec is None:
            only = "__none__"          # no recorded savefig wrote that file
        elif rec.get("tag"):
            only = rec["tag"]          # a save beyond the snapshot limit has no tag: no filter
    out = []
    for tag, meta, load in ctx.objects(side, "array", include_files=False):
        if meta.get("source") != "figure":
            continue
        if only and tag != only:
            continue
        if artist and meta.get("artist") != artist:
            continue
        try:
            out.append((tag, meta, load()))
        except Exception:
            continue
    return out


def drawn_column_matches(ctx, side="pred", rtol=1e-6):
    """Identify which captured table columns were drawn as colour values of a collection
    (e.g. GeoDataFrame.plot(column=...)): returns list of (table_name, column, tag).
    A column matches when its non-null values equal, in order, a drawn values array."""
    drawn = [(t, m, np.asarray(a, float).ravel()) for t, m, a in drawn_arrays(ctx, side)
             if m["name"].endswith("_values")]
    hits = []
    if not drawn:
        return hits
    for tag, meta, load in ctx.objects(side, include_files=False):
        if meta.get("kind") not in ("gdf", "df"):
            continue
        try:
            obj = load()
        except Exception:
            continue
        # GeoDataFrame.plot draws one patch per polygon PART: expand values by part count
        parts = None
        if "geometry" in obj.columns:
            try:
                parts = np.array([len(getattr(g, "geoms", [g])) if g is not None else 1 for g in obj.geometry])
            except Exception:
                parts = None
        for c in obj.columns:
            if c == "geometry" or getattr(obj[c], "dtype", None) is None or obj[c].dtype.kind not in "fiub":
                continue
            raw = obj[c].astype(float).to_numpy()
            cands = [raw[np.isfinite(raw)]]
            if parts is not None and (parts > 1).any():
                ex = np.repeat(raw, parts)
                cands.append(ex[np.isfinite(ex)])
            for dt, dm, d in drawn:
                d2 = d[np.isfinite(d)]
                if any(len(d2) and len(d2) == len(v) and np.allclose(v, d2, rtol=rtol, atol=1e-9) for v in cands):
                    hits.append((meta["name"], c, tag))
                    break
    return hits


def drawn_image_agreement(ctx, gold, side="pred"):
    """Best agreement (raster_agreement) between a drawn image and the 2-D gold array, using
    only the orientation implied by how it was drawn: imshow arrays as stored (origin
    'upper'), flipped vertically when origin='lower'. No arbitrary flips/transposes, so a
    mis-oriented map does not pass. Returns (agreement dict, meta) or (None, None)."""
    best = (None, None)
    for tag, meta, a in drawn_arrays(ctx, side, "image"):
        a = np.asarray(a, float)
        if a.ndim == 3:
            continue  # RGB(A) rendering, not data
        if meta.get("origin") == "lower":
            a = a[::-1]
        try:
            ag = raster_agreement(a, gold)
        except Exception:
            continue
        sc = ag.get("spearman", float("nan"))
        if sc == sc and (best[0] is None or sc > best[0].get("spearman", -2)):
            best = (ag, meta)
    return best


def parse_number(x):
    """Lenient numeric parse: '47.5%', '47,5', ' 0.475 ' -> float (percent sign kept as value)."""
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    try:
        t = str(x).strip().replace(",", ".").replace("%", "")
        return float(t)
    except Exception:
        return float("nan")


def _cmap_positions(colors):
    """Map RGBA face colours back to positions (0..1) along the best-fitting registered
    matplotlib colormap. Returns (positions, cmap_name, mean_residual).
    Reversed maps ('*_r') are not searched: they are exact mirrors, so a map drawn with
    'viridis_r' is fitted by 'viridis' with the same residual and positions 1 - p.
    Positions are therefore only defined up to direction: callers must compare |rho|
    (see drawn_color_column_matches(abs_rho=True), rank_fit_to_positions)."""
    import matplotlib
    cols = np.asarray(colors, float)[:, :3]
    uniq, inv = np.unique(np.round(cols, 4), axis=0, return_inverse=True)
    best = (None, None, np.inf)
    for name in matplotlib.colormaps():
        if name.endswith("_r"):
            continue
        lut = matplotlib.colormaps[name](np.linspace(0, 1, 256))[:, :3]
        d = ((uniq[:, None, :] - lut[None, :, :]) ** 2).sum(-1)
        idx = d.argmin(1)
        res = float(np.sqrt(d[np.arange(len(uniq)), idx]).mean())
        if res < best[2]:
            best = (idx / 255.0, name, res)
    if best[0] is None:
        return None, None, np.inf
    return best[0][np.ravel(inv)], best[1], best[2]


def rank_fit_to_positions(v, pos, tol=1e-3):
    """How well does column v explain drawn colormap positions pos (same length, finite)?
    Returns (rho, ceiling, class_iou):
      rho      spearman(v, pos), signed (sign = colormap direction, see _cmap_positions);
      ceiling  the best |rho| any column could reach with the drawn class sizes: the drawn
               positions re-assigned to v in sorted order (continuous maps -> ~1; a k-class
               quantile / natural-breaks / equal-interval map -> < 1, e.g. 0.43 for 5 equal
               intervals on lognormal data);
      class_iou mean over drawn classes of the IoU between the rows drawn in that class and
               the rows the sorted re-assignment puts there (1.0 when the drawn classes are
               exactly value intervals of v, i.e. v is the column that was classified)."""
    from scipy.stats import spearmanr
    v, pos = np.asarray(v, float), np.asarray(pos, float)
    if len(v) != len(pos) or len(v) < 3 or np.std(v) == 0 or np.std(pos) == 0:
        return float("nan"), float("nan"), float("nan")
    rho = float(spearmanr(v, pos).correlation)
    ps = np.sort(pos) if rho >= 0 else np.sort(pos)[::-1]
    ideal = np.empty_like(pos)
    ideal[np.argsort(v, kind="stable")] = ps
    ceiling = abs(float(spearmanr(v, ideal).correlation))
    q, qi = np.round(pos / tol).astype(np.int64), np.round(ideal / tol).astype(np.int64)
    ious = []
    for c in np.unique(q):
        a, b = q == c, qi == c
        ious.append((a & b).sum() / max(1, (a | b).sum()))
    return rho, ceiling, float(np.mean(ious))


def drawn_color_column_matches(ctx, side="pred", min_rho=0.95, max_residual=0.02, abs_rho=True,
                               min_class_iou=0.9, max_classes=12):
    """Which table columns were drawn as a choropleth whose colours were computed by the
    plotting library (GeoDataFrame.plot passes colours, not values)? For every collection
    with >= 3 distinct face colours, invert the colours to colormap positions and compare
    their ranks with each numeric column of tables of matching length (also expanded by
    polygon part count). A column matches when
      * |rho| >= min_rho (continuous maps; abs because reversed colormaps invert positions;
        abs_rho=False tests the signed rho), or
      * the map is classified (3..max_classes distinct colours: quantiles, natural breaks,
        equal interval, ...) and the drawn classes are value intervals of the column:
        mean class IoU >= min_class_iou (see rank_fit_to_positions). A fixed rho threshold
        does not work here: its ceiling with k classes can be far below min_rho on skewed data.
    Returns list of (table, column, rho, cmap, tag); rho is signed (negative = reversed map)."""
    hits = []
    fcs = [(t, m, np.asarray(a, float)) for t, m, a in drawn_arrays(ctx, side) if m["name"].endswith("_facecolors")]
    if not fcs:
        return hits
    inverted = []
    for t, m, fc in fcs:
        if fc.ndim != 2 or len(np.unique(np.round(fc[:, :3], 4), axis=0)) < 3:
            continue
        pos, name, res = _cmap_positions(fc)
        if pos is not None and res <= max_residual:
            inverted.append((t, m, pos, name, len(np.unique(np.round(pos, 3)))))
    if not inverted:
        return hits
    for tag, meta, load in ctx.objects(side, include_files=False):
        if meta.get("kind") not in ("gdf", "df"):
            continue
        try:
            obj = load()
        except Exception:
            continue
        parts = None
        if "geometry" in obj.columns:
            try:
                parts = np.array([len(getattr(g, "geoms", [g])) if g is not None else 1 for g in obj.geometry])
            except Exception:
                parts = None
        for c in obj.columns:
            if c == "geometry" or getattr(obj[c], "dtype", None) is None or obj[c].dtype.kind not in "fiub":
                continue
            raw = obj[c].astype(float).to_numpy()
            fin = np.isfinite(raw)
            # (values, index of the first drawn patch of each feature or None): one patch per
            # feature, or one per polygon PART. Part-expanded comparisons are collapsed back to
            # one value per feature, so multi-part features (e.g. Alaska: 32 parts in T49) do not
            # dominate the rank correlation.
            variants = [(raw[fin], None)]
            if parts is not None and (parts > 1).any():
                p_fin = parts[fin]
                variants.append((raw[fin], np.r_[0, np.cumsum(p_fin)[:-1]], int(p_fin.sum())))
            done = False
            for ft, fm, pos, cname, ncls in inverted:
                for var in variants:
                    v, first = var[0], var[1]
                    if first is None:
                        if len(v) != len(pos):
                            continue
                        pv = pos
                    else:
                        if var[2] != len(pos):
                            continue
                        pv = pos[first]
                    if np.std(v) == 0:
                        continue
                    rho, ceiling, ciou = rank_fit_to_positions(v, pv)
                    if rho != rho:
                        continue
                    r = abs(rho) if abs_rho else rho
                    classed = 3 <= ncls <= max_classes and ciou >= min_class_iou
                    if r >= min_rho or (classed and (abs_rho or rho > 0)):
                        hits.append((meta["name"], c, round(rho, 4), cname, tag))
                        done = True
                        break
                if done:
                    break
    return hits


cmap_positions = _cmap_positions  # public name


# ----------------------------------------------------------------------------- uniform rules
# Rules applied to every task by build_gold.evaluate, on top of the task's own checks:
#   1. a geographic deliverable FILE without a CRS fails the run (one required check);
#   2. a file that is just a copy of an input dataset is not a result: it is hidden from the
#      task's checks (byte copy, or same geometries + attributes / same raster values).
# Deliverables are declared per task in benchmark/specs/<id>/contract.json (scoring side
# only; never shown to the agent). In-memory captured objects are not affected by rule 1.
import fnmatch as _fnmatch
import hashlib as _hashlib

VECTOR_EXT = (".shp", ".geojson", ".gpkg", ".json")
RASTER_EXT = (".tif", ".tiff")
GEOJSON_TYPES = {"FeatureCollection", "Feature", "Point", "MultiPoint", "LineString", "MultiLineString",
                 "Polygon", "MultiPolygon", "GeometryCollection"}
SHP_PARTS = (".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".sbn", ".sbx", ".qmd", ".shp.xml")
CRS_CHECK_NAME = "deliverable files carry a CRS (uniform rule)"
COPY_INFO_NAME = "copies of input datasets not counted as results (uniform rule)"


def contract_path(tid):
    return os.path.join(paths.SPECS, str(tid), "contract.json")


def load_contract(tid):
    """The task's declarative deliverable contract, or None if the task has none."""
    p = contract_path(tid)
    try:
        return json.load(open(p)) if os.path.exists(p) else None
    except (OSError, ValueError):
        return None


def matches(name, patterns):
    """Case-insensitive fnmatch of a basename against a list of patterns."""
    n = name.lower()
    return any(_fnmatch.fnmatchcase(n, p.lower()) for p in patterns)


def _load_json(path):
    try:
        if os.path.getsize(path) > 2e8:
            return None
        return json.load(open(path, errors="ignore"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def _is_geojson(path):
    """True when a .json/.geojson file is GeoJSON (has a GeoJSON 'type')."""
    try:
        if os.path.getsize(path) > 2e8:
            return path.lower().endswith(".geojson")
    except OSError:
        return False
    d = _load_json(path)
    return isinstance(d, dict) and d.get("type") in GEOJSON_TYPES


def geo_file_crs(path):
    """CRS status of a geographic file. Returns (kind, crs, note):
    kind 'vector' / 'raster', or None when the file is not geographic (e.g. a .json table);
    crs a string, or None when the file has no CRS (or cannot be read).
    GeoJSON without a 'crs' member is WGS 84 by definition (RFC 7946) when its coordinates
    are longitude/latitude; projected coordinates without a 'crs' member have no CRS."""
    low = path.lower()
    if low.endswith(RASTER_EXT):
        try:
            import rasterio
            with rasterio.open(path) as r:
                return "raster", (r.crs.to_string() if r.crs else None), "" if r.crs else "GeoTIFF without CRS"
        except Exception as e:
            return "raster", None, "unreadable raster: " + repr(e)[:120]
    if not low.endswith(VECTOR_EXT):
        return None, None, ""
    import pyogrio
    if low.endswith((".json", ".geojson")):
        if not _is_geojson(path):
            return None, None, "not GeoJSON"
        d = _load_json(path)
        if isinstance(d, dict) and d.get("crs"):
            try:
                c = pyogrio.read_info(path).get("crs")
            except Exception:
                c = None
            return "vector", c or json.dumps(d["crs"])[:80], ""
        try:
            b = pyogrio.read_info(path, force_total_bounds=True).get("total_bounds")
        except Exception as e:
            return "vector", None, "unreadable GeoJSON: " + repr(e)[:120]
        if b is None or not np.all(np.isfinite(b)):
            return "vector", "EPSG:4326 (RFC 7946, empty)", ""
        if -180.0001 <= b[0] and b[2] <= 180.0001 and -90.0001 <= b[1] and b[3] <= 90.0001:
            return "vector", "EPSG:4326 (RFC 7946 default)", ""
        return "vector", None, "GeoJSON with projected coordinates and no 'crs' member"
    try:
        layers = pyogrio.list_layers(path)
    except Exception as e:
        return "vector", None, "unreadable vector: " + repr(e)[:120]
    bad, crs = [], None
    for lyr, gtype in layers:
        if gtype is None:          # attribute-only table inside a GeoPackage
            continue
        try:
            c = pyogrio.read_info(path, layer=lyr).get("crs")
        except Exception:
            c = None
        if c:
            crs = crs or c
        else:
            bad.append(str(lyr))
    if bad:
        return "vector", None, ("no CRS (.prj missing or empty)" if low.endswith(".shp")
                                else f"layer(s) without CRS: {bad}")
    if crs is None:
        return None, None, "no spatial layer"
    return "vector", crs, ""


def contract_geo_files(pred_dir, contract):
    """Geographic files in pred_dir that the contract declares as deliverables with
    crs_required. Patterns match the base name; sub-folders are included (several checks glob
    pred_results recursively). Returns sorted list of (path relative to pred_dir, deliverable id)."""
    out = []
    if not contract or not pred_dir or not os.path.isdir(pred_dir):
        return out
    names = []
    for root, _, files in os.walk(pred_dir, followlinks=True):
        for f in files:
            names.append(os.path.relpath(os.path.join(root, f), pred_dir))
    for d in contract.get("deliverables", []):
        if not d.get("crs_required"):
            continue
        for n in sorted(names):
            b = os.path.basename(n)
            if b.lower().endswith(VECTOR_EXT + RASTER_EXT) and matches(b, d.get("patterns", [])):
                out.append((n, d.get("id")))
    return sorted(set(out))


def deliverable_crs_check(pred_dir, contract):
    """The uniform required check: every geographic deliverable file must carry a CRS.
    Returns a result dict, or None when the contract declares no geographic deliverable."""
    if not contract or not any(d.get("crs_required") for d in contract.get("deliverables", [])):
        return None
    files = sorted({n for n, _ in contract_geo_files(pred_dir, contract)})
    missing, ok = [], []
    for n in files:
        kind, crs, note = geo_file_crs(os.path.join(pred_dir, n))
        if kind is None:
            continue
        if crs:
            ok.append(n)
        else:
            missing.append(f"{n}: {note}")
    thr = "every geographic deliverable file has a CRS"
    if not ok and not missing:
        return result(CRS_CHECK_NAME, True, value=[], threshold=thr,
                      note="no geographic deliverable file written (presence is judged by the task checks)")
    return result(CRS_CHECK_NAME, not missing, value=missing or ok, threshold=thr,
                  note=("deliverable file(s) without CRS: " + "; ".join(missing)) if missing else
                  f"{len(ok)} geographic deliverable file(s), all with a CRS")


# --- copies of input datasets --------------------------------------------------------------
_INPUT_INDEX = {}
_SHA_CACHE = {}


def _file_sha(p):
    k = (p, os.path.getsize(p), os.path.getmtime(p))
    if k not in _SHA_CACHE:
        h = _hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _SHA_CACHE[k] = h.hexdigest()
    return _SHA_CACHE[k]


def _input_index(inputs_dir):
    """Input files of the task: by size (for byte copies), vector layers by feature count,
    rasters by (bands, height, width), csv tables."""
    if inputs_dir in _INPUT_INDEX:
        return _INPUT_INDEX[inputs_dir]
    idx = {"by_size": {}, "vectors": [], "rasters": [], "tables": []}
    if inputs_dir and os.path.isdir(inputs_dir):
        import pyogrio
        for root, _, files in os.walk(inputs_dir, followlinks=True):
            for f in files:
                p = os.path.join(root, f)
                try:
                    sz = os.path.getsize(p)
                except OSError:
                    continue
                idx["by_size"].setdefault(sz, []).append(p)
                low = f.lower()
                if low.endswith((".shp", ".geojson", ".gpkg")) or (low.endswith(".json") and _is_geojson(p)):
                    try:
                        for lyr, gt in pyogrio.list_layers(p):
                            if gt is not None:
                                n = pyogrio.read_info(p, layer=lyr)["features"]
                                idx["vectors"].append((p, lyr, n, sz))
                    except Exception:
                        pass
                elif low.endswith((".tif", ".tiff", ".img", ".asc")):
                    try:
                        import rasterio
                        with rasterio.open(p) as r:
                            idx["rasters"].append((p, (r.count, r.height, r.width)))
                    except Exception:
                        pass
                elif low.endswith(".csv") and sz < 2e8:
                    idx["tables"].append(p)
    _INPUT_INDEX[inputs_dir] = idx
    return idx


def _byte_copy(p, idx):
    try:
        sz = os.path.getsize(p)
    except OSError:
        return None
    for q in idx["by_size"].get(sz, []):
        try:
            if _file_sha(q) == _file_sha(p):
                return q
        except OSError:
            continue
    return None


def _same_values(a, b):
    try:
        if len(a) != len(b):
            return False
        if a.dtype.kind in "fiub" and b.dtype.kind in "fiub":
            x, y = a.astype(float).to_numpy(), b.astype(float).to_numpy()
            return bool(np.allclose(x, y, rtol=1e-9, atol=1e-12, equal_nan=True))
        return bool((a.astype(str).reset_index(drop=True) == b.astype(str).reset_index(drop=True)).all())
    except Exception:
        return False


def _vector_copy(p, idx, layer=None):
    """Input layer with the same geometries (same order, within float noise) whose attributes
    include every attribute column of p with equal values; else None."""
    import geopandas as gpd
    import pyogrio
    import shapely
    try:
        n = pyogrio.read_info(p, layer=layer)["features"]
    except Exception:
        return None
    cands = [(q, lyr) for q, lyr, m, sz in idx["vectors"] if m == n and sz < 5e8 and n > 0]
    if not cands:
        return None
    try:
        a = gpd.read_file(p, layer=layer)
    except Exception:
        return None
    for q, lyr in cands:
        try:
            b = gpd.read_file(q, layer=lyr)
        except Exception:
            continue
        aa = a
        if aa.crs is not None and b.crs is not None and aa.crs != b.crs:
            try:
                aa = aa.to_crs(b.crs)
            except Exception:
                continue
        x0, y0, x1, y1 = b.total_bounds if len(b) else (0, 0, 1, 1)
        tol = 1e-7 * max(abs(x1 - x0), abs(y1 - y0), 1e-9) + 1e-9
        try:
            same = shapely.equals_exact(aa.geometry.values, b.geometry.values, tolerance=tol)
            both_na = aa.geometry.isna().to_numpy() & b.geometry.isna().to_numpy()
            if not bool(np.all(same | both_na)):
                continue
        except Exception:
            continue
        bcols = {str(c).lower(): c for c in b.columns if c != b.geometry.name}
        ok = True
        for c in a.columns:
            if c == a.geometry.name:
                continue
            cl = str(c).lower()
            cand = [bcols[k] for k in bcols if k == cl or (len(cl) >= 10 and k.startswith(cl))]
            if not any(_same_values(a[c], b[m]) for m in cand):
                ok = False
                break
        if ok:
            return q
    return None


def _raster_copy(p, idx):
    try:
        import rasterio
        with rasterio.open(p) as r:
            shp = (r.count, r.height, r.width)
            cands = [q for q, s in idx["rasters"] if s == shp]
            if not cands or r.count * r.height * r.width > 4e8:
                return None
            a = r.read().astype(float)
            if r.nodata is not None:
                a[a == r.nodata] = np.nan
        for q in cands:
            with rasterio.open(q) as s:
                b = s.read().astype(float)
                if s.nodata is not None:
                    b[b == s.nodata] = np.nan
            if np.array_equal(a, b, equal_nan=True):
                return q
    except Exception:
        return None
    return None


def _table_copy(p, idx):
    import pandas as pd
    if not idx["tables"]:
        return None
    try:
        a = pd.read_csv(p, sep=None, engine="python")
    except Exception:
        return None
    for q in idx["tables"]:
        try:
            b = pd.read_csv(q, sep=None, engine="python")
        except Exception:
            continue
        if a.shape == b.shape and [str(c).lower() for c in a.columns] == [str(c).lower() for c in b.columns] \
                and all(_same_values(a[c], b[d]) for c, d in zip(a.columns, b.columns)):
            return q
    return None


def input_copies(pred_dir, inputs_dir):
    """Files at the top level of pred_dir that are just copies of an input dataset:
    {file name: input file it copies (relative to inputs_dir)}. A shapefile copy hides
    all of its sidecar files."""
    out = {}
    if not pred_dir or not os.path.isdir(pred_dir) or not inputs_dir or not os.path.isdir(inputs_dir):
        return out
    idx = _input_index(inputs_dir)
    names = sorted(n for n in os.listdir(pred_dir) if os.path.isfile(os.path.join(pred_dir, n)))
    for n in names:
        p = os.path.join(pred_dir, n)
        low = n.lower()
        if low.endswith(".shp"):
            stem = n[:-4]
            src = _byte_copy(p, idx)
            dbf = os.path.join(pred_dir, stem + ".dbf")
            if src and os.path.exists(dbf) and not _byte_copy(dbf, idx):
                src = None
            src = src or _vector_copy(p, idx)
            if src:
                for m in names:
                    if m.lower().startswith(stem.lower() + ".") and m.lower()[len(stem):] in SHP_PARTS:
                        out[m] = src
            continue
        if low.endswith(SHP_PARTS[1:]):
            continue  # sidecars follow their .shp
        src = _byte_copy(p, idx)
        if src is None:
            if low.endswith(".gpkg"):
                try:
                    import pyogrio
                    lyrs = [lyr for lyr, g in pyogrio.list_layers(p) if g is not None]
                    hits = [_vector_copy(p, idx, layer=lyr) for lyr in lyrs]
                    src = hits[0] if lyrs and all(hits) else None
                except Exception:
                    src = None
            elif low.endswith(".geojson") or (low.endswith(".json") and _is_geojson(p)):
                src = _vector_copy(p, idx)
            elif low.endswith(RASTER_EXT):
                src = _raster_copy(p, idx)
            elif low.endswith(".csv"):
                src = _table_copy(p, idx)
        if src:
            out[n] = src
    return {k: os.path.relpath(v, inputs_dir) for k, v in out.items()}


def scoring_view(pred_dir, hide):
    """A temporary directory mirroring pred_dir through symlinks, without the names in
    `hide`; the caller removes it. Checks that glob the folder directly see the same files
    as ctx.pred_files()."""
    import tempfile
    v = tempfile.mkdtemp(prefix="scoreview_")
    if pred_dir and os.path.isdir(pred_dir):
        for n in os.listdir(pred_dir):
            if n not in hide:
                os.symlink(os.path.abspath(os.path.join(pred_dir, n)), os.path.join(v, n))
    return v
