"""T50 - Seagrass habitat prediction: dissolve, random points, kriging of temp/salinity, MaxEnt.

Task core: 5 000 random presence points inside the dissolved USA seagrass; temperature and
salinity interpolated (EBK in the original) from the ocean measurement points; a presence-only
model (MaxEnt) trained on the two surfaces in the USA study area and applied to predict a global
suitability raster over shallow waters; outputs = training features + prediction raster.

Correctness (required):
  1. presence points: a captured/saved point set of 4 000-5 100 points (the instruction asks
     for 5 000; 0.8x leaves room for points dropped on raster NoData) with >= 97 % inside the
     dissolved seagrass (1 km tolerance). Points drawn in the shallow-water study area or in a
     bounding box fail;
  2. training features saved: a point vector file (>= 1 000 points) carrying explanatory values
     that agree with the gold kriged surfaces at those points (Pearson >= 0.8 for temperature and
     >= 0.6 for salinity; salinity has a large nugget). A CSV with x/y (input CRS assumed) or
     lon/lat (EPSG:4326) columns plus the values counts as a point file. Wrong measurement columns (e.g. dissolved
     O2 or depth instead of salinity), swapped or missing values fail;
  3. prediction raster saved: georeferenced, covering >= 80 % of the gold prediction cells whose
     centre lies inside Global_shallow_waters (~46.2k cells; the gold grid itself is built with
     all_touched=True, of which only ~62 % have their centre inside, so a centre-based mask as in
     the original ArcGIS mask environment could not reach 80 % of the full gold grid), Spearman >= 0.5 with the gold probability surface at all gold
     cells the raster covers. A prediction limited to
     the USA, or a surface unrelated to temperature/salinity suitability, fails. Rasters that
     reproduce a kriged explanatory surface (|Spearman| >= 0.95 with the gold kriged temperature
     or salinity) are not candidates for the prediction;
  4. the prediction depends on BOTH explanatory variables: ranks of the predicted values at gold
     shallow-water cells are explained by joint temperature x salinity bins (gold kriged
     surfaces, 15 quantile bins each) with R2 >= 0.5, and the joint fit beats temperature-only and
     salinity-only fits by >= 0.05 each. Gold: R2 0.73 (T) / 0.19 (S) / 0.96 (T x S); the raw
     temperature surface or a temperature-only model gain <= 0.01 from salinity.
  5. (required: the instruction prescribes EBK) kriging-type
     interpolation, code inspection with comments removed: pykrige OK/UK, gstools, GP regression,
     any *krig*/*ebk*/*vario*() call, or a hand-rolled kriging (fitted variogram/covariance model
     + linear system solve). EBK itself is ArcGIS-only. IDW / nearest / RBF / ML regression fail;
  6. (required: the instruction prescribes MaxEnt) presence-only
     MaxEnt-type model: maxnet / elapid / maxent*() or a presence-vs-background logistic/Poisson
     GLM (IWLR, the standard MaxEnt equivalent). Random forest / boosting fail. A self-defined
     *maxent*() function (fit_maxent, ...) counts when its own body
     (ast) fits a likelihood (minimize / least_squares / logsumexp) or a GLM and uses no other
     classifier (tree/ensemble, *Classifier, SVC, MLP, KNN).
Reported, not required:
  7. spatial (block) cross-validation or thinning mentioned - random CV on 25 km rasters with
     5 000 clustered points is optimistic (gold: random AUC 0.815 vs spatial-block 0.751).
"""
import glob
import os

import numpy as np

try:
    import rasterio as _rio
    _pd = os.path.join(os.path.dirname(_rio.__file__), "proj_data")
    if os.path.isdir(_pd):
        os.environ["PROJ_DATA"] = _pd
except Exception:
    pass

from checklib import result

_C = {}
TEMP_KEYS = ("temp",)
N_MIN = 4000   # presence points: the instruction asks for 5 000
SAL_KEYS = ("sal",)


def _ctx_inputs(ctx):
    if _C.get("dir") == ctx.inputs_dir + ctx.gold_dir:
        return _C
    import geopandas as gpd
    import rasterio
    from rasterio.features import geometry_mask
    sg = gpd.read_file(f"{ctx.inputs_dir}/USA_seagrass.shp")
    sgb = sg.geometry.union_all().buffer(1000)
    G = {}
    for nm in ("EBK_temp", "EBK_salinity", "seagrass_predict2"):
        with rasterio.open(ctx.gold(nm + ".tif")) as r:
            G[nm] = (r.read(1).astype(float), r.transform, r.crs.to_wkt())
    # coverage denominator: gold prediction cells whose centre is inside Global_shallow_waters
    P = G["seagrass_predict2"]
    gsw = gpd.read_file(f"{ctx.inputs_dir}/Global_shallow_waters.shp").to_crs(P[2])
    inside = ~geometry_mask(list(gsw.geometry), P[0].shape, P[1], all_touched=False)
    rr, cc = np.nonzero(np.isfinite(P[0]))
    _C.update(dir=ctx.inputs_dir + ctx.gold_dir, sg_crs=sg.crs, sgb=sgb, G=G, centre_in=inside[rr, cc])
    return _C


def _sample_arr(arr, T, x, y):
    c, r = ~T * (np.asarray(x), np.asarray(y))
    c, r = np.floor(c).astype(int), np.floor(r).astype(int)
    ok = (r >= 0) & (r < arr.shape[0]) & (c >= 0) & (c < arr.shape[1])
    v = np.full(len(c), np.nan)
    v[ok] = arr[r[ok], c[ok]]
    return v


def _vec_files(ctx):
    import geopandas as gpd
    out = []
    for pat in ("*.shp", "*.geojson", "*.gpkg", "*.json"):
        for p in sorted(glob.glob(os.path.join(ctx.pred_dir, "**", pat), recursive=True)):
            try:
                out.append((os.path.basename(p), gpd.read_file(p)))
            except Exception:
                pass
    return out


def _csv_points(ctx, crs):
    """Point tables saved as CSV: x/y (or lon/lat) columns plus covariates.
    The CSV carries no CRS: lon/lat columns are read as EPSG:4326, x/y columns as the input CRS
    (the CRS of USA_seagrass); >= 90 % of the points must lie in the USA study-area extent."""
    import geopandas as gpd
    import pandas as pd
    out, box = [], None
    for p in sorted(glob.glob(os.path.join(ctx.pred_dir, "**", "*.csv"), recursive=True)):
        try:
            if os.path.getsize(p) > 5e8:
                continue
            d = pd.read_csv(p, low_memory=False)
        except Exception:
            continue
        low = {str(c).strip().lower(): c for c in d.columns}
        for xs, ys, c_crs in ((("x", "point_x", "easting"), ("y", "point_y", "northing"), crs),
                              (("lon", "longitude", "lng"), ("lat", "latitude"), "EPSG:4326")):
            cx = next((low[k] for k in xs if k in low), None)
            cy = next((low[k] for k in ys if k in low), None)
            if cx is None or cy is None:
                continue
            x, y = pd.to_numeric(d[cx], errors="coerce"), pd.to_numeric(d[cy], errors="coerce")
            ok = (x.notna() & y.notna()).to_numpy()
            if ok.sum() < 1000:
                continue
            # the CRS is assumed: >= 90 % of the points must fall in the USA study area extent
            # (+50 km), else the assumption does not hold and the table is skipped
            if box is None:
                try:
                    u = gpd.read_file(f"{ctx.inputs_dir}/USA_shallow_waters.shp")
                    box = (u.crs, u.total_bounds)
                except Exception:
                    box = (None, None)
            if box[0] is not None:
                pts = gpd.GeoSeries(gpd.points_from_xy(x[ok], y[ok]), crs=c_crs).to_crs(box[0])
                b = box[1]
                inb = ((pts.x >= b[0] - 5e4) & (pts.x <= b[2] + 5e4) & (pts.y >= b[1] - 5e4) & (pts.y <= b[3] + 5e4))
                if inb.mean() < 0.9:
                    continue
            rest = d.loc[ok, [c for c in d.columns if c not in (cx, cy)]].reset_index(drop=True)
            out.append((os.path.basename(p), gpd.GeoDataFrame(rest, geometry=gpd.points_from_xy(x[ok], y[ok]),
                                                              crs=c_crs)))
            break
    return out


def _pearson(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def checks(ctx):
    import geopandas as gpd
    out = []
    try:
        I = _ctx_inputs(ctx)
    except Exception as e:
        return [result("gold/inputs readable", False, note=repr(e)[:200])]
    sgb, crs = I["sgb"], I["sg_crs"]
    files = _vec_files(ctx)

    # 1 presence points
    best = (0.0, None)
    sets = [(n, g) for n, g in files]
    for tag, meta, load in ctx.objects("pred", "gdf"):
        if 1000 <= meta.get("rows", 0) <= 20000:
            try:
                sets.append((meta["name"] + "@" + tag, load()))
            except Exception:
                pass
    for name, g in sets:
        try:
            if g.crs is None or len(g) == 0 or not (g.geom_type == "Point").all():
                continue
            g = g.to_crs(crs)
            for lab, sub in _presence_subsets(g):
                if not (N_MIN <= len(sub) <= 5100):
                    continue
                share = float(sub.within(sgb).mean())
                if share > best[0]:
                    best = (share, f"{name}{lab} ({len(sub)} pts)")
        except Exception:
            continue
    out.append(result("presence points inside dissolved seagrass (>= 97 %)", best[0] >= 0.97,
                      value=round(best[0], 3), threshold=0.97, note=best[1] or "no point set of 4 000-5 100 points"))

    # 2 training features with explanatory values
    Tg, Sg = I["G"]["EBK_temp"], I["G"]["EBK_salinity"]
    bt, bs, bn = -1.0, -1.0, None
    for name, g in files + _csv_points(ctx, crs):
        try:
            if g.crs is None or len(g) < 1000 or not (g.geom_type == "Point").all():
                continue
            gg = g.to_crs(crs)
            x, y = gg.geometry.x.to_numpy(), gg.geometry.y.to_numpy()
            gt = _sample_arr(Tg[0], Tg[1], x, y)
            gs = _sample_arr(Sg[0], Sg[1], x, y)
            num = [c for c in g.columns if c != "geometry" and g[c].dtype.kind in "fi"]
            rt = max([_pearson(g[c].to_numpy(float), gt) for c in num if any(k in c.lower() for k in TEMP_KEYS)] +
                     [-1.0])
            rs = max([_pearson(g[c].to_numpy(float), gs) for c in num if any(k in c.lower() for k in SAL_KEYS)] +
                     [-1.0])
            if not any(k in c.lower() for c in num for k in TEMP_KEYS + SAL_KEYS):  # unnamed columns: best any
                rt = max([_pearson(g[c].to_numpy(float), gt) for c in num] + [-1.0])
                rs = max([_pearson(g[c].to_numpy(float), gs) for c in num] + [-1.0])
            rt, rs = (rt if np.isfinite(rt) else -1.0), (rs if np.isfinite(rs) else -1.0)
            if rt + rs > bt + bs:
                bt, bs, bn = rt, rs, name
        except Exception:
            continue
    out.append(result("training features saved with kriged temp/salinity values (r >= 0.8 / 0.6)",
                      bt >= 0.8 and bs >= 0.6, value=[round(bt, 3), round(bs, 3)], threshold=[0.8, 0.6],
                      note=bn or "no saved point file with >= 1 000 points"))

    # 3 prediction raster
    P = I["G"]["seagrass_predict2"]
    gv = np.isfinite(P[0])
    rr, cc = np.nonzero(gv)
    xs, ys = P[1] * (cc + 0.5, rr + 0.5)
    gold_vals = P[0][rr, cc]
    best3 = (-1.0, None, 0.0)
    best_v = None
    best_rho = (-2.0, None, 0.0)  # best-rho candidate, reported in the note when nothing passes
    centre_in = I["centre_in"]
    # the kriged explanatory surfaces saved next to the prediction are not candidates for "the
    # prediction" (the kriged temperature correlates 0.83 with the gold suitability and would
    # otherwise be picked instead of the model's raster)
    gt3 = _sample_arr(Tg[0], Tg[1], xs, ys)
    gs3 = _sample_arr(Sg[0], Sg[1], xs, ys)
    skipped = []
    import rasterio
    from pyproj import Transformer
    from scipy.stats import spearmanr
    for pat in ("*.tif", "*.tiff", "*.TIF", "*.img"):
        for p in glob.glob(os.path.join(ctx.pred_dir, "**", pat), recursive=True):
            try:
                with rasterio.open(p) as r:
                    a = r.read(1).astype(float)
                    if r.nodata is not None and np.isfinite(r.nodata):
                        a[a == r.nodata] = np.nan
                    T = r.transform
                    pc = r.crs.to_wkt() if r.crs else None
                if pc is None:
                    continue
                px, py = Transformer.from_crs(P[2], pc, always_xy=True).transform(xs, ys)
                v = _sample_arr(a, T, px, py)
                m = np.isfinite(v)
                cover = float(m[centre_in].mean())
                if m.sum() < 100 or np.nanstd(v[m]) == 0:
                    continue
                if _is_covariate_surface(v, gt3, gs3):
                    skipped.append(os.path.basename(p))
                    continue
                rho = float(spearmanr(v[m], gold_vals[m]).correlation)
                score = rho if cover >= 0.8 else min(rho, 0.0)
                if rho > best_rho[0]:
                    best_rho = (rho, os.path.basename(p), cover)
                if score > best3[0]:
                    best3 = (score, os.path.basename(p), cover)
                    best_v = v
            except Exception:
                continue
    out.append(result("global prediction raster agrees with gold suitability (coverage >= 0.8, Spearman >= 0.5)",
                      best3[0] >= 0.5, value=[round(best3[0], 3), round(best3[2], 3)], threshold=[0.5, 0.8],
                      note=(f"{best3[1]} (rho {best3[0]:.3f}, coverage {best3[2]:.3f})" if best3[0] >= 0.5 else
                            f"best candidate {best_rho[1]} (rho {best_rho[0]:.3f}, coverage {best_rho[2]:.3f})"
                            if best_rho[1] else "no georeferenced raster")
                      + "; coverage over gold cells with centre inside Global_shallow_waters"
                      + (f"; not candidates (kriged explanatory surfaces): {', '.join(sorted(skipped))}" if skipped else "")))

    # 4 the prediction is a model of BOTH explanatory variables (the raw temperature surface
    # saved as "prediction" and a temperature-only model both satisfy check 3).
    nm4 = "prediction is a function of both temperature and salinity (rank R2 gain >= 0.05 each)"
    if best_v is None or best3[0] < 0.5:
        out.append(result(nm4, False, note="no acceptable prediction raster"))
    else:
        try:
            gt = _sample_arr(Tg[0], Tg[1], xs, ys)
            gs_ = _sample_arr(Sg[0], Sg[1], xs, ys)
            r_t, r_s, r_ts = _r2_binned(best_v, gt), _r2_binned(best_v, gs_), _r2_binned(best_v, gt, gs_)
            ok = r_ts >= 0.5 and r_ts - r_t >= 0.05 and r_ts - r_s >= 0.05
            out.append(result(nm4, ok, value={"R2_temp": round(r_t, 3), "R2_sal": round(r_s, 3), "R2_both": round(r_ts, 3)},
                              threshold="R2_both >= 0.5, R2_both - R2_temp >= 0.05, R2_both - R2_sal >= 0.05",
                              note="ranks of the prediction explained by 15-bin (2-D) means of the gold kriged surfaces; "
                                   "gold: 0.733/0.189/0.960"))
        except Exception as e:
            out.append(result(nm4, False, note=repr(e)[:200]))

    code = ctx.code_text()
    low = code.lower()
    # the instruction prescribes "the empirical Bayesian kriging geostatistical
    # method" and "presence-only prediction (MaxEnt)" -> both required (code inspection, comments
    # removed). EBK is ArcGIS-only: any kriging-type interpolation counts (pykrige OK/UK, gstools,
    # GP regression, hand-rolled (EBK-style) kriging = a fitted semivariogram/covariance model and a
    # kriging linear system). IDW, nearest neighbour, RBF/griddata or ML regression fail.
    k_ok, k_note = _kriging_used(code)
    out.append(result("method: kriging-type interpolation of temp/salinity (prescribed: EBK)", k_ok, note=k_note))
    m_ok, m_note = _maxent_used(code)
    out.append(result("method: presence-only MaxEnt-type model (prescribed: MaxEnt)", m_ok, note=m_note))
    out.append(result("spatial CV / thinning considered", any(k in low for k in ("block", "spatial cv", "groupkfold",
                                                                                "thin")), required=False))
    return out


def _is_covariate_surface(v, gt, gs, thr=0.95):
    """A raster that reproduces an explanatory surface (|Spearman| >= 0.95 with the gold kriged
    temperature or salinity at the gold shallow-water cells) is that surface, not a prediction."""
    from scipy.stats import spearmanr
    for g in (gt, gs):
        m = np.isfinite(v) & np.isfinite(g)
        if m.sum() >= 100 and np.std(v[m]) > 0 and abs(float(spearmanr(v[m], g[m]).correlation)) >= thr:
            return True
    return False


def _r2_binned(y, *xs_, nb=15):
    """Share of the variance of rank(y) explained by the means of y in quantile bins of the
    covariate(s) (nb bins per covariate, joint bins for two covariates)."""
    from scipy.stats import rankdata
    m = np.isfinite(y)
    for x in xs_:
        m &= np.isfinite(x)
    if m.sum() < 100:
        return float("nan")
    yy = rankdata(y[m]) / m.sum()
    key = np.zeros(m.sum(), int)
    for x in xs_:
        q = np.quantile(x[m], np.linspace(0, 1, nb + 1)[1:-1])
        key = key * (nb + 1) + np.searchsorted(q, x[m])
    _, inv = np.unique(key, return_inverse=True)
    mu = np.bincount(inv, weights=yy) / np.bincount(inv)
    return float(1 - np.var(yy - mu[inv]) / np.var(yy))


def _presence_subsets(g):
    """The whole set, plus the presence subset if a 0/1 class column exists."""
    yield "", g
    for c in g.columns:
        if c == "geometry" or g[c].dtype.kind not in "iufb":
            continue
        vals = set(np.unique(g[c].dropna().astype(float)))
        if vals and vals <= {0.0, 1.0} and len(vals) == 2:
            yield f"[{c}==1]", g[g[c].astype(float) == 1]


_KRIG = r"(Ordinary|Universal)Kriging\w*\(|GaussianProcessRegressor\(|\w*krig\w*\(|\bebk\w*\(|\w*vario\w*\(|\bKrige\("
_SOLVE = r"linalg\.(solve|lstsq|inv|pinv)\(|cho_solve\(|lu_solve\("
_FIT = r"curve_fit\(|least_squares\(|minimize\("
_MAXLIB = r"maxnet|elapid|MaxentModel|\bmaxent\w*\("
_GLM = r"LogisticRegression\w*\(|\w*logit\w*\(|\w*logistic\w*\(|\bGLM\(|PoissonRegressor\(|SGDClassifier\([^)]*log"
_TREE = r"RandomForest\w*\(|GradientBoosting\w*\(|ExtraTrees\w*\(|DecisionTree\w*\(|XGB\w*\(|LGBM\w*\(|CatBoost\w*\("
_OTHER_CLF = _TREE + r"|\w*Classifier\w*\(|\bSVC\(|\bNuSVC\(|MLP\w*\(|KNeighbors\w*\(|OneClassSVM\("
_BG = r"background|pseudo|\bbg\w*|absence"


def _maxent_functions(code):
    """(name, source without comments) of every function definition whose name contains 'maxent'.
    The code is parsed as a whole, else block by block (blank-line separated cells); unparseable
    parts contribute nothing."""
    import ast
    import re
    trees = []
    try:
        trees = [ast.parse(code)]
    except SyntaxError:
        for blk in re.split(r"\n\s*\n\s*\n", code):
            try:
                trees.append(ast.parse(blk))
            except SyntaxError:
                continue
    out = []
    for t in trees:
        for n in ast.walk(t):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and "maxent" in n.name.lower():
                try:
                    out.append((n.name, ast.unparse(n)))
                except Exception:
                    continue
    return out


def _strip_comments(code):
    import io
    import tokenize
    try:
        return tokenize.untokenize(t for t in tokenize.generate_tokens(io.StringIO(code).readline)
                                   if t.type != tokenize.COMMENT)
    except Exception:
        return "\n".join(l for l in code.splitlines() if not l.lstrip().startswith("#"))


def _kriging_used(code):
    """Kriging-type interpolation in the code (comments removed): a kriging/GP/variogram call, or a
    hand-rolled kriging = a fitted variogram/covariance model (curve_fit / least_squares /
    minimize) together with a solved linear system (the kriging weights)."""
    import re
    c = _strip_comments(code)
    m = re.search(_KRIG, c, re.I)
    if m:
        return True, f"call {m.group(0)}"
    if re.search(_SOLVE, c) and re.search(_FIT, c):
        return True, "hand-rolled kriging: fitted model + linear system solve"
    return False, "no kriging / GP / variogram call and no hand-rolled kriging system (IDW / NN / RBF used?)"


def _maxent_used(code):
    """Presence-only MaxEnt-type model: a MaxEnt implementation (maxnet / elapid / maxent*()),
    or its standard equivalent, a (weighted) logistic / Poisson GLM of presence vs background
    (IWLR). Random forest, boosting or one-class models fail."""
    import re
    c = _strip_comments(code)
    m = re.search(_MAXLIB, c, re.I)
    if m:
        return True, f"MaxEnt implementation: {m.group(0)}"
    # a self-defined *maxent*() function that the pattern '\bmaxent\w*\(' does not match
    # (fit_maxent, run_maxent, ...) counts only by its own body: it must fit a likelihood
    # (minimize / least_squares / logsumexp) or a GLM, and use no other classifier
    # (tree/ensemble, *Classifier, SVC, MLP, KNN). The body is tested, not the whole code, because
    # any variogram curve_fit elsewhere would satisfy a whole-code test.
    for fn, body in _maxent_functions(code):
        if re.search(_OTHER_CLF, body):
            continue
        if re.search(r"minimize\(|least_squares\(|logsumexp|" + _GLM, body):
            return True, f"self-defined MaxEnt {fn}() fitting a likelihood / GLM in its body"
    g = re.search(_GLM, c)
    if g and re.search(_BG, c, re.I):
        return True, f"presence-background GLM (IWLR / MaxEnt-equivalent): {g.group(0)}"
    return False, "no MaxEnt implementation and no presence-vs-background logistic/Poisson GLM"
