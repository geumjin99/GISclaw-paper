"""Task 48 - interest-rate hot spots (Gi*) and GWR on LendingClub ZIP3 tracts.

The instruction asks to keep the tracts with at least 30 loans (AcceptedLo >= 30; 808 of 878),
to find interest-rate hot spots (Gi* on AveInteres) on the selected tracts, and to fit a GWR
with a neighbourhood of 22 neighbours.

Required:
  1. selection: a captured/saved table of exactly the 808 tracts with AcceptedLo >= 30
     (">" gives 808 too here, so both readings pass; no selection = 878 fails);
  2. hot spots: a per-tract z-like column/array (centred, negative values present) with
     Pearson >= 0.85 to a Gi(*) z of AveInteres computed by the checker on the 808 tracts under
     a common conceptualisation (ArcGIS default distance band 202.6 km, Incremental Spatial
     Autocorrelation first-peak band 415 km, KNN 8, Queen contiguity; with/without self; binary
     or row-standardised). A candidate that matches a Gi* of AveLoanGra at >= 0.95 and better
     than every Gi* of AveInteres is rejected as hot spots of the wrong field (the two Gi*
     fields correlate at about 0.97, so a looser rule would reject legitimate rate variants).
     Discrete columns (<= 10 distinct values, e.g. Gi_Bin) are a classification and are not z
     candidates. Local Moran's I values or z-scores of the raw rate fail;
  3. GWR with 22 neighbours: a per-tract local slope with Pearson >= 0.95 to the gold
     (AveLoanGra, adaptive bisquare, 22 NN) or to any checker fit of AveInteres with 22 NN
     (Gaussian or bisquare; covariates: any non-empty subset of the five numeric attributes) -
     the instruction does not name the explanatory variable. Only local slopes are compared:
     an intercept-only GWR (local weighted mean) has no slope and is not credited. An optimised
     bandwidth (46 NN, r 0.90), a fixed huge bandwidth or OLS fail.
Reported, not required:
  4. local R2 present and high (gold mean 0.97);
  5. FDR correction mentioned in code.
"""
import numpy as np

from checklib import result, key_values, scorer_error_result

N = 808
_C = {}


def _one_thread():
    """Limit BLAS to one thread for the reference GWR fits: with n_jobs=1 the fits otherwise
    spread over every core of the scoring machine."""
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(1)
    except Exception:
        import contextlib
        return contextlib.nullcontext()


def _moran_z(B, x):
    """Global Moran's I z (normality assumption) with row-standardised weights from binary B."""
    import scipy.sparse as sp
    n = len(x)
    rs = np.asarray(B.sum(axis=1)).ravel()
    W = sp.diags(np.where(rs > 0, 1 / np.where(rs > 0, rs, 1), 0)) @ B
    z = x - x.mean()
    S0 = W.sum()
    I = n / S0 * float(z @ (W @ z)) / float(z @ z)
    WT = W + W.T
    S1 = 0.5 * WT.multiply(WT).sum()
    S2 = float(((np.asarray(W.sum(axis=1)).ravel() + np.asarray(W.sum(axis=0)).ravel()) ** 2).sum())
    EI = -1.0 / (n - 1)
    V = (n * n * S1 - n * S2 + 3 * S0 * S0) / ((n * n - 1) * S0 * S0) - EI * EI
    return (I - EI) / np.sqrt(V)


def _isa_peak_band(tree, cent, x, nbands=10):
    import scipy.sparse as sp
    try:
        d = tree.query(cent, k=2)[0][:, 1]
        start, inc = float(d.max()), float(d.mean())
        n = len(x)
        bands, zs = [], []
        for i in range(nbands):
            b = start + i * inc
            pr = tree.query_pairs(b, output_type="ndarray")
            B = sp.coo_matrix((np.ones(2 * len(pr)), (np.r_[pr[:, 0], pr[:, 1]], np.r_[pr[:, 1], pr[:, 0]])),
                              shape=(n, n)).tocsr()
            bands.append(b); zs.append(_moran_z(B, x))
        for i in range(1, nbands - 1):
            if zs[i] > zs[i - 1] and zs[i] >= zs[i + 1]:
                return bands[i]
        return bands[int(np.argmax(zs))]
    except Exception:
        return None


def _inputs(ctx):
    if _C.get("dir") == ctx.inputs_dir:
        return _C
    import geopandas as gpd
    import scipy.sparse as sp
    from scipy.spatial import cKDTree
    from libpysal.weights import Queen
    allg = gpd.read_file(f"{ctx.inputs_dir}/loan_data.shp")
    s = allg[allg.AcceptedLo >= 30].reset_index(drop=True)
    cent = np.c_[s.centroid.x, s.centroid.y]
    n = len(s)
    tree = cKDTree(cent)
    band = float(tree.query(cent, k=2)[0][:, 1].max())
    mats = {}
    pr = tree.query_pairs(band, output_type="ndarray")
    mats["band"] = sp.coo_matrix((np.ones(2 * len(pr)), (np.r_[pr[:, 0], pr[:, 1]], np.r_[pr[:, 1], pr[:, 0]])),
                                 shape=(n, n)).tocsr()
    # Incremental Spatial Autocorrelation band (ArcGIS workflow for choosing the Gi* distance):
    # 10 bands from the max NN distance in steps of the mean NN distance, global Moran's I of
    # AveInteres (binary, row-standardised, normality z); the first peak of z (the maximum if
    # there is no interior peak). Here 415 km.
    isa = _isa_peak_band(tree, cent, s.AveInteres.to_numpy(float))
    if isa is not None:
        pr = tree.query_pairs(isa, output_type="ndarray")
        mats["isa_peak"] = sp.coo_matrix((np.ones(2 * len(pr)), (np.r_[pr[:, 0], pr[:, 1]], np.r_[pr[:, 1], pr[:, 0]])),
                                         shape=(n, n)).tocsr()
    _, idx = tree.query(cent, k=9)
    mats["knn8"] = sp.coo_matrix((np.ones(n * 8), (np.repeat(np.arange(n), 8), idx[:, 1:].ravel())), shape=(n, n)).tocsr()
    try:
        q = Queen.from_dataframe(s, use_index=False, silence_warnings=True)
        mats["queen"] = q.sparse.tocsr()
    except Exception:
        pass
    x = s.AveInteres.to_numpy(float)
    variants = {}
    for nm, B in mats.items():
        for selfw in (0, 1):
            Bs = B + sp.identity(n) * selfw
            for std in ("bin", "row"):
                M = Bs
                if std == "row":
                    rs = np.asarray(Bs.sum(axis=1)).ravel()
                    M = sp.diags(np.where(rs > 0, 1 / np.where(rs > 0, rs, 1), 0)) @ Bs
                a1 = np.asarray(M.sum(axis=1)).ravel(); a2 = np.asarray(M.multiply(M).sum(axis=1)).ravel()
                with np.errstate(divide="ignore", invalid="ignore"):
                    variants[f"{nm}/self{selfw}/{std}"] = (M @ x - x.mean() * a1) / (x.std() * np.sqrt((n * a2 - a1 ** 2) / (n - 1)))
    gw = {}
    gw_err = []   # scorer-side failures of the reference fits
    try:
        # The instruction does not name the explanatory variable (only the benchmark's script
        # uses AveLoanGra), so any 22-NN adaptive GWR of AveInteres on the layer's attributes
        # with a bisquare or Gaussian kernel is accepted; every slope surface of these fits is
        # a reference.
        from mgwr.gwr import GWR
        y = x.reshape(-1, 1)
        # Every non-empty subset of the five numeric attributes (31 sets), so e.g.
        # AcceptedLo + pHHLoanRat is a reference too. Slopes only - an intercept-only GWR has
        # no slope and is not credited by this check.
        from itertools import combinations
        attrs = ["AveLoanGra", "TotalLoanA", "AcceptedLo", "TOTHH_CY", "pHHLoanRat"]
        sets = [list(c) for k in range(1, len(attrs) + 1) for c in combinations(attrs, k)]
        for cols_ in sets:
            X = s[cols_].to_numpy(float)
            X = (X - X.mean(0)) / X.std(0)
            for kern in ("gaussian", "bisquare"):
                if cols_ == ["AveLoanGra"] and kern == "bisquare":
                    continue  # the gold itself
                try:
                    with _one_thread():
                        pr = GWR(cent, y, X, bw=22, fixed=False, kernel=kern, n_jobs=1).fit().params
                except Exception as e:
                    if not isinstance(e, (np.linalg.LinAlgError, ValueError, ZeroDivisionError)):
                        gw_err.append(repr(e)[:120])   # scorer-side (e.g. joblib pool), not numerical
                    continue
                for j, c in enumerate(cols_):
                    gw[f"{kern}22 {'+'.join(cols_)}: b[{c}]"] = pr[:, j + 1]
    except Exception as e:
        gw_err.append(repr(e)[:120])
    gvar = {}
    try:  # Gi* of the WRONG field (loan grade) to make sure the hot spots are of the interest rate
        xg = s.AveLoanGra.to_numpy(float)
        for nm, B in mats.items():
            for selfw in (0, 1):
                Bs = B + sp.identity(n) * selfw
                for std in ("bin", "row"):
                    M = Bs
                    if std == "row":
                        rs = np.asarray(Bs.sum(axis=1)).ravel()
                        M = sp.diags(np.where(rs > 0, 1 / np.where(rs > 0, rs, 1), 0)) @ Bs
                    a1 = np.asarray(M.sum(axis=1)).ravel(); a2 = np.asarray(M.multiply(M).sum(axis=1)).ravel()
                    with np.errstate(divide="ignore", invalid="ignore"):
                        gvar[f"{nm}/self{selfw}/{std}"] = (M @ xg - xg.mean() * a1) / (xg.std() * np.sqrt((n * a2 - a1 ** 2) / (n - 1)))
    except Exception:
        pass
    _C.update(dir=ctx.inputs_dir, sel=s, variants=variants, gw=gw, gvar=gvar, n_all=len(allg), gw_err=gw_err)
    return _C


def _pearson(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 0.9 * len(b) or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _columns(ctx, sel):
    """(name, vector aligned to sel row order) for numeric per-tract columns / arrays."""
    out = []
    zips = sel.ZIP3.astype(str).to_numpy()
    pos = {z: i for i, z in enumerate(zips)}
    tables = []
    for tag, meta, load in ctx.objects("pred"):
        if meta.get("kind") in ("gdf", "df") and abs(meta.get("rows", 0) - N) <= 2:
            try:
                tables.append((meta["name"] + "@" + tag, load()))
            except Exception:
                pass
        elif meta.get("kind") == "array":
            shp = meta.get("shape") or []
            if shp and shp[0] == N and len(shp) <= 2:
                try:
                    a = load().astype(float)
                except Exception:
                    continue
                if a.ndim == 1:
                    out.append((meta["name"] + "@" + tag, a))
                else:
                    for j in range(min(a.shape[1], 4)):
                        out.append((f"{meta['name']}[:,{j}]@{tag}", a[:, j]))
    import glob, os, geopandas as gpd
    for p in glob.glob(os.path.join(ctx.pred_dir, "**", "*.*"), recursive=True):
        if p.endswith((".shp", ".geojson", ".gpkg", ".csv")):
            try:
                import pandas as pd
                t = pd.read_csv(p) if p.endswith(".csv") else gpd.read_file(p)
                if abs(len(t) - N) <= 2:
                    tables.append((os.path.basename(p), t))
            except Exception:
                pass
    for name, t in tables:
        key = None
        for c in ("ZIP3", "ZIP3_left", "zip3", "ZIP3NUM"):
            if c in t.columns:
                key = t[c].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(3).to_numpy()
                break
        if key is None and len(t) == N:
            key = zips
        if key is None:
            continue
        ix = np.array([pos.get(k, -1) for k in key])
        ok = ix >= 0
        for c in t.columns:
            if c in ("geometry", "ZIP3", "ZIP3NUM", "index") or t[c].dtype.kind not in "fiu":
                continue
            v = np.full(N, np.nan)
            v[ix[ok]] = t[c].to_numpy(float)[ok]
            out.append((f"{c} in {name}", v))
    return out, tables


def checks(ctx):
    out = []
    try:
        I = _inputs(ctx)
    except Exception as e:
        return [result("inputs readable", False, note=repr(e)[:200])]
    sel, variants, gw = I["sel"], I["variants"], I["gw"]
    cols, tables = _columns(ctx, sel)

    # 1 selection
    ok_sel, note = False, "no 808-row table"
    for name, t in tables:
        if len(t) == N and "AcceptedLo" in t.columns and (t.AcceptedLo >= 30).all():
            ok_sel, note = True, name
            break
    out.append(result("selection AcceptedLo >= 30 (808 tracts)", ok_sel, value=note, threshold=N))

    # 2 Gi* hot spots (of the interest rate, not of the loan grade: a candidate must not agree
    # better with the AveLoanGra Gi* than with the AveInteres Gi*)
    best = (-1.0, None)
    grade_hit = None
    for name, v in cols:
        f = v[np.isfinite(v)]
        if len(f) < 0.9 * N or f.std() == 0 or f.min() >= 0 or abs(f.mean()) > 0.5 * f.std():
            continue
        if len(np.unique(f)) <= 10:
            continue  # confidence bins (Gi_Bin -3..3) are a classification, not a z-score
        rr = max([_pearson(v, z) for z in variants.values()] + [-1.0])
        rg = max([_pearson(v, z) for z in I["gvar"].values()] + [-1.0])
        rr = rr if np.isfinite(rr) else -1.0
        # Rate and grade Gi* correlate ~0.97, so a rate Gi* (e.g. KNN-22) can sit 0.003 closer to
        # a grade variant; reject as "grade hot spots" only when the match with a grade Gi* is
        # itself near-exact (>= 0.95) and better than with every rate Gi*.
        if rg > rr and rg >= 0.95:
            if rr >= 0.85:
                grade_hit = f"{name}: r {rr:.3f} with AveInteres Gi*, {rg:.3f} with AveLoanGra Gi*"
            continue
        if rr > best[0]:
            vn = max(variants, key=lambda k: _pearson(v, variants[k]) if np.isfinite(_pearson(v, variants[k])) else -9)
            best = (rr, f"{name} ~ Gi[{vn}]")
    out.append(result("hot spots: Gi* z of AveInteres (Pearson >= 0.85 with a Gi* variant, closer than to Gi* of AveLoanGra)",
                      best[0] >= 0.85, value=round(best[0], 3), threshold=0.85,
                      note=(best[1] or "no z-like per-tract column") + (f"; rejected as grade hot spots: {grade_hit}" if grade_hit else "")))

    # 3 GWR local coefficient
    gold_coef = None
    try:
        import geopandas as gpd
        gg = gpd.read_file(ctx.gold("gwr_result.shp"))
        gold_coef = gg.set_index(gg.ZIP3.astype(str))["C1_AveLoan"].reindex(sel.ZIP3.astype(str)).to_numpy(float)
    except Exception:
        pass
    refs = {"gold bisquare22": gold_coef} if gold_coef is not None else {}
    refs.update(gw)
    bestc = (-1.0, None)
    for name, v in cols:
        for rn, rv in refs.items():
            r = _pearson(v, rv)
            if np.isfinite(r) and r > bestc[0]:
                bestc = (r, f"{name} ~ {rn}")
    gname = "GWR (22 NN) of AveInteres: a local slope matches a 22-NN reference fit (Pearson >= 0.95)"
    gerr = _C.get("gw_err") or []
    if bestc[0] < 0.95 and gerr:
        # a reference fit that crashed in the scorer (e.g. a shut-down joblib pool under
        # parallel scoring) leaves no verdict instead of a failed check
        out.append(scorer_error_result(f"{gname}: {len(gerr)} reference GWR fit(s) failed: {gerr[0]}"[:300]))
    else:
        out.append(result(gname, bestc[0] >= 0.95,
                          value=round(bestc[0], 3), threshold=0.95, note=bestc[1] or "no per-tract coefficient"))

    lr2 = [np.nanmean(v) for n, v in cols if "r2" in n.lower() and np.nanmax(v) <= 1.0001]
    out.append(result("local R2 reported", bool(lr2), value=round(float(max(lr2)), 3) if lr2 else None, required=False))
    code = ctx.code_text()
    out.append(result("FDR correction (code)", any(k in code.lower() for k in ("fdr", "benjamini", "multipletests")),
                      required=False))
    return out
