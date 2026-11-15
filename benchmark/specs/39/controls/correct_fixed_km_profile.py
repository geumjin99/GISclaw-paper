"""Correct: self-written fixed-bisquare GWR with a bandwidth search over 20-600 km; the AIC / AICc /
BIC / CV profiles are drawn with the bandwidth in km. The script is a sequence of blocks run in
order; blocks that raise an error are wrapped in try/except.
"""


import os; os.makedirs('pred_results', exist_ok=True)

# --- block 1 ---
counties = gpd.read_file('dataset/G_utm.shp')
print(f"Rows: {len(counties)}")
print(f"CRS: {counties.crs.to_string() if counties.crs is not None else None}")
print(f"Geometry types: { {k: int(c) for k, c in counties.geom_type.value_counts().items()} }")
print(f"Bounds (minx, miny, maxx, maxy): {[round(float(_b), 4) for _b in counties.total_bounds]}")
_cols = [c for c in counties.columns if c != counties.geometry.name]
print(f"Columns ({len(_cols)}): {_cols}")
import pandas as _pd
with _pd.option_context('display.max_columns', 30, 'display.width', 200, 'display.max_colwidth', 40):
    print(counties[_cols].head(3).to_string())

# --- block 2 ---
import pandas as pd
df = pd.read_csv('dataset/GData_utm.csv')
print(df.shape)
print(df.head())
print(df.dtypes)
print("dup AreaKey in csv:", df['AreaKey'].duplicated().sum())
print("nulls:\n", df[['PctBach','PctFB','PctBlack','PctRural']].isna().sum())

# --- block 3 (wrapped in try/except) ---
try:
    import pandas as pd
    import geopandas as gpd

    # Join CSV attributes to shapefile by AreaKey
    merged = counties.merge(df[['AreaKey','Latitude','Longitud','TotPop90','PctRural','PctBach','PctEld','PctFB','PctPov','PctBlack']], on='AreaKey', how='left')
    print(merged.shape)
    print("missing after join:", merged[['PctBach','PctFB','PctBlack','PctRural']].isna().sum().sum())
    # check consistency of X,Y between shp and csv
    print("X match:", (merged['X'] - merged['Latitude'].isna()*0).notna().all())
    # use X,Y point coordinates (centroids)
    import numpy as np
    X = merged['X'].values.astype(float)
    Y = merged['Y'].values.astype(float)
    z = merged['PctBach'].values.astype(float)
    print("coords range:", X.min(), X.max(), Y.min(), Y.max())
    print("z range:", z.min(), z.max(), z.mean())
    merged.to_file  # keep
    # save for later
    merged.to_pickle('/tmp/merged.pkl')
    print("OK")
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 4 ---
print(list(merged.columns))
print(merged[['X','Y']].head())

# --- block 5 ---
import numpy as np
# compare csv vs shp values
m = merged.drop(columns=['geometry'])
for col in ['PctBach','PctFB','PctBlack','PctRural']:
    diff = np.abs(m[col+'_x'] - m[col+'_y'])
    print(col, "max diff shp vs csv:", diff.max())
# Use CSV values (per task) - rename
data = merged[['AreaKey','X','Y','PctBach_y','PctFB_y','PctBlack_y','PctRural_y','geometry']].copy()
data = data.rename(columns={'PctBach_y':'PctBach','PctFB_y':'PctFB','PctBlack_y':'PctBlack','PctRural_y':'PctRural'})
data.to_pickle('/tmp/data.pkl')
X = data['X'].values.astype(float)
Y = data['Y'].values.astype(float)
z = data['PctBach'].values.astype(float)
print("n =", len(z))
print("coords:", X.min(), X.max(), Y.min(), Y.max())
print("z:", z.min(), z.max(), round(z.mean(),2))
for c in ['PctFB','PctBlack','PctRural']:
    print(c, data[c].min(), data[c].max(), round(data[c].mean(),2))

# --- block 6 ---
import numpy as np

# Model matrix
X_mat = np.column_stack([np.ones(len(z)), data['PctFB'].values.astype(float),
                         data['PctBlack'].values.astype(float),
                         data['PctRural'].values.astype(float)])
X_pts = np.column_stack([X, Y])

def gwr_fit(X_pts, X_mat, z, b, kernel='bisquare'):
    n, p = X_mat.shape
    beta = np.empty((n, p))
    zhat = np.empty(n)
    trH = 0.0
    for i in range(n):
        d = np.sqrt(((X_pts - X_pts[i])**2).sum(axis=1))
        if kernel == 'bisquare':
            w = np.zeros(n); u = d/b; m = u < 1
            w[m] = (1 - u[m]**2)**2
        else:  # gaussian
            w = np.exp(-0.5*(d/b)**2)
        W = np.diag(w)
        A = X_mat.T @ W @ X_mat + 1e-9*np.eye(p)
        bi = np.linalg.solve(A, X_mat.T @ W @ z)
        beta[i] = bi
        zhat[i] = X_mat[i] @ bi
        trH += X_mat[i] @ np.linalg.solve(A, X_mat[i])
    sse = float(((z - zhat)**2).sum())
    aic = n*np.log(sse/n) + n*np.log(2*np.pi) + n + 2*trH
    return beta, zhat, sse, aic, trH

# quick test
beta, zhat, sse, aic, trH = gwr_fit(X_pts, X_mat, z, b=150000.0)
print("test b=150km: SSE=%.1f AIC=%.1f trH=%.1f" % (sse, aic, trH))
print("local beta means:", beta.mean(axis=0).round(3))

# --- block 7 ---
import numpy as np
bw_list = np.arange(20000, 810000, 10000.0)  # 20 km .. 800 km
results = []
for b in bw_list:
    beta, zhat, sse, aic, trH = gwr_fit(X_pts, X_mat, z, b=b, kernel='bisquare')
    results.append((b, sse, aic, trH))
    if len(results) % 10 == 1:
        print("b=%6.0f km  AIC=%7.2f" % (b/1000, aic))
res = np.array(results)
i_opt = res[:,2].argmin()
b_opt = res[i_opt,0]
print("\nOptimal bandwidth (AIC): %.1f km, AIC=%.2f" % (b_opt/1000, res[i_opt,2]))
print("AIC at ends:", res[0,2], res[-1,2])

# --- block 8 (wrapped in try/except) ---
try:
    import mgwr
    print("mgwr version:", mgwr.__version__ if hasattr(mgwr,'__version__') else 'unknown')
    from mgwr.gwr_selector import GWRSelector
    from mgwr.gwr_model import GWRR
    import inspect
    print(inspect.signature(GWRSelector.__init__))
    print(inspect.signature(GWRSelector.CV))
    print(inspect.signature(GWRR.__init__))
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 9 (wrapped in try/except) ---
try:
    import mgwr
    print(dir(mgwr))
    import mgwr.gwr_model as gm
    print([x for x in dir(gm) if not x.startswith('_')])
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 10 ---
import mgwr
from mgwr import gwr, search, sel_bw, kernels
print("gwr:", [x for x in dir(gwr) if not x.startswith('_')])
print("search:", [x for x in dir(search) if not x.startswith('_')])
print("sel_bw:", [x for x in dir(sel_bw) if not x.startswith('_')])
print("kernels:", [x for x in dir(kernels) if not x.startswith('_')])

# --- block 11 (wrapped in try/except) ---
try:
    import inspect
    from mgwr.gwr import GWR, GWRResults
    from mgwr.sel_bw import Sel_BW, get_AIC, get_AICc, get_CV
    print(inspect.signature(GWR.__init__))
    print(inspect.signature(GWR.fit))
    print(inspect.signature(GWRResults.__init__))
    print(inspect.signature(Sel_BW.__init__))
    print(inspect.signature(Sel_BW.get_best))
    print(inspect.signature(get_CV))
    print(inspect.signature(get_AICc))
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 12 (wrapped in try/except) ---
try:
    import inspect
    from mgwr.sel_bw import Sel_BW
    methods = [x for x in dir(Sel_BW) if not x.startswith('_')]
    print(methods)
    print(inspect.signature(Sel_BW._get_best))
    print(inspect.signature(Sel_BW._get_best_AIC))
    print(inspect.signature(Sel_BW._get_best_CV))
    print(inspect.signature(Sel_BW._get_best_AICc))
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 13 ---
import inspect
from mgwr.sel_bw import Sel_BW
print(inspect.signature(Sel_BW.search))
print(inspect.getdoc(Sel_BW.search))

# --- block 14 (wrapped in try/except) ---
try:
    import numpy as np
    from mgwr.gwr import GWR
    from mgwr.sel_bw import Sel_BW

    coords = X_pts  # (159,2)
    y = z
    X_loc = X_mat   # includes intercept column

    # Try a single fit first to verify mgwr works
    g = GWR(coords, y, X_loc, bw=150000.0, kernel='bisquare', constant=True)
    res_gwr = g.fit()
    print(res_gwr)
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 15 (wrapped in try/except) ---
try:
    from mgwr.gwr import GWR
    g = GWR(coords, y, X_loc, bw=50, kernel='bisquare', constant=True)
    res_gwr = g.fit()
    print(res_gwr)
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 16 ---
import inspect
from mgwr.gwr import GWR
src = inspect.getsource(GWR._local_fit)
print(src[:3000])

# --- block 17 ---
import inspect
from mgwr.gwr import GWR
print(inspect.getsource(GWR._build_wi))
from mgwr.kernels import Kernel
print(inspect.getsource(Kernel.__init__))

# --- block 18 (wrapped in try/except) ---
try:
    from mgwr.gwr import GWR
    g = GWR(coords, y, X_loc, bw=150000.0, kernel='bisquare', constant=True, fixed=True)
    res_gwr = g.fit()
    print(res_gwr)
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 19 ---
import inspect
import mgwr.gwr as gw
src = inspect.getsource(gw)
# find _compute_betas_gwr
i = src.find('def _compute_betas_gwr')
print(src[i:i+1200])

# --- block 20 ---
import inspect
import mgwr.gwr as gw
src = inspect.getsource(gw)
i = src.find('_compute_betas_gwr')
print(src[max(0,i-200):i+1500])

# --- block 21 (wrapped in try/except) ---
try:
    import numpy as np
    X_no_const = X_mat[:, 1:]  # drop the intercept column

    from mgwr.gwr import GWR
    g = GWR(coords, y, X_no_const, bw=150000.0, kernel='bisquare', constant=True, fixed=True)
    res_gwr = g.fit()
    print(res_gwr)
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 22 ---
import inspect
from spglm.iwls import _compute_betas_gwr
print(inspect.getsource(_compute_betas_gwr))

# --- block 23 ---
import inspect
import mgwr.gwr as gw
src = inspect.getsource(gw)
i = src.find('def __init__')
print(src[i:i+2500])

# --- block 24 ---
import numpy as np
y2 = y.reshape(-1,1)
from mgwr.gwr import GWR
g = GWR(coords, y2, X_no_const, bw=150000.0, kernel='bisquare', constant=True, fixed=True)
res_gwr = g.fit()
print(res_gwr)

# --- block 25 (wrapped in try/except) ---
try:
    import numpy as np
    r = res_gwr
    print("predy:", r.predy.mean().round(2), "R2 local:", r.R2loc.round(3), "R2 global:", r.R2glob.round(3))
    print("params shape:", r.params.shape)
    print("params mean:", r.params.mean(axis=0).round(3))
    # compare with manual
    beta_m, zhat_m, sse_m, aic_m, trH_m = gwr_fit(X_pts, X_mat, z, b=150000.0)
    print("manual SSE:", sse_m, "mgwr SSE:", float(((y2 - r.predy)**2).sum()))
    print("manual aic:", aic_m, "mgwr AIC:", r.AIC, "AICc:", r.AICc)
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 26 ---
r = res_gwr
attrs = [x for x in dir(r) if not x.startswith('_')]
print(attrs)

# --- block 27 ---
r = res_gwr
print("predy mean:", r.predy.mean().round(2), "localR2:", r.localR2.round(3), "global R2:", r.R2.round(3))
print("params shape:", r.params.shape, "params mean:", r.params.mean(axis=0).round(3))
print("mgwr AIC:", r.aic, "AICc:", r.aicc, "BIC:", r.bic)
beta_m, zhat_m, sse_m, aic_m, trH_m = gwr_fit(X_pts, X_mat, z, b=150000.0)
print("manual AIC:", aic_m, "manual SSE:", sse_m)
print("mgwr SSE:", float(((y2 - r.predy)**2).sum()))
print("max diff zhat manual vs mgwr:", np.abs(zhat_m - r.predy).max())

# --- block 28 ---
import numpy as np
d = np.abs(zhat_m - r.predy.ravel())
idx = d.argmax()
print("max diff at idx", idx, "z=", z[idx], "manual=", zhat_m[idx], "mgwr=", r.predy.ravel()[idx])
print("num diffs > 1e-6:", (d > 1e-6).sum())
print("r.tr_S:", r.tr_S, "r.tr_STS:", r.tr_STS)
# corrected manual trH: sum_i w_i * x_i A x_i
def gwr_fit2(X_pts, X_mat, z, b, kernel='bisquare'):
    n, p = X_mat.shape
    beta = np.empty((n, p)); zhat = np.empty(n); trH = 0.0
    for i in range(n):
        dist = np.sqrt(((X_pts - X_pts[i])**2).sum(axis=1))
        w = np.zeros(n); u = dist/b
        w[u < 1] = (1 - u[u < 1]**2)**2
        A = X_mat.T @ (X_mat * w[:,None]) 
        A = A + 1e-9*np.eye(p)
        bi = np.linalg.solve(A, X_mat.T @ (w*z))
        beta[i] = bi; zhat[i] = X_mat[i] @ bi
        trH += w[i] * X_mat[i] @ bi  # w_i * x_i' A x_i
    # note: x_i' A x_i = x_i' (A x_i), need A x_i not A(X'w z)
    return beta, zhat, trH
print("need proper A*x_i")

# --- block 29 ---
import inspect
from mgwr.sel_bw import Sel_BW
print(inspect.getsource(Sel_BW.search))

# --- block 30 ---
import inspect
from mgwr.sel_bw import Sel_BW
src = inspect.getsource(Sel_BW)
i = src.find('def _bw')
print(src[i:i+2200])

# --- block 31 ---
import inspect
from mgwr.sel_bw import Sel_BW
src = inspect.getsource(Sel_BW)
i = src.find('def _init_section')
print(src[i:i+1200])
j = src.find('def __init__')
print(src[j:j+1500])

# --- block 32 ---
import inspect
from mgwr import search
print(inspect.getsource(search.equal_interval))
print(inspect.getsource(search.golden_section))

# --- block 33 (wrapped in try/except) ---
try:
    import numpy as np
    from mgwr.sel_bw import Sel_BW
    from mgwr.gwr import GWR

    # Interval search to trace the curve: AICc criterion (recommended)
    sel = Sel_BW(coords, y2, X_no_const, kernel='bisquare', fixed=True, constant=True)
    bw_opt, opt_score, hist = sel.search(search_method='interval', criterion='AICc',
                                         bw_min=10000, bw_max=700000, interval=10000, verbose=False)
    print("AICc-optimal bw:", bw_opt, "score:", opt_score)
    print("num evals:", len(hist))
    curve_aicc = np.array([[b, s[0]] for b, s in hist])
    import numpy as np
    i = curve_aicc[:,1].argmin()
    print("min on grid:", curve_aicc[i])
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 34 ---
import numpy as np
from mgwr.sel_bw import get_AICc, get_AIC
import inspect
print(inspect.getsource(get_AICc))
print(inspect.getsource(get_AIC))

# --- block 35 ---
import numpy as np
from scipy.spatial.distance import pdist
d = pdist(coords)
print("min pairwise dist:", d.min(), "= %.1f km" % (d.min()/1000))
print("max pairwise dist:", d.max(), "= %.1f km" % (d.max()/1000))
# min nearest-neighbor distance
from scipy.spatial import cKDTree
tr = cKDTree(coords)
dd, ii = tr.query(coords, k=2)
nn = dd[:,1]
print("min nearest-neighbor dist:", nn.min(), "= %.1f km" % (nn.min()/1000))

# --- block 36 (wrapped in try/except) ---
try:
    import numpy as np
    from mgwr.gwr import GWR
    from mgwr.sel_bw import get_AIC, get_AICc, get_BIC

    def fit_metric(bw, criterion='AICc', lite=True):
        g = GWR(coords, y2, X_no_const, bw=bw, kernel='bisquare',
                constant=True, fixed=True, n_jobs=-1)
        r = g.fit(lite=lite)
        if criterion=='AIC': return get_AIC(r)
        if criterion=='AICc': return get_AICc(r)
        if criterion=='BIC': return get_BIC(r)

    # quick check of a few values
    for bw in [30000, 100000, 200000, 300000]:
        print(bw/1000, "km  AICc=%.1f" % fit_metric(bw))
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 37 ---
import numpy as np
r = res_gwr
n = len(z)
sse = float(((y2 - r.predy)**2).sum())
trS = r.tr_S
# mgwr llf
print("mgwr llf:", r.llf)
llf_manual = -0.5*n*np.log(2*np.pi) - 0.5*n*np.log(sse/n) - 0.5*n
print("manual llf:", llf_manual)
# check AIC/AICc from these
aic = -2*llf_manual + 2*(trS+1)
aicc = -2*llf_manual + 2*n*(trS+1)/(n-trS-2)
bic = -2*llf_manual + (trS+1)*np.log(n)
print("manual aic:", aic, "mgwr aic:", r.aic)
print("manual aicc:", aicc, "mgwr aicc:", r.aicc)
print("manual bic:", bic, "mgwr bic:", r.bic)

# --- block 38 ---
import numpy as np

def gwr_full(X_pts, X_mat, z, b, ridge=1e-8):
    """Returns beta (n,p), zhat, and diagnostics; trH = tr(S) = sum_i w_i * x_i' A_i^{-1} x_i"""
    n, p = X_mat.shape
    beta = np.empty((n, p)); zhat = np.empty(n); trH = 0.0
    for i in range(n):
        dist = np.sqrt(((X_pts - X_pts[i])**2).sum(axis=1))
        w = np.zeros(n); u = dist/b
        m = u < 1.0
        w[m] = (1.0 - u[m]**2)**2
        Xw = X_mat * w[:, None]
        A = X_mat.T @ Xw + ridge*np.eye(p)
        invA_xi = np.linalg.solve(A, X_mat[i])      # A^{-1} x_i
        bi = X_mat[i] @ invA_xi * 0  # placeholder
        # beta_i = A^{-1} (X' w z)
        bi = np.linalg.solve(A, X_mat.T @ (w*z))
        beta[i] = bi
        zhat[i] = X_mat[i] @ bi
        trH += w[i] * X_mat[i] @ invA_xi
    sse = float(((z - zhat)**2).sum())
    return beta, zhat, sse, trH

def metrics(sse, trH):
    n = len(z); k = trH
    llf = -0.5*n*np.log(2*np.pi) - 0.5*n*np.log(sse/n) - 0.5*n
    aic  = -2*llf + 2*(k+1)
    aicc = -2*llf + 2*n*(k+1)/(n-k-2)
    bic  = -2*llf + (k+1)*np.log(n)
    return aic, aicc, bic

# validate corrected trH vs mgwr at b=150000
beta, zhat, sse, trH = gwr_full(X_pts, X_mat, z, 150000.0)
print("corrected trH:", trH, "mgwr tr_S:", res_gwr.tr_S)
aic,aicc,bic = metrics(sse, trH)
print("aic %.4f vs %.4f | aicc %.4f vs %.4f | bic %.4f vs %.4f" %
      (aic, res_gwr.aic, aicc, res_gwr.aicc, bic, res_gwr.bic))

# --- block 39 ---
import numpy as np
bws = np.arange(20000, 600000+1, 10000.0)
rows = []
for b in bws:
    beta, zhat, sse, trH = gwr_full(X_pts, X_mat, z, b)
    aic, aicc, bic = metrics(sse, trH)
    rows.append((b, aic, aicc, bic, trH))
    print("b=%6.0f km  AIC=%8.2f  AICc=%8.2f  BIC=%8.2f  trS=%.1f" % (b/1000, aic, aicc, bic, trH))
arr = np.array(rows)
print("\nOptimal by AIC:   %.0f km (AIC=%.2f)" % (arr[arr[:,1].argmin(),0]/1000, arr[:,1].min()))
print("Optimal by AICc:  %.0f km (AICc=%.2f)" % (arr[arr[:,2].argmin(),0]/1000, arr[:,2].min()))
print("Optimal by BIC:   %.0f km (BIC=%.2f)" % (arr[arr[:,3].argmin(),0]/1000, arr[:,3].min()))

# --- block 40 ---
import numpy as np

def gwr_full2(X_pts, X_mat, z, b, ridge=0.0):
    n, p = X_mat.shape
    beta = np.empty((n, p)); zhat = np.empty(n); trH = 0.0; h_ii = np.empty(n)
    for i in range(n):
        dist = np.sqrt(((X_pts - X_pts[i])**2).sum(axis=1))
        w = np.zeros(n); u = dist/b
        m = u < 1.0
        w[m] = (1.0 - u[m]**2)**2
        Xw = X_mat * w[:, None]
        A = X_mat.T @ Xw + (ridge*np.eye(p) if ridge else 0.0)
        invA_xi = np.linalg.solve(A, X_mat[i])
        bi = np.linalg.solve(A, X_mat.T @ (w*z))
        beta[i] = bi
        zhat[i] = X_mat[i] @ bi
        h_ii[i] = w[i] * X_mat[i] @ invA_xi
        trH += h_ii[i]
    sse = float(((z - zhat)**2).sum())
    return beta, zhat, sse, trH, h_ii

# find minimum bandwidth where all local windows are full rank (no ridge)
for b in [30000, 35000, 40000, 45000, 50000]:
    try:
        beta, zhat, sse, trH, hii = gwr_full2(X_pts, X_mat, z, b, ridge=0.0)
        print("b=%dkm: OK  SSE=%.1f trS=%.1f min_rank_ok" % (b//1000, sse, trH))
    except np.linalg.LinAlgError:
        print("b=%dkm: SINGULAR" % (b//1000))

# --- block 41 ---
import numpy as np
# LOO CV (PRESS) curve using hat-diagonal
bws2 = np.arange(45000, 600000+1, 10000.0)
rows2 = []
for b in bws2:
    beta, zhat, sse, trH, hii = gwr_full2(X_pts, X_mat, z, b, ridge=0.0)
    e = z - zhat
    press = float(np.sum((e/(1-hii))**2))
    cv = press/len(z)
    aic, aicc, bic = metrics(sse, trH)
    rows2.append((b, aic, aicc, bic, cv, trH))
    print("b=%6.0f km  AIC=%8.2f AICc=%9.2f BIC=%8.2f CV=%.3f trS=%5.1f" %
          (b/1000, aic, aicc, bic, cv, trH))
arr2 = np.array(rows2)
print("\nCV min at %.0f km (CV=%.3f)" % (arr2[arr2[:,4].argmin(),0]/1000, arr2[:,4].min()))
print("AICc min at %.0f km (AICc=%.2f)" % (arr2[arr2[:,2].argmin(),0]/1000, arr2[:,2].min()))
print("BIC min at %.0f km (BIC=%.2f)" % (arr2[arr2[:,3].argmin(),0]/1000, arr2[:,3].min()))

# --- block 42 ---
import numpy as np
bws3 = np.arange(200000, 400000+1, 5000.0)
rows3 = []
for b in bws3:
    beta, zhat, sse, trH, hii = gwr_full2(X_pts, X_mat, z, b, ridge=0.0)
    e = z - zhat
    cv = float(np.sum((e/(1-hii))**2))/len(z)
    aic, aicc, bic = metrics(sse, trH)
    rows3.append((b, aic, aicc, bic, cv))
arr3 = np.array(rows3)
iAIC  = arr3[:,1].argmin(); iAICc = arr3[:,2].argmin()
iBIC  = arr3[:,3].argmin(); iCV   = arr3[:,4].argmin()
print("Fine grid (5km step) optima:")
print("  AIC:  %.0f km, AIC=%.2f"  % (arr3[iAIC,0]/1000, arr3[iAIC,1]))
print("  AICc: %.0f km, AICc=%.2f" % (arr3[iAICc,0]/1000, arr3[iAICc,2]))
print("  BIC:  %.0f km, BIC=%.2f"  % (arr3[iBIC,0]/1000, arr3[iBIC,3]))
print("  CV:   %.0f km, CV=%.3f"   % (arr3[iCV,0]/1000, arr3[iCV,4]))
# save fine curve
np.save('/tmp/curve_fine.npy', arr3)

# --- block 43 ---
import numpy as np
# county outlines for background context
import shapely
outs = []
for i, g in enumerate(data.geometry):
    if g.geom_type == 'MultiPolygon':
        for p in g.geoms:
            outs.append(p.exterior.coords)
    else:
        outs.append(g.exterior.coords)
outline = np.vstack(outs)
print("outline pts:", len(outline))
np.save('/tmp/outline.npy', outline)
import os
print(os.path.exists('pred_results'))

# --- block 44 (wrapped in try/except) ---
try:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    # ---- combine the two bandwidth-search sweeps (coarse + fine) ----
    coarse = np.array([[b, aic, aicc, bic, cv] for (b, aic, aicc, bic, trS, cv) in rows2])
    fine   = arr3[:, :5]  # b, AIC, AICc, BIC, CV

    bws_all = np.concatenate([coarse[:,0], fine[:,0]])
    aic_all = np.concatenate([coarse[:,1], fine[:,1]])
    aicc_all= np.concatenate([coarse[:,2], fine[:,2]])
    bic_all = np.concatenate([coarse[:,3], fine[:,3]])
    cv_all  = np.concatenate([np.nan, np.nan, np.nan, np.nan,
                              coarse[:,4][1:], fine[:,4]])  # coarse[0] had nan CV
    # reorder
    cv_all = np.concatenate([coarse[:,4], fine[:,4]])
    order  = np.argsort(bws_all)
    bws_all, aic_all, aicc_all, bic_all, cv_all = [c[order] for c in
            [bws_all, aic_all, aicc_all, bic_all, cv_all]]

    bkm = bws_all/1000.0

    optAIC  = arr3[iAIC,0];  optAICc = arr3[iAICc,0]
    optBIC  = arr3[iBIC,0];  optCV   = arr3[iCV,0]
    valAIC  = arr3[iAIC,1];  valAICc = arr3[iAICc,2]
    valBIC  = arr3[iBIC,3];  valCV   = arr3[iCV,4]

    fig = plt.figure(figsize=(15,10))
    gs  = fig.add_gridspec(2,2, height_ratios=[1,0.62], width_ratios=[1.15,1],
                           hspace=0.38, wspace=0.30)

    # ============ Panel A: bandwidth search curves (all criteria, one graph) ============
    ax1 = fig.add_subplot(gs[0,:])
    c_aic='#d62728'; c_aicc='#1f77b4'; c_bic='#2ca02c'; c_cv='#9467bd'
    ax1.plot(bkm, aic_all,  '-', color=c_aic,  lw=1.6, label='AIC')
    ax1.plot(bkm, aicc_all, '-', color=c_aicc, lw=1.6, label='AICc')
    ax1.plot(bkm, bic_all,  '-', color=c_bic,  lw=1.6, label='BIC')
    ax1.plot(bkm, cv_all,   '-', color=c_cv,   lw=1.6, label='CV (PRESS / n)')

    for bx,by,col,lab in [
        (optAIC/1000,  valAIC,  c_aic,  'AIC'),
        (optAICc/1000, valAICc, c_aicc, 'AICc'),
        (optBIC/1000,  valBIC,  c_bic,  'BIC'),
        (optCV/1000,   valCV,   c_cv,   'CV')]:
        ax1.axvline(bx, color=col, ls=':', lw=1.0, alpha=0.55)
        ax1.plot([bx],[by], 'o', ms=8, color=col, zorder=5)
        ax1.annotate(lab, (bx,by), textcoords='offset points',
                     xytext=(4,8), fontsize=8.5, color=col)

    ax1.set_title('GWR Bandwidth Search — PctBach ~ PctFB + PctBlack + PctRural (Georgia, n=159)',
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Bandwidth (km)', fontsize=11)
    ax1.set_ylabel('Model selection criterion (lower = better)', fontsize=11)
    ax1.legend(loc='upper center', ncol=4, fontsize=9.5, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(40, 600)

    # ============ Panel B: AICc curve (primary criterion) ============
    ax2 = fig.add_subplot(gs[1,0])
    ax2.plot(bkm, aicc_all, '-', color=c_aicc, lw=2.0)
    ax2.axvline(optAICc/1000, color='red', ls='--', lw=1.2, alpha=0.7)
    ax2.plot([optAICc/1000],[valAICc], 'o', ms=9, color='red', zorder=5)
    ax2.annotate('Optimal AICc bw = %.0f km\nAICc = %.1f' % (optAICc/1000, valAICc),
                 (optAICc/1000, valAICc), textcoords='offset points',
                 xytext=(-40, 22), fontsize=9, color='red',
                 arrowprops=dict(arrowstyle='->', color='red', lw=0.9))
    ax2.set_title('AICc vs Bandwidth (primary criterion)', fontsize=10.5, fontweight='bold')
    ax2.set_xlabel('Bandwidth (km)'); ax2.set_ylabel('AICc')
    ax2.grid(True, alpha=0.3); ax2.set_xlim(40,600)

    # ============ Panel C: model comparison (fit vs actual, GWR @ opt bw) ============
    beta_opt, zhat_opt, sse_opt, trS_opt, hii_opt = gwr_full2(X_pts, X_mat, z, optAICc, ridge=0.0)
    ax3 = fig.add_subplot(gs[1,1])
    sc = ax3.scatter(z, zhat_opt, s=22, c='steelblue', edgecolors='k',
                     linewidths=0.4, alpha=0.85, label='GWR, bw=%.0f km' % (optAICc/1000))
    lims=[0, 40]
    ax3.plot(lims, lims, 'r--', lw=1.4, label='1:1 line')
    ax3.set_xlabel('Observed PctBach (%)', fontsize=10)
    ax3.set_ylabel('GWR Predicted PctBach (%)', fontsize=10)
    ax3.set_title('GWR (optimal bw) — Predicted vs Observed', fontsize=10.5, fontweight='bold')
    ax3.set_xlim(0,40); ax3.set_ylim(0,40); ax3.grid(True, alpha=0.3)
    r2 = 1 - sse_opt/np.sum((z-z.mean())**2)
    ax3.text(0.05,0.95, 'Global R² = %.3f\ntr(S) = %.1f' % (r2, trS_opt),
             transform=ax3.transAxes, va='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    # overall legend / footer
    fig.text(0.5, 0.01,
      'GWR model: PctBach ~ PctFB + PctBlack + PctRural | Bisquare kernel | '
      'Optimal bandwidths: AIC %.0f km, AICc %.0f km, BIC %.0f km, CV %.0f km'
      % (optAIC/1000, optAICc/1000, optBIC/1000, optCV/1000),
      ha='center', fontsize=9, style='italic', color='#444')

    plt.savefig('pred_results/GA_GWR.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved figure.")
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 45 ---
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# combine coarse + fine sweeps  (cols: b, AIC, AICc, BIC, CV)
coarse = np.array([[b, aic, aicc, bic, cv] for (b, aic, aicc, bic, trS, cv) in rows2])
fine   = arr3[:, :5]
bws_all  = np.concatenate([coarse[:,0], fine[:,0]])
aic_all  = np.concatenate([coarse[:,1], fine[:,1]])
aicc_all = np.concatenate([coarse[:,2], fine[:,2]])
bic_all  = np.concatenate([coarse[:,3], fine[:,3]])
cv_all   = np.concatenate([coarse[:,4], fine[:,4]])
order = np.argsort(bws_all)
bws_all, aic_all, aicc_all, bic_all, cv_all = [c[order] for c in
        [bws_all, aic_all, aicc_all, bic_all, cv_all]]
bkm = bws_all/1000.0
print("combined curve:", len(bkm), "points; cv nan count:", np.isnan(cv_all).sum())

optAIC  = arr3[iAIC,0];  optAICc = arr3[iAICc,0]
optBIC  = arr3[iBIC,0];  optCV   = arr3[iCV,0]
valAIC  = arr3[iAIC,1];  valAICc = arr3[iAICc,2]
valBIC  = arr3[iBIC,3];  valCV   = arr3[iCV,4]

c_aic='#d62728'; c_aicc='#1f77b4'; c_bic='#2ca02c'; c_cv='#9467bd'

fig = plt.figure(figsize=(15,10))
gs  = fig.add_gridspec(2,2, height_ratios=[1,0.62], width_ratios=[1.15,1],
                       hspace=0.42, wspace=0.30)

# Panel A: all bandwidth-search curves in one graph
ax1 = fig.add_subplot(gs[0,:])
ax1.plot(bkm, aic_all,  '-', color=c_aic,  lw=1.7, label='AIC')
ax1.plot(bkm, aicc_all, '-', color=c_aicc, lw=1.7, label='AICc')
ax1.plot(bkm, bic_all,  '-', color=c_bic,  lw=1.7, label='BIC')
ax1.plot(bkm, cv_all,   '-', color=c_cv,   lw=1.7, label='CV (PRESS / n)')
for bx,by,col,lab in [(optAIC/1000,  valAIC,  c_aic,  'AIC'),
                      (optAICc/1000, valAICc, c_aicc, 'AICc'),
                      (optBIC/1000,  valBIC,  c_bic,  'BIC'),
                      (optCV/1000,   valCV,   c_cv,   'CV')]:
    ax1.axvline(bx, color=col, ls=':', lw=1.0, alpha=0.5)
    ax1.plot([bx],[by], 'o', ms=8, color=col, zorder=5)
    ax1.annotate(lab, (bx,by), textcoords='offset points',
                 xytext=(4,8), fontsize=8.5, color=col)
ax1.set_title('GWR Bandwidth Search — PctBach ~ PctFB + PctBlack + PctRural (Georgia, n=159)',
              fontsize=12.5, fontweight='bold')
ax1.set_xlabel('Bandwidth (km)', fontsize=11)
ax1.set_ylabel('Model selection criterion (lower = better)', fontsize=11)
ax1.legend(loc='upper center', ncol=4, fontsize=9.5, framealpha=0.9)
ax1.grid(True, alpha=0.3); ax1.set_xlim(40, 600)

# Panel B: AICc curve with optimum marked
ax2 = fig.add_subplot(gs[1,0])
ax2.plot(bkm, aicc_all, '-', color=c_aicc, lw=2.0)
ax2.axvline(optAICc/1000, color='red', ls='--', lw=1.2, alpha=0.7)
ax2.plot([optAICc/1000],[valAICc], 'o', ms=9, color='red', zorder=5)
ax2.annotate('Optimal AICc bw = %.0f km\nAICc = %.1f' % (optAICc/1000, valAICc),
             (optAICc/1000, valAICc), textcoords='offset points',
             xytext=(-55, 24), fontsize=9, color='red',
             arrowprops=dict(arrowstyle='->', color='red', lw=0.9))
ax2.set_title('AICc vs Bandwidth (primary criterion)', fontsize=10.5, fontweight='bold')
ax2.set_xlabel('Bandwidth (km)'); ax2.set_ylabel('AICc')
ax2.grid(True, alpha=0.3); ax2.set_xlim(40,600)

# Panel C: GWR @ optimal AICc bandwidth — predicted vs observed
beta_opt, zhat_opt, sse_opt, trS_opt, hii_opt = gwr_full2(X_pts, X_mat, z, optAICc, ridge=0.0)
ax3 = fig.add_subplot(gs[1,1])
ax3.scatter(z, zhat_opt, s=22, c='steelblue', edgecolors='k',
            linewidths=0.4, alpha=0.85)
ax3.plot([0,40],[0,40], 'r--', lw=1.4, label='1:1 line')
r2 = 1 - sse_opt/np.sum((z-z.mean())**2)
ax3.set_xlabel('Observed PctBach (%)', fontsize=10)
ax3.set_ylabel('GWR Predicted PctBach (%)', fontsize=10)
ax3.set_title('GWR (optimal AICc bandwidth) — Predicted vs Observed',
              fontsize=10.5, fontweight='bold')
ax3.set_xlim(0,40); ax3.set_ylim(0,40); ax3.grid(True, alpha=0.3)
ax3.text(0.04,0.96, 'Global R² = %.3f   |   tr(S) = %.1f\nbandwidth = %.0f km'
        % (r2, trS_opt, optAICc/1000),
         transform=ax3.transAxes, va='top', fontsize=9,
         bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
ax3.legend(loc='lower right', fontsize=9)

fig.text(0.5, 0.005,
  'GWR model: PctBach ~ PctFB + PctBlack + PctRural | Bisquare kernel | '
  'Optimal bandwidths: AIC %.0f km, AICc %.0f km, BIC %.0f km, CV %.0f km'
  % (optAIC/1000, optAICc/1000, optBIC/1000, optCV/1000),
  ha='center', fontsize=9, style='italic', color='#444')

plt.savefig('pred_results/GA_GWR.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved pred_results/GA_GWR.png")

# --- block 46 ---
import matplotlib
matplotlib.use('Agg')
from PIL import Image
import numpy as np
im = Image.open('pred_results/GA_GWR.png')
print("size:", im.size, "mode:", im.mode)
a = np.array(im.convert('L'))
print("pixel value range:", a.min(), a.max())
print("fraction non-white pixels: %.3f" % ((a<250).mean()))
